from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import signal
import sys
from typing import Tuple, Dict, Any, Optional
from ..utils import kill_python_subprocess_processes

import random

# Timeout for code execution in seconds
TIMEOUT = 5

def check_forbidden_imports(code: str) -> bool:
    """
    Checks if the code contains imports of potentially dangerous packages.
    
    Args:
        code: Python code string to analyze
        
    Returns:
        Boolean indicating if the code contains forbidden imports
    """
    # List of potentially dangerous modules that could affect the host system
    forbidden_modules = [
        'subprocess', 'multiprocessing', 'threading',
        'socket', 'psutil', 'resource', 'ctypes'
    ]
    
    # Simple string-based check for import statements
    for module in forbidden_modules:
        if f"import {module}" in code or f"from {module}" in code:
            return True
    
    # Check for os.system, os.popen, and similar dangerous calls
    dangerous_patterns = [
        "os.system", "os.popen", "os.spawn", "os.fork", 
        "os.exec", "sys.exit", "os._exit", "os.kill"
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code:
            return True
    
    return False
    
def execute_python_in_firejail(code: str, timeout: int=TIMEOUT, stdin: Optional[str] = None) -> str:
    """
    Execute Python code in a Firejail sandbox with a timeout.
    
    Args:
        code: Python code string to execute
        stdin: Optional input to provide to the executed code
        
    Returns:
        String containing execution output or error message
    """
    # Check for forbidden imports first
    if check_forbidden_imports(code):
        return "Execution blocked: Code contains potentially dangerous operations or imports."
    
    # Create a minimal environment instead of copying everything
    original_env = os.environ.copy()
    env = {}
    
    # Core system variables
    essential_vars = [
        "PATH", "HOME", "USER", "SHELL", 
        "LANG", "LC_ALL", "LC_CTYPE", "TERM",
        # Python-specific
        "PYTHONIOENCODING", "PYTHONUNBUFFERED", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
        # Runtime optimization
        "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        # Temp directories
        "TMPDIR", "TEMP", "TMP",
        # Display if needed
        "DISPLAY", "XAUTHORITY"
    ]
    
    # Copy only essential variables if they exist
    for var in essential_vars:
        if var in original_env:
            env[var] = original_env[var]
    
    # Explicitly set optimization variables
    env["OPENBLAS_NUM_THREADS"] = "1"
    
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]
    
    # Build the firejail command with resource limits
    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=2m",  # Limit file size
        "--rlimit-as=1096m",
    ]
    command.extend(["python3", "-c", code])
    
    try:
        result = subprocess.run(
            command,
            input=stdin if stdin else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            timeout=timeout
        )
        
        stdout = result.stdout
        stderr = result.stderr.strip()
        
        result = f"{stdout}\nError:\n{stderr}" if stderr else stdout
        if result:
            result = result.strip()
    except subprocess.TimeoutExpired:
        result = f"Execution timed out after {timeout} seconds.\n"
    return result

@register_tool
class FirejailPythonCodeTool(BaseTool):
    tool_type = "firejail_python_code"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>"]
    
    def get_usage_inst(self):
        return "You are able to write and execute Python code securely inside a Firejail sandbox."
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing Python code
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        # Try to find Python code in various formats
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
        
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```python(.*?)```", action, re.DOTALL)
        
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```(.*?)```", action, re.DOTALL)
        
        if len(all_valid_python_code) == 0:
            return "", False
        
        # Use the first code block found (we could extend this to support multiple blocks)
        parsed_code = all_valid_python_code[0].strip()
        
        return parsed_code, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed action in a Firejail sandbox.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_action, is_valid = self.parse_action(action)
        
        heuristic_sentences = {
            "empty": [
                "Hmm, no output at all. Since nothing broke, I'll draft a few edge-case inputs to see if the function ever emits data:",
                "The run is silent—no errors, no text. I’ll broaden the test suite and watch for any change in behaviour:",
                "Blank output suggests the happy path passed; I’ll now probe unusual parameters to confirm:",
                "Nothing was printed, yet the call completed. Let me invent some stress tests to check hidden branches:",
                "Zero output but no crash: time to craft randomized cases and observe whether output appears under different conditions:"
            ],
            "timeout": [
                "The call never returned—likely stuck in a heavy loop. I’ll scan the control flow and think about where it could stall.",
                "Timeout reached. That hints at an expensive section or possible infinite recursion; I’ll trace the algorithm paths and rethink them.",
                "Execution exceeded the limit. I’ll review the data size assumptions and consider simpler test inputs first.",
                "Ran out of time—maybe I’m missing a termination condition. I’ll inspect loops and add safeguards before retrying.",
                "Process froze long enough to trigger a timeout; I’ll look for bottlenecks and refactor the slow part."
            ],
            "error": [
                "It crashed with error message: `<<ERR>>`. I’ll read the stack trace, locate the failing line, and reason out a fix.",
                "Error message captured: `<<ERR>>`. I’ll match it against the code section and adjust the logic.",
                "The run ended in an exception (<<ERR>>). I’ll rethink the assumptions that lead to that problem.",
                "Received `<<ERR>>`. I’ll double-check code logic and try to fix the root cause.",
                "Execution failed: `<<ERR>>`. I’ll analyze the traceback and locate the error source.",
            ],
            "success": [
                "Output obtained: `<<OUT>>`. I’ll cross-check it with expectations and decide if more cases are needed.",
                "Call succeeded with result `<<OUT>>`; next I’ll test boundary values to be thorough.",
                "I see `<<OUT>>` as the response. I’ll verify its plausibility and consider edge cases.",
                "Successful run—output reads `<<OUT>>`. I’ll think about corner cases the current test didn’t cover.",
                "Result `<<OUT>>` returned without errors. I’ll validate it and add more tests if needed.",
            ]
        }


        
        if not is_valid:
            # observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            observation = "No valid Python code found. Please provide code in ```python...``` code blocks."
            done = True
            valid = False
        else:
            
            # Extract stdin if provided in extra_field
            stdin = extra_field.get("stdin", None) if extra_field else None
            
            # Execute the code
            execution_result = execute_python_in_firejail(parsed_action, self.timeout, stdin)
            
            # use heuristics to reformat the results, add sentences to encourage models continue thinking
            # case: empty (correctly runned or the test case does not have output, need to check)
            if execution_result == "":
                # randomly select a sentence from the empty heuristic sentences
                idx = random.randint(0, len(heuristic_sentences["empty"]) - 1)
                observation = heuristic_sentences["empty"][idx]
            
            # case: execution timed out, need to check if the code is correct
            elif "Execution timed out" in execution_result:
                # observation = execution_result
                idx = random.randint(0, len(heuristic_sentences["timeout"]) - 1)
                observation = heuristic_sentences["timeout"][idx]
            
            # case: execution ends with error, need to look back and fix the bug
            elif "ERROR:" in execution_result:
                # observation = f"Execution completed with errors:\n{execution_result}"
                idx = random.randint(0, len(heuristic_sentences["error"]) - 1)
                observation = heuristic_sentences["error"][idx].replace("<<ERR>>", execution_result)            
            # case: generated output without error, need to check the code's output
            else:
                # observation = f"Execution result:\n{execution_result}"
                idx = random.randint(0, len(heuristic_sentences["success"]) - 1)
                observation = heuristic_sentences["success"][idx].replace("<<OUT>>", execution_result)                
            done = False
            valid = True
        
        if action.endswith("```output"):
            observation = observation + "\n```"
        if action.endswith("<output>"):
            observation = observation + "\n</output>"
        
        observation = "\n" + observation + "\n"
        return observation, done, valid
        