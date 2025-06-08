"""
add-apt-repository ppa:deki/firejail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install firejail firejail-profiles
"""
from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import signal
import sys
import json
import uuid
import hashlib
import textwrap
from typing import Tuple, Dict, Any, Optional, Union, List

import random

# Timeout for code execution in seconds
TIMEOUT = 5
PRE_IMPORT_LIBS = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n\n"

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

def wrap_code_blocks(code: Union[str, List[str]]) -> str:
    """
    Wraps the provided code blocks with try-except to handle exceptions including syntax errors.
    For previous codes, redirect stdout and stderr to null and export defined functions and variables.
    
    Args:
        code: List of code strings to wrap
        
    Returns:
        Wrapped code string
    """
    wrapped_code = ""
    
    # Convert single string to list for consistent handling
    if isinstance(code, str):
        code = [code]
    
    # Import needed at the top
    wrapped_code += "import sys, os, io, ast\n\n"
    
    # Add the safe_exec_with_exports function
    wrapped_code += """
def parse_and_exec_salvageable(code_string):
    # Split the code into lines
    lines = code_string.splitlines()
    
    # Try to execute code incrementally, line by line or in blocks
    current_block = ""
    local_namespace = {}
    
    for line in lines:
        # Add the current line to our accumulating block
        if current_block:
            current_block += "\\n" + line
        else:
            current_block = line
            
        # Skip empty lines or comments
        if not line.strip() or line.strip().startswith('#'):
            continue
            
        # Try to parse the current block to check for syntax
        try:
            ast.parse(current_block)
            
            # If it parses successfully, try to execute it
            try:
                # Create a new local namespace for this execution
                exec(current_block, globals(), local_namespace)
                
                # Clear the block after successful execution
                current_block = ""
            except Exception as e:
                print(f"Runtime error in block: {e}")
                current_block = ""  # Reset the block after a runtime error
                
        except SyntaxError:
            # If we have a syntax error in the accumulated block,
            # don't reset yet - we might need more lines to complete the syntax
            pass
    
    return local_namespace
"""
    
    for i, block in enumerate(code):
        is_last_block = i == len(code) - 1
        
        # For all blocks except the last, use safe_exec_with_exports
        if not is_last_block:
            wrapped_block = (
                f"\n# Code block {i+1} (previous)\n"
                f"original_stdout, original_stderr = sys.stdout, sys.stderr\n"
                f"sys.stdout, sys.stderr = io.StringIO(), io.StringIO()\n"
                f"try:\n"
                f"    exported_vars = parse_and_exec_salvageable('''{block}''')\n"
                f"finally:\n"
                f"    sys.stdout, sys.stderr = original_stdout, original_stderr\n\n"
                f"    for name, value in exported_vars.items():\n"
                f"        globals()[name] = value\n"
            )
        else:
            # For the last (current) block, just include the code directly
            wrapped_block = f"\n# Code block {i+1} (current)\n{block}\n"
        
        wrapped_code += wrapped_block
    
    return wrapped_code
    
def execute_python_in_firejail(code: Union[str, List[str]], timeout: int=TIMEOUT, stdin: Optional[str] = None, python_path: str = None, pre_import_lib: bool = False) -> Tuple[str, bool]:
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
        return "Execution blocked: Code contains potentially dangerous operations or imports.", True
    
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
    # set cwd to be a temp dir
    # cwd = os.path.join(os.getcwd(), "/tmp/firejail")
    cwd = "/tmp/firejail"
    if not os.path.exists(cwd):
        os.makedirs(cwd, exist_ok=True)
    # write code to a temp file
    # file_name = f"code_{hashlib.md5(code.encode()).hexdigest()}.py"
    file_name = f"code_{uuid.uuid4().hex}.py"
    file_path = os.path.join(cwd, file_name)
    code = wrap_code_blocks(code)
    with open(file_path, "w") as f:
        f.write(code)
    if pre_import_lib:
        code = PRE_IMPORT_LIBS + code
    # command.extend(["python3", "-c", code])
    # command.extend(["python3", file_path])
    if not python_path:
        python_path = "python3"
    else:
        assert os.path.exists(python_path), f"Python path {python_path} does not exist."
    command.extend([python_path, file_path])
    has_error = False
    try:
        # Execute the command
        result = subprocess.run(
            command,
            input=stdin if stdin else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        
        stdout = result.stdout
        stderr = result.stderr.strip()
        if stderr:
            has_error = True
        
        result = f"{stdout}\nError:\n{stderr}" if stderr else stdout
        if result:
            result = result.strip()
    except subprocess.TimeoutExpired:
        has_error = True
        result = f"Execution timed out after {timeout} seconds.\n"
    # Clean up the temporary file
    try:
        os.remove(file_path)
    except Exception as e:
        pass
    return result, has_error

@register_tool
class FirejailPythonCodeWithTestTool(BaseTool):
    tool_type = "firejail_python_code_with_test"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>", "<tool_call>"]
    enable_history_code_execution = True
    enable_mannual_reflection = False
    force_run_test_cases = True
    done_without_error = False
    python_path = None
    pre_import_lib = False
    
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
            all_valid_python_code = re.findall(r"```\n?python(.*?)```", action, re.DOTALL)
        
        # if not all_valid_python_code:
        #     all_valid_python_code = re.findall(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)

        if len(all_valid_python_code) == 0:
            return "", False
        
        # # Use the first code block found (we could extend this to support multiple blocks)
        # parsed_code = all_valid_python_code[0].strip()
        
        # use all the code blocks
        parsed_code = "\n".join([code.strip() for code in all_valid_python_code])
        
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
        env = self.load_env(trajectory_id)
        
        heuristic_sentences = {
            "empty": [
                "Hmm, no output at all. Since nothing broke, I'll draft a few edge-case inputs to see if the function ever emits data:",
                "The run is silent—no errors, no text. I'll broaden the test suite and watch for any change in behaviour:",
                "Blank output suggests the happy path passed; I'll now probe unusual parameters to confirm:",
                "Nothing was printed, yet the call completed. Let me invent some stress tests to check hidden branches:",
                "Zero output but no crash: time to craft randomized cases and observe whether output appears under different conditions:"
            ], # normally when there is just a function of code, that's the coding question most of the time
            "timeout": [
                "The call never returned—likely stuck in a heavy loop. I'll scan the control flow and think about where it could stall.",
                "Timeout reached. That hints at an expensive section or possible infinite recursion; I'll trace the algorithm paths and rethink them.",
                "Execution exceeded the limit. I'll review the data size assumptions and consider simpler test inputs first.",
                "Ran out of time—maybe I'm missing a termination condition. I'll inspect loops and add safeguards before retrying.",
                "Process froze long enough to trigger a timeout; I'll look for bottlenecks and refactor the slow part."
            ],
            "error": [
                "The code seems crashed. I'll read the stack trace, locate the failing line, and reason out a fix. Based on the error, I think",
                "Wait, I got some errors of my previous code, I'll double-check code logic and try to fix the root cause. Based on the error, I think",
                "Oops, the code crashed. I'll analyze the error message and see if I can fix it. Based on the error, I think",
            ],
            "success": [
                "Code executed successfully! I'll cross-check it with expectations and decide if more cases are needed. Based on the output, I think",
                "Now I have the execution result. After checking the output, I think",
                "Good, the code run successfully without error. However, are there any more corner cases that didn't cover?",
                "It looks like the code is working! But does this match my expected output?",
            ]
        }
        
        if not is_valid:
            # observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            observation = ""
            execution_result = ""
            done = False
            valid = False
        else:
            # Extract stdin if provided in extra_field
            stdin = extra_field.get("stdin", "") if extra_field else None
            
            test_input = re.findall(r"```input\n(.*?)\n```", action, re.DOTALL)
            if len(test_input) > 0:
                stdin = test_input[0].strip()
            
            new_code = parsed_action # 
            if self.enable_history_code_execution:
                # Execute the code
                # previous_parsed_code = [obs["action"] for obs in env["previous_obs"] if obs["is_valid"] and "error:" not in obs["observation"].lower()]
                # code_to_execute = "\n".join(previous_parsed_code) + "\n" + parsed_action
                # execution_result, has_error = execute_python_in_firejail(code_to_execute, self.timeout, stdin, self.python_path, self.pre_import_lib)
                
                previous_parsed_code = [obs["action"] for obs in env["previous_obs"]]
                code_to_execute = wrap_code_blocks(previous_parsed_code + [parsed_action])
                execution_result, has_error = execute_python_in_firejail(code_to_execute, self.timeout, stdin, self.python_path, self.pre_import_lib)
                
                # print("------")
                # print(code_to_execute)
                # print("------")
                # print("------")
                # print(execution_result)
                # print("------")
                # print("----") 
                # print([obs["observation"] for obs in env["previous_obs"]])
                # print("----")
                # for previous_obs in env["previous_obs"]:
                #     if previous_obs["is_valid"] and "error:" not in previous_obs["observation"].lower():
                #         execution_result = execution_result.replace(previous_obs["observation"], "", 1)
            else:
                code_to_execute = parsed_action
                execution_result, has_error = execute_python_in_firejail(code_to_execute, self.timeout, stdin, self.python_path, self.pre_import_lib)
                
            execution_result = execution_result.lstrip(' \n')
                        
            # Format the result
            if "Execution timed out" in execution_result:
                observation = execution_result
            else:
                observation = f"{execution_result}"
            
            if action.endswith("```output"):
                observation = "\n" + observation + "\n```\n"
            elif action.endswith("</tool_call>"):
                observation = "\n```output\n" + observation + "\n```\n"
            elif action.endswith("<output>"):
                observation = "\n" + observation + "\n</output>\n"
            elif action.endswith("</python>") or "</python>" in action:
                observation = "\n<output>\n" + observation + "\n</output>\n"
            elif "<|calling system for feedback|>" in action:
                if "```python" in action:
                    observation = "\n```output\n" + observation + "\n```\n"
                elif "<python>" in action:
                    observation = "\n<output>\n" + observation + "\n</output>\n"
                else:
                    observation = "\n" + observation + "\n"
            elif action.strip(' \n').endswith("```") or "```python" in action:
                if action.count("```") % 2 == 0:
                    observation = "\n```output\n" + observation + "\n```\n"
                else:
                    observation = "output\n" + observation + "\n```\n"
            else:
                observation = "\n" + observation + "\n"
            
            if not has_error and self.force_run_test_cases:
                test_cases = extra_field.get("public_tests", None) if extra_field else None
                if test_cases:
                    if isinstance(test_cases, str):
                        test_cases = json.loads(test_cases)[:10] # debug
                    # execute the public test cases
                    if isinstance(test_cases, list):
                        # list of assert
                        test_cases_code = "\n".join(test_cases)
                        if test_cases_code in new_code:
                            # already tested, pass
                            test_result = ""
                        else:
                            test_codes = code_to_execute + "\n" + test_cases_code
                            test_execution_result, has_error = execute_python_in_firejail(test_codes, self.timeout, stdin, self.python_path, self.pre_import_lib)
                            test_execution_result = test_execution_result.replace(execution_result, "", 1)
                            test_result = f"Testing the above code with the following test cases:\n```python\n{test_cases_code}\n```\n\nTest result:\n```output\n{test_execution_result}\n```\n"
                            if not test_execution_result:
                                test_result += "\nAll public test cases passed!\nSo the final code solution that is ready to submit is (copy the code above):\n```python\n"
                            elif has_error:
                                test_result += "\nOops, it seems one of the test cases failed and that means the code still has logical bugs."
                            else:
                                test_result += "\nI get the execution result and does the execution result match my expectations? Let me analyze one by one."
                    elif isinstance(test_cases, dict):
                        assert "inputs" in test_cases and "outputs" in test_cases, f"Invalid test cases format: {test_cases.keys()}"
                        test_result = ""
                        test_cases_passed = True
                        for i in range(len(test_cases["inputs"])):
                            input_case = test_cases["inputs"][i]
                            output_case = test_cases["outputs"][i]
                            test_codes = code_to_execute
                            test_stdin = (stdin + "\n" + input_case)
                            test_execution_result, has_error = execute_python_in_firejail(test_codes, self.timeout, test_stdin, self.python_path, self.pre_import_lib)
                            test_execution_result = test_execution_result.replace(execution_result, "", 1)
                            test_case_output_match = test_execution_result == output_case
                            if not test_case_output_match:
                                test_cases_passed = False
                            test_result += f"Testing the above code with the following test case:\n```python\n{test_codes}\n```\n\nTest input:\n```input\n{input_case}\n```\n\nExpected output:\n```expected_output\n{output_case}\n```\n\nTest result:\n```output\n{test_execution_result}\n```\nMatching expected output: {test_case_output_match}\n"
                        if test_cases_passed:
                            test_result += "All public test cases passed!\n"
                    else:
                        raise ValueError(f"Invalid test cases format: {test_cases}")
                    observation = observation + "\n" + test_result
            if self.enable_mannual_reflection:
                # case: empty (correctly runned or the test case does not have output, need to check)
                if execution_result == "":
                    # randomly select a sentence from the empty heuristic sentences
                    idx = random.randint(0, len(heuristic_sentences["empty"]) - 1)
                    observation += heuristic_sentences["empty"][idx]
                # case: execution timed out, need to check if the code is correct
                elif "execution timed out" in observation.lower():
                    # observation = execution_result
                    idx = random.randint(0, len(heuristic_sentences["timeout"]) - 1)
                    observation += heuristic_sentences["timeout"][idx]
                # case: execution ends with error, need to look back and fix the bug
                elif has_error:
                    # observation = f"Execution completed with errors:\n{execution_result}"
                    idx = random.randint(0, len(heuristic_sentences["error"]) - 1)
                    observation += heuristic_sentences["error"][idx]     
                # case: generated output without error, need to check the code's output
                else:
                    # observation = f"Execution result:\n{execution_result}"
                    idx = random.randint(0, len(heuristic_sentences["success"]) - 1)
                    observation += heuristic_sentences["success"][idx]

            if self.done_without_error:
                if has_error:
                    done = False
                else:
                    done = True
            else: 
                done = False
            valid = True
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
        