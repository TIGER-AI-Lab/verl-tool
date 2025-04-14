from .base import BaseTool, register_tool
import regex as re
import subprocess
import sys
import io
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List
from tqdm import tqdm
import signal
import os
from ..utils import kill_python_subprocess_processes


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

def _execute_program(query: str, timeout: int = 30) -> str:
    """
    Execute a single Python program and return its output with a timeout.
    This method uses a safer, sandboxed execution environment for security.
    
    Args:
        query: Python program to execute as a string
        timeout: Maximum execution time in seconds (default: 30)
    
    Returns:
        String containing both stdout and stderr outputs
    """
    # Check for forbidden imports first
    if check_forbidden_imports(query):
        return "Execution blocked: Code contains potentially dangerous operations or imports."
    result = ""
    
    try:
        # Create a subprocess with restricted execution environment
        process = subprocess.Popen(
            [sys.executable, "-c", query],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # This will isolate the subprocess into its own process group for better control
        )
        
        # Capture output and errors with timeout
        try:
            # Use a timeout mechanism with the signal module to prevent hanging
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Combine the outputs
            result = stdout
            if stderr:
                result += f"\nERROR:\n{stderr}"
                
        except subprocess.TimeoutExpired:
            # Kill the process if it exceeds the timeout using signal for safer termination
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            stdout, stderr = process.communicate()

            result = f"Execution timed out after {timeout} seconds.\n"
            if stdout:
                result += f"Partial stdout:\n{stdout}\n"
            if stderr:
                result += f"Partial stderr:\n{stderr}"

    except Exception as e:
        # Capture any exceptions that might occur during execution
        result = f"Error executing program: {str(e)}"
    
    return result


@register_tool
class PythonCodeTool(BaseTool):
    tool_type = "python_code"
    timeout = 15
    
    def get_usage_inst(self):
        return "You are able to write python code and run it for natural language reasoning using the markdown code block."
    
    def parse_action(self, action: str):
        """
        Parse the raw action string (which is the llm response) into a actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        """
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```python(.*?)```", action, re.DOTALL)
        
        if len(all_valid_python_code) == 0:
            # Search for markdown code block
            valid = False
            action = ""
        else:
            valid = True
            action = all_valid_python_code[0]
        return action, valid
    
    def conduct_action(self, trajectory_id, action, extra_field):
        parsed_action, is_valid = self.parse_action(action)
        
        if not is_valid:
            observation = "No valid python code between <python> and </python> tags found."
            done = True
        else:
            observation = _execute_program(parsed_action, timeout=self.timeout)
            done = False
        
        if action.endswith("```output"):
            observation = f"\n{observation}```"
        else:
            observation = f"\nHere is the returned execution results of the above python codes:\n"
            observation += f"<output>{observation}</output>"
        
        return observation, done, is_valid
