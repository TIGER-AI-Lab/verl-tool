from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import signal
import sys
from typing import Tuple, Dict, Any, Optional
from ..utils import kill_python_subprocess_processes

# Timeout for code execution in seconds
TIMEOUT = 10

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

def execute_python_in_firejail(code: str, stdin: Optional[str] = None) -> str:
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
    
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]  # avoid importing wrong stuff
    
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
        "--rlimit-as=4096m",
    ]
    command.extend(["python3", "-c", code])
    
    try:
        # Start process in its own process group for better control
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if stdin else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=False,
            preexec_fn=os.setsid  # Creates a new process group
        )
        
        # If stdin provided, write it to the process
        if stdin:
            process.stdin.write(stdin.encode())
            process.stdin.close()
        
        # Capture output and errors with timeout
        try:
            stdout, stderr = process.communicate(timeout=TIMEOUT)
            stdout_str = stdout.decode()
            stderr_str = stderr.decode().strip()
            
            if process.returncode == 0:
                return stdout_str
            
            return f"{stdout_str}\nERROR:\n{stderr_str}"
            
        except subprocess.TimeoutExpired:
            # Kill the entire process group if timeout occurs
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            # Try to get any partial output
            try:
                stdout, stderr = process.communicate(timeout=0.5)
                stdout_str = stdout.decode() if stdout else ""
                stderr_str = stderr.decode().strip() if stderr else ""
            except:
                stdout_str = ""
                stderr_str = ""
            
            # Force kill if still running
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            
            result = f"Execution timed out after {TIMEOUT} seconds.\n"
            if stdout_str:
                result += f"Partial stdout:\n{stdout_str}\n"
            if stderr_str:
                result += f"Partial stderr:\n{stderr_str}"
            
            return result
            
    except Exception as e:
        return f"Error executing program: {str(e)}"


@register_tool
class FirejailPythonCodeTool(BaseTool):
    tool_type = "firejail_python_code"
    timeout = TIMEOUT
    
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
        
        if not is_valid:
            observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            return observation, True, False
        
        # Extract stdin if provided in extra_field
        stdin = extra_field.get("stdin", None) if extra_field else None
        
        # Execute the code using firejail
        execution_result = execute_python_in_firejail(parsed_action, stdin)
        
        # Format the result
        if "Execution timed out" in execution_result:
            observation = execution_result
        elif "ERROR:" in execution_result:
            observation = f"Execution completed with errors:\n{execution_result}"
        else:
            observation = f"Execution result:\n{execution_result}"
            
        return observation, False, True
        