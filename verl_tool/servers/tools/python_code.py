from .base import BaseTool, register_tool
import regex as re
import threading
import subprocess
import sys
import io
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List
from tqdm import tqdm


def _execute_program(query: str, timeout: int = 30) -> str:
    """
    Execute a single Python program and return its output with a timeout.
    
    Args:
        query: Python program to execute as a string
        timeout: Maximum execution time in seconds (default: 30)
    
    Returns:
        String containing both stdout and stderr outputs
    """
    result = ""
    
    try:
        # Create a separate process for execution using subprocess
        process = subprocess.Popen(
            [sys.executable, "-c", query],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Capture output and errors with timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Combine the outputs
            result = stdout
            if stderr:
                result += f"\nERROR:\n{stderr}"
                
        except subprocess.TimeoutExpired:
            # Kill the process if it exceeds the timeout
            process.kill()
            
            # Clean up any remaining output
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
    timeout = 10
    
    def get_usage_inst(self):
        return "You are able to run the python code in your responses that are enclosed in <python> and </python> tags. The output of the code (stdout and stderr) will be returned between <output> and </output> tags."
    
    def parse_action(self, action:str):
        """
        Parse the raw action string (which is the llm response) into a actual action and it's contents
        """
        all_valid_python_code = re.findall(r"<python>((.|\n)*?)</python>", action, re.DOTALL)
        if len(all_valid_python_code) == 0:
            valid = False
            action = ""
        else:
            valid = True
            action = all_valid_python_code[0][0]
        return action, valid
    
    def conduct_action(self, trajectory_id, action, extra_field):
        action, is_valid = self.parse_action(action)
        if not is_valid:
            observation = "No valid python code between <python> and </python> tags found."
            done = True
        else:
            observation = _execute_program(action, timeout=self.timeout)
            done = False
        
        # if the output is empty, encourage the model to contibue generating more test cases
        if observation == "":
            assembled_observation = "\nThe execution results of the above python codes is empty. I will think about how to test the code differently.\n" 
        else:
            assembled_observation = f"\nHere is the returned execution results of the above python codes:\n<output>{observation}</output>\n"
        
        
        
        return assembled_observation, done, is_valid
    