from .base import BaseTool, register_tool
import regex as re
import os
import json
import uuid
import textwrap
from typing import Tuple, Dict, Any, Optional, Union, List
import contextlib
from io import StringIO
import sys
import pickle
import base64

# Try to import IPython components
try:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython import get_ipython
    from IPython.core.magic import register_line_magic, register_cell_magic
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

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

class IPythonSession:
    """Manages an IPython session with state persistence"""
    
    def __init__(self, trajectory_id: str, pre_import_lib: bool = False):
        self.trajectory_id = trajectory_id
        self.execution_count = 0
        
        if not IPYTHON_AVAILABLE:
            raise RuntimeError("IPython is not available. Please install it with: pip install ipython")
        
        # Create a new IPython shell instance
        self.shell = TerminalInteractiveShell.instance(config=None)
        
        # Pre-import common libraries if requested
        if pre_import_lib:
            self.shell.run_cell(PRE_IMPORT_LIBS, silent=True)
    
    def execute_cell(self, code: str, stdin: Optional[str] = None) -> Tuple[str, str, bool]:
        """
        Execute code in the IPython session and capture output.
        
        Args:
            code: Python code to execute
            stdin: Optional input (not directly supported in IPython, but we can simulate)
            
        Returns:
            Tuple of (stdout, stderr, has_error)
        """
        self.execution_count += 1
        
        # Set up input simulation if needed
        if stdin:
            # Replace input() calls with predefined responses
            # This is a simple approach - you might want to make it more sophisticated
            stdin_lines = stdin.strip().split('\n')
            stdin_iterator = iter(stdin_lines)
            
            def mock_input(prompt=''):
                try:
                    value = next(stdin_iterator)
                    print(f"{prompt}{value}")  # Echo the input like real input()
                    return value
                except StopIteration:
                    return ''
            
            # Temporarily replace input function
            original_input = self.shell.user_ns.get('input', __builtins__['input'])
            self.shell.user_ns['input'] = mock_input
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Execute the code
            result = self.shell.run_cell(code, store_history=True)
            
            # Get captured output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # Check for errors
            has_error = not result.success
            if result.error_before_exec:
                stderr_output += str(result.error_before_exec) + '\n'
            if result.error_in_exec:
                stderr_output += str(result.error_in_exec) + '\n'
            
            # If there's a result to display, add it to stdout
            if result.result is not None:
                stdout_output += str(result.result) + '\n'
                
        except Exception as e:
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue() + str(e) + '\n'
            has_error = True
        
        finally:
            # Restore original streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Restore original input function if it was replaced
            if stdin:
                if 'input' in self.shell.user_ns:
                    if original_input != self.shell.user_ns['input']:
                        self.shell.user_ns['input'] = original_input
        
        return stdout_output, stderr_output, has_error
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the IPython session for persistence.
        
        Returns:
            Dictionary containing the session state
        """
        # Get user namespace (variables)
        user_vars = {}
        for name, value in self.shell.user_ns.items():
            # Skip private variables and built-ins
            if not name.startswith('_') and name not in __builtins__:
                try:
                    # Try to pickle the value to ensure it's serializable
                    pickled = pickle.dumps(value)
                    user_vars[name] = {
                        'value': base64.b64encode(pickled).decode('utf-8'),
                        'type': str(type(value).__name__)
                    }
                except:
                    # If we can't pickle it, store a string representation
                    user_vars[name] = {
                        'value': str(value),
                        'type': str(type(value).__name__),
                        'unpicklable': True
                    }
        
        return {
            'trajectory_id': self.trajectory_id,
            'execution_count': self.execution_count,
            'user_vars': user_vars,
            'history': list(self.shell.history_manager.get_range())
        }
    
    def restore_state(self, state: Dict[str, Any]):
        """
        Restore the IPython session from a saved state.
        
        Args:
            state: Dictionary containing the session state
        """
        self.execution_count = state.get('execution_count', 0)
        
        # Restore user variables
        user_vars = state.get('user_vars', {})
        for name, var_info in user_vars.items():
            try:
                if var_info.get('unpicklable', False):
                    # Skip unpicklable variables
                    continue
                
                # Restore pickled value
                pickled_data = base64.b64decode(var_info['value'].encode('utf-8'))
                value = pickle.loads(pickled_data)
                self.shell.user_ns[name] = value
            except Exception as e:
                # If restoration fails, skip this variable
                print(f"Warning: Could not restore variable '{name}': {e}")
    
    def reset(self):
        """Reset the IPython session"""
        self.shell.reset(new_session=True)
        self.execution_count = 0

def execute_python_ipython(code: Union[str, List[str]], trajectory_id: str, timeout: int = TIMEOUT, 
                          stdin: Optional[str] = None, pre_import_lib: bool = False, 
                          session_cache: Dict = None) -> Tuple[str, str, bool, IPythonSession]:
    """
    Execute Python code using IPython with session persistence.
    
    Args:
        code: Python code string or list of code blocks to execute
        trajectory_id: Unique identifier for the session
        timeout: Execution timeout (not directly implemented in IPython)
        stdin: Optional input to provide to the executed code
        pre_import_lib: Whether to pre-import common libraries
        session_cache: Cache to store IPython sessions
        
    Returns:
        Tuple containing (stdout, stderr, has_error, session)
    """
    if session_cache is None:
        session_cache = {}
    
    # Check for forbidden imports first
    code_str = code if isinstance(code, str) else '\n'.join(code)
    if check_forbidden_imports(code_str):
        return "", "Execution blocked: Code contains potentially dangerous operations or imports.", True, None
    
    # Get or create IPython session
    if trajectory_id in session_cache:
        session = session_cache[trajectory_id]
    else:
        session = IPythonSession(trajectory_id, pre_import_lib)
        session_cache[trajectory_id] = session
    
    # Execute the code
    if isinstance(code, list):
        # Execute multiple code blocks
        combined_stdout = ""
        combined_stderr = ""
        has_any_error = False
        
        for block in code:
            stdout, stderr, has_error = session.execute_cell(block, stdin)
            combined_stdout += stdout
            combined_stderr += stderr
            if has_error:
                has_any_error = True
                break  # Stop on first error
        
        return combined_stdout, combined_stderr, has_any_error, session
    else:
        # Execute single code block
        stdout, stderr, has_error = session.execute_cell(code, stdin)
        return stdout, stderr, has_error, session

@register_tool
class IPythonTool(BaseTool):
    tool_type = "ipython_code"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>", "<tool_call>"]
    enable_history_code_execution = False
    done_without_error = False
    pre_import_lib = False
    
    def __init__(self):
        super().__init__()
        self.ipython_sessions = {}  # Cache for IPython sessions
    
    def get_usage_inst(self):
        return "You are able to write and execute Python code using IPython with persistent state across executions."
    
    def has_env(self, trajectory_id):
        """
        Check if the environment for the given trajectory_id exists
        """
        return trajectory_id in self.env_cache
    
    def load_env(self, trajectory_id):
        """
        Load the environment for the given trajectory_id
        """
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {
                    "turns": 0,
                },
                "previous_obs": [],
                "ipython_state": None,  # Store IPython session state
            }
        
        # Restore IPython session if state exists
        if env.get("ipython_state") and trajectory_id not in self.ipython_sessions:
            try:
                session = IPythonSession(trajectory_id, self.pre_import_lib)
                session.restore_state(env["ipython_state"])
                self.ipython_sessions[trajectory_id] = session
            except Exception as e:
                print(f"Warning: Could not restore IPython session: {e}")
        
        return env
    
    def save_env(self, trajectory_id, env):
        """
        Save the environment for the given trajectory_id
        """
        # Save IPython session state
        if trajectory_id in self.ipython_sessions:
            try:
                session = self.ipython_sessions[trajectory_id]
                env["ipython_state"] = session.get_state()
            except Exception as e:
                print(f"Warning: Could not save IPython session state: {e}")
        
        self.env_cache[trajectory_id] = env
    
    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        """
        Update the environment for the given trajectory_id
        """
        env["metadata"]["turns"] += 1
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })
    
    def delete_env(self, trajectory_id):
        """
        Delete the environment for the given trajectory_id
        """
        if trajectory_id in self.env_cache:
            del self.env_cache[trajectory_id]
        
        if trajectory_id in self.ipython_sessions:
            del self.ipython_sessions[trajectory_id]
    
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
        
        if len(all_valid_python_code) == 0:
            return "", False
        
        # Use all the code blocks
        parsed_code = "\n".join([code.strip() for code in all_valid_python_code])
        
        return parsed_code, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed action using IPython.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
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
            
            # Determine what code to execute
            if self.enable_history_code_execution:
                previous_parsed_code = [obs["action"] for obs in env["previous_obs"] if obs["is_valid"]]
                code_to_execute = previous_parsed_code + [parsed_action]
            else:
                code_to_execute = parsed_action
            
            # Execute using IPython
            stdout, stderr, has_error, session = execute_python_ipython(
                code_to_execute, 
                trajectory_id, 
                self.timeout, 
                stdin, 
                self.pre_import_lib, 
                self.ipython_sessions
            )
            
            execution_result = stdout + "\n" + stderr
            execution_result = execution_result.lstrip(' \n')
            observation = execution_result
            
            # Format the observation based on the action type
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