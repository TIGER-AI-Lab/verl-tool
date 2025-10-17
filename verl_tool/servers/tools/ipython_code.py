import ray
from .base import BaseTool, register_tool
import regex as re
import os
import uuid
from typing import Tuple, Dict, Any, Optional, Union, List
from .utils.ipython_kernel import call_python_script_with_ipython, get_kernel_stats, cleanup_all_kernels, remove_kernel

# Timeout for code execution in seconds
TIMEOUT = 10

# Pre-imported libraries for convenience
PRE_IMPORT_LIBS = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.setrecursionlimit(6*10**5)

"""

@register_tool
class IPythonTool(BaseTool):
    """
    Tool for executing Python code using IPython kernels with persistent state.
    Each trajectory maintains its own IPython kernel for stateful execution.
    """
    
    tool_type = "ipython_code"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>", "<tool_call>"]
    enable_history_code_execution = False
    done_without_error = False
    pre_import_lib = False
    
    def get_usage_inst(self):
        return "You are able to write and execute Python code using IPython kernels with persistent state across executions."
    
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
                    "kernel_initialized": False,
                },
                "previous_obs": [],
                "code_history": [],  # Track executed code for this trajectory
            }
        return env
    
    def save_env(self, trajectory_id, env):
        """
        Save the environment for the given trajectory_id
        """
        self.env_cache[trajectory_id] = env
    
    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        """
        Update the environment for the given trajectory_id
        """
        env["metadata"]["turns"] += 1
        if is_valid and action:
            env["code_history"].append(action)
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })
    
    def delete_env(self, trajectory_id):
        """
        Delete the environment for the given trajectory_id and clean up kernel
        """
        if trajectory_id in self.env_cache:
            # Clean up the IPython kernel for this trajectory
            try:
                remove_kernel(trajectory_id)
            except Exception as e:
                pass
                # print(f"Warning: Failed to remove kernel for {trajectory_id}: {e}")
            try:
                del self.env_cache[trajectory_id]
            except Exception as e:
                pass
                # print(f"Warning: Failed to delete env for {trajectory_id}: {e}")
                
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        
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

    def postprocess_observation(
        self,
        action: str, 
        observation: Union[str, Dict[str, Any]], 
        output_tag: str = "result"
    ) -> Union[str, Dict[str, Any]]:
        """
        Add output tags to the observation based on action type.
        
        Args:
            action: The action string that determines formatting
            observation: Raw observation (string or dict with 'observation' key)
            output_tag: Type of output tag to use ('output', 'result', 'response', etc.)
        
        Returns:
            Formatted observation with appropriate tags
        """
        # Extract raw observation
        if isinstance(observation, str):
            raw_observation = observation
        elif isinstance(observation, dict):
            raw_observation = observation.get("obs", "")
        else:
            raise ValueError("Observation must be a string or a dictionary with an 'observation' field.")
        
        # Determine format based on action patterns
        if any(pattern in action for pattern in ["```output", "```python"]):
            # Handle code block patterns
            if action.count("```") % 2 == 0:  # Even number of backticks (closed block)
                formatted_obs = f"\n```{output_tag}\n{raw_observation}\n```\n"
            else:  # Odd number (unclosed block)
                formatted_obs = f"\n{raw_observation}\n```\n"
        elif any(pattern in action for pattern in ["</tool_call>"]):
            # Tool call patterns - prefer code blocks, give in <tool_response> format
            formatted_obs = f"\n<tool_response>\n```{output_tag}\n{raw_observation}\n```\n</tool_response>\n"
        elif any(pattern in action for pattern in [f"<{output_tag}>", f"</{output_tag}>", "</python>"]):
            # XML-style tag patterns
            if action.strip(" \n").endswith(f"<{output_tag}>"):
                formatted_obs = f"\n{raw_observation}\n</{output_tag}>\n"
            else:
                formatted_obs = f"\n<{output_tag}>\n{raw_observation}\n</{output_tag}>\n"
        else:
            # Default: simple newline wrapping
            formatted_obs = f"\n<{output_tag}>\n{raw_observation}\n</{output_tag}>\n"
        
        # Return in same format as input
        if isinstance(observation, str):
            return formatted_obs
        else:
            result = observation.copy()
            result['obs'] = formatted_obs
            return result
    
    def _initialize_kernel(self, trajectory_id: str, env: Dict) -> None:
        """
        Initialize the IPython kernel with pre-imported libraries if needed.
        """
        if not env["metadata"].get("kernel_initialized", False) and self.pre_import_lib:
            # Execute pre-import libraries silently
            output, success = call_python_script_with_ipython(
                trajectory_id, 
                PRE_IMPORT_LIBS, 
                timeout=self.timeout
            )
            env["metadata"]["kernel_initialized"] = True
            if not success:
                print(f"Warning: Failed to initialize kernel with pre-imports: {output}")
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed action using IPython kernel.
        
        Args:
            trajectory_id: ID for tracking the action and maintaining kernel state
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
            # Initialize kernel if needed
            self._initialize_kernel(trajectory_id, env)
            
            # Handle stdin if provided
            stdin_content = ""
            if extra_field and "stdin" in extra_field:
                stdin_content = extra_field.get("stdin", "")
            
            # Check for input in the action itself
            test_input = re.findall(r"```input\n(.*?)\n```", action, re.DOTALL)
            if len(test_input) > 0:
                stdin_content = test_input[0].strip()
            
            # Prepare code for execution
            code_to_execute = parsed_action
            
            # If stdin is needed, inject it into the code
            if stdin_content:
                # Wrap the code to mock stdin
                code_with_stdin = f"""
import sys
from io import StringIO

# Mock stdin with provided input
_original_stdin = sys.stdin
sys.stdin = StringIO('''{stdin_content}''')

try:
    # Execute the actual code
{chr(10).join('    ' + line for line in parsed_action.split(chr(10)))}
finally:
    # Restore original stdin
    sys.stdin = _original_stdin
"""
                code_to_execute = code_with_stdin
            
            # Execute code using IPython kernel
            execution_result, success = call_python_script_with_ipython(
                trajectory_id,
                code_to_execute,
                timeout=self.timeout
            )
            
            # Clean up the execution result
            execution_result = execution_result.strip(' \n')
            observation = execution_result
            
            # Apply postprocessing
            # print(f"Execution result for trajectory {trajectory_id}:\n{execution_result}")
            observation = self.postprocess_observation(action, observation)
            
            # Determine if done based on error status
            if self.done_without_error:
                done = success
            else:
                done = False
            
            valid = True
        
        # Update environment with execution details
        self.update_env(
            trajectory_id, 
            env, 
            parsed_action, 
            is_valid, 
            extra_field, 
            execution_result,
            success=valid and not done if self.done_without_error else valid
        )
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    def get_kernel_statistics(self) -> Dict:
        """
        Get statistics about currently active kernels.
        """
        return get_kernel_stats()
    
    def cleanup_all(self) -> None:
        """
        Clean up all kernels and environments.
        """
        # Clean up all trajectories
        trajectory_ids = list(self.env_cache.keys())
        for trajectory_id in trajectory_ids:
            self.delete_env(trajectory_id)
        
        # Clean up all kernels
        cleanup_all_kernels()