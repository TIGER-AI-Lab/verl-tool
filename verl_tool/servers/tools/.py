from .base import BaseTool, register_tool
import os
import subprocess
import timeout_decorator
import subprocess
import sys

TIMEOUT = 10
OBS_START = '```output'
OBS_END = '\n```\n'

def extract_program(result: str, last_only=True):
    """
    extract the program after "```python", and before "```"
    """
    program = ''
    start = False
    for line in result.split('\n'):
        if line.startswith('```python') or line.endswith('```python'):
            if last_only:
                program = ''  # only extract the last program
            else:
                program += '\n# ========\n'
            start = True
        elif line.startswith('```'):
            start = False
        elif start:
            program += line + '\n'
    if start:
        # the code is incomplete
        program = ''
    return program

@register_tool
class PythonCodeTool(BaseTool):
    tool_type = "python_code"
    timeout = 10
    
    def get_usage_inst(self):
        return "You are able to run the python code in your responses that are enclosed in ```python ``` tags. The output of the code (stdout and stderr) will be returned in ```output ``` tags."
    
    # TODO: check and test this
    def parse_action(self, action:str):
        """
        Parse the raw action string (which is the llm response) into a actual action and it's contents
        """
        program = extract_program(action)
        if program:
            valid = True
            action = program
        else:
            valid = False
            action = ""
        return action, valid
    
    # TODO: check here how to use timeout parameter
    @timeout_decorator.timeout(TIMEOUT, use_signals=False)
    def _code_exec_firejail(self, code, stdin: str = None):
        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "1"
        if "PYTHONPATH" in env:
            del env["PYTHONPATH"] # avoid importing wrong stuff

        # Build the firejail command with resource limits and cleanup options
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
        result = subprocess.run(command,
                                input=stdin.encode() if stdin else None,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=env,
                                check=False)
        stdout = result.stdout.decode()
        stderr = result.stderr.decode().strip()

        if result.returncode == 0:
            return stdout
        return f"{stdout}\nERROR:\n{stderr}"

    def code_exec_firejail(self, code, stdin: str = None):
        try:
            return self._code_exec_firejail(code, stdin)
        except Exception as e:
            return f"Exception: {e}"
    
    def conduct_action(self, trajectory_id, action, extra_field):
        action, is_valid = self.parse_action(action)
        if not is_valid:
            observation = "No valid python code between ```python ``` found."
            done = True
        else:
            observation = self.code_exec_firejail(action)
            done = False
        observation = f'{OBS_START}\n{observation}{OBS_END}'
        return observation, done, is_valid
    