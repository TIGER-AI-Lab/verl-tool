# Firejail is a local sandbox and strikes the best balance of reliability, scalability, and security
# https://github.com/netblue30/firejail
# sudo add-apt-repository ppa:deki/firejail
# sudo apt-get update
# sudo apt-get install firejail firejail-profiles
import os
import subprocess
import timeout_decorator

TIMEOUT = 5

@timeout_decorator.timeout(TIMEOUT, use_signals=False)
def _code_exec_firejail(code, stdin: str = None):
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

def code_exec_firejail(code, stdin: str = None):
    try:
        return _code_exec_firejail(code, stdin)
    except Exception as e:
        return f"Exception: {e}"

if __name__ == "__main__":
    print(code_exec_firejail("print('Hello, World!')"))

