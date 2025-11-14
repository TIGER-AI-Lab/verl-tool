import IPython
import threading
import time
import gc
import psutil
import sys
import os
import site
import asyncio
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import logging
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from pathlib import Path
import ray
from ray.exceptions import GetTimeoutError
from traitlets.config import Config

import signal
import threading
from contextlib import contextmanager


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
site_packages_dir = site.getsitepackages()[0]
file_abs_path = os.path.abspath(__file__)
# ==== Ray + Actor-based kernel management =====================================

# How many actors (kernels) we cache
MAX_ACTORS = 1024

# LRU-ish cache of request_id -> KernelActor
_actor_cache: "OrderedDict[str, ray.actor.ActorHandle]" = OrderedDict()
_actor_lock = threading.RLock()


def _ensure_ray_initialized():
    """Initialize Ray once (in local mode by default)."""
    if not ray.is_initialized():
        # You can customize ray.init(...) as needed
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        logger.info("Ray initialized")
_ensure_ray_initialized()


class TimeoutException(Exception):
    """Raised when execution times out"""
    pass

@contextmanager
def time_limit(seconds):
    """Context manager that raises TimeoutException after specified seconds"""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {seconds} seconds")
    
    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
@ray.remote
class KernelActor:
    """
    A Ray actor that owns a single IPython shell in its own process.

    This gives:
    - Per-request session state (variables persist across executes)
    - No stdout/stderr conflicts with other actors
    - True CPU parallelism (separate processes)
    """

    def __init__(self):

        c = Config()
        c.HistoryManager.enabled = False
        shell = TerminalInteractiveShell(colors='NoColor', config=c)

        # Disable history to save memory
        shell.history_manager.enabled = False
        shell.cache_size = 1000

        # Optional: non-interactive backend for matplotlib
        try:    
            import matplotlib
            matplotlib.use('Agg')
        except Exception:
            pass

        self.shell = shell

    def execute(self, script: str, max_output_size: int = 100 * 1024, timeout: int = 120) -> Tuple[str, bool]:
        """
        Run a script in this actor's IPython shell, capturing stdout/stderr.
        
        Timeout is enforced within the actor using SIGALRM (Unix/Linux only).
        """
        if not isinstance(script, str) or not script.strip():
            return "Error: script must be a non-empty string", False

        stdout_capture = StringIO()
        stderr_capture = StringIO()
        success = True

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                with time_limit(timeout):
                    result = self.shell.run_cell(script, silent=False)

                    # Check for errors in IPython's execution result
                    if result is not None:
                        if getattr(result, "error_before_exec", None):
                            print(result.error_before_exec, file=sys.stderr)
                            success = False
                        if getattr(result, "error_in_exec", None):
                            print(result.error_in_exec, file=sys.stderr)
                            success = False

                        # If there's a Python-level result, print it to stdout
                        if getattr(result, "result", None) is not None:
                            print(result.result)
            except TimeoutException as e:
                print(f"\n{str(e)}", file=sys.stderr)
                success = False
            except Exception as e:
                print(e, file=sys.stderr)
                success = False

        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # Replace site-packages path if desired (mirror your old behavior)
        stdout_output = stdout_output.replace(site_packages_dir, "/lib/python3.10/site-packages")
        stderr_output = stderr_output.replace(site_packages_dir, "/lib/python3.10/site-packages")
        
        if "TimeoutException" in stderr_output:
            print(f"Filtered traceback for timeout in {file_abs_path}")
            # On timeout, we remove the lines in the traceback that contain this file's path
            filtered_lines = stderr_output.splitlines()
            idx = None
            for i, line in enumerate(filtered_lines):
                if file_abs_path in line:
                    idx = i
                    break
            if idx is not None:
                # Keep lines before the first occurrence of our file path
                filtered_lines = filtered_lines[:idx]
            stderr_output = "\n".join(filtered_lines)
        if "TimeoutException" in stdout_output:
            filtered_lines = stdout_output.splitlines()
            idx = None
            for i, line in enumerate(filtered_lines):
                if file_abs_path in line:
                    idx = i
                    break
            if idx is not None:
                # Keep lines before the first occurrence of our file path
                filtered_lines = filtered_lines[:idx]
            stdout_output = "\n".join(filtered_lines)

        if success:
            output = stdout_output.strip()
        else:
            # On error, include both stdout and stderr
            combined = stdout_output
            if stderr_output:
                if combined and not combined.endswith("\n"):
                    combined += "\n"
                combined += stderr_output
            output = combined.strip()

        return output[:max_output_size], success

    def reset(self):
        """Optional: clear namespace if you want."""
        try:
            if hasattr(self.shell, "user_ns"):
                self.shell.user_ns.clear()
            if hasattr(self.shell, "user_global_ns"):
                self.shell.user_global_ns.clear()
            if hasattr(self.shell, "reset"):
                self.shell.reset(new_session=False)
        except Exception as e:
            print(f"Error resetting shell: {e}", file=sys.stderr)
    
    def get_namespace_info(self) -> dict:
        """Return information about the current namespace."""
        return {
            'user_variables': list(self.shell.user_ns.keys()),
            'user_globals': list(self.shell.user_global_ns.keys()) if hasattr(self.shell, 'user_global_ns') else [],
            'builtin_count': len([k for k in self.shell.user_ns.keys() if k.startswith('_')])
        }

    def get_variable_names(self) -> list:
        """Return all non-builtin variable names."""
        return [k for k in self.shell.user_ns.keys() if not k.startswith('_')]


def _get_actor(request_id: str) -> "ray.actor.ActorHandle":
    """LRU get/create a Ray actor for a given request_id."""
    with _actor_lock:
        if request_id in _actor_cache:
            actor = _actor_cache[request_id]
            # actor = _actor_cache.pop(request_id)
            # Move to end (most recently used)
            # _actor_cache[request_id] = actor
            return actor

        # Create new actor
        actor = KernelActor.remote()
        _actor_cache[request_id] = actor
        logger.info(f"Created new KernelActor for request_id={request_id}")

        # LRU eviction if too many actors
        if len(_actor_cache) > MAX_ACTORS:
            old_id, old_actor = _actor_cache.popitem(last=False)
            logger.info(f"Evicting KernelActor for request_id={old_id}")
            try:
                ray.kill(old_actor)
            except Exception as e:
                logger.warning(f"Error killing old actor {old_id}: {e}")

        return actor


async def call_python_script_with_ipython_async(
    request_id: str,
    script: str,
    timeout: int = 120,
    max_output_size: int = 100 * 1024,
) -> Tuple[str, bool]:
    """
    Execute a Python script using a Ray-backed IPython actor and return (output, success).

    - request_id: identifies which actor/kernel/session to use.
    - script: code to run.
    - timeout: max time in seconds for this call; enforced via ray.get(..., timeout).
    - max_output_size: truncate output to this many bytes.
    """
    if not isinstance(request_id, str) or not request_id.strip():
        return "Error: request_id must be a non-empty string", False

    # In case model uses stdin.buffer.read (not supported)
    script = script.replace("sys.stdin.buffer.read()", "sys.stdin.read()")
    if not isinstance(script, str) or not script.strip():
        return "Error: script must be a non-empty string", False

    actor = _get_actor(request_id)

    loop = asyncio.get_event_loop()

    def _run():
        # Run the actor execute() with a timeout enforced by ray.get
        future = actor.execute.remote(script, max_output_size, timeout)
        try:
            return ray.get(future, timeout=timeout+120)  # Allow some buffer time
        except GetTimeoutError:
            # Kill the actor on timeout and evict from cache
            logger.warning(f"Execution timeout for request_id={request_id} after {timeout} seconds")
            try:
                ray.kill(actor)
            except Exception as e:
                logger.warning(f"Error killing actor on timeout: {e}")
            with _actor_lock:
                _actor_cache.pop(request_id, None)
            return f"Execution timeout after {timeout} seconds", False

    output, success = await loop.run_in_executor(None, _run)
    
    return output[:max_output_size], success

def call_python_script_with_ipython(
    request_id: str,
    script: str,
    timeout: int = 120,
    max_output_size: int = 100 * 1024,
) -> Tuple[str, bool]:
    """Synchronous wrapper around the async version."""
    output, success = asyncio.run(
        call_python_script_with_ipython_async(
            request_id, script, timeout, max_output_size
        )
    )
    # actor = _get_actor(request_id)
    # name_space_info = actor.get_namespace_info.remote()
    # variable_names = actor.get_variable_names.remote()
    # name_space_info = ray.get(name_space_info)
    # variable_names = ray.get(variable_names)
    # debug_file = Path(f"debug_ipython/{request_id}.txt")
    # debug_file.parent.mkdir(parents=True, exist_ok=True)
    # with debug_file.open("a+") as f:
    #     f.write(f"\n\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
    #     f.write(f"----"* 20 + "\n")
    #     f.write(f"kernel stats: {get_kernel_stats()}\n")
    #     f.write(f"Actor: {actor}\n")
    #     f.write(f"Script executed:\n{script}\n")
    #     f.write(f"----"* 20 + "\n")
    #     f.write(f"Output:\n{output}\n")
    #     f.write(f"Success: {success}\n")
    #     f.write(f"Namespace info: {name_space_info}\n")
    #     f.write(f"Variable names: {variable_names}\n")
    #     f.write(f"----"* 20 + "\n")
    return output[:max_output_size], success

def get_kernel_stats() -> Dict:
    """Get simple stats about current actor usage."""
    with _actor_lock:
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        return {
            "active_kernels": len(_actor_cache),
            "max_kernels": MAX_ACTORS,
            "memory_usage_mb": memory_mb,
            "kernel_ids": list(_actor_cache.keys()),
        }


def cleanup_all_kernels() -> None:
    """Kill all actors and clear the cache."""
    with _actor_lock:
        ids = list(_actor_cache.keys())
        for rid in ids:
            actor = _actor_cache.pop(rid, None)
            if actor is not None:
                try:
                    ray.kill(actor)
                except Exception as e:
                    logger.warning(f"Error killing actor {rid}: {e}")
        gc.collect()
        logger.info("Cleaned up all KernelActors")


def remove_kernel(request_id: str) -> None:
    """Remove a specific kernel by request_id."""
    with _actor_lock:
        actor = _actor_cache.pop(request_id, None)
    if actor is not None:
        try:
            ray.kill(actor)
        except Exception as e:
            logger.warning(f"Error killing actor {request_id}: {e}")
        gc.collect()
        logger.info(f"Removed KernelActor for request_id={request_id}")


# Example usage and configuration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simple hello world
    result, success = call_python_script_with_ipython("test1", "print('Hello World')")
    print(f"[test1] Result: {result!r}, Success: {success}")

    # Define a variable and reuse the same request_id to check state persistence
    call_python_script_with_ipython("test1", "x = 41")
    result, success = call_python_script_with_ipython("test1", "x + 1")
    print(f"[test1] State test: {result!r}, Success: {success}")

    # Timeout test: this should kill the actor for 'test2'
    result, success = call_python_script_with_ipython(
        "test2",
        "a=1\nimport time\nwhile True: time.sleep(1)",
        timeout=2,
    )
    print(f"[test2] Timeout test - Result: {result!r}, Success: {success}")
    
    # Timeout test: this should kill the actor for 'test2'
    result, success = call_python_script_with_ipython(
        "test2",
        "print(a)",
        timeout=2,
    )
    print(f"[test3] Timeout test - Result: {result!r}, Success: {success}")

    # Stats
    stats = get_kernel_stats()
    print(f"Stats: {stats}")

    # Cleanup
    cleanup_all_kernels()
