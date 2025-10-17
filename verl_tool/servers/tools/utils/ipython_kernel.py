#!/usr/bin/env python3
# Option A: In-worker soft timeout (SIGALRM) + reply ID correlation
# Works best on POSIX. On Windows, falls back to manager-side SIGINT.

import IPython
import multiprocessing as mp
import threading
import time
import gc
import psutil
import sys
import os
import signal
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Any
import logging
import pickle
import queue
import platform

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
_IS_POSIX = (os.name == "posix" and platform.system() != "Darwin") or (os.name == "posix")

class IPythonWorker:
    """Worker process that owns an IPython kernel."""
    def __init__(self, conn):
        self.conn = conn
        self.shell = None
        self.setup_shell()

    def setup_shell(self):
        self.shell = IPython.InteractiveShell.instance()
        # Disable history to save memory
        self.shell.history_manager.enabled = False
        # Limit cache sizes
        self.shell.cache_size = 100
        # Non-interactive matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pass

    def run(self):
        while True:
            try:
                if not self.conn.poll(1.0):
                    continue
                msg = self.conn.recv()
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Worker error receiving message: {e}")
                continue

            if not isinstance(msg, dict):
                continue

            cmd = msg.get("cmd")
            if cmd == "exec":
                self.handle_exec(msg)
            elif cmd == "get_var":
                self.handle_get_var(msg)
            elif cmd == "set_var":
                self.handle_set_var(msg)
            elif cmd == "reset":
                self.handle_reset()
            elif cmd == "shutdown":
                break

        self.conn.close()

    # ---------- Option A core: in-worker timeout ----------
    def handle_exec(self, msg):
        """Execute code in the IPython shell with in-worker soft timeout."""
        code = msg.get("code", "")
        exec_id = msg.get("id")
        stdin_content = msg.get("stdin", None)
        timeout = msg.get("timeout", None)  # seconds (int/float)

        stdout_buf = StringIO()
        stderr_buf = StringIO()

        original_stdin = sys.stdin
        if stdin_content is not None:
            mock_stdin = StringIO(stdin_content)
            sys.stdin = mock_stdin

        # POSIX alarm handler
        def _alarm_handler(signum, frame):
            raise TimeoutError(f"Execution exceeded {timeout} seconds")

        old_handler = None
        timer_set = False

        try:
            # Set per-exec alarm only on POSIX and only if timeout > 0
            if timeout and timeout > 0 and _IS_POSIX:
                old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                # Use ITIMER_REAL for sub-second precision
                signal.setitimer(signal.ITIMER_REAL, float(timeout))
                timer_set = True

            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                result = self.shell.run_cell(code, silent=False)

            # If we get here without TimeoutError, cancel any timer
            if timer_set:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                except Exception:
                    pass
                if old_handler is not None:
                    try:
                        signal.signal(signal.SIGALRM, old_handler)
                    except Exception:
                        pass

            result_value = None
            if hasattr(result, 'result') and result.result is not None:
                try:
                    pickle.dumps(result.result)
                    result_value = result.result
                except Exception:
                    result_value = str(result.result)

            self.conn.send({
                "status": "ok",
                "id": exec_id,
                "result": result_value,
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
                "has_error": bool(getattr(result, "error_before_exec", None) or getattr(result, "error_in_exec", None)),
                "error_type": str(type(result.error_in_exec).__name__) if getattr(result, "error_in_exec", None) else None
            })

        except TimeoutError as te:
            # Clean, user-facing timeout output (no internal traceback)
            if timer_set:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                except Exception:
                    pass
                if old_handler is not None:
                    try:
                        signal.signal(signal.SIGALRM, old_handler)
                    except Exception:
                        pass

            # Clear the current traceback so IPython doesn't print the stack
            import traceback
            sys.stderr.flush()
            sys.stdout.flush()
            # Compose a concise, user-level message
            self.conn.send({
                "status": "interrupted",
                "id": exec_id,
                "stdout": stdout_buf.getvalue(),
                "stderr": (
                    stderr_buf.getvalue()
                    + f"\nExecution interrupted: exceeded {timeout} seconds (stopped by timeout)\n"
                )
            })


        except KeyboardInterrupt:
            self.conn.send({
                "status": "interrupted",
                "id": exec_id,
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue() + "\nExecution interrupted by SIGINT\n"
            })

        except Exception as e:
            import traceback
            self.conn.send({
                "status": "error",
                "id": exec_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue()
            })

        finally:
            # Best-effort cleanup of timer/handler if any path above missed it
            if timer_set:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                except Exception:
                    pass
                if old_handler is not None:
                    try:
                        signal.signal(signal.SIGALRM, old_handler)
                    except Exception:
                        pass
            sys.stdin = original_stdin

    def handle_get_var(self, msg):
        var_name = msg.get("name")
        try:
            if var_name in self.shell.user_ns:
                value = self.shell.user_ns[var_name]
                try:
                    pickle.dumps(value)
                    self.conn.send({"status": "ok", "value": value})
                except Exception:
                    self.conn.send({"status": "ok", "value": str(value)})
            else:
                self.conn.send({"status": "not_found"})
        except Exception as e:
            self.conn.send({"status": "error", "error": str(e)})

    def handle_set_var(self, msg):
        var_name = msg.get("name")
        value = msg.get("value")
        try:
            self.shell.user_ns[var_name] = value
            self.conn.send({"status": "ok"})
        except Exception as e:
            self.conn.send({"status": "error", "error": str(e)})

    def handle_reset(self):
        try:
            self.shell.reset(new_session=False)
            self.conn.send({"status": "ok"})
        except Exception as e:
            self.conn.send({"status": "error", "error": str(e)})

def shell_worker_process(conn):
    # Allow SIGINT to interrupt default
    try:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    except Exception:
        pass
    worker = IPythonWorker(conn)
    worker.run()

class IPythonProcessManager:
    """Manager for IPython processes with timeout & ID-correlated replies."""
    def __init__(self, max_kernels: int = 100, kernel_timeout: int = 3600):
        self.max_kernels = max_kernels
        self.kernel_timeout = kernel_timeout
        self._kernels: OrderedDict = OrderedDict()
        self._kernel_last_used: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _create_kernel(self, request_id: str) -> Dict:
        parent_conn, child_conn = mp.Pipe(duplex=True)
        proc = mp.Process(target=shell_worker_process, args=(child_conn,), daemon=True)
        proc.start()
        current_time = time.time()
        kernel_info = {
            "proc": proc,
            "conn": parent_conn,
            "created": current_time,
            "last_used": current_time,
            "lock": threading.Lock()
        }
        logger.info(f"Created new IPython kernel process for request_id: {request_id}")
        return kernel_info

    def get_kernel(self, request_id: str) -> Dict:
        with self._lock:
            current_time = time.time()
            if request_id in self._kernels:
                kernel_info = self._kernels.pop(request_id)
                self._kernels[request_id] = kernel_info
                kernel_info["last_used"] = current_time
                return kernel_info
            kernel_info = self._create_kernel(request_id)
            self._kernels[request_id] = kernel_info
            if len(self._kernels) > self.max_kernels:
                oldest_id = next(iter(self._kernels))
                self._remove_kernel(oldest_id)
            return kernel_info

    def _remove_kernel(self, request_id: str) -> None:
        if request_id in self._kernels:
            kernel_info = self._kernels.pop(request_id)
            self._kernel_last_used.pop(request_id, None)
            proc = kernel_info["proc"]
            conn = kernel_info["conn"]
            try:
                conn.send({"cmd": "shutdown"})
                conn.close()
            except Exception:
                pass
            proc.terminate()
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
            logger.info(f"Removed kernel process for request_id: {request_id}")

    def _cleanup_worker(self) -> None:
        while True:
            try:
                time.sleep(60)
                current_time = time.time()
                with self._lock:
                    expired = []
                    for request_id, kernel_info in list(self._kernels.items()):
                        if current_time - kernel_info["last_used"] > self.kernel_timeout:
                            expired.append(request_id)
                    for rid in expired:
                        self._remove_kernel(rid)
                    if expired:
                        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                        logger.info(f"Cleaned up {len(expired)} kernels. Memory: {memory_mb:.1f}MB")
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")

    # ---------- Reply correlation helpers ----------
    def _drain_all_messages(self, conn):
        drained = []
        try:
            while conn.poll(0):
                drained.append(conn.recv())
        except EOFError:
            pass
        return drained

    def _recv_until_id(self, conn, expected_id, deadline: float):
        while time.time() < deadline:
            if not conn.poll(0.05):
                continue
            try:
                msg = conn.recv()
            except EOFError:
                return None
            if isinstance(msg, dict) and msg.get("id") == expected_id:
                return msg
            # else: stale/old reply — discard
        return None

    def execute_with_timeout(self, request_id: str, code: str, timeout: int = 120,
                             stdin_content: str = None) -> Tuple[str, bool, Dict]:
        """
        Execute code with timeout. On POSIX, the worker will interrupt the cell
        with SIGALRM so work actually stops and kernel state is preserved.
        On Windows, we fall back to manager-side SIGINT (best effort).
        """
        kernel_info = self.get_kernel(request_id)
        proc = kernel_info["proc"]
        conn = kernel_info["conn"]
        lock = kernel_info["lock"]

        exec_id = time.time_ns()

        with lock:
            # Drain stale replies so next read belongs to THIS exec_id
            self._drain_all_messages(conn)

            try:
                conn.send({
                    "cmd": "exec",
                    "id": exec_id,
                    "code": code,
                    "stdin": stdin_content,
                    "timeout": timeout  # passed to worker (used on POSIX)
                })
            except Exception as e:
                return f"Failed to send execution request: {e}", False, {}

            # Wait for our reply: timeout + small grace
            deadline = time.time() + float(timeout) + 2.0
            result = self._recv_until_id(conn, exec_id, deadline)

            if result is not None:
                status = result.get("status")
                if status == "ok":
                    output = result.get("stdout", "")
                    if result.get("stderr"):
                        output += f"\nStderr: {result['stderr']}"
                    success = not result.get("has_error", False)
                    return output or "Script executed successfully (no output)", success, result
                elif status == "interrupted":
                    return "Execution interrupted (kernel state preserved)", False, result
                else:
                    error_msg = result.get("error", "Unknown error")
                    if result.get("traceback"):
                        error_msg = result["traceback"]
                    return f"Execution error: {error_msg}", False, result

            # If we are here: no reply in time. On Windows (no SIGALRM), try SIGINT.
            if os.name == "nt":
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)  # best-effort for console processes
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                return f"Execution exceeded {timeout}s on Windows; sent best-effort interrupt (kernel may be preserved)", False, {"preserved": True}

            # POSIX fallback: best-effort SIGINT to nudge worker (should be rare with SIGALRM)
            try:
                os.kill(proc.pid, signal.SIGINT)
            except Exception:
                try:
                    proc.send_signal(signal.SIGINT)
                except Exception:
                    pass

            # Give a last short grace for an 'interrupted' reply (do not restart)
            grace_deadline = time.time() + 1.0
            late = self._recv_until_id(conn, exec_id, grace_deadline)
            if late is not None and late.get("status") == "interrupted":
                return "Execution interrupted after manager SIGINT (kernel state preserved)", False, late

            # No auto-restart by default; preserve kernel for inspection
            return f"Execution exceeded {timeout}s with no reply. Kernel left running (no auto-restart).", False, {"preserved": True}

    def get_variable(self, request_id: str, var_name: str) -> Any:
        kernel_info = self.get_kernel(request_id)
        conn = kernel_info["conn"]
        lock = kernel_info["lock"]
        with lock:
            try:
                conn.send({"cmd": "get_var", "name": var_name})
                if conn.poll(5.0):
                    result = conn.recv()
                    if result["status"] == "ok":
                        return result["value"]
            except Exception:
                pass
        return None

    def set_variable(self, request_id: str, var_name: str, value: Any) -> bool:
        kernel_info = self.get_kernel(request_id)
        conn = kernel_info["conn"]
        lock = kernel_info["lock"]
        with lock:
            try:
                conn.send({"cmd": "set_var", "name": var_name, "value": value})
                if conn.poll(5.0):
                    result = conn.recv()
                    return result["status"] == "ok"
            except Exception:
                pass
        return False

    def get_stats(self) -> Dict:
        with self._lock:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            return {
                'active_kernels': len(self._kernels),
                'max_kernels': self.max_kernels,
                'memory_usage_mb': memory_mb,
                'kernel_ids': list(self._kernels.keys())
            }

    def cleanup_all(self) -> None:
        with self._lock:
            kernel_ids = list(self._kernels.keys())
            for request_id in kernel_ids:
                self._remove_kernel(request_id)

# Global process manager instance
_process_manager = IPythonProcessManager()

def call_python_script_with_ipython(request_id: str, script: str, timeout: int = 120,
                                    stdin_content: str = None) -> Tuple[str, bool]:
    logger.info(f"Executing script for request_id: {request_id} with timeout {timeout}s")
    if not isinstance(request_id, str) or not request_id.strip():
        return "Error: request_id must be a non-empty string", False
    if not isinstance(script, str) or not script.strip():
        return "Error: script must be a non-empty string", False
    try:
        output, success, metadata = _process_manager.execute_with_timeout(
            request_id, script, timeout, stdin_content
        )
        return output, success
    except Exception as e:
        logger.error(f"Failed to execute script for {request_id}: {e}")
        return f"System error: {str(e)}", False

def get_kernel_stats() -> Dict:
    return _process_manager.get_stats()

def cleanup_all_kernels() -> None:
    _process_manager.cleanup_all()

def remove_kernel(request_id: str) -> None:
    with _process_manager._lock:
        _process_manager._remove_kernel(request_id)

def get_kernel_variable(request_id: str, var_name: str) -> Any:
    return _process_manager.get_variable(request_id, var_name)

def set_kernel_variable(request_id: str, var_name: str, value: Any) -> bool:
    return _process_manager.set_variable(request_id, var_name, value)

# ------------------- Example usage/tests -------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Option A: In-worker Timeout (SIGALRM) Test ===\n")

    # Test 1: Simple execution
    print("Test 1: Simple execution")
    result, success = call_python_script_with_ipython("test1", "print('Hello World')")
    print(f"Result: {result}, Success: {success}\n")

    # Test 2: State preservation across calls
    print("Test 2: State preservation")
    result, success = call_python_script_with_ipython("test2", "x = 42; print(f'Set x = {x}')")
    print(f"Result: {result}, Success: {success}")
    result, success = call_python_script_with_ipython("test2", "print(f'x is still: {x}')")
    print(f"Result: {result}, Success: {success}\n")

    # Test 3: Timeout with sleep (graceful interruption)
    print("Test 3: Timeout with sleep")
    result, success = call_python_script_with_ipython("test3", """
import time
print("Starting sleep...")
y = 100
time.sleep(10)
print("This won't print")
""", timeout=2)
    print(f"Result: {result}, Success: {success}")

    # Give time for any late signals (shouldn't be needed with SIGALRM, but harmless)
    time.sleep(1)

    # Check if state is preserved after timeout
    result, success = call_python_script_with_ipython("test3", "print(f'Can still use kernel: y = {y}')")
    print(f"Result: {result}, Success: {success}\n")

    # Test 4: Infinite loop (should be interrupted by SIGALRM)
    print("Test 4: Infinite loop interruption")
    result, success = call_python_script_with_ipython("test4", """
counter = 0
while True:
    counter += 1
    if counter % 1_000_000 == 0:
        pass
""", timeout=2)
    print(f"Result: {result}, Success: {success}")

    # Test 5: Direct variable access
    print("\nTest 5: Direct variable access")
    call_python_script_with_ipython("test5", "my_data = {'key': 'value', 'number': 123}")
    value = get_kernel_variable("test5", "my_data")
    print(f"Retrieved variable: {value}")

    # Test 6: stdin input
    print("\nTest 6: Stdin input")
    result, success = call_python_script_with_ipython("test6", """
name = input("Enter name: ")
print(f"Hello, {name}!")
""", stdin_content="Alice\n")
    print(f"Result: {result}, Success: {success}")

    # Stats & cleanup
    stats = get_kernel_stats()
    print(f"\nFinal stats: {stats}")
    cleanup_all_kernels()
    print("Cleaned up all kernels")
