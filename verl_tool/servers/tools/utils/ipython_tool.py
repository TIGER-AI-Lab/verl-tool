import IPython
import threading
import time
import gc
import psutil
import sys
import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import weakref
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import multiprocessing as mp

logger = logging.getLogger(__name__)


class IPythonKernelManager:
    def __init__(self, max_kernels: int = 1024, kernel_timeout: int = 3600,
                 max_workers: Optional[int] = None):
        """
        Initialize the kernel manager.
        
        Args:
            max_kernels: Maximum number of kernels to keep in cache
            kernel_timeout: Timeout in seconds after which unused kernels are cleaned up
            max_workers: Maximum thread pool workers. If None, auto-calculated.
        """
        self.max_kernels = max_kernels
        self.kernel_timeout = kernel_timeout
        self._kernels: OrderedDict = OrderedDict()
        self._kernel_last_used: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Auto-calculate workers based on system resources
        if max_workers is None:
            cpu_count = mp.cpu_count()
            # For mixed I/O and CPU workloads: 4-8x CPU cores
            # Cap at 100 to avoid excessive context switching
            max_workers = min(cpu_count * 8, 100)
            logger.info(f"Auto-configured max_workers={max_workers} (CPUs={cpu_count})")
        
        self.max_workers = max_workers
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
    def _create_kernel(self, request_id: str) -> IPython.InteractiveShell:
        """Create a new IPython kernel with memory optimizations."""
        try:
            # Create configuration to disable history
            from IPython.terminal.interactiveshell import TerminalInteractiveShell
            from traitlets.config import Config
            
            c = Config()
            # Disable history completely
            c.HistoryManager.enabled = False
            c.HistoryManager.hist_file = ':memory:'  # Use in-memory SQLite
            
            # Create new shell with configuration
            shell = TerminalInteractiveShell.instance(config=c, colors='NoColor')
            
            # Double-check history is disabled
            if hasattr(shell, 'history_manager'):
                shell.history_manager.enabled = False
                # Close any existing history database connection
                if hasattr(shell.history_manager, 'db'):
                    try:
                        shell.history_manager.db.close()
                    except:
                        pass
            
            # Limit cache sizes
            shell.cache_size = 100
            
            # Configure matplotlib for non-interactive backend if available
            try:
                import matplotlib
                matplotlib.use('Agg')
            except ImportError:
                pass
            
            # Disable atexit hooks for history manager to prevent cleanup errors
            if hasattr(shell, 'history_manager'):
                import atexit
                # Remove history manager's atexit callback
                try:
                    for callback in atexit._exithandlers[:]:
                        if hasattr(callback[0], '__self__') and callback[0].__self__ is shell.history_manager:
                            atexit._exithandlers.remove(callback)
                except:
                    pass
            
            logger.debug(f"Created new IPython kernel for request_id: {request_id}")
            return shell
            
        except Exception as e:
            logger.error(f"Failed to create kernel for {request_id}: {e}")
            raise
    
    def get_kernel(self, request_id: str) -> IPython.InteractiveShell:
        """Get or create a kernel for the given request_id (thread-safe)."""
        with self._lock:
            current_time = time.time()
            
            if request_id in self._kernels:
                # Move to end (most recently used)
                kernel = self._kernels.pop(request_id)
                self._kernels[request_id] = kernel
                self._kernel_last_used[request_id] = current_time
                return kernel
            
            # Create new kernel
            kernel = self._create_kernel(request_id)
            
            # Add to cache
            self._kernels[request_id] = kernel
            self._kernel_last_used[request_id] = current_time
            
            # Remove oldest kernel if cache is full
            if len(self._kernels) > self.max_kernels:
                oldest_id = next(iter(self._kernels))
                self._remove_kernel(oldest_id)
            
            return kernel
    
    def _remove_kernel(self, request_id: str) -> None:
        """Remove a kernel from cache and clean up memory."""
        if request_id in self._kernels:
            kernel = self._kernels.pop(request_id)
            self._kernel_last_used.pop(request_id, None)
            
            try:
                # Disable atexit operations to prevent history manager errors
                if hasattr(kernel, 'atexit_operations'):
                    kernel.atexit_operations = lambda: None
                
                # Close history manager database connection
                if hasattr(kernel, 'history_manager'):
                    try:
                        if hasattr(kernel.history_manager, 'db'):
                            kernel.history_manager.db.close()
                    except:
                        pass
                
                # Clear kernel namespace
                if hasattr(kernel, 'user_ns'):
                    kernel.user_ns.clear()
                if hasattr(kernel, 'user_global_ns'):
                    kernel.user_global_ns.clear()
                
                # Reset kernel state (skip if it would trigger history operations)
                try:
                    if hasattr(kernel, 'reset'):
                        kernel.reset(new_session=False)
                except:
                    pass  # Ignore reset errors
                    
            except Exception as e:
                logger.warning(f"Error cleaning up kernel {request_id}: {e}")
            
            # Force garbage collection
            del kernel
            gc.collect()
            logger.info(f"Removed kernel for request_id: {request_id}")
    
    def _cleanup_worker(self) -> None:
        """Background thread to clean up expired kernels."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                current_time = time.time()
                
                with self._lock:
                    expired_kernels = [
                        request_id for request_id, last_used in self._kernel_last_used.items()
                        if current_time - last_used > self.kernel_timeout
                    ]
                    
                    for request_id in expired_kernels:
                        self._remove_kernel(request_id)
                        
                    # Log memory usage periodically
                    if expired_kernels:
                        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                        logger.info(f"Cleaned up {len(expired_kernels)} kernels. Memory usage: {memory_mb:.1f}MB")
                        
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def get_stats(self) -> Dict:
        """Get current statistics about kernel usage."""
        with self._lock:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            return {
                'active_kernels': len(self._kernels),
                'max_kernels': self.max_kernels,
                'memory_usage_mb': memory_mb,
                'kernel_ids': list(self._kernels.keys())
            }
    
    def cleanup_all(self) -> None:
        """Clean up all kernels."""
        with self._lock:
            kernel_ids = list(self._kernels.keys())
            for request_id in kernel_ids:
                self._remove_kernel(request_id)


# Global kernel manager instance
_kernel_manager = IPythonKernelManager()

# Thread pool executor for running blocking code
_executor = ThreadPoolExecutor(max_workers=_kernel_manager.max_workers)

async def _execute_with_timeout_async(kernel: IPython.InteractiveShell, script: str, timeout: int) -> Tuple[any, bool, str]:
    """Execute script with timeout using asyncio."""
    
    def _run_cell_sync():
        """Synchronous execution wrapper."""
        try:
            result = kernel.run_cell(script, silent=False)
            
            # Check for execution errors
            if result.error_before_exec:
                return result, False, f"Error before execution: {result.error_before_exec}"
            elif result.error_in_exec:
                return result, False, f"Error during execution: {result.error_in_exec}"
            else:
                return result, True, ""
        except Exception as e:
            return None, False, f"Execution error: {str(e)}"
    
    try:
        # Run in executor with timeout
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _run_cell_sync),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        return None, False, f"Execution timeout after {timeout} seconds"
    except Exception as e:
        return None, False, f"Execution error: {str(e)}"


def _execute_with_timeout(kernel: IPython.InteractiveShell, script: str, timeout: int) -> Tuple[any, bool, str]:
    """Synchronous wrapper for async execution with timeout."""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(_execute_with_timeout_async(kernel, script, timeout))
                )
                return future.result()
        else:
            # Run in the existing loop
            return loop.run_until_complete(_execute_with_timeout_async(kernel, script, timeout))
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(_execute_with_timeout_async(kernel, script, timeout))


class NoInputAvailable(Exception):
    """Raised when trying to read from stdin when no input is available"""
    pass

class MockStdin:
    """Mock stdin that raises an exception if no input is pre-written"""
    def __init__(self):
        self._buffer = StringIO()
        self._has_content = False
    
    def write(self, content):
        """Allow pre-writing content to stdin"""
        self._buffer.write(content)
        self._has_content = True
        self._buffer.seek(0)  # Reset to beginning for reading
    
    def read(self, size=-1):
        if not self._has_content:
            raise NoInputAvailable("Cannot read from stdin: no input available")
        return self._buffer.read(size)
    
    def readline(self, size=-1):
        if not self._has_content:
            raise NoInputAvailable("Cannot read from stdin: no input available")
        return self._buffer.readline(size)
    
    def readlines(self, hint=-1):
        if not self._has_content:
            raise NoInputAvailable("Cannot read from stdin: no input available")
        return self._buffer.readlines(hint)
    
    def __iter__(self):
        if not self._has_content:
            raise NoInputAvailable("Cannot read from stdin: no input available")
        return iter(self._buffer)

def call_python_script_with_ipython(request_id: str, script: str, timeout: int = 120, max_output_size: int = 100 * 1024) -> Tuple[str, bool]:
    """
    Execute a Python script using IPython and return the output.
    Thread-safe with memory management and timeout protection.
    
    Args:
        request_id: Unique identifier for the request
        script: Python script to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Tuple of (output, success) where success indicates if execution completed without error
    """
    if not isinstance(request_id, str) or not request_id.strip():
        return "Error: request_id must be a non-empty string", False
    script = script.replace("sys.stdin.buffer.read()", "sys.stdin.read()") # in case model uses buffer read which we don't support
    if not isinstance(script, str) or not script.strip():
        return "Error: script must be a non-empty string", False
    
    try:
        kernel = _kernel_manager.get_kernel(request_id)
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        # Capture both stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        # Execute with timeout protection
        success = True
        
        
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Execute the code
            result, success, error_output = _execute_with_timeout(kernel, script, timeout)
            
            # Get captured output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # Check for errors
            has_error = not result.success if result else True
            if result and result.error_before_exec:
                stderr_output += str(result.error_before_exec) + '\n'
            if result and result.error_in_exec:
                stderr_output += str(result.error_in_exec) + '\n'
            
            # If there's a result to display, add it to stdout
            if result and result.result is not None:
                stdout_output += str(result.result) + '\n'
                
        except Exception as e:
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue() + str(e) + '\n'
            has_error = True
        
        finally:
            # Restore original streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        if not result:
            output = error_output
        else:
            output = stdout_output.rstrip('\n')
        return output[:max_output_size], success

    except Exception as e:
        logger.error(f"Failed to execute script for {request_id}: {e}")
        return f"System error: {str(e)}", False

def get_kernel_stats() -> Dict:
    """Get current kernel manager statistics."""
    return _kernel_manager.get_stats()


def cleanup_all_kernels() -> None:
    """Clean up all cached kernels."""
    _kernel_manager.cleanup_all()
    
def remove_kernel(request_id: str) -> None:
    """Remove a specific kernel by request_id."""
    _kernel_manager._remove_kernel(request_id)


# Example usage and configuration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    result, success = call_python_script_with_ipython("test1", "print('Hello World')")
    print(f"Result: {result}, Success: {success}")
    
    # Test timeout functionality
    result, success = call_python_script_with_ipython("test2", "import time; time.sleep(5)", timeout=2)
    print(f"Timeout test - Result: {result}, Success: {success}")
    
    # Check stats
    stats = get_kernel_stats()
    print(f"Stats: {stats}")