"""
Tool Server - Fixed version with proper async handling and resource management
"""
import asyncio, inspect
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from tqdm import tqdm
import regex as re
import json
import fire
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .utils import hash_requests
import weakref
import time

from .tools import get_tool_cls, ALL_TOOLS, set_use_tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

    
class AgentResponse(BaseModel):
    """Model for outgoing agent responses"""
    observations: List[Union[str, dict]]
    dones: List[bool]
    valids: List[bool]


class AsyncToolManager:
    """Manages all tools and their execution using asyncio"""
    
    def __init__(self, tool_types: Tuple[str], num_workers_per_tool: int = 4, use_tqdm: bool = False, done_if_invalid: bool = False, thread_pool_size: int = 512):
        """Initialize the tool manager with specified tools"""
        self.tools: Dict[str, Any] = {}
        self.use_tqdm = use_tqdm
        set_use_tqdm(use_tqdm)
        self.done_if_invalid = done_if_invalid
        self._initialize_tools(tool_types, num_workers_per_tool)
        
        # Configure asyncio thread pool with proper sizing
        import concurrent.futures
        # Increase thread pool size significantly for high concurrency
        actual_pool_size = max(thread_pool_size, num_workers_per_tool * len(self.tools) * 2)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=actual_pool_size,
            thread_name_prefix="tool_server"
        )
        logger.info(f"Thread pool initialized with {actual_pool_size} workers")
        
    def _initialize_tools(self, tool_types: Tuple[str], num_workers: int) -> None:
        """Initialize all tools based on tool types"""
        if "finish" in tool_types:
            tool_types = tuple(t for t in tool_types if t != "finish")
            tool_types = tool_types + ("finish",)
            
        print(f"Initializing tools: {tool_types}")
        for tool_type in tool_types:
            try:
                tool_cls = get_tool_cls(tool_type)
                self.tools[tool_type] = tool_cls(num_workers=num_workers)
                print(f"Initialized tool: {tool_type}")
            except Exception as e:
                logger.error(f"Failed to initialize tool {tool_type}: {e}")
        
        finish_tool = get_tool_cls("finish")
        self.tools["finish"] = finish_tool(num_workers=num_workers, other_tools=list(self.tools.values()))
                
        print("Available Tools:")
        for tool in ALL_TOOLS:
            if tool in self.tools:
                print(f"  - {tool}: active 🟢")
            else:
                print(f"  - {tool}: inactive ⚪")
    
    def get_tool_usage_instructions(self) -> str:
        """Get usage instructions for all available tools"""
        usage_instructions = {}
        for tool_type, tool in self.tools.items():
            if tool_type not in ["finish", "base"]:
                usage_instructions[tool_type] = tool.get_usage_inst()
                
        message = "\nYour action did not match any of the available tools, please use one of the following tools: \n"
        message += "\n".join([f"- {tool_type}: {usage_instructions[tool_type]}" for tool_type in usage_instructions])
        return message
    
    def identify_tool_for_action(self, action: str, extra_field: Dict[str, Any]) -> Optional[str]:
        """Identify which tool should process a given action - SYNCHRONOUS VERSION"""
        if extra_field.get("finish", False):
            return "finish"
            
        if len(self.tools) == 1:
            return list(self.tools.keys())[0]
        
        for tool_type, tool in self.tools.items():
            if tool_type in ["finish", 'mcp_interface']:
                continue
            try:
                _, valid = tool.parse_action(action)
                if valid:
                    return tool_type
            except Exception as e:
                logger.debug(f"Tool {tool_type} failed to parse action: {e}")
                continue
        
        if "mcp_interface" in self.tools:
            try:
                tool = self.tools["mcp_interface"]
                _, valid = tool.parse_action(action)
                if valid:
                    return "mcp_interface"
            except Exception as e:
                logger.debug(f"MCP interface tool failed to parse action: {e}")

        return None
    
    async def identify_tool_types_batch(self, actions: List[str], extra_fields: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Efficiently identify tools for a batch of actions using CPU-bound processing"""
        def process_batch(batch_data):
            """CPU-bound function to process a batch"""
            batch_actions, batch_extra_fields = batch_data
            results = []
            for action, extra_field in zip(batch_actions, batch_extra_fields):
                tool_type = self.identify_tool_for_action(action, extra_field)
                results.append(tool_type)
            return results
        
        # Process in smaller chunks to avoid blocking
        chunk_size = min(50, len(actions))
        tool_types = []
        
        for i in range(0, len(actions), chunk_size):
            chunk_end = min(i + chunk_size, len(actions))
            chunk_actions = actions[i:chunk_end]
            chunk_extra_fields = extra_fields[i:chunk_end]
            
            # Process chunk in thread pool
            chunk_results = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                process_batch,
                (chunk_actions, chunk_extra_fields)
            )
            tool_types.extend(chunk_results)
            
            # Yield control periodically
            if i % (chunk_size * 4) == 0:
                await asyncio.sleep(0.001)
        
        logger.debug(f"Identified tool types for {len(actions)} actions")
        return tool_types
    
    async def process_actions(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """Process a batch of actions asynchronously using appropriate tools"""
        
        # Use the more efficient batch processing
        tool_types = await self.identify_tool_types_batch(actions, extra_fields)
        
        # Prepare result containers
        all_observations = [None] * len(actions)
        all_dones = [False] * len(actions)
        all_valids = [False] * len(actions)
        
        # Group actions by tool type for batch processing
        unique_tool_types: Set[Optional[str]] = set(tool_types)
        
        # Create tasks for each tool type
        tasks = []
        indices_by_tool = {}
        
        for tool_type in unique_tool_types:
            indices = [i for i, t in enumerate(tool_types) if t == tool_type]
            indices_by_tool[tool_type] = indices
            
            if tool_type is None:
                continue
                
            tool = self.tools[tool_type]
            tool_trajectory_ids = [trajectory_ids[i] for i in indices]
            tool_actions = [actions[i] for i in indices]
            tool_extra_fields = [extra_fields[i] for i in indices]
            
            # Create task for tool processing with proper async/thread handling
            if hasattr(tool, "aget_observations") and inspect.iscoroutinefunction(tool.aget_observations):
                task = await tool.aget_observations(tool_trajectory_ids, tool_actions, tool_extra_fields)
                # True async method
                # task = asyncio.create_task(
                #     tool.aget_observations(tool_trajectory_ids, tool_actions, tool_extra_fields)
                # )
            else:
                task = tool.get_observations(tool_trajectory_ids, tool_actions, tool_extra_fields)
                # # Blocking method - use thread pool
                # task = asyncio.get_event_loop().run_in_executor(
                #     self.thread_pool,
                #     tool.get_observations,
                #     tool_trajectory_ids,
                #     tool_actions,
                #     tool_extra_fields,
                # )
            tasks.append((tool_type, task))
        
        # Process all non-matching actions
        if None in indices_by_tool:
            usage_instructions = self.get_tool_usage_instructions()
            indices = indices_by_tool[None]
            for idx in indices:
                all_observations[idx] = {"obs": "", "invalid_reason": "no valid tool found for action"}
                all_valids[idx] = False
                all_dones[idx] = self.done_if_invalid
        
        # Await all tool processing tasks with proper error handling
        for tool_type, task in tasks:
            try:
                if isinstance(task, asyncio.Task) or inspect.isawaitable(task):
                    observations, dones, valids = await task
                else:
                    observations, dones, valids = task
                
                indices = indices_by_tool[tool_type]
                for idx_pos, result_idx in enumerate(indices):
                    all_observations[result_idx] = observations[idx_pos]
                    all_dones[result_idx] = dones[idx_pos]
                    all_valids[result_idx] = valids[idx_pos]
                    
            except Exception as e:
                logger.error(f"Tool {tool_type} processing failed: {e}", exc_info=True)
                # Handle failed tool processing
                indices = indices_by_tool[tool_type]
                for result_idx in indices:
                    all_observations[result_idx] = {"obs": "", "error": f"Tool processing failed: {str(e)}"}
                    all_dones[result_idx] = True
                    all_valids[result_idx] = False
                
        return all_observations, all_dones, all_valids


class AsyncToolServer:
    """Server to handle tool execution requests using asyncio"""
    
    def __init__(
        self,
        tool_types: Tuple[str],
        host: str = "0.0.0.0",
        port: int = 5000,
        workers_per_tool: int = 32,
        max_concurrent_requests: int = 64,
        use_tqdm: bool = False,
        done_if_invalid: bool = False,
        use_ray: bool = False,
        enable_hashing: bool = False,
        request_timeout: float = 60.0,
        thread_pool_size: int = None
    ):
        """Initialize the tool server"""
        self.host = host
        self.port = port
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_hashing = enable_hashing
        self.request_timeout = request_timeout
        self.active_requests = 0
        
        # Critical: Ensure thread pool is large enough
        if thread_pool_size is None:
            thread_pool_size = max(max_concurrent_requests * 3, 1024)  # 3x concurrency, min 1024
        
        if not use_ray:
            self.tool_manager = AsyncToolManager(tool_types, workers_per_tool, use_tqdm, done_if_invalid, thread_pool_size)
        else:
            from .ray_utils import RayToolManager
            self.tool_manager = RayToolManager(tool_types, workers_per_tool, use_tqdm, done_if_invalid)
        
        # Create FastAPI app with optimized settings
        self.app = FastAPI(
            title="Async Tool Server",
            description="A server for executing tools based on agent requests using asyncio",
            version="1.0.0",
        )
        
        # Use WeakValueDictionary for automatic cleanup
        self.processing_tasks = weakref.WeakValueDictionary()
        self._task_cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        
        self._configure_app()
        
    async def _cleanup_old_tasks(self):
        """Clean up old completed tasks periodically"""
        current_time = time.time()
        if current_time - self._last_cleanup > self._task_cleanup_interval:
            # The WeakValueDictionary will automatically clean up unreferenced tasks
            # But we can also manually clean up old completed tasks
            to_remove = []
            for hash_str, task_info in list(self.processing_tasks.items()):
                if (hasattr(task_info, 'get') and 
                    task_info.get('completed_time') and 
                    current_time - task_info['completed_time'] > 60):  # Remove after 1 minute
                    to_remove.append(hash_str)
            
            for hash_str in to_remove:
                self.processing_tasks.pop(hash_str, None)
                
            self._last_cleanup = current_time
            logger.debug(f"Cleaned up {len(to_remove)} old tasks")
        
    def _configure_app(self):
        """Configure FastAPI app with routes and event handlers"""
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        @self.app.post("/get_observation", response_model=AgentResponse)
        async def handle_observation_request(request: Request, background_tasks: BackgroundTasks):
            """Handle incoming observation requests"""
            request_start_time = time.time()
            self.active_requests += 1
            logger.debug(f"Request started. Active: {self.active_requests}")
            
            try:
                # Acquire semaphore with timeout
                try:
                    await asyncio.wait_for(semaphore.acquire(), timeout=30.0)
                except asyncio.TimeoutError:
                    raise HTTPException(status_code=429, detail="Server overloaded - semaphore timeout")
                
                try:
                    # Process request with timeout
                    response = await asyncio.wait_for(
                        self._process_request(request, background_tasks),
                        timeout=self.request_timeout
                    )
                    
                    processing_time = time.time() - request_start_time
                    logger.debug(f"Request completed in {processing_time:.2f}s")
                    return response
                    
                finally:
                    semaphore.release()
                    
            except asyncio.TimeoutError:
                logger.warning(f"Request timed out after {self.request_timeout}s")
                raise HTTPException(status_code=408, detail="Request timeout")
            except Exception as e:
                logger.error(f"Request failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
            finally:
                self.active_requests -= 1
                # Periodic cleanup
                await self._cleanup_old_tasks()
        
        async def _process_request(self, request: Request, background_tasks: BackgroundTasks):
            """Process the actual request logic"""
            try:
                # Parse request data
                data = await asyncio.wait_for(request.json(), timeout=5.0)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=400, detail="Request parsing timeout")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
            
            data_hash_str = hash_requests(data) if self.enable_hashing else str(id(data))
            
            # Check for duplicate processing (if hashing enabled)
            if self.enable_hashing and data_hash_str in self.processing_tasks:
                logger.debug(f"Duplicate request detected: {data_hash_str}")
                task_info = self.processing_tasks[data_hash_str]
                task_info['ref_count'] = task_info.get('ref_count', 0) + 1
                
                # Wait for result with proper timeout
                for _ in range(int(self.request_timeout * 2)):  # Check every 0.5s
                    if task_info.get('result') is not None:
                        return task_info['result']
                    await asyncio.sleep(0.5)
                
                # Timeout waiting for duplicate
                raise HTTPException(status_code=408, detail="Timeout waiting for duplicate request")
            
            # Create new task entry
            task_info = {"ref_count": 1, "result": None, "start_time": time.time()}
            if self.enable_hashing:
                self.processing_tasks[data_hash_str] = task_info
            
            try:
                # Validate and process request data
                trajectory_ids = [str(tid) for tid in data.get("trajectory_ids", [])]
                actions = data.get("actions", [])
                
                if not trajectory_ids or not actions:
                    raise HTTPException(status_code=400, detail="Missing trajectory_ids or actions")
                
                if len(trajectory_ids) != len(actions):
                    raise HTTPException(status_code=400, detail="trajectory_ids and actions length mismatch")
                
                # Process extra fields
                if 'extra_fields' in data:
                    extra_fields = data['extra_fields']
                    for key in data:
                        if key not in ["trajectory_ids", "actions", "extra_fields"]:
                            if len(data[key]) != len(trajectory_ids):
                                raise HTTPException(status_code=400, detail=f"Length mismatch for {key}")
                            for i in range(len(trajectory_ids)):
                                extra_fields[i][key] = data[key][i]
                else:
                    extra_keys = [k for k in data.keys() if k not in ["trajectory_ids", "actions"]]
                    extra_fields = [
                        {key: data[key][i] for key in extra_keys} 
                        for i in range(len(trajectory_ids))
                    ]
                
                logger.debug(f"Processing {len(actions)} actions")
                
                # Process actions
                observations, dones, valids = await self.tool_manager.process_actions(
                    trajectory_ids, actions, extra_fields
                )
                
                # Create and store response
                response = AgentResponse(
                    observations=observations,
                    dones=dones,
                    valids=valids
                )
                
                if self.enable_hashing:
                    task_info['result'] = response
                    task_info['completed_time'] = time.time()
                
                return response
                
            except Exception as e:
                # Clean up on error
                if self.enable_hashing and data_hash_str in self.processing_tasks:
                    del self.processing_tasks[data_hash_str]
                logger.error(f"Error processing request: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
        # Bind method
        self._process_request = _process_request.__get__(self, type(self))
            
        # Enhanced health check
        @self.app.get("/health")
        async def health_check():
            thread_count = "unknown"
            if hasattr(self.tool_manager, 'thread_pool'):
                try:
                    thread_count = len(self.tool_manager.thread_pool._threads)
                except:
                    thread_count = "error"
            
            return {
                "status": "healthy", 
                "concurrent_requests": self.active_requests,
                "thread_pool_threads": thread_count,
                "processing_tasks": len(self.processing_tasks),
                "max_concurrent": self.max_concurrent_requests,
                "tools": list(self.tool_manager.tools.keys())
            }
    
    def start(self, log_level: str = "error"):
        """Start the server with optimized uvicorn settings"""
        logger.info(f"Starting server on {self.host}:{self.port}")
        logger.info(f"Max concurrent requests: {self.max_concurrent_requests}")
        logger.info(f"Request timeout: {self.request_timeout}s")
        logger.info(f"Thread pool size: {getattr(self.tool_manager, 'thread_pool', {})._max_workers if hasattr(self.tool_manager, 'thread_pool') else 'unknown'}")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=log_level,
            access_log=False,  # Disable access logs for performance
            # Optimize for high concurrency
            loop="uvloop" if self._has_uvloop() else "asyncio",
            http="httptools",
            lifespan="on",
            timeout_keep_alive=30,
            timeout_graceful_shutdown=30,
        )
    
    def _has_uvloop(self):
        """Check if uvloop is available"""
        try:
            import uvloop
            return True
        except ImportError:
            return False


def main(
    tool_type: Union[str, Tuple[str]] = "base",
    host: str = "0.0.0.0",
    port: int = 5000,
    workers_per_tool: int = None,
    max_concurrent_requests: int = 128,
    use_tqdm: bool = False,
    log_level: str = "info",
    slient=False,
    done_if_invalid=False,
    use_ray: bool = False,
    enable_hashing: bool = False,
    request_timeout: float = 30.0,
    thread_pool_size: int = None
):
    """Start the async tool server with improved defaults"""
    
    if workers_per_tool is None:
        workers_per_tool = max_concurrent_requests
    
    # For 256 concurrent requests, ensure adequate thread pool
    if thread_pool_size is None and max_concurrent_requests >= 256:
        thread_pool_size = max_concurrent_requests * 4  # 4x for safety
    
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.basicConfig(level=numeric_level)
        logger.setLevel(numeric_level)
    
    # Convert tool_type string to tuple
    if isinstance(tool_type, str):
        if "," in tool_type:
            tool_type = tuple(t.strip() for t in tool_type.split(","))
        else:
            tool_type = (tool_type,)
    
    # Create and start server
    server = AsyncToolServer(
        tool_types=tool_type,
        host=host,
        port=port,
        workers_per_tool=workers_per_tool,
        max_concurrent_requests=max_concurrent_requests,
        use_tqdm=use_tqdm,
        done_if_invalid=done_if_invalid,
        use_ray=use_ray,
        enable_hashing=enable_hashing,
        request_timeout=request_timeout,
        thread_pool_size=thread_pool_size
    )
    
    if slient:
        import sys
        import os
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    server.start(log_level=log_level)


if __name__ == "__main__":
    fire.Fire(main)