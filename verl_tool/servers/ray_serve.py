import ray
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from fastapi import FastAPI, Request
import uvicorn
import time
from collections import defaultdict
from .tools import get_tool_cls

# Initialize Ray
if not ray.is_initialized():
    print("Ray not initialized")
    try:
        ray.init(ignore_reinit_error=True)
    except:
        # Connect to existing Ray cluster
        ray.init(address="auto", ignore_reinit_error=True)

# Import your tool classes
from .tools import get_tool_cls, ALL_TOOLS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
        
@ray.remote(num_cpus=0)
def ray_execute(tool, trajectory_id: str, action: str, extra_field: Dict[str, Any]):
    """
    Execute a single tool action.
    
    Args:
        trajectory_id: Unique identifier for the trajectory
        action: The action string to execute
        extra_field: Additional data for the action
        
    Returns:
        tuple: (observation, done, valid) result of the action
    """
    return tool.conduct_action(trajectory_id, action, extra_field)
    
@ray.remote(num_cpus=0)
def ray_parse_action(tool, action: str):
    """
    Check if this tool can handle the action.
    
    Args:
        action: The action string to parse
        
    Returns:
        tuple: (parsed_action, valid)
    """
    return tool.parse_action(action)

@ray.remote(num_cpus=0)
def ray_get_usage_inst(tool):
    """
    Get usage instructions for this tool.
    
    Returns:
        str: The usage instructions
    """
    return tool.get_usage_inst()


class RayToolManager:
    """Tool manager that uses Ray for distributed execution"""
    
    def __init__(self, tool_types: Tuple[str], num_workers_per_tool: int = 4):
        """
        Initialize tool workers as Ray actors.
        
        Args:
            tool_types: Types of tools to initialize
            num_workers_per_tool: Number of Ray workers to create per tool type
        """
        self.tool_types = tool_types
        self.workers_per_tool = num_workers_per_tool
        self.tool_workers = {}
        
        # Make sure Ray is initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        self._initialize_tools()
        
    def _initialize_tools(self):
        """Initialize Ray actors for each tool type"""
        for tool_type in self.tool_types:
            if tool_type == "finish":
                continue  # Handle finish tool separately
                
            # Create multiple workers for each tool type for parallelization
            self.tool_workers[tool_type] = get_tool_cls(tool_type)()
        
        # Initialize finish tool (if needed)
        if "finish" not in self.tool_workers:
            self.tool_workers["finish"] = get_tool_cls("finish")()
            
        # Log available vs. active tools with emoji indicators
        logger.info("Available Tools:")
        for tool in ALL_TOOLS:
            if tool in self.tool_workers:
                status = "active 🟢"  # Green circle for active tools
                logger.info(f"  - {tool}: {status}")
            else:
                status = "inactive ⚪"  # White circle for inactive tools
                logger.info(f"  - {tool}: {status}")
    
    async def identify_tool_for_action(self, action: str, extra_field: Dict[str, Any]) -> Optional[str]:
        """
        Identify which tool type can handle this action.
        
        Args:
            action: The action string
            extra_field: Additional data for the action
            
        Returns:
            str or None: The identified tool type, or None if no tool matches
        """
        # Check for finish condition
        if extra_field.get("finish", False):
            return "finish"
            
        # Try each tool type with Ray
        for tool_type, worker in self.tool_workers.items():
            if tool_type == "finish":
                continue
                
            # Use the first worker to check validity
            _, valid_ref = await asyncio.to_thread(
                ray.get, ray_parse_action.remote(worker, action)
            )
            
            if valid_ref:
                return tool_type
                
        return None
        
    def get_tool_usage_instructions(self) -> str:
        """
        Get usage instructions for all tools.
        
        Returns:
            str: Combined usage instructions for all tools
        """
        # Collect usage instructions from one worker of each type
        futures = {
            tool_type: ray_get_usage_inst.remote(worker)
            for tool_type, worker in self.tool_workers.items()
            if tool_type != "finish"
        }
        
        instructions = ray.get(list(futures.values()))
        tool_types = list(futures.keys())
        
        message = "\nYour action did not match any of the available tools, please use one of the following tools: \n"
        message += "\n".join([f"- {tool_types[i]}: {instructions[i]}" for i in range(len(tool_types))])
        return message
        
    async def process_actions(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """
        Process actions using Ray workers in parallel.
        
        Args:
            trajectory_ids: List of trajectory IDs
            actions: List of actions corresponding to each trajectory
            extra_fields: List of extra data for each action
            
        Returns:
            tuple: (observations, dones, valids) lists for all actions
        """
        # Identify tool types for actions
        tool_types = []
        for i in range(len(actions)):
            tool_type = await self.identify_tool_for_action(actions[i], extra_fields[i])
            tool_types.append(tool_type)
            
        # Prepare results
        observations = [None] * len(actions)
        dones = [False] * len(actions)
        valids = [False] * len(actions)
        
        @ray.remote(num_cpus=0)
        def non_tool_action(trajectory_id: str, action: str, extra_field: Dict[str, Any]):
            return "", False, False # no observation if no tool matched, [obs, done, valid]
        
        pending_refs = []
        for i, tool_type in enumerate(tool_types):
            trajectory_id = trajectory_ids[i]
            action = actions[i]
            extra_field = extra_fields[i]
            
            if tool_type is None:
                # Handle actions with no matching tool
                result_ref = non_tool_action.remote(trajectory_id, action, extra_field)
            else:
                worker = self.tool_workers[tool_type]
                result_ref = ray_execute.remote(worker, trajectory_id, action, extra_field)
            pending_refs.append(result_ref)
        
        # Get results as they complete
        if pending_refs:
            # Use asyncio to wait for Ray tasks
            results = [ray.get(ref) for ref in pending_refs]
            for i, result in enumerate(results):
                observation, done, valid = result
                observations[i] = observation
                dones[i] = done
                valids[i] = valid
        assert observations.count(None) == 0, f"{observations.count(None)} actions did not return an observation"
        return observations, dones, valids


# FastAPI server
class RayToolServer:
    """FastAPI server that uses Ray for distributed tool execution"""
    
    def __init__(
        self, 
        tool_types: Tuple[str], 
        host: str = "0.0.0.0", 
        port: int = 5000,
        workers_per_tool: int = 4,
        max_concurrent_requests: int = 64
    ):
        """
        Initialize the server.
        
        Args:
            tool_types: Types of tools to initialize
            host: Host address
            port: Port number
            workers_per_tool: Number of Ray workers per tool type
            max_concurrent_requests: Maximum number of concurrent requests
        """
        self.host = host
        self.port = port
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize Ray tool manager
        self.tool_manager = RayToolManager(tool_types, workers_per_tool)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Ray Tool Server",
            description="Tool server using Ray for distributed execution",
            version="1.0.0"
        )
        
        # Initialize processing tasks dictionary for handling duplicates
        self.processing_tasks = {}
        
        # Configure routes
        self._configure_app()
        
    def _configure_app(self):
        """Set up API routes"""
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        @self.app.post("/get_observation")
        async def handle_observation_request(request: Request):
            async with semaphore:
                # Parse request
                data = await request.json()
                
                # Generate hash for duplicate detection
                from .utils import hash_requests  # Assuming the same hash_requests function
                data_hash_str = hash_requests(data)
                logger.debug(f"Request hash: {data_hash_str}")
                
                # Check for duplicate requests
                if data_hash_str in self.processing_tasks:
                    logger.info(f"Duplicate request detected: {data_hash_str}")
                    # Wait for the original request to complete
                    task_info = self.processing_tasks[data_hash_str]
                    # Increment reference count
                    task_info["ref_count"] += 1
                    
                    # Wait for result
                    while task_info["result"] is None:
                        if task_info.get("error"):
                            # Original request failed, re-process this one
                            logger.warning(f"Original request failed, processing duplicate: {data_hash_str}")
                            break
                        # Wait before checking again
                        await asyncio.sleep(0.2)
                    
                    # If we have a valid result, return it
                    if task_info["result"] is not None:
                        logger.debug(f"Returning cached result for {data_hash_str}")
                        # Schedule reference count decrement
                        asyncio.create_task(self._decrement_ref_count(data_hash_str))
                        return task_info["result"]
                
                # Register new request or process duplicate after original failed
                if data_hash_str not in self.processing_tasks:
                    self.processing_tasks[data_hash_str] = {
                        "ref_count": 1,
                        "result": None,
                        "error": None,
                        "start_time": time.time()
                    }
                
                try:
                    # Normalize trajectory IDs
                    if "trajectory_ids" in data:
                        data["trajectory_ids"] = [str(tid) if not isinstance(tid, str) else tid 
                                                for tid in data.get("trajectory_ids", [])]
                        
                    # Validate and process request
                    trajectory_ids = data.get("trajectory_ids", [])
                    actions = data.get("actions", [])
                    
                    # Extract extra fields
                    if 'extra_fields' in data.keys():
                        extra_fields = data['extra_fields']
                        for key in data.keys():
                            assert len(data[key]) == len(trajectory_ids), f"Length of {key} ({len(data[key])}) does not match trajectory_ids ({len(trajectory_ids)})"
                            if key not in ["trajectory_ids", "actions", "extra_fields"]:
                                for i in range(len(trajectory_ids)):
                                    extra_fields[i][key] = data[key][i]
                        assert len(extra_fields) == len(trajectory_ids)
                    else:
                        extra_keys = [k for k in data.keys() if k not in ["trajectory_ids", "actions"]]
                        extra_fields = [
                            {key: data[key][i] for key in extra_keys} 
                            for i in range(len(trajectory_ids))
                        ]
                    
                    # Process with Ray
                    start = time.time()
                    logger.info(f"Processing {len(trajectory_ids)} actions")
                    observations, dones, valids = await self.tool_manager.process_actions(
                        trajectory_ids, actions, extra_fields
                    )
                    end = time.time() - start
                    logger.info(f"Processed {len(trajectory_ids)} actions in {end:.2f} seconds")
                    
                    # Create response
                    response = {
                        "observations": observations, 
                        "dones": dones, 
                        "valids": valids
                    }
                    
                    # Store result for duplicates
                    if data_hash_str in self.processing_tasks:
                        self.processing_tasks[data_hash_str]["result"] = response
                        # Schedule reference count decrement
                        asyncio.create_task(self._decrement_ref_count(data_hash_str))
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Error processing request: {e}", exc_info=True)
                    # Mark task as failed
                    if data_hash_str in self.processing_tasks:
                        self.processing_tasks[data_hash_str]["error"] = str(e)
                        # Schedule task removal
                        asyncio.create_task(self._decrement_ref_count(data_hash_str))
                    return {"error": str(e)}, 500
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy", 
                "ray_status": "connected" if ray.is_initialized() else "disconnected",
                "active_tasks": len(self.processing_tasks)
            }
    
    async def _decrement_ref_count(self, data_hash_str):
        """Decrement reference count and clean up completed tasks"""
        if data_hash_str in self.processing_tasks:
            self.processing_tasks[data_hash_str]["ref_count"] -= 1
            
            # If no more references, remove the task
            if self.processing_tasks[data_hash_str]["ref_count"] <= 0:
                # Calculate processing time
                start_time = self.processing_tasks[data_hash_str].get("start_time", 0)
                processing_time = time.time() - start_time
                
                # Log cleanup
                logger.debug(f"Removing completed task {data_hash_str} (processed in {processing_time:.2f}s)")
                del self.processing_tasks[data_hash_str]
                
                # Optionally, you could run garbage collection here if memory usage is a concern
                # import gc
                # gc.collect()
                
    def start(self):
        """Start the server"""
        logger.info(f"Starting Ray Tool Server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

# CLI entry point
def main(
    tool_type: Union[str, Tuple[str]] = "base",
    host: str = "0.0.0.0",
    port: int = 5000,
    workers_per_tool: int = 16,
    max_concurrent_requests: int = 64,
    ray_address: Optional[str] = None,
    slient=False,
):
    """
    Start the Ray Tool Server.
    
    Args:
        tool_type: Tool type(s) to initialize
        host: Host address
        port: Port number
        workers_per_tool: Number of Ray workers per tool type
        max_concurrent_requests: Maximum number of concurrent requests
        ray_address: Ray cluster address
    """
    # Initialize Ray
    if not ray.is_initialized():
        if ray_address:
            ray.init(address=ray_address)
        else:
            ray.init()
    
    # Convert tool_type to tuple if needed
    if isinstance(tool_type, str):
        if "," in tool_type:
            tool_type = tuple(t.strip() for t in tool_type.split(","))
        else:
            tool_type = (tool_type,)
    
    # Create and start server
    server = RayToolServer(
        tool_types=tool_type,
        host=host,
        port=port,
        workers_per_tool=workers_per_tool,
        max_concurrent_requests=max_concurrent_requests
    )
    if slient:
        import sys
        import os
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    server.start()


if __name__ == "__main__":
    import fire
    fire.Fire(main)