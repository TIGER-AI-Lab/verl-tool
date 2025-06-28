"""
Tool Server - A FastAPI server to manage and execute tools based on incoming requests.
Using asyncio for concurrent processing.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from tqdm import tqdm

import fire
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .utils import hash_requests
from collections import defaultdict

from .tools import get_tool_cls, ALL_TOOLS, set_use_tqdm
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DictObservation:
    """Dataclass for structured observations"""
    obs: str
    reward: Union[int, float, None] = None
    
class AgentResponse(BaseModel):
    """Model for outgoing agent responses"""
    observations: List[Union[str,  dict]]
    dones: List[bool]
    valids: List[bool]


# ---- Tool Management ----

class AsyncToolManager:
    """Manages all tools and their execution using asyncio"""
    
    def __init__(self, tool_types: Tuple[str], num_workers_per_tool: int = 4, use_tqdm: bool = False, done_if_invalid: bool = False):
        """
        Initialize the tool manager with specified tools
        
        Args:
            tool_types: Tuple of tool type names to initialize
            num_workers_per_tool: Number of workers for each tool
        """
        self.tools: Dict[str, Any] = {}
        self.use_tqdm = use_tqdm
        set_use_tqdm(use_tqdm)
        self.done_if_invalid = done_if_invalid
        self._initialize_tools(tool_types, num_workers_per_tool)
        
    def _initialize_tools(self, tool_types: Tuple[str], num_workers: int) -> None:
        """Initialize all tools based on tool types"""
        # Ensure we have the finish tool
        if "finish" in tool_types:
            tool_types = tuple(t for t in tool_types if t != "finish")
            tool_types = tool_types + ("finish",)
            
        logger.info(f"Initializing tools: {tool_types}")
        for tool_type in tool_types:
            try:
                tool_cls = get_tool_cls(tool_type)
                self.tools[tool_type] = tool_cls(num_workers=num_workers)
                logger.info(f"Initialized tool: {tool_type}")
            except Exception as e:
                logger.error(f"Failed to initialize tool {tool_type}: {e}")
        
        # initialize the finish tool
        finish_tool = get_tool_cls("finish")
        self.tools["finish"] = finish_tool(num_workers=num_workers, other_tools=list(self.tools.values()))
                
        # Log available vs. active tools with emoji indicators
        logger.info("Available Tools:")
        for tool in ALL_TOOLS:
            if tool in self.tools:
                status = "active 🟢"  # Green circle for active tools
                logger.info(f"  - {tool}: {status}")
            else:
                status = "inactive ⚪"  # White circle for inactive tools
                logger.info(f"  - {tool}: {status}")
    
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
        """
        Identify which tool should process a given action
        
        Args:
            action: The action string to process
            extra_field: Extra fields associated with the action
            
        Returns:
            The identified tool type or None if no tool matches
        """
        # Check for finish condition
        if extra_field.get("finish", False):
            return "finish"
            
        # If only one tool available, use it
        if len(self.tools) == 1:
            return list(self.tools.keys())[0]
        # # Try to find matching tool
        for tool_type, tool in self.tools.items():
            if tool_type == "finish":
                continue
            _, valid = tool.parse_action(action)
            if valid:
                return tool_type
                
        return None
    
    async def identify_tool_types(self, actions: List[str], extra_fields: List[Dict[str, Any]]) -> List[Optional[str]]:
        """
        Asynchronously identify tools for a batch of actions
        
        Args:
            actions: List of action strings
            extra_fields: List of extra fields for each action
            
        Returns:
            List of identified tool types
        """
        # The issue with the previous implementation is that asyncio.to_thread can be inefficient
        # for quick CPU-bound operations and might get stuck in some environments.
        # Instead, we'll create a more direct approach by processing items in batches
        
        tool_types = []
        
        # Process in small batches to avoid blocking the event loop
        batch_size = 10
        for i in range(0, len(actions), batch_size):
            batch_end = min(i + batch_size, len(actions))
            batch_actions = actions[i:batch_end]
            batch_extra_fields = extra_fields[i:batch_end]
            
            # Process this batch
            batch_results = []
            for j in range(len(batch_actions)):
                # Yield control back to event loop periodically
                if j % 3 == 0:
                    await asyncio.sleep(0)
                
                tool_type = self.identify_tool_for_action(batch_actions[j], batch_extra_fields[j])
                batch_results.append(tool_type)
            
            tool_types.extend(batch_results)
            
            # Yield control back to event loop between batches
            await asyncio.sleep(0)
        
        logger.debug(f"Identified tool types: {tool_types}")
        return tool_types
    
    async def process_actions(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """
        Process a batch of actions asynchronously using appropriate tools
        
        Args:
            trajectory_ids: List of trajectory IDs
            actions: List of action strings
            extra_fields: List of extra fields for each action
            
        Returns:
            Tuple of (observations, dones, valids) lists
        """
        # Identify which tool should process each action
        # tool_types = await self.identify_tool_types(actions, extra_fields)
        # just use a tqdm for loop
        tool_types = []
        for i in tqdm(range(len(actions)), desc="Identifying tool types", unit="action", disable=True):
            tool_type = self.identify_tool_for_action(actions[i], extra_fields[i])
            tool_types.append(tool_type)
        
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
            # Get indices of actions for this tool type
            indices = [i for i, t in enumerate(tool_types) if t == tool_type]
            indices_by_tool[tool_type] = indices
            
            if tool_type is None:
                # No processing needed for actions that don't match any tool
                continue
                
            # Process with the appropriate tool
            tool = self.tools[tool_type]
            tool_trajectory_ids = [trajectory_ids[i] for i in indices]
            tool_actions = [actions[i] for i in indices]
            tool_extra_fields = [extra_fields[i] for i in indices]
            
            # Create task for tool processing
            # We use asyncio.to_thread for potentially blocking operations
            task = asyncio.to_thread(
                tool.get_observations,
                tool_trajectory_ids, 
                tool_actions, 
                tool_extra_fields
            )
            tasks.append((tool_type, task))
        
        # Process all non-matching actions
        if None in indices_by_tool:
            usage_instructions = self.get_tool_usage_instructions()
            indices = indices_by_tool[None]
            for idx in indices:
                # all_observations[idx] = usage_instructions
                all_observations[idx] = "" # no observation
                # all_observations[idx] = "\nNo valid action found\n" # no observation
                all_valids[idx] = False
                if self.done_if_invalid:
                    all_dones[idx] = True
                else:
                    all_dones[idx] = False
        
        # Await all tool processing tasks
        for tool_type, task in tasks:
            observations, dones, valids = await task
            
            # Store results in the appropriate positions
            indices = indices_by_tool[tool_type]
            for idx_pos, result_idx in enumerate(indices):
                all_observations[result_idx] = observations[idx_pos]
                all_dones[result_idx] = dones[idx_pos]
                all_valids[result_idx] = valids[idx_pos]
                
        return all_observations, all_dones, all_valids


# ---- Server Implementation ----

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
    ):
        """
        Initialize the tool server
        
        Args:
            tool_types: Tool types to initialize
            host: Server host
            port: Server port
            workers_per_tool: Number of workers per tool
            max_concurrent_requests: Maximum number of concurrent requests
        """
        self.host = host
        self.port = port
        self.max_concurrent_requests = max_concurrent_requests
        
        if not use_ray:
            # Initialize async tool manager
            self.tool_manager = AsyncToolManager(tool_types, workers_per_tool, use_tqdm, done_if_invalid)
        else:
            from .ray_utils import RayToolManager
            self.tool_manager = RayToolManager(tool_types, workers_per_tool, use_tqdm, done_if_invalid)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Async Tool Server",
            description="A server for executing tools based on agent requests using asyncio",
            version="1.0.0",
        )
        self.processing_tasks = {}
        
        # Set up routes and event handlers
        self._configure_app()
        
    async def _decrement_reference_counter(self, data_hash_str):
        """Decrement reference counter and clean up if no more references"""
        if data_hash_str in self.processing_tasks:
            self.processing_tasks[data_hash_str]['ref_count'] -= 1
            # If no more references, remove from processing tasks
            if self.processing_tasks[data_hash_str]['ref_count'] <= 0:
                del self.processing_tasks[data_hash_str]
                logger.debug(f"Cleaned up completed task: {data_hash_str}")
        
    def _configure_app(self):
        """Configure FastAPI app with routes and event handlers"""
        
        # Create a semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        @self.app.post("/get_observation", response_model=AgentResponse)
        async def handle_observation_request(request: Request, background_tasks: BackgroundTasks):
            async with semaphore:
                # Parse request
                data = await request.json()
                data_hash_str = hash_requests(data)
                logger.debug(f"Request hash: {data_hash_str}")
                
                # Check if this request is already being processed
                if data_hash_str in self.processing_tasks:
                    self.processing_tasks[data_hash_str]["ref_count"] += 1
                    logger.debug(f"Duplicate request detected: {data_hash_str}")
                    # Wait for the original request to complete
                    while True:
                        # Check if result is available
                        if self.processing_tasks[data_hash_str]['result'] is not None:
                            logger.debug(f"Result for duplicate request {data_hash_str} is ready")
                            response = self.processing_tasks[data_hash_str]['result']
                            # Schedule background task to decrement reference counter
                            background_tasks.add_task(self._decrement_reference_counter, data_hash_str)
                            return response
                        # Wait a bit before checking again
                        await asyncio.sleep(0.5)
                else:
                    self.processing_tasks[data_hash_str] = {"ref_count": 1, "result": None}
                    try:
                        # Handle raw request data first for more flexible input handling
                        # Convert any numeric trajectory_ids to strings
                        if "trajectory_ids" in data:
                            data["trajectory_ids"] = [str(tid) if not isinstance(tid, str) else tid 
                                                    for tid in data.get("trajectory_ids", [])]
                        
                        # Validate and process request
                        trajectory_ids = data.get("trajectory_ids", [])
                        actions = data.get("actions", [])
                        if 'extra_fields' in data.keys():
                            extra_fields = data['extra_fields']
                            for key in data.keys():
                                assert len(data[key]) == len(trajectory_ids), f"Length of {key} ({len(data[key])}) does not match trajectory_ids ({len(trajectory_ids)})"
                                if key not in ["trajectory_ids", "actions", "extra_fields"]:
                                    for i in range(len(trajectory_ids)):
                                        extra_fields[i][key] = data[key][i]
                            assert len(extra_fields) == len(trajectory_ids), f"Length of extra_fields ({len(extra_fields)}) does not match trajectory_ids ({len(trajectory_ids)})"
                        else:
                            extra_keys = [k for k in data.keys() if k not in ["trajectory_ids", "actions"]]
                            extra_fields = [
                                {key: data[key][i] for key in extra_keys} 
                                for i in range(len(trajectory_ids))
                            ]
                        
                        observations, dones, valids = await self.tool_manager.process_actions(
                            trajectory_ids,
                            actions,
                            extra_fields
                        )
                        
                        # Create response
                        response = AgentResponse(
                            observations=observations,
                            dones=dones,
                            valids=valids
                        )
                        # import json
                        # with open(f"tmp_requests/request_response_{data_hash_str}.json", "w") as f:
                        #     json.dump([
                        #         {
                        #             "trajectory_id": trajectory_ids[i],
                        #             "action": actions[i],
                        #             "extra_field": extra_fields[i],
                        #             "observation": observations[i],
                        #             "done": dones[i],
                        #             "valid": valids[i]
                        #         } for i in range(len(trajectory_ids))
                        #     ], f, indent=4)
                        logger.debug(f"Sending response: {response}")
                        # Store the result for potential duplicate requests
                        self.processing_tasks[data_hash_str]['result'] = response
                        # Schedule background task to decrement reference counter
                        background_tasks.add_task(self._decrement_reference_counter, data_hash_str)
                        return response
                        
                    except Exception as e:
                        # On error, remove the task from processing
                        if data_hash_str in self.processing_tasks:
                            del self.processing_tasks[data_hash_str]
                        logger.error(f"Error processing request: {e}", exc_info=True)
                        return JSONResponse(
                            status_code=500,
                            content={"error": f"Failed to process request: {str(e)}"}
                        )
            
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}
            
    
    def start(self, log_level: str = "error"):
        """Start the server"""
        logger.info(f"Starting async server on {self.host}:{self.port}")
        logger.info(f"Server configured for up to {self.max_concurrent_requests} concurrent requests")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=log_level,
        )


# ---- CLI Entry Point ----

def main(
    tool_type: Union[str, Tuple[str]] = "base",
    host: str = "0.0.0.0",
    port: int = 5000,
    workers_per_tool: int = None,
    max_concurrent_requests: int = 128,
    use_tqdm: bool = False,
    log_level: str = "error",
    slient=False,
    done_if_invalid=False,
    use_ray: bool = False,
):
    """
    Start the async tool server
    
    Args:
        host: The host address
        port: The port number
        workers_per_tool: Number of workers per tool
        max_concurrent_requests: Maximum number of concurrent requests
        tool_type: Tool type(s) to use (comma-separated string or tuple)
        log_level: Logging level (debug, info, warning, error)
    """
    if workers_per_tool is None:
        workers_per_tool = max_concurrent_requests
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.basicConfig(level=numeric_level)
        logger.setLevel(numeric_level)
    
    # Convert string to tuple of tool types if needed
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
    )
    if slient:
        import sys
        import os
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    server.start(log_level=log_level)


if __name__ == "__main__":
    fire.Fire(main)
    
    
"""
python -m verl_tool.servers.serve --tool_type "python_code" --workers_per_tool 64
"""