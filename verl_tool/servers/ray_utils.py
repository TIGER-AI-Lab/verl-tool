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
        
@ray.remote
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

class RayToolManager:
    """Tool manager that uses Ray for distributed execution"""
    
    def __init__(self, tool_types: Tuple[str], num_workers_per_tool: int = 4, use_tqdm=False, done_if_invalid=False):
        """
        Initialize tool workers as Ray actors.
        
        Args:
            tool_types: Types of tools to initialize
            num_workers_per_tool: Number of Ray workers to create per tool type
        """
        self.tool_types = tool_types
        self.workers_per_tool = num_workers_per_tool
        self.tools = {}
        self.use_tqdm = use_tqdm
        self.done_if_invalid = done_if_invalid
        
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
            self.tools[tool_type] = get_tool_cls(tool_type)()
        
        # Initialize finish tool (if needed)
        if "finish" not in self.tools:
            self.tools["finish"] = get_tool_cls("finish")()
            
        # Log available vs. active tools with emoji indicators
        logger.info("Available Tools:")
        for tool in ALL_TOOLS:
            if tool in self.tools:
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
        for tool_type, tool in self.tools.items():
            if tool_type == "finish":
                continue
                
            _, valid_ref = tool.parse_action(action)
            
            if valid_ref:
                return tool_type
                
        return None
    
    def get_tool_usage_instructions(self) -> str:
        """Get usage instructions for all available tools"""
        usage_instructions = {}
        for tool_type, tool in self.tools.items():
            if tool_type not in ["finish", "base"]:
                usage_instructions[tool_type] = tool.get_usage_inst()
                
        message = "\nYour action did not match any of the available tools, please use one of the following tools: \n"
        message += "\n".join([f"- {tool_type}: {usage_instructions[tool_type]}" for tool_type in usage_instructions])
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
            if self.done_if_invalid:
                return "", True, False
            else:
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
                tool = self.tools[tool_type]
                result_ref = ray_execute.remote(tool, trajectory_id, action, extra_field)
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
