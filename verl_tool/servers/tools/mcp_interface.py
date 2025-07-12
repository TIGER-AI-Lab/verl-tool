"""
add-apt-repository ppa:deki/firejail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install firejail firejail-profiles
"""
import os
import json
from .base import BaseTool, register_tool
import regex as re

from typing import Tuple, Dict, Any, Optional, Union, List

@register_tool
class MCPInterfaceTool(BaseTool):
    tool_type = "mcp_interface"
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    tool_schema_path = os.getenv("MCP_TOOL_SCHEMA_PATH", "verl_tool/servers/tools/mcp_interface_schema.json")
    def __init__(self, num_workers=1):
        super().__init__(num_workers=num_workers)
        self.mcp_tools = {}
        # Load MCP tool schema

    def get_usage_inst(self):
        return "You are able to write and execute Python code securely inside a Firejail sandbox."
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing Python code
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        has_tool_call = False
        if action.endswith("</tool_call>"):
            # Extract the JSON part from the action
            json_part = re.search(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
            if json_part:
                action = json_part.group(1)
                action = action.strip()
                # Parse the JSON string
                action = json.loads(action)
                assert "name" in action, "Action JSON must contain 'name' field"
                assert "arguments" in action, "Action JSON must contain 'arguments' field"
                action_name = action["name"]
                if action_name in self.mcp_tools:
                    has_tool_call = True
        if not has_tool_call:
            return "", False
        
        return action, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        call the action with the given arguments
        """
        raise NotImplementedError("MCPInterfaceTool does not implement conduct_action method. Use MCPInterfaceToolServer instead.")
