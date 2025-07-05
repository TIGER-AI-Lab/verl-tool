from .base import BaseTool, register_tool
import regex as re



@register_tool
class FinishTool(BaseTool):
    tool_type = "finish"
    timeout = 10
    
    def __init__(self, num_workers=1, other_tools:list = []):
        super().__init__(num_workers)
        self.other_tools = other_tools
    
    def get_usage_inst(self):
        return ""
    
    def parse_action(self, action:str):
        """
        Parse the raw action string to check for answer tags or finish conditions.
        Implements the finish condition logic that was originally in serve.py lines 107-109.
        """
        # Default behavior - trajectory ends without explicit answer
        return "", False
    
    def conduct_action(self, trajectory_id, action, extra_data):
        action, is_valid = self.parse_action(action)
        
        observation = ""
        done = True
        
        # Clean up environments for all tools
        for tool in self.other_tools:
            if tool.has_env(trajectory_id):
                tool.delete_env(trajectory_id)
        
        return observation, done, is_valid
    
