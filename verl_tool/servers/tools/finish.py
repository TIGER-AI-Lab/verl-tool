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
        # Priority logic moved from serve.py: check for finish condition (</answer> tag)
        # This implements the original logic: if "</answer>" in action
        if "</answer>" in action:
            # Check for <answer> tags (Search-R1 style)
            answer_matches = re.findall(r"<answer>(.*?)</answer>", action, re.DOTALL)
            if len(answer_matches) > 0:
                final_answer = answer_matches[-1].strip()
                return final_answer, True
        
        # Default behavior - trajectory ends without explicit answer
        return "", False
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        """
        Get priority for handling this action. Finish tool has highest priority for answer tags.
        This moves the finish condition logic from serve.py lines 107-109.
        
        Args:
            action: The raw action string
            extra_field: Extra fields associated with the action
        Returns:
            priority: Integer priority (-1 means cannot handle, higher numbers = higher priority)
        """
        # Highest priority for explicit finish flag (original logic from serve.py line 108)
        if extra_field.get("finish", False):
            return 1000  # Highest priority
        
        # High priority for </answer> tags (original logic from serve.py line 107)
        if "</answer>" in action:
            _, valid = self.parse_action(action)
            if valid:
                return 999  # Very high priority for answer tags
        
        # Finish tool doesn't handle other actions
        return -1
    
    def conduct_action(self, trajectory_id, action, extra_data):
        action, is_valid = self.parse_action(action)
        
        # For Search-R1 style, we don't add observation when finishing with <answer>
        # The final answer is already in the LLM response
        observation = ""
        done = True
        
        # Clean up environments for all tools
        for tool in self.other_tools:
            if tool.has_env(trajectory_id):
                tool.delete_env(trajectory_id)
        
        return observation, done, is_valid
    