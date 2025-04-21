from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

class ModelConfig(BaseModel):
    model_path: str
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    temperature: float = 0
    top_p: float = 0.9

class ToolConfig(BaseModel):
    tool_server_url: str = "http://localhost:30150/get_observation"
    valid_actions: List[str] = ["python"]  # list of valid tool actions, will automatically add "```"
    stop_tokens: List[str] = ["```output"]
    action_stop_token: str = "```output"
    max_turns: int = 5  # max generation turns
    no_action_as_stop: bool = True  # if no action, stop generation
    min_action_num: int = 1  # minimun number of tool-calling actions required
    truncate_obs_side: str = "left"  # "left" or "right", which side to truncate when the observation is too long
    max_prompt_length: int = 4096  # maximum length of prompt
    max_obs_length: int = 1024  # maximum length of observation
    max_start_length: int = 4096  
    max_response_length: int = 4096

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    llm_config: ModelConfig
    tool_config: ToolConfig
    execution_prompt: str = """\
Answer the given coding question. You must conduct reasoning inside <think> and </think> first before you can finally output the final program. During the thinking, you can test your program by writing it inside <python> and </python> tags. The code will be executed, and the terminal output (standard output and standard error) will be returned between <output> and </output>. Each program between <python> and </python> tags are independent program. You can run Python code as many times as you want. If you find no further code execution needed, you can then give the final program in a markdown code block like this: ```python\\nyour code here\\n```. The final program will be evaluated against the test cases.
"""