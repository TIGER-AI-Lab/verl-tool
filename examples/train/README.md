# Model Training Scripts

This folder contains scripts for calling the `verl-tool` pipeline for training models with tool-calling capability. The training configuration can be found in [ppo_trainer.yaml](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/trainer/config/ppo_trainer.yaml) and [config.py](https://github.com/TIGER-AI-Lab/verl-tool/blob/dev/train/verl_tool/llm_agent/config.py)

Specifically, `acecoder` is used for training tool-calling coding models. `torl` is used to train tool-calling mathematical models. Other folders are currently under development.

|Model  Name   |Tool            |Task Type|Link|
|--------------|----------------|---------|----|
|Verl-Tool-Math|Code Interpreter|Math     | [`torl`](./torl)   |
|Verl-Tool-Code|Code Interpreter|Code     |  [`acecoder`](./acecoder)  |




## Config Explanation

+ For ppo_trainer.yaml, refer to the [VERL config documentation](https://verl.readthedocs.io/en/latest/examples/config.html) for configuration details.
+ For Agent config [config.py](https://github.com/TIGER-AI-Lab/verl-tool/blob/dev/train/verl_tool/llm_agent/config.py). Below is the explanation.

```python
from dataclasses import dataclass

@dataclass
class AgentActorConfig:
    # Whether to enable the agent
    enable_agent: bool = True
    
    # Maximum number of interaction turns
    max_turns: int = 0
    
    # Maximum token length of the initial prompt (before any turns)
    max_start_length: int = None
    
    # Maximum total token length of the prompt (including turns)
    max_prompt_length: int = None
    
    # Maximum token length of the response (e.g., LLM response + observation)
    max_response_length: int = None
    
    # Maximum token length of the observation from environment
    max_obs_length: int = None
    
    # Maximum token length of the action (e.g., LLM response)
    max_action_length: int = None
    
    # Number of GPUs used for inference or training
    num_gpus: int = 1
    
    # URL of the tool server to call tools or APIs
    tool_server_url: str = None
    
    # Number of response samples 
    n: int = 1
    
    # Truncation direction for observations if they exceed length limit ('left' or 'right')
    truncate_obs_side: str = 'left'
    
    # Truncation direction for responses if they exceed length limit ('left' or 'right')
    truncate_response_side: str = 'left'
    
    # Directory to save agent records (e.g., logs or results)
    agent_records_dir: str = None
    
    # If rolling_with_prompt is True, then we will keep the system prompt when truncation 
    rolling_with_prompt: bool = False
    
    # Whether to call tool before generating the response
    call_tool_first: bool = False
    
    # Minimum number of actions required before allowing the agent to finish
    min_turns: int = 0
    
    # List of stop tokens that indicate the end of an action
    action_stop_tokens: list = None
    
    # List of additional token treated as end-of-sequence
    additional_eos_token_ids: list = None
    
    # Whether to mask (hide) observations from the model input
    mask_observations: bool = True
    
    # Force the agent to end after the last turn
    force_finish_for_last_turn: bool = False
    
    # Whether to enable multi-turn reinforcement learning
    enable_mtrl: bool = False
    
    # Separator format used in MTRL mode
    mtrl_sep: str = None  # Example: "\n<|im_start|>system\n{obs}<|im_end|>\n<|im_start|>assistant\n"
    
    # Token used to mark the end of each turn
    turn_end_token: str = "<|im_end|>"


```
