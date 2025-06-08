from dataclasses import dataclass

@dataclass
class AgentActorConfig:
    enable_agent: bool=True
    max_turns: int=0
    max_start_length: int=None
    max_prompt_length: int=None
    max_response_length: int=None
    max_obs_length: int=None
    max_action_length: int=None
    num_gpus: int=1
    tool_server_url: str = None
    n: int=1
    truncate_obs_side: str='left'
    truncate_response_side: str='left'
    agent_records_dir: str=None
    rolling_with_prompt: bool=False
    call_tool_first: bool=False
    min_action_num: int=0
    action_stop_tokens: list=None
    additional_eos_token_ids: list=None
    mask_observations: bool=True
    force_finish_for_last_turn: bool=False
    enable_mtrl: bool=False
    mtrl_sep: str=None # "\n<|im_start|>system\n{obs}<|im_end|>\n<|im_start|>assistant\n"
    turn_end_token: str="<|im_end|>"
    include_stop_str_in_output: bool=True