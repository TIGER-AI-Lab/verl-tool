from dataclasses import dataclass

@dataclass
class AgentActorConfig:
    enable_agent: bool=True
    max_turns: int=0
    min_turns: int=0
    max_start_length: int=None
    max_prompt_length: int=None
    max_response_length: int=None
    max_model_len: int=None  # Maximum model length, used for async rollout to limit the input length.
    max_obs_length: int=None
    max_action_length: int=None
    tool_server_url: str = None
    n: int=1
    truncate_obs_side: str='left'
    truncate_response_side: str='left'
    rolling_with_prompt: bool=False
    call_tool_first: bool=False
    action_stop_tokens: list=None
    additional_eos_token_ids: list=None
    mask_observations: bool=True
    force_finish_for_last_turn: bool=False
    enable_mtrl: bool=False
    mtrl_role: str="user"
    mtrl_sep: str=None # "\n<|im_start|>system\n{obs}<|im_end|>\n<|im_start|>assistant\n"
    assistant_role: str="assistant"
    turn_end_token: str="<|im_end|>"
    rollout_mode: str="async" # "sync" or "async"
    mask_overlong_loss: bool=False # whether to mask the overlong trajectory to not train on it
    max_concurrent_trajectories: int=256 # Maximum number of concurrent trajectories for async rollout. If None, no limit is applied.
    enable_tqdm: bool=True # Whether to enable tqdm for async rollout.
    over_sampling: bool=False # Whether to over-sample the trajectories in async rollout.
    tool_call_time_out: int=None # Timeout for tool calls in async rollout.
    tool_call_max_retries: int=5 # Maximum number of retries for tool calls in async rollout.