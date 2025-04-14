from dataclasses import dataclass

@dataclass
class AgentActorConfig:
    enable_agent: bool=True
    max_turns: int=0
    max_start_length: int=None
    max_prompt_length: int=None
    max_response_length: int=None
    max_obs_length: int=None
    # logging: dict
    num_gpus: int=1
    tool_server_url: str = None
    n: int=1
    truncate_obs_side: str='left'
    agent_records_dir: str=None
    valid_actions: list=None
    no_action_as_stop: bool=True
    min_action_num: int=0
    action_stop_tokens: list=None
