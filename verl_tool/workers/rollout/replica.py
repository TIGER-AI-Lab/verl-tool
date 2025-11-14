from verl.workers.rollout.replica import get_rollout_replica_class, TokenOutput
from typing import Any, Optional

class VerlToolTokenOutput(TokenOutput):
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None
    text: Optional[str] = None
    finished: Optional[bool] = None
    
    
