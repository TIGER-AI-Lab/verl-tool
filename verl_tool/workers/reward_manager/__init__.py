from verl.workers.reward_manager.registry import register, REWARD_MANAGER_REGISTRY
from pathlib import Path

def get_reward_manager_cls(name):
    """Get the reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.

    Returns:
        `(type)`: The reward manager class.
    """
    if name not in REWARD_MANAGER_REGISTRY:
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER_REGISTRY[name]

# search current directory for reward manager classes
current_dir = Path(__file__).parent
for file in current_dir.glob("*.py"):
    if file.name == "__init__.py":
        continue
    try:
        # import
        module = __import__(f"verl_tool.workers.reward_manager.{file.stem}", fromlist=[file.stem])
    except ImportError as e:
        pass
        # print(f"[Warning] Failed to import {file.stem} reward manager due to ImportError: {e}")
        # raise e