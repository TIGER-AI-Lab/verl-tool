import time
from collections import defaultdict
from typing import Dict

def nested_copy(obj):
    """
    Recursively copy nested objects (lists, dicts, etc.) to avoid reference issues.
    """
    if isinstance(obj, dict):
        return {k: nested_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nested_copy(item) for item in obj]
    elif hasattr(obj, 'copy'):
        return obj.copy()
    else:
        return obj
class PerformanceTimer:
    """Helper class to track execution times"""
    def __init__(self, do_timer: bool = True):
        self.timings = defaultdict(list)
        self.do_timer = do_timer # whether to actually track timings
        self.start_times = {}
    
    def start(self, operation: str):
        """Start timing an operation"""
        if not self.do_timer:
            return
        self.start_times[operation] = time.perf_counter()
    
    def end(self, operation: str):
        """End timing an operation and record the duration"""
        if not self.do_timer:
            return
        if operation in self.start_times:
            duration = time.perf_counter() - self.start_times[operation]
            self.timings[operation].append(duration)
            del self.start_times[operation]
            return duration
        return None
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics"""
        stats = {}
        for operation, times in self.timings.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        return stats
    
    def log_stats(self, logger, prefix=""):
        """Log timing statistics"""
        if not self.do_timer:
            return
        stats = self.get_stats()
        if stats:
            logger.info(f"{prefix}Performance Statistics:")
            for operation, stat in stats.items():
                logger.info(f"  {operation}: count={stat['count']}, total={stat['total']:.4f}s, "
                          f"mean={stat['mean']:.4f}s, min={stat['min']:.4f}s, max={stat['max']:.4f}s")
                