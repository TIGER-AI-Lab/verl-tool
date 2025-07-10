from verl.workers.fsdp_workers import ActorRolloutRefWorker, Worker, DictConfig
from verl.workers.fsdp_workers import *
from verl.utils.debug.performance import simple_timer
from verl.protocol import DataProto
from ..llm_agent.config import AgentActorConfig
from ..llm_agent.manager import AgentActorManager
from .utils import SiblingMetaClass, SiblingMarker


def dispatch_no_change(worker_group, *args, **kwargs):
    return args, kwargs

def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup
    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return output

class AgentActorRolloutRefWorker(Worker, ActorRolloutRefWorker, SiblingMarker, metaclass=SiblingMetaClass):
    def __init__(self, config: DictConfig, role: str):
        self.config = config
        self.role = role
        self.manager = AgentActorManager.from_rollout_config(self, self.config, rollout_mode="sync")
        
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                if not self.agent_config.enable_agent:
                    # old behavior
                    output = self.rollout.generate_sequences(prompts=prompts)
                else:
                    # agent behavior
                    output = self.manager.run_llm_loop(prompts) # our agent behavior

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output