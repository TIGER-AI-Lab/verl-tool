from verl.workers.fsdp_workers import ActorRolloutRefWorker, Worker, DictConfig
from verl.workers.fsdp_workers import *
from verl.single_controller.base.decorator import Execute, collect_all_to_all
from functools import partial
from ..llm_agent.config import AgentActorConfig
from ..llm_agent.manager import AgentActorManager

import inspect
import re
import textwrap

class AgentActorRolloutRefWorkerMeta(type):
    def __new__(mcs, name, bases, attrs):
        if ActorRolloutRefWorker in bases:
            # Create a dictionary to store super methods
            super_methods_record = {}
            
            # First pass: get methods defined in AgentActorRolloutRefWorker
            agent_methods = {method_name for method_name, method in attrs.items() 
                           if callable(method) and not method_name.startswith('__')}
            
            # Check which methods also exist in Worker
            for method_name in agent_methods:
                if method_name not in super_methods_record and hasattr(Worker, method_name):
                    super_methods_record[method_name] = Worker.__dict__.get(method_name)
                    
            # Check which methods also exist in ActorRolloutRefWorker
            for method_name in agent_methods:
                if hasattr(ActorRolloutRefWorker, method_name):
                    super_methods_record[method_name] = ActorRolloutRefWorker.__dict__.get(method_name)
            
            # Store the dictionary in the class
            attrs['super_methods_record'] = super_methods_record
            
            
            agent_init = attrs.get('__agent_init__')
            
            # Get the source code of ActorRolloutRefWorker.__init__
            init_source = inspect.getsource(ActorRolloutRefWorker.__init__)
            
            # Remove the super().__init__() call using regex
            # This pattern matches "super().__init__()" with optional arguments and whitespace
            modified_source = re.sub(r'super\(\)\.__init__\(.*?\)', '', init_source)
            
            # Create the combined init function
            def combined_init(self, *args, **kwargs):
                # First call Worker.__init__ directly
                Worker.__init__(self)
                
                # Create a local namespace for execution
                local_vars = {'self': self}
                local_vars.update(kwargs)
                
                # Execute the modified init body (skipping the def line and indentation)
                # This executes all the code from ActorRolloutRefWorker.__init__ except super().__init__()
                exec(textwrap.dedent(modified_source.split('\n', 1)[1]), globals(), local_vars)
                
                # Call the agent_init if it exists
                if agent_init:
                    agent_init(self, *args, **kwargs)
            
            attrs['__init__'] = combined_init
            
            # Copy other methods
            for method_name, method in ActorRolloutRefWorker.__dict__.items():
                if not method_name.startswith('__') and method_name not in attrs:
                    attrs[method_name] = method
            
            # Fix bases to avoid duplication
            new_bases = []
            for base in bases:
                if base is ActorRolloutRefWorker:
                    if Worker not in new_bases:
                        new_bases.append(Worker)
                elif base not in new_bases:
                    new_bases.append(base)
            
            bases = tuple(new_bases)
        
        return super().__new__(mcs, name, bases, attrs)


def dispatch_no_change(worker_group, *args, **kwargs):
    return args, kwargs

def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup
    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return output


# # Create a decorator function
# def extend_hello(original_method):
#     def wrapped_method(self):
#         original_method(self)  # Call the original
#         print("Additional code added via decorator")
#     return wrapped_method

# def extend_generate_sequences(ori_gen_seq_func, agent_gen_seq_func):
#     """
#     This wraps the self.rollout.generate_sequences function with agent behavior.
#     """
#     def wrapped_method(self, prompts: DataProto, *args, **kwargs):
#         if
#         original_method(self
#         print("Additional code added via decorator")
#     return wrapped_method
class AgentActorRolloutRefWorker(Worker, ActorRolloutRefWorker, metaclass=AgentActorRolloutRefWorkerMeta):
    def __agent_init__(self, config: DictConfig, role: str):
        self.config = config
        self.role = role
        self.agent_config = AgentActorConfig()
        # for key in get(self.config.agent_config
        for key in getattr(self.config, 'agent', {}).keys():
            if key in self.agent_config.__dict__.keys():
                setattr(self.agent_config, key, self.config.agent[key])
        setattr(self.agent_config, 'n', self.config.rollout.n)
        print(f"AgentActorRolloutRefWorker: {self.agent_config}")
        self.model_path = self.config.model.path
        self.manager = AgentActorManager(self.model_path, self, self.agent_config)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(torch.cuda.current_device())

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        meta_info = {
            'eos_token_id':
                self.generation_config.eos_token_id
                if self.generation_config is not None else self.tokenizer.eos_token_id,
            'pad_token_id':
                self.generation_config.pad_token_id
                if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:

            # after parameters sync with rollout, offload actor model to CPU
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)

            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            if not self.agent_config.enable_agent:
                # old behavior
                output = self.rollout.generate_sequences(prompts=prompts)
            else:
                # agent behavior
                output = self.manager.run_llm_loop(prompts) # our agent behavior
            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to('cpu')
        # clear kv cache
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output