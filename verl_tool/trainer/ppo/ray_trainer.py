import ray
import uuid
import torch
import os
import json
import numpy as np
from copy import deepcopy
from collections import defaultdict
from typing import Optional
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
) # for train and validate
from verl.trainer.ppo.ray_trainer import (
    DataProto,
) # for init
from verl.utils.debug import marked_timer


##############################################################################
#### Replace the original classes/functions with verl-tool customized ones ####
import verl.experimental.agent_loop
from verl_tool.agent_loop import AgentLoopManager
import verl.trainer.ppo.ray_trainer
from .reward import compute_reward, compute_reward_async
from verl_tool.workers.rollout.vllm_rollout.vllm_async_server import VerlToolvLLMHttpServer
import verl.workers.rollout.vllm_rollout.vllm_async_server
verl.experimental.agent_loop.AgentLoopManager = AgentLoopManager
verl.trainer.ppo.ray_trainer.compute_reward = compute_reward
verl.trainer.ppo.ray_trainer.compute_reward_async = compute_reward_async
verl.workers.rollout.vllm_rollout.vllm_async_server.vLLMHttpServer = VerlToolvLLMHttpServer
##############################################################################

class AgentRayPPOTrainer(RayPPOTrainer):

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs_attention_masks = batch.batch['attention_mask'][:, :batch.batch['prompts'].shape[1]]
            outputs_attention_masks = batch.batch['attention_mask'][:, batch.batch['prompts'].shape[1]:]
            inputs = [self.tokenizer.decode(batch.batch["prompts"][i][inputs_attention_masks[i]==1], skip_special_tokens=False) for i in range(batch.batch["prompts"].shape[0])]
            outputs = [self.tokenizer.decode(batch.batch["responses"][i][outputs_attention_masks[i]==1], skip_special_tokens=False) for i in range(batch.batch["responses"].shape[0])]
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            
            tool_interact_info = batch.non_tensor_batch.get("tool_interact_info", None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            if tool_interact_info:
                for tool_interact in tool_interact_info:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
                reward_extra_infos_to_dump.update({
                    "tool_interact_info": tool_interact_info,
                })

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )