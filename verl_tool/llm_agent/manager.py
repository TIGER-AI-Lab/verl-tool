import torch
import os
import re
import time
import ray
import uuid
import json
import random
import logging
import asyncio
import aiohttp
import regex as re
import numpy as np
import requests
import omegaconf
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.utils import hf_tokenizer
from verl.utils.model import get_generation_config
from tqdm import tqdm
from typing import List, Union
from .config import AgentActorConfig
from .tensor_helper import TensorHelper, TensorConfig
from .utils import PerformanceTimer
logger = logging.getLogger(__file__)

# 1) A sanitizer that strips all embedded NULs (and, optionally, any
#    other C0 control characters except common whitespace).
CONTROL_CHAR_RE = re.compile(
    # this matches U+0000 through U+001F, excluding tab(09), LF(0A), CR(0D)
    r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
)

def sanitize_request(obj: Any) -> Any:
    """
    Recursively walk through obj and:
      - For dicts: sanitize each value
      - For lists/tuples: sanitize each element
      - For strings: remove embedded nulls (and other control chars)
      - Leave other types untouched
    """
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {sanitize_request(key): sanitize_request(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_request(item) for item in obj)
    elif isinstance(obj, str):
        # strip NUL (\x00) and other C0 control chars
        return CONTROL_CHAR_RE.sub('', obj)
    else:
        return obj

class AgentActorManager:
    def __init__(
        self,
        model_path,
        actor_rollout_wg,
        config: AgentActorConfig,
        is_validation: bool = False,
    ):
        self.model_path = model_path
        self.tokenizer = hf_tokenizer(self.model_path)
        self.generation_config = get_generation_config(self.model_path)
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation
        self.eos_token_id = self.generation_config.eos_token_id \
            if self.generation_config is not None else self.tokenizer.eos_token_id
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length,
            max_response_length=config.max_response_length,
        ))
        if self.config.action_stop_tokens is not None:
            if os.path.exists(self.config.action_stop_tokens):
                with open(self.config.action_stop_tokens, 'r') as f:
                    self.action_stop_tokens = [x for x in f.read().split(',') if x]
                logger.info(f"Using action stop tokens: {self.action_stop_tokens}")
            else:
                raise ValueError(f"action_stop_tokens file not found: {self.config.action_stop_tokens}")
        else:
            self.action_stop_tokens = []
        self.additional_eos_token_ids = self.config.additional_eos_token_ids
        if isinstance(self.additional_eos_token_ids, str):
            self.additional_eos_token_ids = [int(x) for x in self.additional_eos_token_ids.split(',')]
        elif isinstance(self.additional_eos_token_ids, list) or isinstance(self.additional_eos_token_ids, omegaconf.listconfig.ListConfig):
            self.additional_eos_token_ids = [int(x) for x in self.additional_eos_token_ids]
        elif self.additional_eos_token_ids is None:
            self.additional_eos_token_ids = []
        if self.config.mtrl_sep is None:
            messages = [{"role": "system", "content": "{obs}"}]
            self.config.mtrl_sep = "\n" + self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self.config.mtrl_sep = self.config.mtrl_sep.replace("system", self.config.mtrl_role)
        self.max_action_length = self.config.max_action_length if self.config.max_action_length is not None else 0
        self.max_model_len = int(config.max_model_len or config.max_prompt_length + config.max_response_length)
        self.tokenizer_lock = asyncio.Lock()

        if self.config.rollout_mode == "sync":
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )['input_ids']

    def repeat_inputs_by_n(self, inputs: DataProto):
        """
        this version verl do not repeat the input by n times, so we manually repeat the input by n times
        """
        if inputs.meta_info.get("is_repeated_by_n", False):
            # if the inputs are already repeated by n times, we do not need to repeat again
            return inputs

        # we manually repeat the input by n times if needed since every trajectory is independent
        do_sample = inputs.meta_info.get("do_sample", True)
        assert 'traj_ids' in inputs.non_tensor_batch, "traj_ids should be claimed univerally in the ray trainer"
        ori_len = len(inputs.batch['input_ids'])
        if not do_sample:
            n = 1
        else:
            n = self.config.n
            inputs = inputs.repeat(n, interleave=True)
        # add "_{i}" for each trajectory to the traj_ids
        for i in range(ori_len):
            for j in range(n):
                inputs.non_tensor_batch['traj_ids'][i*n+j] += f"_{j}"
        inputs.meta_info['is_repeated_by_n'] = True
        return inputs

    async def _postprocess_responses(self, responses: Union[torch.Tensor, List[str]], action_step: int) -> torch.Tensor:
        """Process responses to stop at python operation or answer operation."""
        effective_lens = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
        do_actions = []
        async with self.tokenizer_lock:
            if self.config.enable_mtrl:
                if isinstance(responses, torch.Tensor):
                    responses_str = [self.tokenizer.decode(responses[i][:effective_lens[i]], skip_special_tokens=False) for i in range(responses.shape[0])]
                else:
                    responses_str = responses
                for i in range(len(responses_str)):
                    if action_step >= self.config.min_turns:
                        if self.action_stop_tokens:
                            if any([action_stop_token in responses_str[i] for action_stop_token in self.action_stop_tokens]):
                                do_action = True
                                # replace other action stop tokens with the first one
                                for j in range(1, len(self.action_stop_tokens)):
                                    if self.action_stop_tokens[j] in responses_str[i]:
                                        responses_str[i] = responses_str[i].replace(self.action_stop_tokens[j], self.action_stop_tokens[0])
                                if not responses_str[i].endswith(self.config.turn_end_token):
                                    responses_str[i] += self.config.turn_end_token
                            else:
                                do_action = False
                        else:
                            if not responses_str[i].endswith(self.config.turn_end_token):
                                responses_str[i] += self.config.turn_end_token
                            do_action = True
                    else:
                        # always do action, decided by the server about whether an action stops
                        for j in range(1, len(self.action_stop_tokens)):
                            if self.action_stop_tokens[j] in responses_str[i]:
                                responses_str[i] = responses_str[i].replace(self.action_stop_tokens[j], self.action_stop_tokens[0])
                        turn_end_token_idx = responses_str[i].rfind(self.config.turn_end_token)
                        if self.action_stop_tokens and not self.action_stop_tokens[0] in responses_str[i]:
                            if turn_end_token_idx != -1:
                                responses_str[i] = responses_str[i][:turn_end_token_idx] + self.action_stop_tokens[0] + self.config.turn_end_token
                            else:
                                responses_str[i] = responses_str[i] + self.action_stop_tokens[0] + self.config.turn_end_token
                        else:
                            if turn_end_token_idx == -1:
                                responses_str[i] += self.config.turn_end_token
                        do_action = True
                    do_actions.append(do_action)
            else:
                if isinstance(responses, torch.Tensor):
                    responses_str = self.tokenizer.batch_decode(
                        responses,
                        skip_special_tokens=True
                    )
                else:
                    responses_str = responses
                for i, resp in enumerate(responses_str):
                    # resp = resp.strip(' \n')
                    has_action = False
                    for j in range(len(self.action_stop_tokens)):
                        if self.action_stop_tokens[j] in resp:
                        # if resp.endswith(self.action_stop_tokens[j]):
                        # if self.action_stop_tokens[j] in resp[-(len(self.action_stop_tokens[j]) + 3):]: # 5 for some action token tokens not indepdently decoded
                            has_action = True
                            responses_str[i] = resp.split(self.action_stop_tokens[j])[0] + self.action_stop_tokens[j]
                            break
                    if not has_action and action_step < self.config.min_turns:
                        has_action = True
                        responses_str[i] = resp + self.action_stop_tokens[0]
                    do_actions.append(has_action)
                for i in range(len(responses_str)):
                    if not do_actions[i]:
                        responses_str[i] = self.tokenizer.decode(responses[i][:effective_lens[i]], skip_special_tokens=False) # preserve eos token
            # with open(f"temp-{action_step}.json", 'w') as f:
            #     json.dump([{
            #         "responses_str": responses_str[i],
            #         "do_action": do_actions[i],
            #     } for i in range(len(responses_str))], f, indent=4)
            responses = self._batch_tokenize(responses_str).to(torch.int64)
        # overlong dones
        effective_lens = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
        for i in range(len(responses_str)):
            if effective_lens[i] >= self.config.max_response_length:
                logger.warning(f"[WARNING] Response too long, consider changing your config, {effective_lens[i]} & {self.config.max_response_length}")
                do_actions[i] = False
        return responses, responses_str, do_actions

    async def _process_next_obs(self, next_obs: List[str], dones: List[bool], valid_action: List[bool], finishs: List[bool]) -> torch.Tensor:
        """Process next observations from environment."""
        async with self.tokenizer_lock:
            mtrl_sep = self.config.mtrl_sep
            next_obs = [obs if not done else "" for obs, done in zip(next_obs, dones)]
            if self.config.truncate_obs_side == 'left':
                next_obs_ids = self.tokenizer(
                    next_obs,
                    padding='longest',
                    return_tensors='pt',
                    add_special_tokens=False,  # Prevents adding special tokens
                    padding_side='left',
                )['input_ids'].to(torch.int64)
                if next_obs_ids.shape[1] > self.config.max_obs_length:
                    logger.warning(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")
                    next_obs_ids = next_obs_ids[:, -self.config.max_obs_length:]
            elif self.config.truncate_obs_side == 'right':
                next_obs_ids = self.tokenizer(
                    next_obs,
                    padding='longest',
                    return_tensors='pt',
                    add_special_tokens=False,  # Prevents adding special tokens
                    padding_side='right',
                )['input_ids'].to(torch.int64)
                if next_obs_ids.shape[1] > self.config.max_obs_length:
                    logger.warning(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")
                    next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
            else:
                raise ValueError(f"Invalid truncate_obs_side: {self.config.truncate_obs_side}")
            if self.config.enable_mtrl:
                next_obs = self.tokenizer.batch_decode(
                    next_obs_ids,
                    skip_special_tokens=True
                )
                processed_next_obs = []
                for i in range(len(next_obs)):
                    if finishs[i] or dones[i]:
                        # do action is false
                        assert next_obs[i] == "", f"next_obs should be empty when finishs is True, but got {next_obs[i]}"
                        processed_next_obs.append("")
                    elif valid_action[i]:
                        processed_next_obs.append(mtrl_sep.format(obs=next_obs[i]))
                    else:
                        processed_next_obs.append(mtrl_sep.format(obs="Your action is not valid, please check the format and try again." + next_obs[i]))
                next_obs = processed_next_obs
                next_obs_ids = self.tokenizer(
                    next_obs,
                    padding='longest',
                    return_tensors='pt',
                    add_special_tokens=False,  # Prevents adding special tokens
                )['input_ids'].to(torch.int64)

        return next_obs_ids

    def _update_rolling_state(self, left_side, rollings, cur_responses: torch.Tensor,
                              next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""

        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_lens = new_attention_mask.sum(dim=1)
        effective_len = effective_lens.max()
        min_effective_len = effective_lens.min().item()
        # max_len = min(self.config.max_prompt_length, effective_len)
        max_len = min(self.config.max_prompt_length+self.config.max_response_length, effective_len)
        available_context_budget = max(0, self.config.max_prompt_length+self.config.max_response_length - min_effective_len)
        assert isinstance(available_context_budget, int), f"available_context_budget should be int, but got {type(available_context_budget)}"
        if getattr(self.config, "rolling_with_prompt", False):
            # if rolling_with_prompt is True, then we need to keep the system prompt
            if isinstance(left_side, dict):
                left_ids = left_side["input_ids"]
            else:
                left_ids = left_side.batch["input_ids"]

            left_len = left_ids.size(1)

            if left_len >= max_len:
                final_input_ids = left_ids[:, -max_len:]
            else:
                right_budget = max_len - left_len
                right_ids_full = new_input_ids[:, left_len:]
                right_ids = right_ids_full[:, -right_budget:] if right_budget < right_ids_full.size(1) else right_ids_full
                final_input_ids = torch.cat([left_ids, right_ids], dim=1)

            final_attention_mask = self.tensor_fn.create_attention_mask(final_input_ids)
            final_position_ids = self.tensor_fn.create_position_ids(final_attention_mask)

            new_rollings = DataProto.from_dict(
                {
                    "input_ids": final_input_ids,
                    "position_ids": final_position_ids,
                    "attention_mask": final_attention_mask,
                }
            )
        else: # By default keep the right side
            new_rollings = DataProto.from_dict(
                {
                    "input_ids": new_input_ids[:, -max_len:],
                    "position_ids": new_position_ids[:, -max_len:],
                    "attention_mask": new_attention_mask[:, -max_len:],
                }
            )
        new_rollings.non_tensor_batch = rollings.non_tensor_batch.copy()
        new_rollings.meta_info.update(rollings.meta_info)
        
        # update raw_prompt_ids, required for vllm inference
        ray_prompt_ids = []
        for i in range(new_rollings.batch['input_ids'].size(0)):
            non_pad_index = torch.nonzero(new_rollings.batch['input_ids'][i] != self.tokenizer.pad_token_id, as_tuple=False)[0][0]
            ray_prompt_ids.append(new_rollings.batch['input_ids'][i][non_pad_index:].tolist())
        new_rollings.non_tensor_batch['raw_prompt_ids'] = np.array(ray_prompt_ids, dtype=object)

        return new_rollings, available_context_budget

    def _loss_masked_concatenate_with_padding(self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
        pad_to_left: bool = True
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (loss_mask) to cover the information block if it exists."""
        # move `response` and `info` tensor to the same device as `prompt`
        response = response.to(prompt.device)
        if info is not None:
            info = info.to(prompt.device)

        # set padding ids
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]

        # info: observations, need to be masked
        if info is not None:
            # for non-masked tensors, just append the observation
            tensors.append(info)

            # assemble the mask for the observation part
            loss_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)  # information mask
            # extend the mask for the observation part, to update masked tensors
            tensors_with_mask.append(loss_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)

        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(
        self,
        right_side: Dict,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor = None
    ) -> Dict:
        """Update right side state."""

        # observation exists, perform concatenation and masked concatenation
        if next_obs_ids != None:
            responses, responses_with_loss_mask = self._loss_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_loss_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False
            )
        else:
            # no observation, only concatenate the response with generated response
            responses, responses_with_loss_mask = self._loss_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_loss_mask'],
                    cur_responses,
                    pad_to_left=False
                )

        effective_lens = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
        effective_len = effective_lens.max()

        max_len = min(self.config.max_response_length, effective_len)

        # return the updated responses along with its masked version
        if self.config.truncate_response_side == 'left':
            # it should be left most of the time.
            return {'responses': responses[:, :max_len],
                    'responses_with_loss_mask': responses_with_loss_mask[:, :max_len]}
        elif self.config.truncate_response_side == 'right':
            return {'responses': responses[:, -max_len:],
                    'responses_with_loss_mask': responses_with_loss_mask[:, -max_len:]}
        else:
            raise ValueError(
                f"Invalid truncate_response_side: {self.config.truncate_response_side}. Allowed options are 'left' or 'right'.")

    async def generate_sequences(self, prompts: DataProto, **sampling_params: Dict[str, Any]) -> DataProto:
        if self.config.rollout_mode == "async":
            return await self.actor_rollout_wg.simple_generate_sequences(prompts, **sampling_params)
        elif self.config.rollout_mode == "sync":
            with self.actor_rollout_wg.rollout.update_sampling_params(**sampling_params):
                gen_output = self.actor_rollout_wg.rollout.generate_sequences(prompts, **sampling_params) # [active_size, response_length]
            return gen_output
        else:
            raise ValueError(f"Invalid rollout_mode: {self.config.rollout_mode}. Allowed options are 'async' or 'sync'.")

    # Instead of creating new masks repeatedly
    def _update_active_mask_inplace(self, active_mask: torch.Tensor, new_conditions: torch.Tensor):
        """Update active mask in-place to avoid memory allocation"""
        active_mask &= new_conditions
        return active_mask.sum().item()  # Return count for logging

    async def run_llm_loop_async(self, gen_batch: DataProto, **sampling_params: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        perf_timer = PerformanceTimer(do_timer=False)
        perf_timer.start('run_llm_loop_total')
        perf_timer.start('initialization')
        
        ori_meta_info = gen_batch.meta_info
        if 'eos_token_id' not in ori_meta_info:
            stop_token_ids = self.tokenizer.eos_token_id + self.additional_eos_token_ids if isinstance(self.tokenizer.eos_token_id, list) else [self.tokenizer.eos_token_id] + self.additional_eos_token_ids
        elif isinstance(ori_meta_info['eos_token_id'], list):
            stop_token_ids = ori_meta_info['eos_token_id'] + self.additional_eos_token_ids
        else:
            stop_token_ids = [ori_meta_info['eos_token_id']] + self.additional_eos_token_ids
        gen_batch = self.repeat_inputs_by_n(gen_batch)

        initial_input_ids = gen_batch.batch['input_ids'][:, -self.config.max_start_length:].clone()

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []],
                               'responses_with_loss_mask': initial_input_ids[:, []]}

        turns_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool) # [bs*n]
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        traj_ids = gen_batch.non_tensor_batch['traj_ids']

        turns_stats_extra_keys = ['action_lengths', 'obs_lengths', 'rewards', 'tool_interact_info']
        turns_stats_extra = {}
        for key in turns_stats_extra_keys:
            turns_stats_extra[key] = np.empty((gen_batch.batch['input_ids'].shape[0],), dtype=object)  # rewards can be None, so we use object type
            for i in range(gen_batch.batch['input_ids'].shape[0]):
                turns_stats_extra[key][i] = []
        agent_sampling_params = sampling_params.copy()
        agent_sampling_params.update({
            "n": 1,  # already repeated by n times in repeat_inputs_by_n
            "stop": self.action_stop_tokens,  # stop when generated an end of action
            "include_stop_str_in_output": True,
            "detokenize": True,
            "stop_token_ids": stop_token_ids,
            # "allowed_token_ids": list(range(self.tokenizer.vocab_size)) # see vllm issue: # 1398
        })
        available_context_budget = self.config.max_response_length
        available_context_budget = min(available_context_budget, self.config.max_action_length)
        agent_sampling_params['max_tokens'] = available_context_budget # for vllm
        agent_sampling_params['max_new_tokens'] = available_context_budget # for sglang

        perf_timer.end('initialization')

        if self.config.call_tool_first:
            perf_timer.start('initial_tool_call')
            # Added Zhiheng: Add initial observation to the prompt from server, use response=""
            do_actions = [True] * len(traj_ids)
            responses_str = [''] * len(traj_ids)
            responses_ids = torch.zeros((len(traj_ids), 1), dtype=torch.int64)
            active_uids = [traj_ids[i] for i in range(len(traj_ids)) if active_mask[i]]
            next_obs, dones, valid_action, finishs, rewards, tool_interact_info = await self.interact_with_tool_server(
                active_uids, responses_str, do_actions, active_mask,
                extra_fields=rollings.non_tensor_batch.get('extra_info', None)
            )
            for i, reward in enumerate(rewards):
                if rewards[i] is not None and active_mask[i]:
                    turns_stats_extra["rewards"][i].append(reward)
                turns_stats_extra["tool_interact_info"][i].append(tool_interact_info[i])
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_num_list.append(self._update_active_mask_inplace(active_mask, curr_active_mask))
            # turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            next_obs_ids = await self._process_next_obs(next_obs, dones, valid_action, finishs) # [active_size, obs_length]

            obs_idx = 0
            for i, active in enumerate(active_mask):
                if i >= len(turns_stats_extra["obs_lengths"]):
                    break
                if active:
                    obs_length = next_obs_ids[obs_idx].shape[0]
                    turns_stats_extra["obs_lengths"][i].append(int(obs_length))
                    obs_idx += 1
                else:
                    turns_stats_extra["obs_lengths"][i].append(0)

            rollings, available_context_budget = self._update_rolling_state(
                original_left_side,
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            agent_sampling_params['max_tokens'] = available_context_budget
            active_num_list.append(active_mask.sum().item())
            perf_timer.end('initial_tool_call')

        # Main generation loop
        perf_timer.start('main_generation_loop')
        for step in range(self.config.max_turns+1):
            if not active_mask.any():
                break

            step_timer_key = f'step_{step}'
            perf_timer.start(step_timer_key)
            perf_timer.start(f'step_{step}_preparation')

            logger.info(f"Action step {step}/{self.config.max_turns}")
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            ) # TODO: delete
            rollings_active = DataProto.from_dict(
                {k: v[active_mask] for k, v in rollings.batch.items()},
                {k: v[active_mask.numpy()] for k, v in rollings.non_tensor_batch.items()},
                meta_info=ori_meta_info
            )
            if step == self.config.max_turns and self.config.force_finish_for_last_turn:
                # remove the action stop tokens in the last turn to force a finish
                agent_sampling_params.pop('stop')
            
            perf_timer.end(f'step_{step}_preparation')
            
            # Time the generation
            perf_timer.start(f'step_{step}_generation')
            gen_output = await self.generate_sequences(rollings_active, **agent_sampling_params) # [active_size, response_length]
            perf_timer.end(f'step_{step}_generation')

            # Time the postprocessing
            perf_timer.start(f'step_{step}_postprocess')
            responses_ids, responses_str, do_actions = await self._postprocess_responses(gen_output.batch['responses'], step) # [active_size, ...]
            responses_ids, _ = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask) # [bs*n, response_length]
            perf_timer.end(f'step_{step}_postprocess')

            logger.info(f"Number of active trajectories: {active_mask.sum().item()}")
            logger.info(f"Length of responses: {responses_ids.shape[1]}")

            perf_timer.start(f'step_{step}_action_length_tracking')
            async with self.tokenizer_lock:
                idx = 0
                for i, active in enumerate(active_mask):
                    if active:
                        action_length = len(self.tokenizer.encode(responses_str[idx], add_special_tokens=False))
                        turns_stats_extra["action_lengths"][i].append(action_length)
                        idx += 1
                    else:
                        turns_stats_extra["action_lengths"][i].append(0)
            perf_timer.end(f'step_{step}_action_length_tracking')

            # Execute in environment and process observations
            perf_timer.start(f'step_{step}_tool_interaction')
            active_uids = [traj_ids[i] for i in range(len(traj_ids)) if active_mask[i]]
            next_obs, dones, valid_action, finishs, rewards, tool_interact_info = await self.interact_with_tool_server(
                active_uids, responses_str, do_actions, active_mask,
                extra_fields=rollings_active.non_tensor_batch.get('extra_info', None),
                is_last_step=(step == self.config.max_turns)
            )
            for i, reward in enumerate(rewards):
                if rewards[i] is not None and active_mask[i]:
                    turns_stats_extra["rewards"][i].append(reward)
                turns_stats_extra["tool_interact_info"][i].append(tool_interact_info[i])
            perf_timer.end(f'step_{step}_tool_interaction')

            perf_timer.start(f'step_{step}_state_updates')
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            self._update_active_mask_inplace(active_mask, curr_active_mask)
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)

            next_obs_ids = await self._process_next_obs(next_obs, dones, valid_action, finishs) # [active_size, obs_length]

            obs_idx = 0
            for i, active in enumerate(active_mask):
                if i >= len(turns_stats_extra["obs_lengths"]):
                    break
                if active:
                    obs_length = next_obs_ids[obs_idx].shape[0]
                    turns_stats_extra["obs_lengths"][i].append(int(obs_length))
                    obs_idx += 1 
                else:
                    turns_stats_extra["obs_lengths"][i].append(0)

            # Update states
            rollings, available_context_budget = self._update_rolling_state(
                original_left_side,
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            available_context_budget = min(available_context_budget, self.config.max_action_length)
            agent_sampling_params['max_tokens'] = available_context_budget # for vllm
            agent_sampling_params['max_new_tokens'] = available_context_budget # for sglang
            perf_timer.end(f'step_{step}_state_updates')
            
            perf_timer.end(step_timer_key)

        perf_timer.end('main_generation_loop')

        perf_timer.start('final_composition')
        non_tensors = {
            'traj_ids': traj_ids.tolist(),
            'turns_stats': turns_stats.tolist(),
            'valid_action_stats': valid_action_stats.tolist(),
            'active_mask': active_mask.tolist(),
            'action_lengths': turns_stats_extra["action_lengths"],
            'obs_lengths': turns_stats_extra["obs_lengths"],
            'turn_rewards': turns_stats_extra["rewards"],
            'tool_interact_info': turns_stats_extra["tool_interact_info"],
        }

        logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")

        results = self._compose_final_output(original_left_side, original_right_side, non_tensors, ori_meta_info)
        perf_timer.end('final_composition')
        
        perf_timer.end('run_llm_loop_total')
        
        # Log performance statistics
        perf_timer.log_stats(logger, f"[PERF] Batch size: {gen_batch.batch['input_ids'].shape[0]} - ")
        
        return results
    
    def run_llm_loop(self, gen_batch: DataProto, **sampling_params: Dict[str, Any]) -> Tuple[Dict, Dict]:
        return asyncio.run(self.run_llm_loop_async(gen_batch, **sampling_params))

    def _compose_final_output(
        self,
        left_side: Dict,
        right_side: Dict,
        non_tensors: Dict,
        meta_info: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Compose the final output of the rollout by merging prompt and response
        components, padding sequences as needed, and ensuring all turn-level
        non-tensor fields are aligned in shape for safe concatenation across samples.
        """
        # ---------- 1. Pad turn-level lists to the same length ----------
        pad_len = self.config.max_turns + 2  # buffer to avoid mismatch

        def _pad(seq_list, fill_value=0):
            """
            Pad or truncate a list to match pad_len.
            This is used for per-turn statistics like action_lengths or obs_lengths.
            """
            if len(seq_list) < pad_len:
                seq_list += [fill_value] * (pad_len - len(seq_list))
            else:
                seq_list[:] = seq_list[:pad_len]
            return seq_list

        if "action_lengths" in non_tensors:
            non_tensors["action_lengths"] = [
                _pad(traj, 0) for traj in non_tensors["action_lengths"]
            ]
        if "obs_lengths" in non_tensors:
            non_tensors["obs_lengths"] = [
                _pad(traj, 0) for traj in non_tensors["obs_lengths"]
            ]

        # ---------- 2. Build final tensor fields ----------
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids'] # [bs*n, prompt_length]

        # padding responses length to max_response_length
        if final_output['responses'].shape[1] < self.config.max_response_length:
            final_output['responses'] = self.tensor_fn.pad_tensor(
                final_output['responses'],
                max_length=self.config.max_response_length,
                padding_side='right'
            ) # [bs*n, max_response_length]

        # padding response_with_loss_mask length to max_response_length
        if final_output['responses_with_loss_mask'].shape[1] < self.config.max_response_length:
            final_output['responses_with_loss_mask'] = self.tensor_fn.pad_tensor(
                final_output['responses_with_loss_mask'],
                max_length=self.config.max_response_length,
                padding_side='right'
            ) # [bs*n, max_response_length]

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            final_output['responses']
        ], dim=1) # [bs*n, prompt_length + max_response_length]

        # Create attention mask
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1) # [bs*n, prompt_length + max_response_length]

        # Create observation mask
        if self.config.mask_observations:
            final_output['loss_mask'] = torch.cat([
                torch.zeros_like(left_side['input_ids']), # do not train on prompt
                self.tensor_fn.create_attention_mask(final_output['responses_with_loss_mask'])
            ], dim=1) # [bs*n, prompt_length + max_response_length]
        else:
            final_output['loss_mask'] = final_output['attention_mask']
        
        # if mask overlong trajectory is enabled, we need to mask the overlong trajectory
        if self.config.mask_overlong_loss:
            # set loss_mask to 0 for those overlong trajectories
            effective_lens = self.tensor_fn.create_attention_mask(final_output['responses']).sum(dim=1)
            overlong_mask = effective_lens >= self.config.max_response_length
            final_output['loss_mask'][overlong_mask] = 0
            logger.debug(f"Masked {overlong_mask.sum().item()}/{final_output['loss_mask'].shape[0]} overlong trajectories.")

        # Create position ids
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        ) # [bs*n, prompt_length + max_response_length]

        # ---------- 3. Create and return DataProto ----------
        final_output = DataProto.from_dict(final_output, non_tensors=non_tensors)
        final_output.meta_info.update(meta_info)

        return final_output

    def send_batch_requests(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send batch requests to the tool server.
        Args:
            batch_data: Batch data to send
        Returns:
            response: Response from the tool server
        """
        safe_payload = sanitize_request(batch_data)
        response = requests.post(self.config.tool_server_url, json=safe_payload)
        if response.status_code != 200:
            os.mkdir('tmp', exist_ok=True)  # Ensure tmp directory exists
            with open("tmp/error_data.json", 'w') as f:
                json.dump(batch_data, f, indent=4)
            try:
                # Try to decode as utf-8 for error message
                error_text = response.text
                logger.error(f"Error: {response.status_code}, {error_text}")
            except UnicodeDecodeError:
                # If decoding fails, show raw content and encoding
                logger.error(f"Error: {response.status_code}, Binary response, encoding: {response.encoding}")
                logger.error(f"Raw content (first 100 bytes): {response.content[:100]}")
            raise ValueError(f"Error: {response.status_code}, Response could not be decoded as UTF-8")
        
        try:
            return response.json()
        except ValueError as e:

            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response content type: {response.headers.get('Content-Type')}")
            logger.error(f"First 100 chars of response: {response.text[:100]}")
            raise
    
    async def _aiohttp_request(self, data):
        try:
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=self.config.tool_server_url,
                json=data,
            ) as resp:
                data = await resp.json()
                return data
        finally:
            await session.close()
        
    async def send_batch_requests_async(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Robust version with retry logic"""
        safe_payload = sanitize_request(batch_data)
        
        try:
            return await self._aiohttp_request(safe_payload)
        except Exception as e:
            # Log error with context
            logging.error(f"Failed to send batch request after all retries: {e}")
            logging.error(f"Payload size: {len(str(safe_payload))} chars")
            
            # Save error data for debugging
            if not os.path.exists('tmp'):
                os.mkdir('tmp')  # Ensure tmp directory exists
            error_file = f"tmp/error_data_{uuid.uuid4().hex[:8]}.json"
            with open(error_file, 'w') as f:
                json.dump(safe_payload, f, indent=2)
            logging.error(f"Error data saved to {error_file} for debugging.")
            
            raise ValueError(f"Tool server communication failed: {e}")
        
    async def interact_with_tool_server(
        self,
        active_uids:List[str],
        responses: List[str],
        do_actions:List[bool],
        active_mask=None,
        extra_fields=None,
        is_last_step=False,
    ) -> List[str]:
        """
        Call tool server for queries.
        Args:
            batch: batch of data
            resposnes: responses from the model
            pad_token: pad token
            active_mask: active mask
        Returns: (All of length of active_mask, which is the original batch size)
            observations (List[str]): observations from the tool server. None if the the query do not need to do any action.
            dones (List[bool]): dones
            valid_actions (List[bool]): valid actions
            _finishs (List[bool]): whether the trajectory is finished for eos for all trajectories (including those that are not active)
            rewards (List[float]): rewards for the trajectories, None if not applicable
            tool_interact_info (List[Dict]): tool interaction info for each trajectory, None if not applicable
        """
        finishs = [not do_action for do_action in do_actions]
        batch_data = {
            "trajectory_ids": active_uids,
            "actions": responses,
            "finish": finishs, # if do_action is False, then it is a finish action, finishing the trajectory,
            "is_last_step": [is_last_step] * len(finishs)
        }
        if extra_fields is not None:
            batch_data['extra_fields'] = extra_fields.tolist() if isinstance(extra_fields, np.ndarray) else extra_fields
        logger.info(f" - Number of finished responses: {len([x for x in do_actions if not x])} / {len(do_actions)}")
        response = await self.send_batch_requests_async(batch_data)
        active_observations = response['observations']
        active_dones = [int(x) for x in response['dones']]
        active_valid_actions = [int(x) for x in response['valids']]

        logger.debug(f"Received observations from tool server. Samples: {len(active_observations)}")
        logger.info(f" - Number of valid actions (exclusing finish action): {len([x for x in active_valid_actions if x])} / {len(active_valid_actions)}")
        logger.info(f" - Number of dones: {len([x for x in active_dones if x])} / {len(active_dones)}")
        logger.debug("Example observations:")
        non_empty_observations = [obs for obs in active_observations if obs]
        if len(non_empty_observations) > 0:
            logger.debug(f"{non_empty_observations[0]}")
        else:
            logger.debug("No non-empty observations.")

        next_obs, dones, valid_action, _finishs = [], [], [], []
        for i, active in enumerate(active_mask):
            if active:
                next_obs.append(active_observations.pop(0))
                dones.append(active_dones.pop(0)) # whether the trajectory is finished for eos or considered done by the remote server
                valid_action.append(active_valid_actions.pop(0))
                _finishs.append(finishs.pop(0)) # whether the trajectory is finished for eos
            else:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                _finishs.append(1)

        assert len(active_observations) == 0
        
        # postprocess next_obs. For now we support two types of observations:
        # 1. string observations, which will be the most common case
        # 2. dict observations, e.g. {"obs": "some observation", "reward": 1.0}
        #     for now we only support "obs" and "reward" keys, but can be extended later
        processed_next_obs = []
        rewards = []
        tool_interact_info = []
        allowed_keys = ['obs', 'reward']
        for i, obs in enumerate(next_obs):
            if isinstance(obs, str):
                # can be invalid
                processed_next_obs.append(obs)
                rewards.append(None)
                tool_interact_info.append({})
            elif isinstance(obs, dict):
                assert "obs" in obs, f"Observation dict must contain 'obs' key, but got {obs.keys()}"
                _obs = obs.get('obs', '')
                _reward = obs.get('reward', None)
                assert isinstance(_obs, str), f"Expected 'obs' to be a string, but got {type(_obs)}"
                assert _reward is None or isinstance(_reward, (int, float)), f"Expected 'reward' to be None, int, or float, but got {type(_reward)}"
                processed_next_obs.append(_obs)
                rewards.append(_reward)
                # store tool interaction info if exists
                tool_interact_info.append({k: v for k, v in obs.items() if k not in allowed_keys})
            else:
                raise ValueError(f"Invalid observation type: {type(obs)}. Expected str or dict.")
        next_obs = processed_next_obs
        return next_obs, dones, valid_action, _finishs, rewards, tool_interact_info

     # Step 4: Add cleanup method (optional but recommended)
    async def cleanup(self):
        """Clean up HTTP session"""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
    
    def __del__(self):
        """Ensure session is closed when object is destroyed"""
        if self._http_session and not self._http_session.closed:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._http_session.close())
                else:
                    loop.run_until_complete(self._http_session.close())
            except:
                pass  # Best effort cleanup