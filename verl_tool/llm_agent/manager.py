import torch
import os
import re
import uuid
import json
import numpy as np
import requests
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.utils import hf_tokenizer
from verl.utils.model import get_generation_config
from tqdm import tqdm
from typing import List
from .config import AgentActorConfig
from .tensor_helper import TensorHelper, TensorConfig

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
        if os.path.exists(self.config.action_stop_tokens):
            with open(self.config.action_stop_tokens, 'r') as f:
                self.action_stop_tokens = f.read().strip('\n').split(',')
            print(f"Using action stop tokens: {self.action_stop_tokens}")
        else:
            raise FileNotFoundError(f"Action stop tokens file '{self.config.action_stop_tokens}' not found.")

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _preprocess_inputs(self, inputs: DataProto):
        """
        this version verl do not repeat the input by n times, so we manually repeat the input by n times
        
        """
        # we manually repeat the input by n times if needed since every trajectory is independent
        do_sample = inputs.meta_info.get("do_sample", True)
        if not do_sample:
            n = 1
        else:
            n = self.config.n 
            inputs = inputs.repeat(n)
        inputs.non_tensor_batch['traj_ids'] = np.array([str(uuid.uuid4()) for _ in range(len(inputs.batch))], dtype=object)
        return inputs
        
    def _postprocess_responses(self, responses: torch.Tensor, action_step: int) -> torch.Tensor:
        """Process responses to stop at python operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        do_actions = []
        for i, resp in enumerate(responses_str):
            resp = resp.strip(' \n')
            if self.config.no_action_as_stop and action_step >= self.config.min_action_num:
                has_action = False
                for j in range(len(self.action_stop_tokens)):
                    if resp.endswith(self.action_stop_tokens[j]):
                        has_action = True
                        responses_str[i] = resp.split(self.action_stop_tokens[j])[0] + self.action_stop_tokens[j]
                        break
            else:
                has_action = True
            do_actions.append(has_action)
        responses = self._batch_tokenize(responses_str).to(torch.int64)
        return responses, responses_str, do_actions

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids'].to(torch.int64)

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            if self.config.truncate_obs_side == 'left':
                next_obs_ids = next_obs_ids[:, -self.config.max_obs_length:]
            elif self.config.truncate_obs_side == 'right':
                next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
            else:
                raise ValueError(f"Invalid truncate_obs_side: {self.config.truncate_obs_side}")

        return next_obs_ids

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
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
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings
    
    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        
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
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            # extend the mask for the observation part, to update masked tensors
            tensors_with_mask.append(info_mask)    
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                cur_responses: torch.Tensor,
                next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        
        # observation exists, perform concatenation and masked concatenation
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            # no observation, only concatenate the response with generated response
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
            
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_response_length, effective_len)
        
        # return the updated responses along with its masked version
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}
        

    def run_llm_loop(self, gen_batch: DataProto) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        ori_meta_info = gen_batch.meta_info
        gen_batch = self._preprocess_inputs(gen_batch)
        
        initial_input_ids = gen_batch.batch['input_ids'][:, -self.config.max_start_length:].clone()
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        # original_right_side = {'responses': initial_input_ids[:, []]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        turns_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        traj_ids = gen_batch.non_tensor_batch['traj_ids']
        
        agent_sampling_params = {
            "n": 1, # already repeated by n times in _preprocess_inputs
            "stop": self.action_stop_tokens, # stop when generated an end of action
            "include_stop_str_in_output": True,
            "detokenize": True
        }
        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                print("All trajectories are done.")
                break
            print(f"Action step {step+1}/{self.config.max_turns}")
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            }, meta_info=ori_meta_info)
            with self.actor_rollout_wg.rollout.update_sampling_params(**agent_sampling_params):
                gen_output = self.actor_rollout_wg.rollout.generate_sequences(rollings_active)
            
            meta_info = gen_output.meta_info            
            responses_ids, responses_str, do_actions = self._postprocess_responses(gen_output.batch['responses'], step)
            responses_ids, _ = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            print(f"Number of active trajectories: {active_mask.sum().item()}")
            print(f"Length of responses: {responses_ids.shape[1]}")

            # Execute in environment and process observations
            active_uids = [traj_ids[i] for i in range(len(traj_ids)) if active_mask[i]]
            next_obs, dones, valid_action = self.interact_with_tool_server(active_uids, responses_str, do_actions, active_mask)
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        
        agent_sampling_params = {
            "n": 1, # already repeated by n times in _preprocess_inputs
        } # reomve stop related params in the last call
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            }, meta_info=ori_meta_info)
            with self.actor_rollout_wg.rollout.update_sampling_params(**agent_sampling_params):
                gen_output = self.actor_rollout_wg.rollout.generate_sequences(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str, do_actions = self._postprocess_responses(gen_output.batch['responses'], step)
            responses_ids, _ = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            dones = []
            j=0
            for i, active in enumerate(active_mask):
                if not active:
                    dones.append(1)
                else:
                    if do_actions[j]:
                        dones.append(0)
                    else:
                        dones.append(1)
                    j += 1
            assert j == len(do_actions), f"j: {j}, len(do_actions): {len(do_actions)}"

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
            
        # meta_info['turns_stats'] = turns_stats.tolist()
        # meta_info['active_mask'] = active_mask.tolist()
        # meta_info['valid_action_stats'] = valid_action_stats.tolist()
        non_tensors = {
            'traj_ids': traj_ids.tolist(),
            'turns_stats': turns_stats.tolist(),
            'valid_action_stats': valid_action_stats.tolist(),
            'active_mask': active_mask.tolist(),
        }
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        results = self._compose_final_output(original_left_side, original_right_side, non_tensors, meta_info)
        return results

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            non_tensors: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # padding responses length to max_response_length
        if final_output['responses'].shape[1] < self.config.max_response_length:
            final_output['responses'] = self.tensor_fn.pad_tensor(
                final_output['responses'],
                max_length=self.config.max_response_length,
                padding_side='right'
            )
        
        # padding response_with_info_mask length to max_response_length 
        if final_output['responses_with_info_mask'].shape[1] < self.config.max_response_length:
            final_output['responses_with_info_mask'] = self.tensor_fn.pad_tensor(
                final_output['responses_with_info_mask'],
                max_length=self.config.max_response_length,
                padding_side='right'
            )
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            final_output['responses']
        ], dim=1)
        
        # Create attention mask 
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        # Create observation mask 
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        # Create position ids
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output, non_tensors=non_tensors)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def interact_with_tool_server(self, active_uids:List[str], responses: List[str], do_actions:List[bool], active_mask=None) -> List[str]:
        """
        Call tool server for queries.
        Args:
            batch: batch of data
            resposnes: responses from the model
            pad_token: pad token
            active_mask: active mask
        Returns:
            observations: observations from the tool server. None if the the query do not need to do any action.
            dones: dones
            valid_actions: valid actions
        """
        assert len(active_uids) == len(responses) == len(do_actions), f"Length mismatch: {len(active_uids)}, {len(responses)}, {len(do_actions)}"
        data = {
            "trajectory_ids": active_uids,
            "actions": responses,
            "finish": [not do_action for do_action in do_actions], # if do_action is False, then it is a finish action, finishing the trajectory,
        }
        print(f"Sending request to {self.config.tool_server_url}")
        print(f" - Number of non-finished actions: {len([x for x in do_actions if not x])} / {len(do_actions)}")
        response = requests.post(self.config.tool_server_url, json=data)
        active_observations = response.json()['observations']
        active_dones = [int(x) for x in response.json()['dones']]
        active_valid_actions = [int(x) for x in response.json()['valids']]
        print("Received observations from tool server. Samples:", len(active_observations))
        print(f" - Number of valid actions (exclusing finish action): {len([x for x in active_valid_actions if x])} / {len(active_valid_actions)}")
        print(f" - Number of dones: {len([x for x in active_dones if x])} / {len(active_dones)}")
        print("Example observations:")
        non_empty_observations = [obs for obs in active_observations if obs]
        if len(non_empty_observations) > 0:
            print(f"{non_empty_observations[0]}")
        else:
            print("No non-empty observations.")
        
        next_obs, dones, valid_action = [], [], []
        for i, active in enumerate(active_mask):
            if active:
                next_obs.append(active_observations.pop(0))
                dones.append(active_dones.pop(0))
                valid_action.append(active_valid_actions.pop(0))
            else:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
        
        assert len(active_observations) == 0
        return next_obs, dones, valid_action