import torch
import os
import re
import uuid
import json
import logging
import asyncio
import aiohttp
import regex as re
import numpy as np
import requests
import omegaconf
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.model import get_generation_config
from tqdm import tqdm
from typing import List, Union, Optional, Any
from pathlib import Path
from .config import AgentActorConfig
from .tensor_helper import TensorHelper, TensorConfig
from PIL import Image
from .utils import PerformanceTimer, nested_copy
from .vision_utils import encode_image, encode_image_url, encode_video_url, decode_image_url, decode_video_url

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
    elif isinstance(obj,Image.Image):
        return encode_image(obj)
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
        self.processor = hf_processor(self.model_path)
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
        # for multimodal processing
        if self.processor:
            self.mm_prefix, self.mm_postfix = self.processor.apply_chat_template(
                [{"role": "system", "content": [{"type": "text", "text": "|||"}]}],
                tokenize=False, add_generation_prompt=False).split("|||") # this is used to create the correct multi-modal prompt
        else:
            self.mm_prefix = ""
            self.mm_postfix = ""
        if self.config.rollout_mode == "sync":
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

    @classmethod
    def from_rollout_config(cls, actor_rollout_wg, rollout_config, rollout_mode="async"):
        agent_config = AgentActorConfig()
        for key in getattr(rollout_config, 'agent', {}).keys():
            if key in agent_config.__dict__.keys():
                setattr(agent_config, key, rollout_config.agent[key])
        setattr(agent_config, 'n', rollout_config.rollout.n)
        setattr(agent_config, 'max_model_len', rollout_config.rollout.max_model_len)
        model_path = rollout_config.model.path
        agent_config.rollout_mode = rollout_mode
        print(f"AgentAsyncActorRolloutRefWorker: {agent_config}")
        agent_actor_manager = cls(model_path, actor_rollout_wg, agent_config)
        return agent_actor_manager
    
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
            if inputs.meta_info.get("validate", False):
                n = self.config.val_kwargs.n
            else:
                n = self.config.n

            inputs = inputs.repeat(n, interleave=True)
        # add "_{i}" for each trajectory to the traj_ids
        for i in range(ori_len):
            for j in range(n):
                inputs.non_tensor_batch['traj_ids'][i*n+j] += f"_{j}"
                # deepcopy to avoid reference bug
                for key in inputs.non_tensor_batch.keys():
                    # # check if it's the same reference as the inputs.non_tensor_batch[key][i]
                    inputs.non_tensor_batch[key][i*n+j] = nested_copy(inputs.non_tensor_batch[key][i*n])
        inputs.meta_info['is_repeated_by_n'] = True
        return inputs

    async def _postprocess_responses(self, responses: Union[torch.Tensor, List[str]], action_step: int, rollout_messages: list) -> torch.Tensor:
        """Process responses to stop at python operation or answer operation.
        Args:
            responses (Union[torch.Tensor, List[str]]): Responses from the model, either as a tensor or a list of strings. of length sum(active_mask), which <= batch_size
            action_step (int): Current action step in the interaction.
            rollout_messages (list): List of rollout messages to update with new responses.
            active_mask (torch.Tensor): A mask indicating which responses are active, of shape [batch_size].
        Returns:
            responses (torch.Tensor): Processed responses as a tensor.
            responses_str (List[str]): List of processed response strings.
            do_actions (List[bool]): List indicating whether to perform actions based on the responses.
            rollings (DataProto): Updated rolling state with new responses.
        """
        effective_lens = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
        do_actions = []
        async with self.tokenizer_lock:
            if isinstance(responses, torch.Tensor):
                responses_str = self.tokenizer.batch_decode(
                    responses,
                    skip_special_tokens=True
                )
            else:
                responses_str = responses
            # update messages with new responses
            if rollout_messages is not None:
                for i in range(len(responses_str)):
                    rollout_messages[i].update_rollout_messages(
                        {
                            "role": self.config.assistant_role,
                            "content": responses_str[i]
                        }
                    )
                    
            for i in range(len(responses_str)):
                # check if the response contains action stop tokens
                has_action = False
                for j in range(len(self.action_stop_tokens)):
                    if self.action_stop_tokens[j] in responses_str[i]:
                        responses_str[i] = responses_str[i].split(self.action_stop_tokens[j])[0] + self.action_stop_tokens[j]
                        has_action = True
                        break
                
                # judge whether do action or not
                if action_step >= self.config.min_turns:
                    # do action if there are action stop tokens in the response
                    do_action = has_action or (self.config.enable_mtrl and not self.action_stop_tokens)
                else:
                    # always do action, decided by the server about whether an action stops
                    do_action = True
                    if self.action_stop_tokens and not has_action:
                        # force add a action stop token for those responses that do not have action stop tokens
                        turn_end_token_idx = responses_str[i].rfind(self.config.turn_end_token)
                        if turn_end_token_idx != -1:
                            responses_str[i] = responses_str[i][:turn_end_token_idx] + self.action_stop_tokens[0]
                        else:
                            responses_str[i] = responses_str[i] + self.action_stop_tokens[0]
                
                # now if do action, responses_str[i] should end with a action stop token, if not do action, we use the original response
                if do_action:
                    if self.config.enable_mtrl:
                        # add turn end token
                        responses_str[i] += self.config.turn_end_token
                else:
                    # preserve eos token
                    responses_str[i] = self.tokenizer.decode(responses[i][:effective_lens[i]], skip_special_tokens=False)
                do_actions.append(do_action)     

            responses = self._batch_tokenize(responses_str).to(torch.int64)
        return responses, responses_str, do_actions, rollout_messages

    async def _process_next_obs(self, next_obs: List[str], dones: List[bool], valid_action: List[bool], finishs: List[bool], tool_interact_info: List[dict], rollings: DataProto) -> Tuple[torch.Tensor, List[dict]]:
        """Process next observations from environment.
        Args:
            next_obs (List[str]): List of next observations, only the text part.
            dones (List[bool]): List of done flags for each observation.
            valid_action (List[bool]): List of valid action flags for each observation.
            finishs (List[bool]): List of finish flags for each observation.
            tool_interact_info (List[dict]): List of tool interaction information for each observation, also include multi modal keys like 'image' and 'video'.
            rollings (DataProto): Current rolling state containing input_ids, position_ids, attention_mask, and non_tensor_batch.
        Returns:
            next_obs_ids (torch.Tensor): Tokenized next observations.
            rollings (DataProto): Updated rolling state with new observations.
        """
        has_multi_modal_data = "multi_modal_data" in rollings.non_tensor_batch and rollings.non_tensor_batch['multi_modal_data'] is not None
        mm_data_list = None
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
            next_obs = self.tokenizer.batch_decode(
                next_obs_ids,
                skip_special_tokens=True
            )

            if not has_multi_modal_data:
                
                if self.config.enable_mtrl:
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

                # update rollout messages with next_obs
                if "rollout_messages" in rollings.non_tensor_batch:
                    for i in range(len(next_obs)):
                        if next_obs[i]:
                            rollings.non_tensor_batch['rollout_messages'][i].update_rollout_messages(
                                {
                                    "role": self.config.mtrl_role if self.config.enable_mtrl else self.config.assistant_role,
                                    "content": next_obs[i]
                                }
                            )
            else:
                mm_data_list = []
                raw_prompts = []
                
                for k, tool_interact_info_k in enumerate(tool_interact_info):
                    multi_modal_data = {}
                    next_obs_image = tool_interact_info_k.get('image', [])
                    if not isinstance(next_obs_image, list):
                        next_obs_image = [next_obs_image]
                    next_obs_image = [decode_image_url(img) for img in next_obs_image]
                    multi_modal_data["image"] = next_obs_image
                    
                    next_obs_video = tool_interact_info_k.get('video', [])
                    if not isinstance(next_obs_video, list):
                        next_obs_video = [next_obs_video]
                    next_obs_video = [decode_video_url(video) for video in next_obs_video]
                    multi_modal_data["video"] = [video.numpy() for video in next_obs_video]

                    # add additional <image> and <video> placeholder to next_obs[k]
                    next_obs_k = next_obs[k]
                    if not valid_action[k] and not (dones[k] or finishs[k]):
                        next_obs_k = "Your action is not valid, please check the format and try again." + next_obs_k
                    if next_obs_image:
                        image_placeholder_count = next_obs_k.count("<image>")
                        if image_placeholder_count < len(next_obs_image):
                            next_obs_k = "<image>" * (len(next_obs_image) - image_placeholder_count) + next_obs_k
                        elif image_placeholder_count > len(next_obs_image):
                            next_obs_k = next_obs_k.replace("<image>", "", image_placeholder_count - len(next_obs_image))
                    if next_obs_video:
                        video_placeholder_count = next_obs_k.count("<video>")
                        if video_placeholder_count < len(next_obs_video):
                            next_obs_k = "<video>" * (len(next_obs_video) - video_placeholder_count) + next_obs_k
                        elif video_placeholder_count > len(next_obs_video):
                            next_obs_k = next_obs_k.replace("<video>", "", video_placeholder_count - len(next_obs_video))
                    
                    content_list = []
                    segments = re.split("(<image>|<video>)", next_obs_k)
                    segments = [item for item in segments]
                    segment_idx = defaultdict(int)
                    for segment in segments:
                        if segment == "<image>":
                            content_list.append({"type": "image"})
                            segment_idx[segment] += 1
                        elif segment == "<video>":
                            content_list.append({"type": "video"})
                            segment_idx[segment] += 1
                        else:
                            content_list.append({"type": "text", "text": segment})
                    if content_list and not dones[k] and not finishs[k]:
                        next_obs_message = [{"role": "system", "content": content_list}]
                        if not self.config.enable_mtrl:
                            raw_prompt = self.processor.apply_chat_template(
                                next_obs_message, add_generation_prompt=False, tokenize=False, continue_final_message=True
                            )
                            # remove mm_prefix, only keep the part after <im_start>, the system will not appear
                            raw_prompt = raw_prompt.replace(self.mm_prefix, "")
                        else:
                            raw_prompt = self.processor.apply_chat_template(
                                next_obs_message, add_generation_prompt=True, tokenize=False, continue_final_message=False
                            )
                            # change system role to mtrl_role
                            raw_prompt = "\n" + raw_prompt.replace("system", self.config.mtrl_role, 1)
                    else:
                        raw_prompt = ""

                    # udpate rollout messages with next_obs
                    if "rollout_messages" in rollings.non_tensor_batch and raw_prompt:
                        content_list = []
                        segment_idx = defaultdict(int)
                        for segment in segments:
                            if segment == "<image>":
                                content_list.append({"type": "image_url", "image_url": {"url": encode_image_url(next_obs_image[segment_idx[segment]])}})
                                segment_idx[segment] += 1
                            elif segment == "<video>":
                                content_list.append({"type": "video_url", "video_url": {"url": encode_video_url(next_obs_video[segment_idx[segment]])}})
                                segment_idx[segment] += 1
                            else:
                                content_list.append({"type": "text", "text": segment})
                        rollings.non_tensor_batch['rollout_messages'][k].update_rollout_messages(
                            {
                                "role": self.config.mtrl_role if self.config.enable_mtrl else self.config.assistant_role,
                                "content": content_list
                            }
                        )   
                    mm_data_list.append(multi_modal_data)
                    raw_prompts.append(raw_prompt)
                    
                next_obs_ids = self.processor(
                    text=raw_prompts, 
                    images=[mm_data_list[i]['image'] for i in range(len(mm_data_list)) if 'image' in mm_data_list[i] and mm_data_list[i]['image']] or None,
                    videos=[mm_data_list[i]['video'] for i in range(len(mm_data_list)) if 'video' in mm_data_list[i] and mm_data_list[i]['video']] or None,
                    padding='longest',
                    return_tensors='pt',
                    add_special_tokens=False,  # Prevents adding special tokens
                )['input_ids'].to(torch.int64)
        
        if mm_data_list is not None and "multi_modal_data" in rollings.non_tensor_batch:
            for i in range(len(rollings.non_tensor_batch['multi_modal_data'])):
                next_mm_data_i = mm_data_list[i]
                if 'image' in next_mm_data_i and next_mm_data_i['image'] :
                    rollings.non_tensor_batch['multi_modal_data'][i]['image'].extend(next_mm_data_i['image'])
                if 'video' in next_mm_data_i and next_mm_data_i['video']:
                    rollings.non_tensor_batch['multi_modal_data'][i]['video'].extend(next_mm_data_i['video'])

        return next_obs_ids, rollings

    def _update_rolling_state(self, 
        left_side, 
        rollings, 
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor,
        active_mask: torch.Tensor
    ) -> Dict:
        """Update rolling state with new responses and observations."""

        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ], pad_to_left=False)

        max_len = self.max_model_len
        
        if getattr(self.config, "rolling_with_prompt", False):
            # if rolling_with_prompt is True, then we need to keep the system prompt, and keep the right side
            if isinstance(left_side, dict):
                left_ids = left_side["input_ids"]
            else:
                left_ids = left_side.batch["input_ids"]

            left_len = left_ids.size(1)

            new_input_ids, _ = self.tensor_fn.convert_pad_structure(new_input_ids, pad_to_left=True)
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
        else: 
            # By default keep the left side
            new_input_ids = new_input_ids[:, :max_len]  # Truncate to max_len
            new_input_ids, _ = self.tensor_fn.convert_pad_structure(new_input_ids, pad_to_left=True)
            # Create attention mask and position ids
            new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
            new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
            new_rollings = DataProto.from_dict(
                {
                    "input_ids": new_input_ids,
                    "position_ids": new_position_ids,
                    "attention_mask": new_attention_mask,
                }
            )
        new_rollings.non_tensor_batch = rollings.non_tensor_batch.copy()
        new_rollings.meta_info.update(rollings.meta_info)
        
        # update raw_prompt_ids, required for vllm inference
        raw_prompt_ids = []
        for i in range(new_rollings.batch['input_ids'].size(0)):
            non_pad_index = torch.nonzero(new_rollings.batch['input_ids'][i] != self.tokenizer.pad_token_id, as_tuple=False)[0][0]
            raw_prompt_ids.append(new_rollings.batch['input_ids'][i][non_pad_index:].tolist())
        new_rollings.non_tensor_batch['raw_prompt_ids'] = np.array(raw_prompt_ids, dtype=object)

        effective_lens = new_attention_mask.sum(dim=1)
        min_effective_len = effective_lens.min().item()
        overlong_traj_mask = (effective_lens >= max_len).cpu().numpy()
        if overlong_traj_mask.sum() > 0:
            overlong_traj_ids = rollings.non_tensor_batch['traj_ids'][overlong_traj_mask]
            self.close_traj_tool_threads(overlong_traj_ids)
            self._update_active_mask_inplace(active_mask, ~overlong_traj_mask)
        available_context_budget = max(0, self.max_model_len - min_effective_len)
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
        """Update active mask in-place to avoid memory allocation, return the count of active trajectories."""
        active_mask &= new_conditions
        return active_mask.sum().item()  # Return count for logging

    async def run_llm_loop_async(self, gen_batch: DataProto, **sampling_params: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        perf_timer = PerformanceTimer(do_timer=False)
        perf_timer.start('run_llm_loop_total')
        perf_timer.start('initialization')
        # only async is supported for multi-modal now
        if "multi_modal_data" in gen_batch.non_tensor_batch and self.config.rollout_mode != "async":
            raise ValueError("Multi-modal data is only supported in async mode, please set rollout_mode to 'async'.")
        
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
            next_obs_ids, rollings = await self._process_next_obs(next_obs, dones, valid_action, finishs, tool_interact_info, rollings)

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
                next_obs_ids,
                active_mask
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            agent_sampling_params['max_tokens'] = available_context_budget # for vllm
            agent_sampling_params['max_new_tokens'] = available_context_budget # for sglang
            active_num_list.append(active_mask.sum().item())
            perf_timer.end('initial_tool_call')
            
        # it seems somehow and sometime the non_tensor_batch will be changed by the generate_sequences. so we save a copy and reassign it later
        if "multi_modal_data" in rollings.non_tensor_batch:
            immutable_non_tensor_batch_keys = ["multi_modal_data", "multi_modal_inputs"]
        else:
            immutable_non_tensor_batch_keys = []
        rollout_messages = deepcopy(rollings.non_tensor_batch.get('rollout_messages', None))
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
            active_idxs = torch.nonzero(active_mask, as_tuple=True)[0]
            rollings_active = DataProto.from_dict(
                {k: v[active_mask] for k, v in rollings.batch.items()},
                {k: v[active_mask.numpy()] for k, v in rollings.non_tensor_batch.items()},
                meta_info=ori_meta_info
            )
            
            active_rollout_messages = [rollout_messages[i] for i in active_idxs] if rollout_messages is not None else None
            immutable_non_tensor_batch_records = {
                key: np.array([nested_copy(rollings.non_tensor_batch[key][i]) for i in range(len(rollings.non_tensor_batch[key]))])
                for key in immutable_non_tensor_batch_keys
            }
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
            responses_ids, responses_str, do_actions, active_rollout_messages = await self._postprocess_responses(gen_output.batch['responses'], step, active_rollout_messages) # [active_size, ...]
            responses_ids, _ = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask) # [bs*n, response_length]
            for i in range(len(active_rollout_messages)):
                rollings.non_tensor_batch['rollout_messages'][active_idxs[i]] = active_rollout_messages[i]
            for key in immutable_non_tensor_batch_keys:
                for i in range(len(rollings.non_tensor_batch[key])):
                    rollings.non_tensor_batch[key][i] = immutable_non_tensor_batch_records[key][i]
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
            
            # Prepare extra fields with turn information
            extra_fields = rollings_active.non_tensor_batch.get('extra_info', None)
            if extra_fields is not None:
                # Add current step and turns_left information to each extra_field entry
                enhanced_extra_fields = []
                for i, extra_field in enumerate(extra_fields):
                    if isinstance(extra_field, dict):
                        enhanced_field = extra_field.copy()
                        enhanced_field['current_step'] = step
                        enhanced_field['max_turns'] = self.config.max_turns
                        enhanced_field['turns_left'] = max(0, self.config.max_turns - step)
                        enhanced_extra_fields.append(enhanced_field)
                    else:
                        # If extra_field is not a dict, create a new dict with turn info
                        enhanced_extra_fields.append({
                            'current_step': step,
                            'max_turns': self.config.max_turns,
                            'turns_left': max(0, self.config.max_turns - step)
                        })
                extra_fields = enhanced_extra_fields
            else:
                # If no extra_fields exist, create them with turn information for each active trajectory
                extra_fields = [
                    {
                        'current_step': step,
                        'max_turns': self.config.max_turns,
                        'turns_left': max(0, self.config.max_turns - step)
                    }
                    for _ in range(len(active_uids))
                ]
            
            next_obs, dones, valid_action, finishs, rewards, tool_interact_info = await self.interact_with_tool_server(
                active_uids, responses_str, do_actions, active_mask,
                extra_fields=extra_fields,
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

            next_obs_ids, rollings = await self._process_next_obs(next_obs, dones, valid_action, finishs, tool_interact_info, rollings)

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
                next_obs_ids,
                active_mask
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            available_context_budget = min(available_context_budget, self.config.max_action_length)
            agent_sampling_params['max_tokens'] = available_context_budget # for vllm
            agent_sampling_params['max_new_tokens'] = available_context_budget # for sglang
            if available_context_budget == 0:
                # update all active_mask to False, since no more context is available
                self.close_traj_tool_threads(traj_ids[active_mask.numpy()])
                self._update_active_mask_inplace(active_mask, torch.zeros_like(active_mask, dtype=torch.bool))
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
        
        if "multi_modal_data" in rollings.non_tensor_batch:
            mm_inputs = await self.get_final_mm_inputs(rollings)
            non_tensors['multi_modal_inputs'] = mm_inputs # used for policy gradient updates

        results = self._compose_final_output(original_left_side, original_right_side, non_tensors, ori_meta_info)
        perf_timer.end('final_composition')
        
        perf_timer.end('run_llm_loop_total')
        
        # Log performance statistics
        perf_timer.log_stats(logger, f"[PERF] Batch size: {gen_batch.batch['input_ids'].shape[0]} - ")
        
        results.save_to_disk("test.pkl")
        return results
    
    def run_llm_loop(self, gen_batch: DataProto, **sampling_params: Dict[str, Any]) -> Tuple[Dict, Dict]:
        return asyncio.run(self.run_llm_loop_async(gen_batch, **sampling_params))

    async def get_final_mm_inputs(self, rollings: DataProto):
        mm_inputs = []
        async with self.tokenizer_lock:
            for i in range(rollings.batch['input_ids'].shape[0]):
                raw_prompt = self.processor.apply_chat_template(rollings.non_tensor_batch['rollout_messages'][i].messages, add_generation_prompt=False, tokenize=False)
                
                images = rollings.non_tensor_batch['multi_modal_data'][i].get('image', None)
                videos = rollings.non_tensor_batch['multi_modal_data'][i].get('video', None)
                model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
                
                # # for debugging, make sure the input_ids from rollout messages match the input_ids maintained in the processor
                rolling_raw_prompt = self.processor.decode(rollings.batch['input_ids'][i].tolist(), skip_special_tokens=False)
                _raw_prompt = self.processor.decode(model_inputs['input_ids'][0].tolist(), skip_special_tokens=False)[:len(rolling_raw_prompt)]
                rolling_raw_prompt = rolling_raw_prompt[:len(_raw_prompt)]
                # if _raw_prompt != rolling_raw_prompt:
                #     logger.warning(f"Raw prompt mismatch for trajectory {i}: \n{_raw_prompt}\n != \n{rolling_raw_prompt}\n")
                #     with open("test.json", "w") as f:
                #         json.dump({
                #             "rollout_messages": rollings.non_tensor_batch['rollout_messages'][i].messages,
                #             "raw_prompt_before_tokenization": raw_prompt,
                #             "raw_prompt": _raw_prompt,
                #             "rolling_raw_prompt": rolling_raw_prompt,
                #             "images": [encode_image_url(img) for img in images] if images else None,
                #             "videos": [encode_video_url(video) for video in videos] if videos else None,
                #         }, f, indent=4)
                #     import pickle
                #     with open("test_mm_data.pkl", "wb") as f:
                #         pickle.dump(rollings.non_tensor_batch['multi_modal_data'][i], f)
                #     raise ValueError(f"Raw prompt mismatch for trajectory {i}, please check the processor and tokenizer settings.")
                input_ids = model_inputs.pop('input_ids')
                attention_mask = model_inputs.pop('attention_mask')
                if "second_per_grid_ts" in model_inputs:
                    model_inputs.pop('second_per_grid_ts')
                mm_inputs.append(dict(model_inputs))
        return mm_inputs
    
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
        # recent (from July 2025) verl uses response_mask for loss_mask
        response_length = final_output['responses'].shape[1]
        final_output['response_mask'] = final_output['loss_mask'][:, -response_length:]  # [bs*n, max_response_length]
        
        # if mask overlong trajectory is enabled, we need to mask the overlong trajectory
        if self.config.mask_overlong_loss:
            # set loss_mask to 0 for those overlong trajectories
            effective_lens = self.tensor_fn.create_attention_mask(final_output['responses']).sum(dim=1)
            overlong_mask = effective_lens >= self.config.max_response_length
            final_output['loss_mask'][overlong_mask] = 0
            num_masked = overlong_mask.sum().item()
            if num_masked > 0:
                logger.warning(f"Masked {num_masked}/{final_output['loss_mask'].shape[0]} overlong trajectories.")

        # Create position ids
        if "multi_modal_inputs" in non_tensors and \
            self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = []
            for i in range(len(non_tensors['multi_modal_inputs'])):
                model_inputs = non_tensors['multi_modal_inputs'][i]
                input_ids_i = final_output['input_ids'][i]
                attention_mask_i = final_output['attention_mask'][i]
                effective_len = attention_mask_i.sum().item()
                final_output_effective_len = final_output['attention_mask'][i].sum().item()
                assert final_output_effective_len == effective_len, \
                    f"Effective length mismatch: {final_output_effective_len} != {effective_len}"
                try:
                    _position_ids = get_rope_index(
                            self.processor,
                            input_ids=input_ids_i,
                            image_grid_thw=model_inputs.get("image_grid_thw"),
                            video_grid_thw=model_inputs.get("video_grid_thw"),
                            second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                            attention_mask=attention_mask_i
                        )
                    position_ids.append(_position_ids)  # (3, seq_len)
                except:
                    logger.error(f"Failed to get position ids for trajectory {i}, input_ids: {input_ids_i}, attention_mask: {attention_mask_i}")
                    torch.save({
                        "final_output": final_output,
                        "model_inputs": model_inputs,
                    }, f"tmp/final_output_{i}.pt")
                    raise 
            final_output['position_ids'] = torch.stack(position_ids, dim=0)  #
        else:
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
        timeout_seconds = self.config.tool_call_time_out
        max_retries = self.config.tool_call_max_retries
        for attempt in range(max_retries):
            session = None
            try:
                timeout = aiohttp.ClientTimeout(total=timeout_seconds)
                session = aiohttp.ClientSession(timeout=timeout)
                async with session.post(
                    url=self.config.tool_server_url,
                    json=data,
                ) as resp:
                    data = await resp.json()
                    return data
            except asyncio.TimeoutError as e:
                if attempt == max_retries - 1:
                    raise e
                logging.warning(f"Attempt {attempt + 1} failed: {e}. traj_id: {data['trajectory_ids']}. Retrying...")
                await asyncio.sleep(1)  # Brief delay before retry
            finally:
                if session:
                    await session.close()
        
        # logging.error(f"Failed to interact after {max_retries} attempts. Ending the trajectory.")
        # # if we reach here, it means all retries failed, we return dummy data
        # num_samples = len(data['trajectory_ids'])
        # return {
        #     "observations": [''] * num_samples,
        #     "dones": [1] * num_samples,
        #     "valids": [0] * num_samples,
        # }
            
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
    
    async def close_traj_tool_threads(
        self,
        active_uids:Union[List[str], np.ndarray]
    ):
        """
            This function is used to close the trajectories that are overlong and clean up the tool server for corresponding tool threads.
        """
        if isinstance(active_uids, np.ndarray):
            active_uids = active_uids.tolist()
        if isinstance(active_uids, str):
            active_uids = [active_uids]
        finishs = [True for _ in active_uids] # all trajectories are finished
        actions = [''] * len(active_uids) # no actions, just finish the trajectories
        is_last_step = True # this is the last step
        batch_data = {
            "trajectory_ids": active_uids,
            "actions": actions,
            "finish": finishs, # if do_action is False, then it is a finish action, finishing the trajectory,
            "is_last_step": [is_last_step] * len(finishs)
        }
        response = await self.send_batch_requests_async(batch_data)
        return 
        
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
        active_idx = 0
        for i, obs in enumerate(next_obs):
            tool_interact_info_i = {}
            if isinstance(obs, str):
                # can be invalid
                processed_next_obs.append(obs)
                rewards.append(None)
                tool_interact_info_i['obs'] = obs
                tool_interact_info_i['reward'] = None
            elif isinstance(obs, dict):
                assert "obs" in obs, f"Observation dict must contain 'obs' key, but got {obs.keys()}"
                _obs = obs.get('obs', '')
                _reward = obs.get('reward', None)
                assert isinstance(_obs, str), f"Expected 'obs' to be a string, but got {type(_obs)}"
                assert _reward is None or isinstance(_reward, (int, float)), f"Expected 'reward' to be None, int, or float, but got {type(_reward)}"
                processed_next_obs.append(_obs)
                rewards.append(_reward)
                # store tool interaction info if exists
                tool_interact_info_i = {k: v for k, v in obs.items()}
                tool_interact_info_i['obs'] = _obs
                tool_interact_info_i['reward'] = _reward
            else:
                raise ValueError(f"Invalid observation type: {type(obs)}. Expected str or dict.")
            tool_interact_info_i['active'] = bool(active_mask[i])
            if active_mask[i]:
                tool_interact_info_i['trajectory_id'] = active_uids[active_idx] if active_idx < len(active_uids) else None
                tool_interact_info_i['action'] = responses[active_idx] if active_idx < len(responses) else None
                tool_interact_info_i['is_last_step'] = is_last_step
                active_idx += 1
            tool_interact_info_i['done'] = dones[i]
            tool_interact_info_i['valid_action'] = valid_action[i]
            tool_interact_info_i['finish'] = _finishs[i]
            tool_interact_info_i['invalid_reason'] = tool_interact_info_i.get('invalid_reason', None)
            tool_interact_info.append(tool_interact_info_i)
        next_obs = processed_next_obs
        return next_obs, dones, valid_action, _finishs, rewards, tool_interact_info