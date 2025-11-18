
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import aiohttp
import asyncio
import numpy as np
import json
import uuid
import threading
import regex as re
from PIL import Image
from typing import Union, Optional, Dict
from typing import Any
from uuid import uuid4
from verl.utils.profiler import simple_timer
from .agent_loop import AgentLoopBase, register, AgentLoopOutput
from .vision_utils import decode_image_url, encode_image, encode_image_url
from verl_tool.utils.dataset.audio_utils import encode_audio_data

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

from dataclasses import dataclass

# 1) A sanitizer that strips all embedded NULs (and, optionally, any
#    other C0 control characters except common whitespace).
CONTROL_CHAR_RE = re.compile(
    # this matches U+0000 through U+001F, excluding tab(09), LF(0A), CR(0D)
    r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
)

@dataclass
class AgentActorConfig:
    enable_agent: bool=True
    max_turns: int=0
    val_max_turns: int=0
    max_start_length: int=None
    max_prompt_length: int=None
    max_response_length: int=None
    train_max_response_length: int=None
    val_max_response_length: int=None
    max_model_len: int=None  # Maximum model length, used for async rollout to limit the input length.
    max_obs_length: int=None
    max_action_length: int=None
    tool_server_url: str = None
    n: int=1
    truncate_obs_side: str='middle'
    truncate_response_side: str='left'
    action_stop_tokens: list=None
    mask_observations: bool=True
    force_finish_for_last_turn: bool=False
    enable_mtrl: bool=False
    mtrl_role: str="user"
    mtrl_sep: str=None # "\n<|im_start|>system\n{obs}<|im_end|>\n<|im_start|>assistant\n"
    assistant_role: str="assistant"
    turn_end_token: str="<|im_end|>"
    rollout_mode: str="async" # "sync" or "async"
    mask_overlong_loss: bool=False # whether to mask the overlong trajectory to not train on it
    enable_tqdm: bool=True # Whether to enable tqdm for async rollout.
    tool_call_time_out: int=None # Timeout for tool calls in async rollout.
    tool_call_max_retries: int=5 # Maximum number of retries for tool calls in async rollout.
    
    # The following fields are to be disgarded, only for compatibility
    rolling_with_prompt: bool=False
    call_tool_first: bool=False
    additional_eos_token_ids: list=None
    max_concurrent_trajectories: int=None # Maximum number of concurrent trajectories for async rollout. If None, no limit is applied.
    over_sampling: bool=False # Whether to over-sample the trajectories in async rollout.
    min_turns: int=0
    retokeniziation: bool=False
    
    
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
    
@register("verltool_agent")
class VerlToolAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""
    _init_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
    
        
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        with cls._init_lock:
            cls.tokenizer = tokenizer
            cls.processor = processor
            
            agent_config = AgentActorConfig()
            rollout_config = config.actor_rollout_ref
            for key in getattr(rollout_config, 'agent', {}).keys():
                if key in agent_config.__dict__.keys():
                    setattr(agent_config, key, rollout_config.agent[key])
            setattr(agent_config, 'n', rollout_config.rollout.n)
            setattr(agent_config, 'max_model_len', rollout_config.rollout.max_model_len)
            setattr(agent_config, 'rollout_mode', "async")
            cls.agent_config = agent_config

            if cls.agent_config.action_stop_tokens is not None:
                if os.path.exists(cls.agent_config.action_stop_tokens):
                    with open(cls.agent_config.action_stop_tokens, 'r') as f:
                        cls.action_stop_tokens = [x for x in f.read().split(',') if x]
                    logger.info(f"Using action stop tokens: {cls.action_stop_tokens}")
                else:
                    raise ValueError(f"action_stop_tokens file not found: {cls.agent_config.action_stop_tokens}")
            else:
                cls.action_stop_tokens = []

            if cls.agent_config.mtrl_sep is None:
                messages = [{"role": "system", "content": "{obs}"}]
                cls.agent_config.mtrl_sep = cls.agent_config.turn_end_token + "\n" + cls.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                cls.agent_config.mtrl_sep = cls.agent_config.mtrl_sep.replace("system", cls.agent_config.mtrl_role)
                cls.mtrl_sep_prefix, cls.mtrl_sep_suffix = cls.agent_config.mtrl_sep.split("{obs}")
                cls.mtrl_sep_prefix_ids = cls.tokenizer.encode(cls.mtrl_sep_prefix)
                cls.mtrl_sep_suffix_ids = cls.tokenizer.encode(cls.mtrl_sep_suffix)
            cls.enable_mtrl = cls.agent_config.enable_mtrl
            cls.mtrl_sep = cls.agent_config.mtrl_sep
            cls.max_model_len = int(cls.agent_config.max_model_len or (cls.agent_config.max_prompt_length + cls.agent_config.max_response_length))
            
            cls.response_length = rollout_config.rollout.response_length
            cls.max_response_length = cls.agent_config.max_response_length if cls.agent_config.max_response_length is not None else cls.response_length
            cls.max_response_length = min(cls.max_response_length, cls.max_model_len)
            cls.val_max_response_length = cls.agent_config.val_max_response_length if cls.agent_config.val_max_response_length is not None else cls.max_response_length
            cls.train_max_response_length = cls.agent_config.train_max_response_length if cls.agent_config.train_max_response_length is not None else cls.max_response_length
            assert cls.response_length >= cls.max_response_length, f"rollout.response_length {cls.response_length} must be >= agent.max_response_length {cls.max_response_length}"
            assert cls.response_length >= cls.val_max_response_length, f"rollout.response_length {cls.response_length} must be >= agent.val_max_response_length {cls.val_max_response_length}"
            assert cls.response_length >= cls.train_max_response_length, f"rollout.response_length {cls.response_length} must be >= agent.train_max_response_length {cls.train_max_response_length}"
            cls.max_action_length = cls.agent_config.max_action_length
            cls.max_obs_length = cls.agent_config.max_obs_length if cls.agent_config.max_obs_length is not None else 512
            cls.max_turns = cls.agent_config.max_turns if cls.agent_config.max_turns > 0 else 10
            cls.val_max_turns = cls.agent_config.val_max_turns if cls.agent_config.val_max_turns > 0 else cls.agent_config.max_turns
            
            
            # for multimodal processing
            cls.qwen_image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
            cls.qwen_video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
            cls.qwen_audio_placeholder = "<|audio_bos|><|AUDIO|><|audio_eos|>"  # for qwen2.5-omni
            cls.non_truncate_tokens = [
                "<|vision_start|>",
                "<|image_pad|>",
                "<|vision_end|>",
                "<|video_pad|>",
                "<|audio_bos|>",
                "<|AUDIO|>",
                "<|audio_eos|>",
            ]
            cls.non_truncate_token_ids = [cls.tokenizer.convert_tokens_to_ids(tok) for tok in cls.non_truncate_tokens]
            
            
            if cls._class_initialized:
                return
            cls._class_initialized = True
        
    async def _aiohttp_request(self, data):
        timeout_seconds = self.agent_config.tool_call_time_out
        max_retries = self.agent_config.tool_call_max_retries
        for attempt in range(max_retries):
            session = None
            try:
                timeout = aiohttp.ClientTimeout(total=timeout_seconds)
                session = aiohttp.ClientSession(timeout=timeout)
                async with session.post(
                    url=self.agent_config.tool_server_url,
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
        traj_id: str,
        action: str,
        do_action: bool,
        extra_fields: Any = None,
        is_last_step: bool = False,
    ) -> list[str]:
        batch_data = {
            "trajectory_ids": [traj_id],
            "actions": [action],
            "finish": [not do_action],
            "is_last_step": [is_last_step],
        }
        if extra_fields is not None:
            batch_data['extra_fields'] = [extra_fields]
        
        response = await self.send_batch_requests_async(batch_data)
        obs = response['observations'][0]
        done = int(response['dones'][0])
        is_valid_action = int(response['valids'][0])
        
        # postprocess next_obs. For now we support two types of observations:
        # 1. string observations, which will be the most common case
        # 2. dict observations, e.g. {"obs": "some observation", "reward": 1.0}
        #     for now we only support "obs" and "reward" keys, but can be extended later
        tool_interact_info = {}
        if isinstance(obs, str):
            # can be invalid
            obs_text = obs
            reward = None
            tool_interact_info['obs'] = obs
            tool_interact_info['reward'] = None
        elif isinstance(obs, dict):
            assert "obs" in obs, f"Observation dict must contain 'obs' key, but got {obs.keys()}"
            obs_text = obs.get('obs', '')
            reward = obs.get('reward', None)
            assert isinstance(obs_text, str), f"Expected 'obs' to be a string, but got {type(obs)}"
            assert reward is None or isinstance(reward, (int, float)), f"Expected 'reward' to be None, int, or float, but got {type(reward)}"
            # store tool interaction info if exists
            tool_interact_info = {k: v for k, v in obs.items()} # image or other info
        else:
            raise ValueError(f"Invalid observation type: {type(obs)}. Expected str or dict.")
        
        tool_interact_info['obs'] = obs_text
        tool_interact_info['reward'] = reward
        tool_interact_info['trajectory_id'] = traj_id
        tool_interact_info['action'] = action
        tool_interact_info['is_last_step'] = is_last_step
        tool_interact_info['done'] = done
        tool_interact_info['valid_action'] = is_valid_action
        tool_interact_info['finish'] = not do_action
        tool_interact_info['invalid_reason'] = tool_interact_info.get('invalid_reason', None)
        
        return tool_interact_info
        

    async def close_traj_tool_threads(
        self,
        request_id: str,
    ):
        """
            This function is used to close the trajectories that are overlong and clean up the tool server for corresponding tool threads.
        """
        finishs = [True] # all trajectories are finished
        actions = [''] # no actions, just finish the trajectories
        is_last_step = True # this is the last step
        batch_data = {
            "trajectory_ids": [request_id],
            "actions": actions,
            "finish": finishs, # if do_action is False, then it is a finish action, finishing the trajectory,
            "is_last_step": [is_last_step] * len(finishs)
        }
        _ = await self.send_batch_requests_async(batch_data)
        return 
        
        
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        prompt_ids = list(kwargs["raw_prompt_ids"])
        multi_modal_data = kwargs.get("multi_modal_data") or {}
        image_data = multi_modal_data.get("image")
        encoded_image_data = [encode_image_url(img) for img in image_data] if image_data is not None else None
        audio_data = multi_modal_data.get("audio")
        encoded_audio_data = [encode_audio_data(audio) for audio in audio_data] if audio_data is not None else None
        use_tool = kwargs.get("use_tool", self.agent_config.enable_agent)
        
        metrics = {}
        request_id = str(uuid4().hex)
        
        stats_dict = {
            "num_turns": 0,
            "empty_responses": 0,
            "valid_action": 0,
            "action_lengths": [],
            "action_logps": [],
            "obs_lengths": [],
            "rewards": [],
            "tool_interact_info": [],
            "is_traj_finished": False,
        }
        
        if image_data or audio_data:
            raw_prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            processor_kwargs = {"text": [raw_prompt_text], "return_tensors": "pt"}
            if image_data:
                processor_kwargs["images"] = image_data
            if audio_data:
                processor_kwargs["audio"] = audio_data
            model_inputs = self.processor(**processor_kwargs)
            prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
        
        max_turns = self.max_turns if not kwargs.get("validate", False) else self.val_max_turns
        max_response_length = self.train_max_response_length if not kwargs.get("validate", False) else self.val_max_response_length
        max_action_length = self.max_action_length or max_response_length
        max_obs_length = self.max_obs_length
        
        agent_sampling_params = sampling_params.copy()
        agent_sampling_params.update({
            "n": 1,  # already repeated by n times in repeat_inputs_by_n
            "stop": self.action_stop_tokens,  # stop when generated an end of action
            "include_stop_str_in_output": True,
        })
        
        running_prompt_ids = prompt_ids.copy()
        running_image_data = image_data.copy() if image_data is not None else None
        running_audio_data = audio_data.copy() if audio_data is not None else None
        response_mask = []
        response_logprobs = []
        traj_stop_reason = ""

        logger.debug(f"Starting agent loop for traj_id={request_id} with use_tool={use_tool}, max_turns={max_turns}, max_response_length={max_response_length}, max_action_length={max_action_length}, max_obs_length={max_obs_length}")
        
        for step in range(max_turns + 1):
            previous_length = len(running_prompt_ids)
            available_length = max(max_response_length - len(running_prompt_ids) + len(prompt_ids), 0)
            max_tokens_for_this_turn = min(max_action_length, available_length)
            is_last_step = (step == max_turns)
            if max_tokens_for_this_turn <= 0:
                traj_stop_reason = "max_model_len_exceeded"
                break
            agent_sampling_params["max_tokens"] = max_tokens_for_this_turn # for vllm
            logger.debug(f"Turn {step}: available_length={available_length}, max_tokens_for_this_turn={max_tokens_for_this_turn}")
            # agent_sampling_params["max_new_tokens"] = max_tokens_for_this_turn # for sglang
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=running_prompt_ids,
                    sampling_params=agent_sampling_params,
                    image_data=running_image_data,
                    audio_data=running_audio_data,
                ) # request_id here should be unique for each generate call, otherwise vllm can generate empty response
                if output.text.strip() == "":
                    logger.warning(f"Turn {step}: Generated empty response for traj_id={request_id}. prompt_ids length: {len(running_prompt_ids)}")
            gen_ids = output.token_ids
            gen_logprobs = output.log_probs or [0.0] * len(gen_ids)
            gen_text = output.text
            running_prompt_ids.extend(gen_ids)
            response_mask.extend([1] * len(gen_ids))
            response_logprobs.extend(gen_logprobs)
            
            
            stats_dict["num_turns"] += 1
            stats_dict["action_lengths"].append(len(gen_ids))
            stats_dict["action_logps"].append(sum(gen_logprobs))
            if gen_text.strip() == "":
                stats_dict["empty_responses"] += 1
            
            finish_reason = output.finish_reason
            stop_reason = output.stop_reason
            
            if not use_tool:
                # only one turn generation
                stats_dict["is_traj_finished"] = True
                traj_stop_reason = f"no_tool-{finish_reason}"
                break
            
            # judge whether to interact with tool or finish
            do_action = False
            action_text = ""
            if finish_reason == "stop" and stop_reason:
                for action_stop_token in self.action_stop_tokens:
                    if action_stop_token in stop_reason:
                        # do action
                        do_action = True
                        action_text = (gen_text.split(action_stop_token)[0] + action_stop_token)
                        break
            # send generated action to tool server
            if do_action and not is_last_step:
                extra_fields = {}
                if encoded_image_data is not None:
                    extra_fields["images"] = encoded_image_data
                if encoded_audio_data is not None:
                    extra_fields["audio"] = encoded_audio_data
                logger.info(f"Turn {step}: finish_reason={finish_reason}, stop_reason={stop_reason}, do_action={do_action}, action_text={json.dumps(action_text[-50:])}")
                with simple_timer("tool_calls", metrics):
                    tool_results = await self.interact_with_tool_server(
                        traj_id=request_id,
                        action=action_text,
                        do_action=do_action,
                        extra_fields=extra_fields,
                        is_last_step=is_last_step,
                    )
                # process observations and prepare for next turn
                obs_text = tool_results['obs']
                if tool_results.get('image', None):
                    images = [tool_results['image']] if not isinstance(tool_results['image'], list) else tool_results['image']
                    decoded_images = [decode_image_url(img_url) for img_url in images]
                    running_image_data.extend(decoded_images)
                    # add image placeholder token ids
                    # first see whether there are <image> tags in obs_text
                    num_image_tags = obs_text.count("<image>")
                    if num_image_tags < len(decoded_images):
                        # append at the end
                        obs_text += "<image>" * (len(decoded_images) - num_image_tags)
                        num_image_tags = len(decoded_images)
                    # now replace <image> tags with image placeholder tokens
                    obs_text = obs_text.replace("<image>", self.qwen_image_placeholder, num_image_tags)
                    obs_token_ids = self.processor(text=[obs_text], images=decoded_images, return_tensors="pt")["input_ids"].squeeze(0).tolist()
                    if max_obs_length < len(obs_token_ids):
                        if self.agent_config.truncate_obs_side == 'left':
                            truncation_index = max_obs_length
                            if obs_token_ids[max_obs_length] in self.non_truncate_token_ids:
                                # find the nearest non-truncate token id before max_obs_length
                                truncation_index = max_obs_length - 1
                                while truncation_index > 0 and obs_token_ids[truncation_index] in self.non_truncate_token_ids:
                                    truncation_index -= 1
                            obs_token_ids = obs_token_ids[:truncation_index]
                            obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
                        else:
                            raise NotImplementedError(f"Only left truncation is supported for multimodal observations for now.")
                elif tool_results.get('audio', None):
                    audios = [tool_results['audio']] if not isinstance(tool_results['audio'], list) else tool_results['audio']
                    decoded_audios = [decode_image_url(audio_url) for audio_url in audios]
                    running_audio_data.extend(decoded_audios)
                    # add audio placeholder token ids
                    # first see whether there are <audio> tags in obs_text
                    num_audio_tags = obs_text.count("<audio>")
                    if num_audio_tags < len(decoded_audios):
                        # append at the end
                        obs_text += "<audio>" * (len(decoded_audios) - num_audio_tags)
                        num_audio_tags = len(decoded_audios)
                    # now replace <audio> tags with audio placeholder tokens
                    obs_text = obs_text.replace("<audio>", self.qwen_audio_placeholder, num_audio_tags)
                    # for mtrl
                    if self.enable_mtrl:
                        obs_text = self.mtrl_sep.format(obs=obs_text)
                    obs_token_ids = self.processor(text=[obs_text], audio=decoded_audios, return_tensors="pt")["input_ids"].squeeze(0).tolist()
                    # obs_token_ids = obs_token_ids[:max_obs_length]
                    if max_obs_length < len(obs_token_ids):
                        if self.agent_config.truncate_obs_side == 'left':
                            truncation_index = max_obs_length
                            if obs_token_ids[max_obs_length] in self.non_truncate_token_ids:
                                # find the nearest non-truncate token id before max_obs_length
                                truncation_index = max_obs_length - 1
                                while truncation_index > 0 and obs_token_ids[truncation_index] in self.non_truncate_token_ids:
                                    truncation_index -= 1
                            obs_token_ids = obs_token_ids[:truncation_index]
                            obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
                        else:
                            raise NotImplementedError(f"Only left truncation is supported for multimodal observations for now.")
                else:
                    obs_token_ids = self.tokenizer.encode(obs_text)
                    if max_obs_length < len(obs_token_ids):
                        if self.agent_config.truncate_obs_side == 'left':
                            obs_token_ids = obs_token_ids[-max_obs_length:]
                            obs_token_ids.extend(self.tokenizer.encode("...(truncated)"))
                        elif self.agent_config.truncate_obs_side == 'right':
                            obs_token_ids = obs_token_ids[:max_obs_length]
                            obs_token_ids.extend(self.tokenizer.encode("(truncated)..."))
                        elif self.agent_config.truncate_obs_side == 'middle':
                            half_len = max_obs_length // 2
                            obs_token_ids = obs_token_ids[:half_len] + self.tokenizer.encode("...(truncated)...") + obs_token_ids[-half_len:]
                        else:
                            raise ValueError(f"Invalid truncate_obs_side: {self.agent_config.truncate_obs_side}")
                
                if self.enable_mtrl:
                    obs_token_ids = self.mtrl_sep_prefix_ids + obs_token_ids + self.mtrl_sep_suffix_ids
                # update stats
                stats_dict["obs_lengths"].append(len(obs_token_ids))
                if 'reward' in tool_results and tool_results['reward'] is not None:
                    stats_dict["rewards"].append(tool_results['reward'] if 'reward' in tool_results else 0.0)
                stats_dict["valid_action"] += tool_results['valid_action'] if 'valid_action' in tool_results else 0
                stats_dict["tool_interact_info"].append(tool_results)
                
                # update running prompt
                running_prompt_ids.extend(obs_token_ids)
                if self.agent_config.mask_observations:
                    response_mask.extend([0] * len(obs_token_ids))
                else:
                    response_mask.extend([1] * len(obs_token_ids))
                response_logprobs.extend([0.0] * len(obs_token_ids)) # pad with 0.0 logprobs for observations
                
                if self.agent_config.retokeniziation:
                    new_text = self.tokenizer.decode(running_prompt_ids[previous_length:], skip_special_tokens=False)
                    new_ids = self.tokenizer.encode(new_text)
                    if len(new_ids) != len(running_prompt_ids) - previous_length:
                        logger.warning(f"Retokenization changed the length from {len(running_prompt_ids) - previous_length} to {len(new_ids)}. traj_id={request_id}, turn={step}")
                    running_prompt_ids = running_prompt_ids[:previous_length] + new_ids
                    if len(response_mask) > running_prompt_ids:
                        response_mask = response_mask[:len(running_prompt_ids)]
                    else:
                        response_mask.extend([response_mask[-1]] * (len(running_prompt_ids) - len(response_mask)))
                    if len(response_logprobs) > running_prompt_ids:
                        response_logprobs = response_logprobs[:len(running_prompt_ids)]
                    else:
                        response_logprobs.extend([0.0] * (len(running_prompt_ids) - len(response_logprobs)))

                if tool_results['done']:
                    # finish the trajectory
                    stats_dict["is_traj_finished"] = True
                    traj_stop_reason = "tool_signaled_done"
                    break
            else:
                # finish the trajectory
                stats_dict["is_traj_finished"] = True
                if finish_reason == "stop":
                    if not stop_reason:
                        if gen_text.strip() == "":
                            traj_stop_reason = "model_chose_to_finish_with_empty_response"
                        else:
                            traj_stop_reason = "model_chose_to_finish"
                    else:
                        if do_action and is_last_step:
                            traj_stop_reason = "max_turns_reached"
                        else:
                            traj_stop_reason = "no_action_stop_token_found_in_stop_reason=" + stop_reason
                elif finish_reason == "length":
                    traj_stop_reason = "max_response_length_reached"
                else:
                    traj_stop_reason = f"finish_reason_{finish_reason}"
                break
        
        logger.debug(f"Trajectory {request_id} finished after {step} turns. Stop reason: {traj_stop_reason}")
        response_ids = running_prompt_ids[len(prompt_ids):]
        assert len(response_ids) == len(response_mask), f"Response ids and mask length mismatch: {len(response_ids)} vs {len(response_mask)}"
        await self.close_traj_tool_threads(request_id=request_id)
            
        verl_tool_metrics = {
            "num_turns": stats_dict["num_turns"],
            "empty_responses": stats_dict["empty_responses"],
            "valid_action": stats_dict["valid_action"],
            "per_action_length": np.mean(stats_dict["action_lengths"]) if len(stats_dict["action_lengths"]) > 0 else 0,
            "per_obs_length": np.mean(stats_dict["obs_lengths"]) if len(stats_dict["obs_lengths"]) > 0 else 0,
            "per_action_logp": np.mean(stats_dict["action_logps"]) if len(stats_dict["action_logps"]) > 0 else 0,
            "per_reward_from_tool": np.mean(stats_dict["rewards"]) if len(stats_dict["rewards"]) > 0 else 0,
            "traj_actions_length": sum(stats_dict["action_lengths"]),
            "traj_obs_length": sum(stats_dict["obs_lengths"]),
            "generated_length": len(output.token_ids),
            "is_traj_finished": float(stats_dict["is_traj_finished"]),
        }
        
        multi_modal_output = {}
        if running_image_data is not None:
            multi_modal_output["image"] = running_image_data
        if running_audio_data is not None:
            multi_modal_output["audio"] = running_audio_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length],
            multi_modal_data=multi_modal_output,
            num_turns=stats_dict["num_turns"],
            metrics=metrics,
            extra_fields={"tool_interact_info": stats_dict.get("tool_interact_info", []), "traj_stop_reason": traj_stop_reason, "verl_tool_metrics": verl_tool_metrics},
        )
        return output
