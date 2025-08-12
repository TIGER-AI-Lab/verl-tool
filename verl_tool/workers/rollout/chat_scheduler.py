import asyncio
import aiohttp
import time
import heapq
import torch
from tqdm.asyncio import tqdm
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler, logger, DictConfig
from openai.types import Completion
from openai.types.chat.chat_completion import ChatCompletion
from openai import AsyncOpenAI
from typing import Union, List, Dict, Any, Iterable
from verl.protocol import DataProto
from verl_tool.llm_agent import AgentActorManager, AgentActorConfig

def print_messages(messages):
    from copy import deepcopy
    messages = deepcopy(messages)
    for message in messages:
        for content in message['content']:
            if content['type'] == 'image_url':
                content['image_url']['url'] = content['image_url']['url'][:100] + "..." if len(content['image_url']['url']) > 100 else content['image_url']['url']
            if content['type'] == 'video_url':
                content['video_url']['url'] = content['video_url']['url'][:100] + "..." if len(content['video_url']['url']) > 100 else content['video_url']['url']
    print(messages)

class VerlToolChatCompletionScheduler(ChatCompletionScheduler):
    """A chat completion scheduler for verl-tool, which is a wrapper around the ChatCompletionScheduler."""

    def __init__(
        self,
        config: DictConfig,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        super().__init__(config, server_addresses)
        rollout_config = config.actor_rollout_ref
        self.agent_actor_manager = AgentActorManager.from_rollout_config(self, rollout_config, rollout_mode="async")
        self.agent_config = self.agent_actor_manager.config
        self.max_model_len = self.agent_actor_manager.max_model_len
        self.max_response_length = self.agent_config.max_response_length
        self.max_concurrent_trajectories = self.agent_config.max_concurrent_trajectories
        self.tokenizer = self.agent_actor_manager.tokenizer
        self.over_sampling = self.agent_config.over_sampling
        print(f"AgentActorManager initialized with config: {self.agent_config}")
    
    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key="token-abc123", timeout=None, max_retries=0)
        return await client.chat.completions.create(**chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        try:
            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers")
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise ValueError(f"Request failed with status {data.get('code', 'unknown')}: {data}")
                return ChatCompletion(**data)
        finally:
            await session.close()

    async def _completions_openai(self, address: str, **complete_request) -> Completion:
        client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key="token-abc123", timeout=None, max_retries=0)
        return await client.completions.create(**complete_request)

    async def _completions_aiohttp(self, address: str, **complete_request) -> Completion:
        try:
            extra_body = complete_request.pop("extra_body", {})
            complete_request.update(extra_body or {})
            extra_headers = complete_request.pop("extra_headers")
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=complete_request,
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise ValueError(f"Request failed with status {data.get('code', 'unknown')}: {data}")
                return Completion(**data)
        finally:
            await session.close()
    
    async def _abort(self, address: str, request_id: str) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        try:
            async with session.post(
                url=f"http://{address}/v1/abort",
                headers={"Authorization": "Bearer token-abc123"},
                json={"request_id": request_id},
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise ValueError(f"Abort request failed with status {data.get('code', 'unknown')}: {data}")
                return data
        finally:
            await session.close()

    async def _submit_completions(
        self,
        prompt: Union[str, List[str], Iterable[int], Iterable[Iterable[int]], None],
        request_id: str,
        info: Dict[str, Any],
    ):
        """Submit chat completion request, wait request finish and do callback."""
        if request_id:
            request_id = request_id.removeprefix("chatcmpl-")
            if request_id not in self.request_id_to_address:
                address = self.weighted_addresses[0][1]
                self.weighted_addresses[0][0] += 1
                heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])
                self.request_id_to_address[request_id] = address
            assert request_id in self.request_id_to_address
            address = self.request_id_to_address.pop(request_id)
        else:
            raise ValueError("request_id must be provided for chat completion requests.")

        # use new request_id to avoid duplicate request_id problem
        self.request_id_to_address[request_id] = address
        openai_completion_allowed_keys = [
            "model", "prompt", "best_of", "echo", "frequency_penalty",
            "logit_bias", "logprobs", "max_tokens", "n", "presence_penalty",
            "seed", "stop", "stream", "stream_options", "suffix", "temperature", "top_p", "user",
            "extra_headers", "extra_query", "extra_body", "timeout"
        ]
        sampling_params = {k: v for k, v in info["__sampling_params__"].items() if k in openai_completion_allowed_keys}
        extra_body = {k: v for k, v in info["__sampling_params__"].items() if k not in openai_completion_allowed_keys}
        completion, exception = None, None
        if "max_tokens" in sampling_params:
            prompt_len = len(prompt)
            if prompt_len + sampling_params["max_tokens"] > self.max_model_len:
                sampling_params["max_tokens"] = self.max_model_len - prompt_len
                if sampling_params["max_tokens"] <= 0:
                    raise ValueError(f"max_tokens {sampling_params['max_tokens']} is too small for prompt length {prompt_len} and max model length {self.max_model_len}.")
                logger.debug(f"Adjusted max_tokens to {sampling_params['max_tokens']} for prompt length {prompt_len} and max model length {self.max_model_len}.")
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completion = await self._completions_aiohttp(
                address,
                prompt=prompt,
                extra_body=extra_body,
                extra_headers={"x-request-id": request_id + f"-{time.time()}"},  # add a unique request id to avoid random duplicate request_id problem, seems to be a bug in VLLM
                **sampling_params,
            )
        except Exception as e:
            # Let user handle the exception
            exception = e
            raise e 

        info["__depth__"] -= 1

        if exception is not None:
            logger.exception(f"chat completion failed with exception: {exception}")

        # No more ongoing completion requests
        if info["__depth__"] == 0:
            info["__done__"].set()
        
        return completion.choices[0].text

    async def _submit_chat_completions(
        self,
        messages: List[Dict[str, str]], 
        request_id: str, 
        info: Dict[str, Any]
    ):
        """Submit chat completion request, wait request finish and do callback."""
        if request_id:
            request_id = request_id.removeprefix("chatcmpl-")
            if request_id not in self.request_id_to_address:
                address = self.weighted_addresses[0][1]
                self.weighted_addresses[0][0] += 1
                heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])
                self.request_id_to_address[request_id] = address
            assert request_id in self.request_id_to_address
            address = self.request_id_to_address.pop(request_id)
        else:
            raise ValueError("request_id must be provided for chat completion requests.")

        # use new request_id to avoid duplicate request_id problem
        self.request_id_to_address[request_id] = address
        openai_chat_completion_allowed_keys = [
            "model", "messages", "audio", "frequency_penalty",
            "function_call", "functions", "logit_bias", "logprobs",
            "max_completion_tokens", "max_tokens", "metadata", "modalities",
            "n", "parallel_tool_calls", "prediction", "presence_penalty",
            "reasoning_effort", "response_format", "seed", "service_tier",
            "stop", "store", "stream", "stream_options", "temperature",
            "tool_choice", "tools", "top_logprobs", "top_p", "user",
            "web_search_options", "extra_headers", "extra_query",
            "extra_body", "timeout"
        ]

        sampling_params = {k: v for k, v in info["__sampling_params__"].items() if k in openai_chat_completion_allowed_keys}
        extra_body = {k: v for k, v in info["__sampling_params__"].items() if k not in openai_chat_completion_allowed_keys}
        chat_completion, exception = None, None

        if messages[-1]["role"] == self.agent_config.assistant_role:
            extra_body['continue_final_message'] = True
            extra_body['add_generation_prompt'] = False
        
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            chat_completion = await self._chat_completions_aiohttp(
                address,
                messages=messages,
                extra_body=extra_body,
                extra_headers={"x-request-id": request_id + f"-{time.time()}"},  # add a unique request id to avoid random duplicate request_id problem, seems to be a bug in VLLM
                **sampling_params,
            )
        except Exception as e:
            with open("error_messages.json", 'w') as f:
                import json
                json.dump(messages, f, indent=4)
            # Let user handle the exception
            exception = e
            raise e 

        info["__depth__"] -= 1

        if exception is not None:
            logger.exception(f"chat completion failed with exception: {exception}")

        # No more ongoing completion requests
        if info["__depth__"] == 0:
            info["__done__"].set()

        if not isinstance(chat_completion, ChatCompletion):
            raise ValueError(f"Expected ChatCompletion, got {type(chat_completion)}")
        
        return chat_completion.choices[0].message.content if chat_completion.choices else None

    def simple_postprocess(self, batch: DataProto, responses: List[str]) -> DataProto:
        prompt_ids = batch.batch["input_ids"]
        prompt_attention_mask = batch.batch["attention_mask"]
        responses = self.tokenizer(responses, return_tensors="pt", padding="max_length", padding_side="right", max_length=self.max_response_length, truncation=True)

        input_ids = torch.cat([prompt_ids, responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch.batch['prompts'] = prompt_ids
        batch.batch['input_ids'] = input_ids
        batch.batch['attention_mask'] = attention_mask
        batch.batch['position_ids'] = position_ids
        batch.batch['responses'] = responses["input_ids"]
        batch.batch['response_mask'] = responses["attention_mask"]
        return batch
    
    def submit_task(
        self, 
        prompt: List[int], 
        messages: List[dict],
        request_id: str, 
        info: Dict[str, Any]
    ) -> asyncio.Task:
        """Submit a task to the agent actor manager."""
        if info['is_multi_modal']:
            return asyncio.create_task(
                self._submit_chat_completions(messages=messages, request_id=request_id, info=info)
            )
        else:
            return asyncio.create_task(
                self._submit_completions(prompt=prompt, request_id=request_id, info=info)
            )
    
    async def simple_generate_sequences(
        self, batch: DataProto, **kwargs
    ) -> DataProto:
        t_start = time.time()
        kwargs.update({
            "model": self.model_name,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        })
        to_remove_keys = ["max_new_tokens", "detokenize"]
        for key in to_remove_keys:
            if key in kwargs:
                kwargs.pop(key)

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        tasks = []
        for batch_index in range(len(batch)):
            prompt = list(batch.non_tensor_batch["raw_prompt_ids"][batch_index]) # change ndarray to list
            rollout_messages = batch.non_tensor_batch["rollout_messages"][batch_index].tolist() # change RolloutMessagesMixin to list
            request_id = batch.non_tensor_batch["traj_ids"][batch_index]
            info = {
                "__sampling_params__": kwargs,
                "__depth__": 1,
                "__done__": asyncio.Event(),
                "is_multi_modal": "multi_modal_data" in batch.non_tensor_batch,
            }
            tasks.append(
                self.submit_task(
                    prompt=prompt,
                    messages=rollout_messages,
                    request_id=request_id,
                    info=info
                )
            )
        responses = await tqdm.gather(*tasks, total=len(tasks), desc="Simple generating sequences", disable=(len(tasks) < 10) or not self.agent_config.enable_tqdm)
        output_batch = self.simple_postprocess(batch, responses)
        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        return output_batch

    async def generate_sequences(self, batch: DataProto, **kwargs) -> DataProto:
        logger.info("[VerlToolChatCompletionScheduler] generate_sequences start")
        t_start = time.time()
        kwargs.update({
            "model": self.model_name,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        })

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature
            n = self.config.val_kwargs.n
        else:
            n = self.config.n
        if not batch.meta_info.get("is_repeated_by_n", False):
            repeated_batch = self.agent_actor_manager.repeat_inputs_by_n(batch)
        else:
            repeated_batch = batch
        repeated_chunk_batch = repeated_batch.chunk(len(repeated_batch))
        # repeated_batch = [repeated_batch] # for debug
        logger.warning(f"[VerlToolChatCompletionScheduler] generate_sequences number of chunks: {len(repeated_chunk_batch)}")
        tasks = []
        if self.agent_config.enable_agent:
            if self.max_concurrent_trajectories is not None and self.max_concurrent_trajectories > 0:
                semaphore = asyncio.Semaphore(self.max_concurrent_trajectories)
                async def run_with_semaphore(batch_index):
                    async with semaphore:
                        return await self.agent_actor_manager.run_llm_loop_async(
                            repeated_chunk_batch[batch_index],
                            **kwargs
                        )
                for batch_index in range(len(repeated_chunk_batch)):
                    tasks.append(
                        asyncio.create_task(
                            run_with_semaphore(batch_index)
                        )
                    )
            else:
                for batch_index in range(len(repeated_chunk_batch)):
                    tasks.append(
                        asyncio.create_task(
                            self.agent_actor_manager.run_llm_loop_async(
                                repeated_chunk_batch[batch_index],
                                **kwargs
                            )
                        )
                    )
            # gen_outputs = await asyncio.gather(*tasks)
            gen_outputs = await tqdm.gather(*tasks, total=len(tasks), desc="Async Generating sequences", disable=not self.agent_config.enable_tqdm)
            output_batch = DataProto.concat(gen_outputs)
        else:
            kwargs["max_tokens"] = self.max_response_length
            output_batch = await self.simple_generate_sequences(
                repeated_batch,
                **kwargs
            )
        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        logger.info("[VerlToolChatCompletionScheduler] generate_sequences for {} number of trajectories done, took {:.2f} seconds".format(
            len(repeated_batch), output_batch.meta_info["timing"]["generate_sequences"]
        ))
        return output_batch