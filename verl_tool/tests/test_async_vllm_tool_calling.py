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
import asyncio
import os
import re
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict
import itertools

import aiohttp
import fastapi
import numpy as np
import ray
import uvicorn
from datasets import load_dataset
from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from starlette.responses import JSONResponse
from verl.utils import hf_tokenizer, hf_processor
import torch
from tensordict import TensorDict
from typing import List

from verl_tool.agent_workers.tool_chat_completion_scheduler import NaiveChatCompletionScheduler
from verl_tool.tests.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto
from verl.trainer.main_ppo import create_rl_dataset
from verl_tool.llm_agent.config import AgentActorConfig

boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})")

def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@ray.remote(num_cpus=1)
class Sandbox:
    """Sandbox to execute python code.

    WARNING: This class is for testing purpose only, do not use it in production.
    Please use a sandbox with strong isolation and security restrictions instead.
    """

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

    async def code_execution(self, request: Request):
        request_json = await request.json()
        code = request_json["code"]
        print(f"execute code:\n{code}")

        _, temp_file = tempfile.mkstemp(suffix=".py", prefix="temp_code", dir=None, text=True)
        with open(temp_file, "w") as f:
            f.write(code)

        return JSONResponse(content={"stdout": "No Result", "stderr": "No Result", "returncode": 0})
        # try:
        #     process = await asyncio.create_subprocess_exec(sys.executable, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

        #     stdout, stderr = await process.communicate()

        #     return JSONResponse(content={"stdout": stdout.decode(), "stderr": stderr.decode(), "returncode": process.returncode})
        # finally:
        #     try:
        #         os.unlink(temp_file)
        #     except:  # noqa: E722
        #         pass

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print("FastAPI startup")
            self.server_ready.set()
            yield

            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/code/execution", self.code_execution, methods=["POST"])

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> str:
        """Get FastAPI server address."""
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"


class ToolChatCompletionScheduler(NaiveChatCompletionScheduler):
    """This is a chat completion scheduler that supports sandbox code execution
    """

    def __init__(self, config, model_path, server_addresses, sandbox_address, agent_config, **kwargs):
        super().__init__(config, model_path, server_addresses, **kwargs)
        self.sandbox_address = sandbox_address
        self.agent_config = agent_config
        print(f"agent_config: {self.agent_config}")
    
    async def parse_code_block(self, action: str) -> str:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing Python code
            
        Returns:
            parsed_code
        """
        # Try to find Python code in various formats
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
        
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```\n?python(.*?)```", action, re.DOTALL)
        
        if len(all_valid_python_code) == 0:
            return ""
        
        # use all the code blocks
        parsed_code = "\n".join([code.strip() for code in all_valid_python_code])
        
        return parsed_code
    
    async def sandbox_code_execution(self, code: str) -> Dict[str, Any]:
        """Execute python code in sandbox."""
        try:
            session = aiohttp.ClientSession()
            async with session.post(
                url=f"http://{self.sandbox_address}/code/execution",
                json={"code": code},
            ) as resp:
                return await resp.json()
        finally:
            await session.close()

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            max_completion_tokens=self.agent_config.max_action_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            include_stop_str_in_output=self.agent_config.include_stop_str_in_output,
            stop=self.agent_config.action_stop_tokens, 
        )

        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature
        
        print(f"[ToolChatCompletionScheduler] generate_sequences sampling params: {kwargs}")
            
            
        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        kwargs.update(sampling_params)

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            new_kwargs = {k: v for k, v in kwargs.items()}

            batch_conversations, batch_index, turn = (
                info["batch_conversations"],
                info["batch_index"],
                info["turn"],
            )
            print(f"[id={completions.id},turn={turn}] {completions}")
            role, content = completions.choices[0].message.role, completions.choices[0].message.content
            batch_conversations[batch_index].append({"role": role, "content": content})


            # STEP 0: if we reach max turns + 1 break and if we reach max turns, remove stop token
            if turn == self.agent_config.max_turns + 1:
                return
            elif turn == self.agent_config.max_turns:
                # if we reach max turns + 1, remove code block stop token
                print(f"[id={completions.id},turn={turn}] new_kwargs 1: {new_kwargs}")
                new_kwargs.pop("stop") # TODO: `force_finish_for_last_turn`
                print(f"[id={completions.id},turn={turn}] new_kwargs 2: {new_kwargs}")

            # STEP 1: check if we got answer # TODO: update error 
            matches = boxed_pattern.findall(content)
            if matches:
                print(f"[id={completions.id},turn={turn}] Got answer: {matches[0]}, done!")
                return

            # STEP 2: check if stop reason is 
            finish_reason = completions.choices[0].finish_reason
            stop_reason = completions.choices[0].stop_reason

            # STEP 3: execute code block in sandbox 
            if finish_reason == "stop" and stop_reason is not None:
                # TODO: send http request 
                code = await self.parse_code_block(content) 
                result = await self.sandbox_code_execution(code) # TODO: use `interact_with_tool_server`
                stdout, stderr = result["stdout"], result["stderr"]
                batch_conversations[batch_index].append({"role": "tool", "content": f"{stdout}{stderr}"}) 

            # STEP 4: resubmit chat completions with code block output 
            extra_headers = {"x-request-id": completions.id}
            await self.submit_chat_completions(
                callback=callback,
                callback_additional_info={
                    "batch_conversations": batch_conversations,
                    "batch_index": batch_index,
                    "turn": turn + 1,
                },
                model=self.model_name,
                messages=batch_conversations[batch_index],
                extra_headers=extra_headers,
                **new_kwargs,
            )

        tasks, batch_conversations = [], [None] * len(batch) * n
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)): 
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = list(conversation)
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "turn": 1, 
                        },
                        model=self.model_name,
                        messages=batch_conversations[batch_index],
                        **kwargs,
                    )
                )
            )

        await asyncio.gather(*tasks)
        print("[ToolChatCompletionScheduler] generate_sequences done")
        
        return self._postprocess(batch, batch_conversations, n) 
    
    
    def _postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]

        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask
        
        return_batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )

        if self.agent_config.mask_observations:
            # Create tool mask directly from tokenized responses
            tool_mask = torch.ones_like(responses["input_ids"])
            tool_begin_tokens = self.tokenizer("<|im_start|>user\n<tool_response>\n", return_tensors="pt", add_special_tokens=False)["input_ids"] # TODO: update to config 
            tool_end_tokens = self.tokenizer("</tool_response><|im_end|>\n", return_tensors="pt", add_special_tokens=False)["input_ids"] # TODO: update to config
            
            # Find and mask tool-related content in each response
            for i in range(len(responses["input_ids"])):
                response_tokens = responses["input_ids"][i]
                pos = 0
                while pos < len(response_tokens):
                    # Look for tool start token
                    if torch.equal(response_tokens[pos:pos+len(tool_begin_tokens[0])], tool_begin_tokens[0]):
                        start_pos = pos
                        # Look for tool end token
                        pos += len(tool_begin_tokens)
                        while pos < len(response_tokens):
                            if torch.equal(response_tokens[pos:pos+len(tool_end_tokens[0])], tool_end_tokens[0]):
                                end_pos = pos + len(tool_end_tokens[0])
                                # Mask the entire tool block
                                tool_mask[i, start_pos:end_pos] = False
                                break
                            pos += 1
                    pos += 1
                
            response_mask = tool_mask * responses["attention_mask"]
            info_mask = torch.cat([prompts["attention_mask"], response_mask], dim=1) # [bs, prompt_length + response_length]
            return_batch["info_mask"] = info_mask
        
        return DataProto(batch=return_batch)


def test_vllm_tool_calling():
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # Load config # TODO: update in .sh script expecially for agent config 
    config = OmegaConf.load("verl_tool/trainer/config/ppo_trainer.yaml")
    config.actor_rollout_ref.model.path = "/map-vepfs/models/Qwen2.5-Math-1.5B"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.chat_scheduler = "verl_tool.tests.test_async_vllm_tool_calling.ToolChatCompletionScheduler"
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 1024
    config.actor_rollout_ref.rollout.n = 2
    config.trainer.n_gpus_per_node = 2
    config.data.train_files = "./data/tests/aime24/train.parquet"
    config.data.return_raw_chat = True

    agent_config = AgentActorConfig()
    for key in getattr(config, 'agent', {}).keys():
        if key in agent_config.__dict__.keys():
            setattr(agent_config, key, config.agent[key])
    setattr(agent_config, 'n', config.actor_rollout_ref.rollout.n)

    agent_config.max_action_length = 512
    agent_config.max_turns = 1
    agent_config.include_stop_str_in_output = False
    agent_config.action_stop_tokens = ["```output"] # TODO: magic string delete after testing # TODO: use ```without python
    # if agent_config.action_stop_tokens is not None:
    #     if os.path.exists(agent_config.action_stop_tokens):
    #         with open(agent_config.action_stop_tokens, 'r') as f:
    #             agent_config.action_stop_tokens = [x for x in f.read().split(',') if x]
    #         print(f"Using action stop tokens: {agent_config.action_stop_tokens}")
    #     else:
    #         raise ValueError(f"action_stop_tokens file not found: {agent_config.action_stop_tokens}")
    # else:
    #     agent_config.action_stop_tokens = []


    # Init sandbox and async rollout manager
    sandbox = Sandbox.options(num_cpus=1).remote()
    sandbox_address = ray.get(sandbox.get_server_address.remote())
    # TODO: update `ray_trainer` init 
    async_rollout_manager = init_async_rollout_manager(config, scheduler_kwargs={"sandbox_address": sandbox_address, "agent_config": agent_config})

    # Build dataset
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=True)
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, None) 
    raw_prompt = train_dataset.dataframe['prompt']

    gen_batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompt),
        },
    )
    result = async_rollout_manager.generate_sequences(prompts=gen_batch)

    print(result)


if __name__ == "__main__":
    test_vllm_tool_calling()

# TODO: move to verl-tool 1. add `raw_prompt` 2. 
"""
# preprocess aime24 data
python examples/data_preprocess/aime24.py --local_dir ./data/tests/aime24 

# run test
python verl_tool/tests/test_async_vllm_tool_calling.py
"""
