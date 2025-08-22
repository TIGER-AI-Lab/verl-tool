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
import logging
from typing import Type
from verl.workers.rollout.async_server import AsyncServerBase, AsyncLLMServerManager
from .chat_scheduler import VerlToolChatCompletionScheduler
from verl.protocol import DataProto
logger = logging.getLogger(__file__)

class VerlToolAsyncLLMServerManager(AsyncLLMServerManager):
    """AsyncLLMServerManager manage a group of vllm instances, i.e AsyncvLLMServer."""

    def _init_chat_scheduler(self):
        self.chat_scheduler_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.chat_scheduler_loop)
        print("VerlToolChatCompletionScheduler")
        self.chat_scheduler = VerlToolChatCompletionScheduler(
            config=self.full_config,
            server_addresses=self.server_addresses,
        )

        self.chat_scheduler_ready.set()
        self.chat_scheduler_loop.run_forever()
    
    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        self.wake_up()
        result = super().generate_sequences(prompts, **sampling_params)
        self.sleep()
        return result

# here are the hacky parts to replace the original AsyncLLMServerManager with VerlToolAsyncLLMServerManager
import verl.experimental.agent_loop
import verl.workers.rollout.async_server
verl.experimental.agent_loop.AgentLoopManager = VerlToolAsyncLLMServerManager # replace the original AgentLoopManager with VerlToolAsyncLLMServerManager
verl.workers.rollout.async_server.AsyncLLMServerManager = VerlToolAsyncLLMServerManager # replace the original AsyncLLMServerManager with VerlToolAsyncLLMServerManager