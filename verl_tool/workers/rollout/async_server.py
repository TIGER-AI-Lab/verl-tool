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
import verl.workers.rollout.async_server
from typing import Type
from verl.workers.rollout.async_server import AsyncServerBase, AsyncLLMServerManager
from .chat_scheduler import VerlToolChatCompletionScheduler
logger = logging.getLogger(__file__)

def async_server_class(rollout_backend: str) -> Type[AsyncServerBase]:
    """Get async server class.

    Args:
        rollout_backend: str, rollout backend, should be "vllm" or "sglang".

    Returns:
        Type[AsyncServerBase]: async server class.
    """
    if rollout_backend == "vllm":
        # from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
        from .vllm_rollout.vllm_async_server import AsyncvLLMServer # added by verl-tool

        return AsyncvLLMServer
    elif rollout_backend == "sglang":
        from verl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSglangServer

        return AsyncSglangServer
    else:
        raise NotImplementedError
verl.workers.rollout.async_server.async_server_class = async_server_class  # for backward compatibility

class VerlToolAsyncLLMServerManager(AsyncLLMServerManager):
    """AsyncLLMServerManager manage a group of vllm instances, i.e AsyncvLLMServer."""

    def _init_chat_scheduler(self):
        self.chat_scheduler_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.chat_scheduler_loop)

        self.chat_scheduler = VerlToolChatCompletionScheduler(
            config=self.full_config,
            server_addresses=self.server_addresses,
        )

        self.chat_scheduler_ready.set()
        self.chat_scheduler_loop.run_forever()