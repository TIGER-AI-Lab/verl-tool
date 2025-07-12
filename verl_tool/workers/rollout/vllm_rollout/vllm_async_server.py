import os
import ray
import fastapi
import uvicorn
from contextlib import asynccontextmanager
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer as VerlAsyncvLLMServer
from verl.workers.rollout.async_server import AsyncServerBase, _get_free_port
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ErrorResponse, CompletionRequest, CompletionResponse
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

@ray.remote(num_cpus=1)
class AsyncvLLMServer(VerlAsyncvLLMServer.__ray_actor_class__):
    async def init_engine(self):
        # self.sibling_methods_record['_init_engine'](self)
        await super().init_engine()
        self.openai_serving_completion = OpenAIServingCompletion(
            self.engine,
            self.engine.model_config,
            self.openai_serving_chat.models,
            request_logger=RequestLogger(max_log_len=4096),
        )

    # added by verl-tool
    async def completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = CompletionRequest(**request_json)
        generator = await self.openai_serving_completion.create_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, CompletionResponse)
            return JSONResponse(content=generator.model_dump())
    
    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print(f"FastAPI listen on {self.address}:{self.port}")
            self.server_ready.set()
            yield

            # There's no way to gracefully restart uvicorn server if port is already in use,
            # so we exit the process directly and let AsyncLLMServerManager restart it.
            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/v1/chat/completions", self.chat_completion, methods=["POST"])
        app.router.add_api_route("/v1/completions", self.completion, methods=["POST"]) # added by verl-tool

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()
