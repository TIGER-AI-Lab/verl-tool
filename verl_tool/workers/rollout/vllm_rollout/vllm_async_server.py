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
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    AsyncEngineArgs,
    copy_to_local,
    ExternalZeroMQDistributedExecutor,
    ExternalRayDistributedExecutor,
    OpenAIServingChat,
    OpenAIServingModels,
    BaseModelPath,
    SamplingParams,
    AsyncLLM,
)
@ray.remote(num_cpus=1)
class AsyncvLLMServer(VerlAsyncvLLMServer.__ray_actor_class__):
    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        self.max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            repetition_penalty=1.0,
            max_new_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"override_generation_config: {kwargs}")

        backend = os.environ.get("VERL_VLLM_DISTRIBUTED_BACKEND", "zeromq")
        if backend == "zeromq":
            distributed_executor_backend = ExternalZeroMQDistributedExecutor
        elif backend == "ray":
            distributed_executor_backend = ExternalRayDistributedExecutor
        else:
            distributed_executor_backend = None

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=config.free_cache_engine,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=self.max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=False, # changed to False by verl-tool for higher output quality
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # init async llm engine
        vllm_config = self._create_engine_config(engine_args)
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

        # build serving chat
        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=RequestLogger(max_log_len=4096),
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=config.multi_turn.tool_config_path is not None,
            tool_parser=config.multi_turn.format,  # hermes, llama3_json, ...
        )
        # added by verl-tool
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
