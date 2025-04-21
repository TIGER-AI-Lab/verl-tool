import uuid, time, torch, re
from typing import List, Dict, Any, Tuple, Optional
from vllm import LLM, SamplingParams

from config import ModelConfig, ToolConfig
from utils import extract_python_tags, call_tool_server

# -------------------------------------------------
#                 ModelService
# -------------------------------------------------
class ModelService:
    """verl‑tool   evaluation‑time inference service"""

    def __init__(self, model_config: ModelConfig, tool_config: ToolConfig):
        self.model_config   = model_config
        self.tool_config    = tool_config
        self.model          = None
        self.tokenizer      = None
        self.stop_tokens    = tool_config.stop_tokens          # list[str]

    # ---------- model loading ---------- #
    def load_model(self):
        print(f"[LOAD] VLLM model: {self.model_config.model_path}")
        self.model = LLM(
            model=self.model_config.model_path,
            tensor_parallel_size=self.model_config.tensor_parallel_size,
            gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            max_model_len=self.model_config.max_model_len
        )
        self.tokenizer = self.model.get_tokenizer()
        return self.model, self.tokenizer

    # ---------- prompt assembling ---------- #
    @staticmethod
    def format_system_user_prompt(system_prompt: List[str], user_msg: str) -> str:
        assembled_system_prompt = '\n'.join(system_prompt)
        return (
            f"<|im_start|>system\n{assembled_system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # ---------- _postprocess_responses() ---------- #
    def _postprocess_response(
        self, resp: str, step_idx: int
    ) -> Tuple[str, bool]:
        """
        return type: (cleaned_resp, do_action)
        - do_action=True when the model need to call tool
        - do_action=False when the model does not need to call tool
        """
        resp = resp.strip(" \n")
        
        # case: if the model has reached the minimum required number of tool-calling actions
        if step_idx >= self.tool_config.min_action_num:
            # check if the model produced the tool-calling request token
            for tok in self.stop_tokens:
                if resp.endswith(tok):
                    return resp, True          # the model wants to call tool
            return resp, False                 # otherwise the model does not want to call tool, halt
        else:
            return resp, True                  # if the current step is less than the minimum required number of tool-calling actions, force one tool call

    # ---------- main inference loop ---------- #
    def generate_with_tools(
        self,
        prompt: str,
        max_total_tokens: int = 4096,
        max_chunk_tokens: int = 1024,
        debug: bool = False,
        verbose: bool = True
    ) -> Dict[str, str]:

        context      = prompt
        traj_id      = str(uuid.uuid4())
        tokens_used  = len(self.tokenizer.encode(prompt)) if not debug else 0
        final_answer = ""

        for step in range(self.tool_config.max_turns):
            if verbose:
                print(f"\n[STEP {step+1}/{self.tool_config.max_turns}] tokens={tokens_used}")

            # generate text chunks
            if not debug:
                if verbose:
                    print(f"  ↳ Context: {context}")
                params = SamplingParams(
                    temperature=self.model_config.temperature,
                    top_p=self.model_config.top_p,
                    max_tokens=max_chunk_tokens,
                    stop=self.stop_tokens,
                    skip_special_tokens=False,
                )
                with torch.no_grad():
                    out = self.model.generate([context], params)
                raw_resp = out[0].outputs[0].text.strip()
            else:
                raw_resp = "```python\nprint(1+1)\n``````output"

            if verbose:
                print(f"  ↳ LLM Response: {raw_resp}")
            
            # according to the current step and LLM response, determine whether to call tool or not
            cleaned, need_tool = self._postprocess_response(raw_resp, step)

            if not need_tool:
                # LLM does not request for the tool call, extract the code block from the last response block
                if verbose:
                    print(f" ↳ LLM does not need tool, returning final answer.")
                context += cleaned
                # extract the last python code block from the last response
                code, found = extract_python_tags(context)
                if found:
                    if verbose:
                        print(f"  ↳ Extracted Code block: {code}")
                    final_answer = code
                else:
                    if verbose:
                        print(f"  ↳ No code block found, returning error.")
                    final_answer = "No code block is found."
                break
            
            
            # LLM request for a tool call, extract the code block from the last response block and send to tool server
            if verbose:
                print(f"  ↳ LLM indicated tool use, extracting code block.")
                print(f"  ↳ Raw Code block: {cleaned}")
            code, found = extract_python_tags(cleaned)
            
            if not found:
                # LLM indicate a tool call but no code block found, extract the last code block
                if verbose:
                    print(f"  ↳ No code block found in the last response block.")
                context += cleaned
                code, found = extract_python_tags(context)
                if not found:
                    if verbose:
                        print(f"  ↳ No code block found, returning error.")
                    final_answer = "No code block is found."
                else:
                    final_answer = code
                break
            
            # found python code block, send it to the tool server    
            print(f"  ↳ Successfully extracted Code block: {code}")
            tool_ret = call_tool_server(
                self.tool_config.tool_server_url,
                traj_id,
                python_code=code,
                finish=False,
            )
            
            # retrieve tool server response
            observation = tool_ret["observation"]
            done_flag   = tool_ret["done"]

            # concatenate the tool server response with the context
            context += cleaned + observation
            # count tokens
            if not debug:
                tokens_used += len(self.tokenizer.encode(cleaned + observation))
            else:
                tokens_used += 1
            if verbose:
                print(f"  ↳ tool‑obs(valid={tool_ret['valid']}): {observation[:120]}")

            # extra ending conditions
            if done_flag or tokens_used >= max_total_tokens:
                final_answer = code
                break
        
        # failsafe: if the model does not actively stops, need to extract the last code block as its final response
        if final_answer == "":
            if verbose:
                print(f"  ↳ No final answer, extracting the last code block.")
            code, found = extract_python_tags(context)
            if found:
                if verbose:
                    print(f"  ↳ Extracted Code block as the final response: {code}")
                final_answer = code
            else:
                if verbose:
                    print(f"  ↳ No code block found, returning error.")
                final_answer = "No code block is found."
        
        return {"full_response": context, "final_response": final_answer}

    # OpenAI-compliant response generation
    def generate_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        system_prompt = [
            'Answer the given coding question. You must conduct reasoning about the problem and then provide the final program as answer.',
            'During the thinking process, you can write test cases or test your current solutions using a testing tool. if you want to test any python code, writing it inside ```python and ``` tags following with "```output".',
            'The code between "```python" and "``````output" will then be executed, and the terminal output (standard output and standard error) will be provided to you.',
            'Each program between ```python and ``` tags are independent program. You can test Python codes as many times as you want. ',
            'If you find no further code execution needed, you can then give your final solution in a markdown code block like this:',
            '```python\nyour code here\n``` without appending anything.',
            'The final program will be evaluated against the hidden test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty.'
        ]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
        if not user_msg:
            raise ValueError("No user message provided.")

        prompt  = self.format_system_user_prompt(system_prompt, user_msg)
        result  = self.generate_with_tools(prompt)

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_config.model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["final_response"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
