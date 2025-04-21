import torch
import re
import time
import uuid
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional, Union
from config import ModelConfig, ToolConfig
from utils import extract_python_tags, call_tool_server

class ModelService:
    """verl-tool model inference service"""
    
    def __init__(self, model_config: ModelConfig, tool_config: ToolConfig):
        """initialize model service"""
        self.model_config = model_config
        self.tool_config = tool_config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """load the model using VLLM backend"""
        print(f"Loading Model using VLLM: {self.model_config.model_path}...")
        self.model = LLM(
            model=self.model_config.model_path,
            tensor_parallel_size=self.model_config.tensor_parallel_size,
            gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            max_model_len=self.model_config.max_model_len
        )
        
        print(f"Loading Tokenizer using VLLM: {self.model_config.model_path}...")
        self.tokenizer = self.model.get_tokenizer()
        
        
        print("Model loaded successfully.")
        return self.model, self.tokenizer
    
    def format_system_user_prompt(self, system_prompt: str, user_message: str) -> str:
        """Format the system and user messages into a single prompt string"""
        
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    def generate_with_tools(self, prompt, max_total_tokens=4096, max_chunk_tokens=1024, debug=False):
        """
        Generate text with tool calls in a multi-turn loop.
        
        Args:
            prompt: Initial prompt for generation
            max_total_tokens: Maximum total tokens to generate
            max_chunk_tokens: Maximum tokens to generate in each chunk
            
        Returns:
            Generated text with tool interactions
        """
        # initialize the context as the passed prompt (system prompt + question)
        context = prompt
        action_step = 0
        
        if not debug:
            total_tokens_used = len(self.tokenizer.encode(context))
        else:
            total_tokens_used = 0
        
        # initialize the response payload
        return_payload = {
            "full_response": "",
            "final_response": ""   
        }
        
        # keep trying to generate the response until reached the token limit or the tool-calling limit
        while total_tokens_used < max_total_tokens and action_step < self.tool_config.max_turns:
            # for each new generation, mark it as the next action step
            action_step += 1
            print(f"Action step {action_step}")
            print(context)
            
            # for testing: in test mode avoid calling LLM
            if not debug:
                # 1. generate the text chunk using the model
                # fix for qwen 
                # REF: https://github.com/vllm-project/vllm/issues/2947#issuecomment-1959569041
                sampling_params = SamplingParams(
                    temperature=0,
                    top_p=0.9,
                    max_tokens=max_chunk_tokens,
                    skip_special_tokens=False, 
                    stop=["<|im_end|>", "<|endoftext|>"]
                )
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        [context],
                        sampling_params
                    )
                
                response = outputs[0].outputs[0].text
                print(f"[DEBUG] Generated text block: {response}")
            else:
                response = """
                this is a dummy response without tool calling.
                """
                
            # next: check if there is tool call, if not then break out the generation loop and try to extract the most recent code block in the entire context
            # if there's tool call in the current block, then extract the code model want to execute, run it, get response and append it to the generated text
            # otherwise stop generating and break out the generation loop as the model does not want to call tool anymore
            
            # set the flag to False, strip the response text chunk
            has_action = False
            response = response.strip(' \n')

            # define the tool-calling token (stop token) 
            stop_token = "```output"
            
            # check if the last part of the response text chunk include (part of) the stop token
            for i in range(len(stop_token), 0, -1):
                if response.endswith(stop_token[:i]):
                    # if the last part of the response text chunk is the stop token
                    # then we need to remove the stop token from the response text chunk
                    trimmed_response = response[:-len(stop_token[:i])] 
                    has_action = True
                    break
            # then need to extract the last python code block
            if has_action:
                extracted_python_code, is_code_found = extract_python_tags(trimmed_response)        
                if is_code_found:
                    print(f"[DEBUG] the extracted code is valid, will call too server")
                    # interact with the tool server and retrieve observation result
                    # create a trajectory id for interaction
                    traj_id = str(uuid.uuid4())
                    tool_response = call_tool_server(
                        self.tool_config.tool_server_url,
                        traj_id, 
                        extracted_python_code, 
                        has_action, 
                    )
                    next_obs = tool_response["observation"]
                    valid_action = tool_response["valid"]
                    
                    print(f"\n[DEBUG] extracted code: {extracted_python_code}\n")
                    print(f"\n[DEBUG] next_obs: {next_obs}, valid_action: {valid_action}\n")

                    if valid_action:
                        # this round is finished, append the code execution result to the trimmed and cleaned response
                        # then update the context
                        response = trimmed_response + next_obs
                        context += response
                        
                        if not debug:
                            total_tokens_used += len(self.tokenizer.encode(response))
                        else:
                            total_tokens_used += 100
            else:
                # the model has not utilized tool calling in this iteration but the number of generation turns has reached the limit
                if action_step > self.tool_config.min_turns:
                    
                    print(f"[DEBUG] current action step: {action_step}, min_turns: {self.tool_config.min_turns}")
                    # no tool-calling token found, the model does not want to call tools anymore
                    # then we try to extract the most recent python code and return the result
                    most_recent_python_code, is_extract_success = extract_python_tags(response)
                    
                    if not is_extract_success:
                        most_recent_python_code = "[ERR] No python code extracted"
                        
                    context += response
                    return_payload = {
                        "full_response": context,
                        "final_response": most_recent_python_code 
                    }                
                    return return_payload    
            
        
        # failsafe: if the loop ends without returning, then extract the most recent python code from the entire context
        most_recent_python_code, is_extract_success = extract_python_tags(context)
        if not is_extract_success:
            most_recent_python_code = "[ERR] No python code extracted"
        
        return_payload = {
            "full_response": context,
            "final_response": most_recent_python_code 
        }
        return return_payload    
            
        
    # TODO: fix generation prompt
    def generate_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """process API request and generate response"""
        system_prompt = \
"""
Answer the given coding question. You must conduct reasoning about the problem and then provide the final program as answer. 
During the thinking process, you can write test cases or test your current solutions using a testing tool. if you want to test any python code, writing it inside ```python and ``` tags following with "```output". 
The code between "```python" and "``````output" will then be executed, and the terminal output (standard output and standard error) will be provided to you. 
Each program between ```python and ``` tags are independent program. You can test Python codes as many times as you want. 
If you find no further code execution needed, you can then give your final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything. 
The final program will be evaluated against the hidden test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty.
"""
        
        user_message = None
        
        # we keep the system message intact, only need to retrieve the user query
        for message in messages:
            if message["role"] == "user":
                if user_message is None:  
                    user_message = message["content"]
        
        if not user_message:
            raise ValueError("No user message found in the request.")
        
        # 1. format the system and user messages into a single prompt string
        prompt = self.format_system_user_prompt(system_prompt, user_message)
        
        # 2. utilize the model to generate the response
        result = self.generate_with_tools(prompt)
        
        # TODO: implement token usage computation
        # format the response into OpenAI-compliant format
        return {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_config.model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"{result['final_response']}"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }