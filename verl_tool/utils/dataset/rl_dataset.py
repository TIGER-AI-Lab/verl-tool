import io
import base64
import numpy as np
import regex as re
import time
import datasets
from verl.utils.dataset.rl_dataset import RLHFDataset
from pathlib import Path
from typing import List
from copy import deepcopy
from collections import defaultdict

def encode_image(img_path: str) -> str:
    with open(img_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        encoded_str = encoded_bytes.decode("utf-8")
        return encoded_str
    
def nested_copy(obj):
    """
    Recursively copy nested objects (lists, dicts, etc.) to avoid reference issues.
    """
    if isinstance(obj, dict):
        return {k: nested_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nested_copy(item) for item in obj]
    elif hasattr(obj, 'copy'):
        return obj.copy()
    else:
        return obj
    
class RolloutMessagesMixin:
    """Mixin class to handle rollout messages in reinforcement learning datasets.

    This mixin provides methods to update and manage rollout messages, which are used
    to store the conversation history and interactions during the reinforcement learning process.
    """
    def __init__(self, messages: List[dict]):
        self.messages = messages if messages is not None else []
    
    def update_rollout_messages(self, new_message: dict) -> List[dict]:
        """Update the rollout messages with new messages."""
        messages = self.messages
        role = new_message['role']
        content_list = new_message['content']
        if isinstance(content_list, str):
            content_list = [{"type": "text", "text": content_list}]
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        assert isinstance(content_list, list), f"content_list should be a list, but got {type(content_list)}"
        
        if messages[-1]['role'] != role:
            messages.append({'role': role, 'content': content_list})
        else:
            for content in content_list:
                if isinstance(content, dict) and content.get('type') == 'text' and messages[-1]['content'][-1].get('type') == 'text':
                    messages[-1]['content'][-1]['text'] += content['text']
                else:
                    messages[-1]['content'].append(content)
        return messages

    def tolist(self):
        """Convert the messages to a list format."""
        return self.messages.copy()
    
    def __copy__(self):
        """Create a shallow copy of the RolloutMessagesMixin instance."""
        return RolloutMessagesMixin(nested_copy(self.messages))
class VerlToolRLHFDataset(RLHFDataset):
    """A dataset class for reinforcement learning tasks in verl-tool.

    This class extends the base RLHFDataset class to provide additional functionality
    specific to verl-tool, such as custom data loading and processing methods.
    """
    
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        start = time.time()
        rollout_messages = self._build_rollout_messages(row_dict)
        # print(f'finish getting {item}-th item rollout messages in {time.time() - start} seconds')
        start = time.time()
        result = super().__getitem__(item)
        result['rollout_messages'] = rollout_messages
        # print(f'finish getting {item}-th item in {time.time() - start} seconds')
        return result
    
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = (
                        [process_image(image) for image in doc[image_key]] if image_key in doc else None # changed to get images from doc
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]] if video_key in doc else None # changed to get videos from doc
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe
    
    def _build_rollout_messages(self, example: dict):
        messages = deepcopy(example[self.prompt_key])

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                try:
                    segments = re.split("(<image>|<video>)", content)
                except Exception as e:
                    raise ValueError(f"Error splitting content: {content}") from e
                segments = [item for item in segments if item != ""]
                segment_idx = defaultdict(int)
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image", "image": example[self.image_key][segment_idx[segment]]["image"]})
                        segment_idx[segment] += 1
                    elif segment == "<video>":
                        content_list.append({"type": "video", "video": example[self.video_key][segment_idx[segment]]["video"]})
                        segment_idx[segment] += 1
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        if self.processor is not None:
            # multi-modal inputs
            from verl_tool.llm_agent.vision_utils import encode_image_url, encode_video_url
            for i, message in enumerate(messages):
                if isinstance(message['content'], list):
                    for j in range(len(message['content'])):
                        content = message['content'][j]
                        if content['type'] == 'image':
                            message['content'][j] = {
                                "type": "image_url",
                                "image_url": {
                                    "url": encode_image_url(content['image']),
                                }
                            }
                            assert Path(content['image']).exists(), f"Image file {content['image']} does not exist."
                        elif content['type'] == 'video':
                            message['content'][j] = {
                                "type": "video_url",
                                "video_url": {
                                    "url": encode_video_url(content['video']),
                                }
                            }
                            assert Path(content['video']).exists(), f"Video file {content['video']} does not exist."
                        elif content['type'] == 'text':
                            message['content'][j] = {
                                "type": "text",
                                "text": content['text']
                            }
                        else:
                            raise ValueError(f"Unknown content element type: {content['type']}")
                elif isinstance(message['content'], str):
                    message['content'] = [{"type": "text", "text": message['content']}]
                else:
                    raise ValueError(f"Unknown content type: {type(message['content'])}")
                    
        return RolloutMessagesMixin(messages)