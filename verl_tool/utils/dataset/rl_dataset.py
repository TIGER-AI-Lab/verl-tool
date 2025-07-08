import io
import base64
import numpy as np
import regex as re
from verl.utils.dataset.rl_dataset import RLHFDataset
from pathlib import Path
from typing import List

def encode_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

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
            messages[-1]['content'].extend(content_list)
        return messages

    def tolist(self):
        """Convert the messages to a list format."""
        return self.messages

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
        rollout_messages = self._build_rollout_messages(row_dict)
        result = super().__getitem__(item)
        result['rollout_messages'] = rollout_messages
        return result
    
    def _build_rollout_messages(self, example: dict):
        messages = example[self.prompt_key]

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        for i, message in enumerate(messages):
            if isinstance(message['content'], list):
                for j in range(len(message['content'])):
                    content = message['content'][j]
                    if content['type'] == 'image':
                        message['content'][j] = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(content['image'])}"
                            }
                        }
                    elif content['type'] == 'video':
                        message['content'][j] = {
                            "type": "video_url",
                            "video_url": {
                                "url": content['video'],
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