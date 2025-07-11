from .base import BaseTool, register_tool
import shutil
import regex as re
import requests
import json
from typing import Tuple, Union
import os

import base64
import io
from PIL import Image
from pathlib import Path
def crop( str_image, bbox_2d,padding=(0.1,0.1)):
    """
    Crop the image based on the bounding box coordinates.
    """
    if isinstance(str_image,list):
        str_image = str_image[0]
    if isinstance(str_image, Path) and str_image.exists() or \
        isinstance(str_image, str) and os.path.exists(str_image):
        # If the image is a file path, open it directly
        image = Image.open(str_image)
    else:
        image = decode_image_url(str_image)
    img_x, img_y = image.size
    padding_tr = (600.0/img_x,600.0/img_y)
    padding = (min(padding[0],padding_tr[0]),min(padding[1],padding_tr[1]))

    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0])-padding[0], float(bbox_2d[1])-padding[1], float(bbox_2d[2])+padding[0], float(bbox_2d[3])+padding[1])
    else:
        normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding[0], float(bbox_2d[1])/img_y-padding[1], float(bbox_2d[2])/img_x+padding[0], float(bbox_2d[3])/img_y+padding[1])
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 =min(max(0, normalized_x1), 1)
    normalized_y1 =min(max(0, normalized_y1), 1)
    normalized_x2 =min(max(0, normalized_x2), 1)
    normalized_y2 =min(max(0, normalized_y2), 1)
    cropped_img = image.crop((int(normalized_x1*img_x), int(normalized_y1*img_y), int(normalized_x2*img_x), int(normalized_y2*img_y)))
    return cropped_img


#only when doing cropping the image is converted to pil
def encode_image(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Create JSON with the encoded image
def decode_image(img_str):
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data))
    return img

def encode_image_url(img: Image.Image) -> str:
    encoded_img = encode_image(img)
    return f"data:image/jpeg;base64,{encoded_img}"  # Assume img is a base64 string or file path

# Create JSON with the encoded image
def decode_image_url(img_str):
    if img_str.startswith("data:image/jpeg;base64,"):
        img_str = img_str.split("data:image/jpeg;base64,")[1]
    return decode_image(img_str)

@register_tool
class PixelReaonerTool(BaseTool):
    tool_type = "pixel_reasoner"

    stop_tokens = [ "</tool_call>"]

    def get_usage_inst(self):
        return ""
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing bbox_2d & target_image
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        # Try to find Python code in various formats
        try:
            call = json.loads(action.split('<tool_call>')[1].split('</tool_call>')[0])
            name = call.get('name', '')
            if name not in ['zoom_in', 'select_frames']:
                return "", False
        except:
            return "", False
        
        return call, True

    def load_env(self, trajectory_id):
        """
        Load the environment for the given trajectory_id
        """
        env = self.env_cache.get(trajectory_id)
        if env == None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {
                    "turns": 0,
                },
                "previous_obs": [],
                "images": None,
                "temporary_images": [],
                "temporary_image_folder": Path(f"tmp/crop_images/{trajectory_id}"),
            }
            env['temporary_image_folder'].mkdir(parents=True, exist_ok=True)
        return env
    
    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        """
        Update the environment for the given trajectory_id
        """
        # save image
        if isinstance(observation, dict) and 'image' in observation:
            if isinstance(observation['image'], str):
                env['images'].append(self.save_image_to_env(trajectory_id, observation['image']))
            elif isinstance(observation['image'], list):
                env['images'].extend([self.save_image_to_env(trajectory_id, img) for img in observation['image']])
        env["metadata"]["turns"] += 1
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })
    
    def delete_env(self, trajectory_id):
        """
        Delete the environment for the given trajectory_id
        """
        # import json
        if trajectory_id in self.env_cache:
            if self.env_cache[trajectory_id]['temporary_image_folder'].exists():
                shutil.rmtree(self.env_cache[trajectory_id]['temporary_image_folder'])
            del self.env_cache[trajectory_id]

    def save_image_to_env(self, trajectory_id, image: Union[Image.Image,str]) -> str:
        """
        Save the image to the environment for the given trajectory_id
        """
        env = self.load_env(trajectory_id)
        temporary_image_folder = env['temporary_image_folder']
        image_path = temporary_image_folder / f"image_{len(env['temporary_images'])}.jpg"
        if isinstance(image, str):
            # If the image is a base64 string, decode it
            image = decode_image_url(image)
        elif isinstance(image, Image.Image):
            # If the image is already a PIL Image, no need to decode
            pass
        else:
            raise ValueError("Image must be a PIL Image or a base64 encoded string.")
        image.save(image_path)
        env['temporary_images'].append(image_path)
        self.save_env(trajectory_id, env)
        return str(image_path.absolute())

    def conduct_zoom_in_action(self, parameters, env):
        """
        Execute the zoom-in action based on the parsed parameters.
        
        Args:
            parameters: Parsed action parameters containing bbox_2d and target_image
            env: Current environment state
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        missing_parameters = []
        if 'bbox_2d' not in parameters:
            missing_parameters.append('bbox_2d')
        if 'target_image' not in parameters:
            missing_parameters.append('target_image')
        if missing_parameters:
            observation = f"Missing parameters: {', '.join(missing_parameters)}"
        elif not isinstance(parameters['bbox_2d'], list) or len(parameters['bbox_2d']) != 4:
            observation = "Invalid bbox_2d format. It should be a list of four numbers."
        elif not isinstance(parameters['target_image'], int) or parameters['target_image'] <= 0 or parameters['target_image'] > len(env['images']):
            observation = f"Invalid target_image index. It should be an integer between 1 and the number of previous images ({len(env['images'])})."
        else:
            try:
                previous_images = env['images']
                img_to_crop = previous_images[parameters['target_image']-1]
                cropped_img = crop(img_to_crop, parameters['bbox_2d'])
                encoded_cropped_img = encode_image_url(cropped_img)
                observation = {
                    'obs': "Here is the cropped image.\n<image>",
                    'image': encoded_cropped_img,
                }
            except Exception as e:
                print(parameters)
                with open('test.json', 'w') as f:
                    json.dump(parameters, f, indent=4)
                observation = ""
                raise e
        return observation
    
    def conduct_select_frames_action(self, parameters, env):
        missing_parameters = []
        if 'target_frames' not in parameters:
            missing_parameters.append('target_frames')
        if missing_parameters:
            observation = f"Missing parameters: {', '.join(missing_parameters)}"
        elif not isinstance(parameters['target_frames'], list):
            observation = "Invalid target_frames format. It should be a list of integers."
        elif not all(isinstance(frame, int) and 1 <= frame <= len(env['images']) for frame in parameters['target_frames']):
            observation = f"Invalid target_frames indices. Each index should be an integer between 1 and the number of previous images ({len(env['images'])})."
        else:
            target_frames = [env['images'][frame - 1] for frame in parameters['target_frames']]
            observation = {
                'obs': "Here are the selected frames.\n"+"<image>"*len(target_frames),
                'images': [encode_image_url(crop(img, (0, 0, 1, 1))) for img in target_frames],
            }
        return observation

    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed action
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """

        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        if env['images'] is None:
            env['images'] = [Path(x) for x in extra_field["images"]]
        
        if not is_valid:
            observation = ""
            done = False
            valid = False
        else:
            if parsed_action['name'] == 'zoom_in':
                observation = self.conduct_zoom_in_action(parsed_action['arguments'], env)
            elif parsed_action['name'] == 'select_frames':
                observation = self.conduct_select_frames_action(parsed_action['arguments'], env)
            else:
                observation = "Unknown action name."
            # warp with <tool_response> and </tool_response>
            if isinstance(observation, dict):
                observation['obs'] = f"\n<tool_response>{observation['obs']}</tool_response>"
            elif isinstance(observation, str):
                observation = f"\n<tool_response>{observation}</tool_response>"
            else:
                raise ValueError("Observation must be a string or a dictionary.")
            done = False
            valid = True

        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    