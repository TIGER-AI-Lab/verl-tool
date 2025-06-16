from .base import BaseTool, register_tool
import regex as re
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
import os
def crop( str_image, bbox_2d,padding=(0.1,0.1)):
    """
    Crop the image based on the bounding box coordinates.
    """
    if isinstance(str_image,list):
        str_image = str_image[0]
    image = decode_image(str_image)
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
    str_cropped_img = encode_image(cropped_img)

    # assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"


    return str_cropped_img  

import base64
import io
from PIL import Image

#only when doing cropping the image is converted to pil
def encode_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Create JSON with the encoded image
def decode_image(img_str):
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data))
    return img

@register_tool
class CropImageTool(BaseTool):
    tool_type = "crop_image"

    stop_tokens = [ "</tool_call>"]
    done_without_error = False

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
        except:
            return "", False
        
        return call, True
    
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
        if not env['previous_obs'] :
            env["previous_obs"].append({
            "action": None,
            "is_valid": None,
            "observation": {'image':extra_field ['image']},
            "extra_field": extra_field,
            
        })
        

        
        if not is_valid:

            observation = ""
            execution_result = ""
            done = False
            valid = False
        else:
            has_error = False
            try:
                img_to_crop = env['previous_obs'][parsed_action['arguments']['target_image']-1]['observation']['image']
                cropped_img = crop(img_to_crop, parsed_action['arguments']['bbox_2d'])
      
                observation = {
                    'text': "Here is the cropped image.<|vision_start|><|image_pad|><|vision_end|>",
                    'image': cropped_img
                }
            except:
                observation = ""
                has_error = True


            if self.done_without_error:
                if has_error:
                    done = False
                else:
                    done = True
            else: 
                done = False
            valid = True
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field,observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    