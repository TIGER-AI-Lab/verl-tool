import io
import base64
import numpy as np
from typing import Union, Optional
from PIL import Image
from verl.utils.dataset.vision_utils import process_image, process_video

def encode_image(img: Image.Image) -> str:
    if isinstance(img, Image.Image):
        buffered = io.BytesIO()
        # convert the image to RGB if it is not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    else:
        raise ValueError(f"Unsupported image type: {type(img)}. Expected str or PIL Image, got {type(img)}.")

def decode_image(img_str):
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data))
    return img

def decode_image_url(img_url: str) -> Image.Image:
    return process_image({"image": img_url})

def encode_image_url(img: Union[str, dict, Image.Image]) -> str:
    if isinstance(img, str):
        img = process_image({"image": img})
    else:
        img = process_image(img)
    encoded_img = encode_image(img)
    return f"data:image/jpeg;base64,{encoded_img}"  # Assume img is a base64 string or file path

def encode_video_url(
    video: Union[list, str, dict, np.ndarray], 
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None
) -> str:
    if isinstance(video, list):
        if all(isinstance(frame, np.ndarray) for frame in video) or \
        isinstance(video, np.ndarray) and video.ndim == 4:  # Assuming video is a list of numpy arrays or a 4D numpy array
            # load from numpy arrays
            frames = [Image.fromarray(frame) for frame in video]
        else:
            frames = [process_image({"image": frame}) for frame in video]
    else:
        if isinstance(video, str):
            video = {"video": video}
        else:
            frames = process_video(video, nframes=nframes, fps=fps, fps_min_frames=fps_min_frames, fps_max_frames=fps_max_frames)
    encoded_frames = [encode_image(frame) for frame in frames]
    return f"data:video/jpeg;base64,{','.join(encoded_frames)}"  # Assume video is a list of processed images

def decode_video_url(video_url: str) -> list:
    if video_url.startswith("data:video/jpeg;base64,"):
        video_data = video_url.split(",")[1]
        frames = [process_image("data:image/jpeg;base64," + frame) for frame in video_data.split(",")]
        return frames
    else:
        return process_video({"video": video_url})