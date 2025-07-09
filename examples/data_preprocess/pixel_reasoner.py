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
"""
Preprocess the GSM8k dataset to parquet format
"""
import fire
import os
import datasets
import zipfile
import cv2
import os
from glob import glob
from pathlib import Path
from huggingface_hub import hf_hub_download

system_prompt = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "zoom_in", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "normalized coordinates for bounding box of the region you want to zoom in. Values should be within [0.0,1.0].", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "select_frames", "description": "Select frames from a video.", "parameters": {"type": "object", "properties": {"target_frames": {"type": "array", "description": "List of frame indices to select from the video (no more than 8 frames in total).", "items": {"type": "integer", "description": "Frame index from 1 to 16."}}}, "required": ["target_frames"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

def images_to_video(image_folder, output_path, fps=24):
    images = sorted(glob(os.path.join(image_folder, "*.jpg")))
    if not images:
        raise ValueError("No .jpg images found in folder.")

    # Read the first image to get size
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

def main(
    dataset_path: str = 'TIGER-Lab/PixelReasoner-RL-Data',
    local_dir: str = 'data/pixel_reasoner',
    seed: int = 42,
):
    local_dir = Path(local_dir)
    local_dir = local_dir / (dataset_path.split('/')[-1].replace('-', '_'))
    local_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = datasets.load_dataset(dataset_path, split='train')

    # 500 examples for testing
    
    dataset = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    # download images and videos
    image_zip_file = hf_hub_download(repo_id=dataset_path, filename='images.zip', repo_type='dataset')
    video_zip_file = hf_hub_download(repo_id=dataset_path, filename='videos.zip', repo_type='dataset')
    # extract the zip files to local_dir/images and local_dir/videos
    image_dir = local_dir / 'images'
    video_dir = local_dir / 'videos'
    image_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    image_extraction_marker = image_dir / 'finish_extracting.txt'
    video_extraction_marker = video_dir / 'finish_extracting.txt'
    if not image_extraction_marker.exists():
        print(f"Extracting images from {image_zip_file} to {image_dir}")
        with zipfile.ZipFile(image_zip_file, 'r') as zip_ref:
            zip_ref.extractall(image_dir)
        with open(image_dir / 'finish_extracting.txt', 'w') as f:
            f.write('Images extracted successfully.')
        print(f"Images extracted successfully to {image_dir}.")
    else:
        print(f"Images already extracted at {image_dir}. Skipping extraction.")
    if not video_extraction_marker.exists():
        print(f"Extracting videos from {video_zip_file} to {video_dir}")
        with zipfile.ZipFile(video_zip_file, 'r') as zip_ref:
            zip_ref.extractall(video_dir)
        with open(video_dir / 'finish_extracting.txt', 'w') as f:
            f.write('Videos extracted successfully.')
        print(f"Videos extracted successfully to {video_dir}.")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            image = example.pop('image')
            is_video = example.pop('is_video')
            answer = example.pop('answer')[0]
            # we use absolute paths for images and videos
            if is_video:
                assert all((video_dir / video).exists() for video in image), f"Some video files do not exist in {video_dir}"
                # mm_content = [{"type": "video", "video": [(video_dir / video).absolute().as_posix() for video in image]}]
                mm_content = [{"type": "image", "image": (video_dir / video).absolute().as_posix()} for video in image]
            else:
                assert (image_dir / image[0]).exists(), f"Image file {image[0]} does not exist in {image_dir}"
                mm_content = [{"type": "image", "image": (image_dir / image[0]).absolute().as_posix()}]

            data = {
                "data_source": dataset_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question_raw},
                            *mm_content,
                        ],
                    }
                ],
                "ability": "visual_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'qid': example.get('qid', f'{split}_{idx}'),
                    'is_video': bool(is_video)
                }
            }
            return data

        return process_fn

    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)
    # split 400 as val
    train_dataset, val_dataset = train_dataset.train_test_split(test_size=400, seed=seed).values()
    
    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(val_dataset)} validation samples")
    print(f"Example of a training sample:")
    print(train_dataset[0])

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))
    print(f"Saved to {len(train_dataset)} training samples to {local_dir}/train.parquet")
    print(f"Saved to {len(val_dataset)} validation samples to {local_dir}/val.parquet")

if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/data_preprocess/pixel_reasoner.py --dataset_path=TIGER-Lab/PixelReasoner-RL-Data --local_dir=data/pixel_reasoner
"""