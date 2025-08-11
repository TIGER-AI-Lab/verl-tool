# AceCoder Training Guide

## Requirements
```python
uv pip install .[acecoder]
```
## Dataset Preparation

```bash

# AceCoder V1 dataset
local_dir = "data/acecoderv1"
dataset_path="TIGER-Lab/AceCode-87K" # V1 dataset

# AceCoder V2 dataset
local_dir = "data/acecoderv2"
dataset_path="TIGER-Lab/AceCode-V2-122K" # V2 dataset
python examples/data_preprocess/acecoder.py --dataset_path ${dataset_path} --local_dir $local_dir --propmt_type complex

```

## Training example
- AceCoder training without tool
```bash
bash examples/train/acecoder/train_no_tool.sh
```

- AceCoder training with tool
```bash
bash examples/train/acecoder/train_with_tool.sh
```