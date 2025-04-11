# AceCoder Training Guide

## Preparation Steps

### 1. Dataset Preparation
Download and process the dataset using the following command:
```bash
python examples/data_preprocess/acecoder.py \
  --dataset_path CodeDPO/AceCoderV2-mini-processed \
  --local_dir data/acecoder \
  --add_execution_prompt
```

### 2. Initialize Git Submodules:
run the following commands:
```bash
git submodule init
git submodule update
```

### 3. Logging Configuration
You **MUST** set the Weights & Biases (wandb) key:
```bash
export WANDB_API_KEY="<your_key>"
```
Alternatively, modify line 65 in `verl-tool/examples/train/train_acecoder.sh`:
- Change:
  ```
  trainer.logger=['console','wandb']
  ```
- To:
  ```
  trainer.logger=['console']
  ```

## Additional Notes

### Data Truncation Adjustment
Due to the following error:
```
NotImplementedError: sequence_length=xxxx is larger than max_length=2048
```

- A data truncation parameter has been added in `train.sh` to handle sequences exceeding the maximum length: `data.truncation='right'`

## Run Experiment
Start training using:
```bash
bash examples/train/acecoder/train.sh
```

