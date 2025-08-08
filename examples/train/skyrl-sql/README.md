## Data Preprocessing

1. Download the processed SkyRL-SQL dataset:

https://huggingface.co/datasets/VerlTool/SkyRL-SQL-Reproduction/tree/main/data

In the dataset, each entry expect a corresponding SQL database file. To download it, run:

```bash
cd <verl-tool-parent-path>
bash ./examples/data_preprocess/skyrl-sql.sh
```

2. Set correct dataset path and model weight path:

In `sql_experiment/verl-tool/examples/train/skyrl-sql/train.sh`, set the path to `train.parquet` and `test.parquet` accordingly:

```bash
train_data=$(pwd)/data/${dataset_name}/train.parquet
val_data=$(pwd)/data/${dataset_name}/test.parquet   # dummy val data
model_name=/map-vepfs/yi/model_weights/Qwen2.5-Coder-7B-Instruct # should use coder model
```

If you want to record the training stats to Weight&Bias, set your wandb API key as environment variable:

```bash
export WANDB_API_KEY="<your_wandb_api_key>"
```