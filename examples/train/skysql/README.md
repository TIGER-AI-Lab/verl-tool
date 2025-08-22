# skysql Training Guide

This guide covers data preprocessing and training setup for the skysql model using the verl-tool framework.

## Prerequisites

⚠️ **Important**: All operations below assume you are in the parent directory of the `verl-tool` repository.

```bash
uv pip install -e .[sql_tool]
```

## 📋 Overview

The skysql training requires two main datasets:
- **Omni-SQL dataset** (~50GB): Contains evaluation data for Spider-series and BIRD datasets, plus the SynSQL-2.5M training dataset
- **Preprocessed training/evaluation datasets**: Ready-to-use parquet files for training

## 🗄️ Dataset Setup

### Step 1: Download Omni-SQL Dataset

**Option A: Manual Download**
1. Download from: https://huggingface.co/datasets/seeklhy/OmniSQL-datasets
2. Decompress to `verl-tool/data/`

**Option B: Automated Script**
```bash
cd <verl-tool-parent-path>
bash ./examples/data_preprocess/skysql/download_skysql.sh
```

### Step 2: Verify Data Structure

After downloading, your folder structure should look like:

```bash
📁 verl-tool/data/synsql/data
├── 📁 bird/
├── 📁 spider/
├── 📁 Spider-DK/
├── 📁 spider-realistic/
├── 📁 Spider-Syn/
├── 📁 ... (other dataset subfolders)
├── 📄 dev_bird.json
├── 📄 dev_spider_dk.json
└── 📄 ... (other evaluation json files)
```

## 🚀 Quick Start: Using Preprocessed Datasets

### Download Ready-to-Use Datasets

Clone the preprocessed datasets directly:

```bash
huggingface-cli download --local-dir "data/skysql" --repo-type dataset VerlTool/SkyRL-SQL-Reproduction train.parquet test.parquet
```

### Configure Training Script

Update the dataset paths in `sql_experiment/verl-tool/examples/train/skysql/train.sh`:

```bash
# Set these paths according to your downloaded data
train_data=$(pwd)/data/${dataset_name}/train.parquet
val_data=$(pwd)/data/${dataset_name}/test.parquet
model_name=/path/to/your/model/weights  # e.g., Qwen2.5-Coder-7B-Instruct
```

### Optional: Enable Weights & Biases Logging

If you want to track training metrics:

```bash
export WANDB_API_KEY="<your_wandb_api_key>"
```

## 🔧 Manual Dataset Preprocessing

If you need to reprocess the datasets from scratch:

### Training Dataset Preprocessing

The training dataset is converted from SynSQL-2.5M format to verl-tool format.

**Script Location**: `verl-tool/examples/data_preprocess/skysql/prepare_train.py`

**Configuration**:
- Modify `DEFAULT_DATABASE_PATH` to point to the SynSQL-2.5M dataset's `databases` subfolder
- If you downloaded Omni-SQL correctly, SynSQL-2.5M should be included as a subfolder

### Evaluation Dataset Preprocessing

The evaluation dataset merges 6 subsets from the Omni-SQL dataset.

**Script Location**: `verl-tool/examples/data_preprocess/skysql/prepare_test.py`

**Before Running**:
1. Check the demo instructions at the bottom of the script (commented out)
2. **Critical**: Update all database paths for `DEV_PROMPT` and `DEV_SCHEMA` (lines 25-41 in the script)

## 📝 Important Notes

- Ensure you have sufficient disk space (~50GB+ for datasets)
- The SynSQL-2.5M dataset is embedded within the Omni-SQL dataset
- All paths in preprocessing scripts must be absolute paths to avoid issues
- Review and update file paths before running any preprocessing scripts

## 🏃‍♂️ Next Steps

After completing the data setup:
1. Verify all paths in your training script
2. Run the training script: `bash sql_experiment/verl-tool/examples/train/skysql/train.sh`
3. Monitor training progress (via W&B if configured)

## 🆘 Troubleshooting

- **Path issues**: Ensure all paths are absolute and point to existing files/directories
- **Missing data**: Verify the Omni-SQL download completed successfully
- **Memory issues**: Ensure sufficient RAM for processing large datasets
- **Permission errors**: Check file permissions and disk space 

(**Note: please set `enable_prefix_caching=False` when running!!!**)

Reference:

- https://github.com/RUCKBReasoning/OmniSQL/tree/main/train_and_evaluate
- https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/eval_open_source_models.py
- https://skyrl.readthedocs.io/en/latest/examples/multi_turn_text2sql.html
- https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train
- https://novasky-ai.notion.site/skyrl-sql