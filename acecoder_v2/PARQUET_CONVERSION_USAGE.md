# Parquet 转换工具使用说明

## 🎯 功能概述

`convert_to_parquet.py` 是一个基于 `hf_dataset_converter.py` 逻辑的数据转换工具，将 AceCoderV2 的 JSONL 格式数据转换为 Parquet 格式，便于高效存储和加载。

## 🚀 主要特性

- ✅ **智能数据提取**：从 `synthesis_result` 中提取生成的复杂问题和测试用例
- ✅ **数据验证**：自动验证数据完整性和格式正确性
- ✅ **去重合并**：支持按问题内容去重和跨轮次合并
- ✅ **灵活过滤**：支持按轮次、最小测试用例数量等条件过滤
- ✅ **统计信息**：提供详细的数据集统计和分析
- ✅ **元数据保留**：完整保留实验轮次、模型信息等元数据

## 📋 使用方法

### 基本用法

```bash
# 基本转换
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir output_dir

# 指定测试集大小
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir output_dir --test_size 1000

# 显示详细统计信息
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir output_dir --stats
```

### 高级用法

```bash
# 只提取特定轮次的数据
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir output_dir --target_round 3

# 设置最小测试用例数量
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir output_dir --min_tests 5

# 禁用去重（保留所有重复项）
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir output_dir --no_deduplicate

# 自定义指令模板
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir output_dir --instruction_template "请解决以下问题："
```

## 📊 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--jsonl_path` | str | `problems_merged.jsonl` | 输入 JSONL 文件路径 |
| `--local_dir` | str | `data/acecoder` | 输出目录 |
| `--test_size` | int | `500` | 测试集大小 |
| `--random_seed` | int | `69` | 随机种子 |
| `--instruction_template` | str | 默认模板 | 自定义指令模板 |
| `--target_round` | int | `None` | 指定轮次（None 表示所有轮次） |
| `--min_tests` | int | `1` | 最小测试用例数量 |
| `--deduplicate` | bool | `True` | 是否去重 |
| `--stats` | bool | `True` | 是否显示统计信息 |

## 📁 输出结构

```
output_dir/
├── train.parquet          # 训练集
├── test.parquet           # 测试集
└── metadata.json          # 元数据信息
```

## 🔍 数据格式

### Parquet 文件结构

每行包含以下字段：

```python
{
    "data_source": "acecoderv2",
    "prompt": [
        {
            "role": "user", 
            "content": "问题描述 + 指令模板"
        }
    ],
    "ability": "code",
    "reward_model": {
        "style": "rule",
        "ground_truth": ""
    },
    "extra_info": {
        "split": "train/test",
        "index": 0,
        "id": "问题ID",
        "question": "原始问题描述",
        "test_cases": ["测试用例1", "测试用例2", ...],
        "inputs_outputs": None,
        "metadata": {
            "experiment_round": "实验轮次",
            "model_name": "模型名称",
            "step_type": "步骤类型",
            "content_type": "generated/original",
            "has_generated_problem": True/False,
            "num_tests": 测试用例数量,
            "num_programs": 程序数量,
            "eval_stats": {...}  # 评估统计
        }
    }
}
```

## 📈 统计信息

工具会提供以下统计信息：

- **基本统计**：总项目数、训练/测试集大小
- **测试用例统计**：平均/最小/最大测试用例数量
- **轮次分布**：各实验轮次的数据分布
- **模型分布**：各模型的数据分布
- **内容类型**：生成内容 vs 原始内容的比例
- **样本展示**：示例数据展示

## 🎯 使用场景

### 1. 数据预处理
```bash
# 将原始数据转换为训练格式
python convert_to_parquet.py --jsonl_path raw_data.jsonl --local_dir processed_data
```

### 2. 特定轮次分析
```bash
# 只分析第3轮的数据
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir round3_data --target_round 3
```

### 3. 高质量数据筛选
```bash
# 只保留测试用例数量 >= 5 的数据
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir high_quality --min_tests 5
```

### 4. 快速测试
```bash
# 小规模测试
python convert_to_parquet.py --jsonl_path data.jsonl --local_dir test_data --test_size 100
```

## 🔧 与 hf_dataset_converter.py 的关系

- **数据提取逻辑**：完全一致，都从 `synthesis_result` 提取生成的问题和测试用例
- **去重合并**：使用相同的去重和合并算法
- **元数据保留**：保留相同的元数据结构
- **输出格式**：转换为 Parquet 格式，便于高效存储和加载

## ⚡ 性能优势

- **存储效率**：Parquet 格式比 JSON 更紧凑
- **加载速度**：Parquet 支持列式存储，加载更快
- **内存友好**：支持分块读取，适合大数据集
- **兼容性**：与 pandas、Dask 等工具完美兼容

## 🚨 注意事项

1. **数据完整性**：确保输入 JSONL 文件包含 `synthesis_result` 字段
2. **内存使用**：大数据集可能需要较多内存
3. **文件路径**：确保输出目录有写入权限
4. **编码格式**：输入文件应为 UTF-8 编码

## 📝 示例输出

```
✅ Conversion completed!
📊 Training samples: 1200
📊 Test samples: 500
📁 Output directory: data/acecoder/processed
📄 Train file: data/acecoder/processed/train.parquet
📄 Test file: data/acecoder/processed/test.parquet

==================================================
📊 DATASET STATISTICS
==================================================
Total items: 1700
Test cases per problem:
  Average: 12.5
  Min: 3
  Max: 25
Distribution by experiment round:
  round0: 200 items
  round1: 300 items
  round2: 400 items
  round3: 500 items
  round4: 300 items
Content type distribution:
  generated: 1600 items
  original: 100 items
Generated problems: 1600/1700 (94.1%)
==================================================
```

## 🎉 总结

`convert_to_parquet.py` 是一个功能强大的数据转换工具，特别适合将 AceCoderV2 数据集转换为高效的 Parquet 格式。它保持了与 `hf_dataset_converter.py` 完全一致的数据处理逻辑，同时提供了更好的存储和加载性能。
