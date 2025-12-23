import json
import fire
import datasets
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

def plot_frequency_distribution(problem_acc_map, save_path='problem_accuracy_distribution.png'):
    accuracies = []
    for problem, stats in problem_acc_map.items():
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total > 0 else 0
        accuracies.append(accuracy)
    
    plt.hist(accuracies, bins=20, range=(0, 1), edgecolor='black')
    plt.title('Distribution of Problem Accuracies')
    plt.xlabel('Accuracy')
    plt.ylabel('Number of Problems')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(save_path)
    print("Saved histogram to problem_accuracy_distribution.png")

def main(
    records_folder: str = "verl_step_records/acetoolreason-fsdp-agent-models_atr_v2.3_math_rl_with_tool_stage2_32k-grpo-n8-b128-t1.0-lr1e-6-v2-math-rl-with-tool-stage3-40k",
    train_dataset_path: str = "data/math_rl/train_tool_stage_3.parquet",
    is_tool_used: bool = True,
):
    records_folder = Path(records_folder)
    dataset = datasets.load_dataset("parquet", data_files={"train": train_dataset_path})["train"]
    all_records = []
    record_files = list(records_folder.glob("*.jsonl"))
    for record_file in tqdm(record_files, desc="Loading record files"):
        with record_file.open("r") as f:
            for line in f:
                all_records.append(json.loads(line))
    if is_tool_used:
        all_records = [record for record in all_records if "# Tools" in record['input']]
    else:
        print("Filtering records without tool usage...")
        print(f"Total records before filtering: {len(all_records)}")
        all_records = [record for record in all_records if "# Tools" not in record['input']]
        print(f"Total records after filtering: {len(all_records)}")
    problem_acc_map = {}
    for record in all_records:
        _input = record["input"]
        acc = record["accuracy"]
        gts = record["gts"]
        problem_start_idx = _input.index("<|im_start|>user\n")
        assert problem_start_idx != -1
        problem_start_idx += len("<|im_start|>user\n")
        problem_end_idx = _input.index("<|im_end|>\n<|im_start|>assistant\n")
        assert problem_end_idx != -1
        problem = _input[problem_start_idx:problem_end_idx].strip()
        if problem not in problem_acc_map:
            problem_acc_map[problem] = {"correct": 0, "total": 0, "gts": gts}
        assert problem_acc_map[problem]["gts"] == gts, f"GTs mismatch for problem: {problem_acc_map[problem]['gts']} vs {gts}"
        problem_acc_map[problem]["total"] += 1
        if acc:
            problem_acc_map[problem]["correct"] += 1

    fig_save_path = Path(train_dataset_path)
    fig_save_path = fig_save_path.parent / "plots" / (fig_save_path.stem + "_problem_accuracy_distribution.png")
    fig_save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frequency_distribution(problem_acc_map, save_path=fig_save_path)
    
    original_dataset_problems = set()
    
    for item in dataset:
        problem = item['prompt'][1]['content'].strip()
        original_dataset_problems.add(problem)
    print(f"Total unique problems in records: {len(problem_acc_map)}")
    print(f"Total unique problems in original dataset: {len(original_dataset_problems)}")
    print("Example problem from records:")
    print(next(iter(problem_acc_map)))
    print("Example problem from original dataset:")
    print(next(iter(original_dataset_problems)))
    # check overlap
    overlap_count = sum(1 for problem in problem_acc_map if problem in original_dataset_problems)
    print(f"Number of problems in records that are also in the original dataset: {overlap_count}")
    overlap_count = sum(1 for problem in original_dataset_problems if problem in problem_acc_map)
    print(f"Number of problems in original dataset that are also in the records: {overlap_count}")
    
    # remove those problems in original dataset that are too easy
    # acc >= 0.9
    easy_problems = {problem for problem, stats in problem_acc_map.items() if (stats["correct"] / stats["total"] >= 0.9)}
    print(f"Number of easy problems (accuracy >= 0.9): {len(easy_problems)}")
    
    def filter_easy_problems(example):
        problem = example['prompt'][1]['content'].strip()
        return problem not in easy_problems
    filtered_dataset = dataset.filter(filter_easy_problems)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    if is_tool_used:
        output_path = Path(train_dataset_path).parent / (Path(train_dataset_path).stem + "_filtered_by_tir.parquet")
    else:
        output_path = Path(train_dataset_path).parent / (Path(train_dataset_path).stem + "_filtered_by_no_tool.parquet")
    filtered_dataset.to_parquet(output_path)
    print(f"Filtered dataset saved to {output_path}")
    
    
if __name__ == "__main__":
    fire.Fire(main)
    
    
"""
# With Tool Stage 1
python scripts/filter_too_easy_prompts.py \
    --records_folder verl_step_records/acetoolreason-fsdp-agent-models_atr_8b_9e-6_v2.3_sft_5epoch-grpo-n8-b128-t1.0-lr1e-6-v2-math-rl-with-tool-stage1-24k \
    --train_dataset_path data/math_rl/train_tool_stage_1.parquet --is_tool_used True

# With Tool Stage 2
python scripts/filter_too_easy_prompts.py \
    --records_folder verl_step_records/acetoolreason-fsdp-agent-models_atr_v2.3_math_rl_with_tool_stage1_24k-grpo-n8-b128-t1.0-lr2e-6-v2-math-rl-with-tool-stage2-32k \
    --train_dataset_path data/math_rl/train_tool_stage_2.parquet --is_tool_used True
    
# With Tool Stage 3
python scripts/filter_too_easy_prompts.py \
    --records_folder verl_step_records/acetoolreason-fsdp-agent-models_atr_v2.3_math_rl_with_tool_stage2_32k-grpo-n8-b128-t1.0-lr1e-6-v2-math-rl-with-tool-stage3-40k \
    --train_dataset_path data/math_rl/train_tool_stage_3.parquet --is_tool_used True
    
# No Tool Stage 1
python scripts/filter_too_easy_prompts.py \
    --records_folder verl_step_records/acetoolreason-fsdp-models_atr_8b_9e-6_v2.3_sft_5epoch-grpo-n8-b128-t1.0-lr1e-6-v2-math-rl-no-tool-stage1-24k \
    --train_dataset_path data/math_rl/train_no_tool_stage_1.parquet --is_tool_used False

# No Tool Stage 2
python scripts/filter_too_easy_prompts.py \
    --records_folder verl_step_records/acetoolreason-fsdp-models_atr_8b_9e-6_v2.3_sft_5epoch-grpo-n8-b128-t1.0-lr1e-6-v2-math-rl-no-tool-stage2-32k \
    --train_dataset_path data/math_rl/train_no_tool_stage_2.parquet --is_tool_used False
    
# No Tool Stage 3
python scripts/filter_too_easy_prompts.py \
    --records_folder verl_step_records/acetoolreason-fsdp-models_atr_v2.3_math_rl_no_tool_stage2_32k-grpo-n8-b128-t1.0-lr1e-6-v2-math-rl-no-tool-stage3-40k \
    --train_dataset_path data/math_rl/train_no_tool_stage_3.parquet --is_tool_used False
    
    
# Code RL mix
python scripts/filter_too_easy_prompts.py \
    --records_folder verl_step_records/acetoolreason-fsdp-agent-models_atr_8b_9e-6_v2.3_sft_5epoch-grpo-n8-b128-t1.0-lr1e-6-code-rl-mix-tool-32k \
    --train_dataset_path data/code_rl/train_no_tool_problems.parquet --is_tool_used False

python scripts/filter_too_easy_prompts.py \
    --records_folder verl_step_records/acetoolreason-fsdp-agent-models_atr_8b_9e-6_v2.3_sft_5epoch-grpo-n8-b128-t1.0-lr1e-6-code-rl-mix-tool-32k \
    --train_dataset_path data/code_rl/train_tool_problems.parquet --is_tool_used True

"""