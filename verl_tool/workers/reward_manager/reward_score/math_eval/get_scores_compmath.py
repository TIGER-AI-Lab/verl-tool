#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import shutil
import sys
from typing import List, Optional
import numpy as np

from .math_eval_utils.grader import math_equal

# NOTE: This file used to reuse the extractor/grader from `get_scores_hmmt.py`.
# To make `get_scores_compmath.py` self-contained, we inline the minimal grader
# logic here (copied from `get_scores_hmmt.py`): answer extraction + cleaning +
# equality checks used by `evaluate_amc23_or_aime24_zeroshot`.

def _supports_color(stream) -> bool:
    try:
        return stream.isatty() and os.environ.get("NO_COLOR") is None
    except Exception:
        return False


def _c(text: str, code: str) -> str:
    # Minimal ANSI coloring (kept optional so logs stay readable when redirected).
    return f"\033[{code}m{text}\033[0m"


def _fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    # Simple ASCII table (no external deps).
    if not headers:
        return
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def fmt_row(r: List[str]) -> str:
        cells = []
        for i, w in enumerate(widths):
            cell = r[i] if i < len(r) else ""
            cells.append(cell.ljust(w))
        return "| " + " | ".join(cells) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep)
    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print(sep)


def _print_dataset_summary(
    dataset: str,
    accs: List[float],
    acc_mean: float,
    acc_std: float,
    inc_mean: Optional[float],
    tok_mean: Optional[float],
) -> None:
    use_color = _supports_color(sys.stdout)
    title = f" {dataset} "
    if use_color:
        title = _c(title, "1")  # bold

    cols = shutil.get_terminal_size(fallback=(100, 20)).columns
    rule = "─" * max(40, min(cols, 140))
    print(rule)
    print(title.center(len(rule), "─"))
    print(rule)

    headers = ["runs", "acc_mean(%)", "acc_std(%)", "incomplete(%)", "avg_tok_len"]
    rows = [[
        str(len(accs)),
        _fmt_pct(acc_mean, digits=2),
        _fmt_pct(acc_std, digits=2),
        _fmt_pct(inc_mean, digits=2) if inc_mean is not None else "n/a",
        f"{tok_mean:.1f}" if tok_mean is not None else "n/a",
    ]]
    _print_table(headers, rows)


def is_completely_wrapped_by_text(input_string):
    pattern = r'^\\text{(.*)}$'
    match = re.match(pattern, input_string)
    if match:
        # input_string is completely wrapped by \text{}
        extracted_content = match.group(1)
        extracted_content = extracted_content.replace("(", "").replace(")", "").replace(",", "")
        return extracted_content
    else:
        return None


def math_answer_cleaning(answer):
    # remove irrelevant text and space to see whether it is exact match
    extracted_content = is_completely_wrapped_by_text(answer)
    answer = extracted_content if extracted_content else answer

    # convert 5,\!460 into 5460; convert 14{,}916 into 14916; convert \$4 into 4; convert 50\\\% into 50
    answer = answer.replace(",\\!", "").replace("{,}", "").replace("\\$", "")
    # convert \dfrac{3}{2} into frac{3}{2}
    answer = answer.replace("dfrac{", "frac{").replace("tfrac{", "frac{")
    # convert 121^\circ into 121
    answer = answer.replace("^\\circ", "")
    answer = answer.replace("^{\\circ}", "")
    # remove \quad
    answer = answer.replace("\\quad", "")
    # convert 558\,\text{nm} into 558
    answer = re.sub(r'\\,\\text\{.*?\}', '', answer)
    # convert 558\text{nm} into 558
    answer = re.sub(r'\\text\{.*?\}', '', answer)
    # convert 2.45e6^{-1} into 2.45e6; "15000^{-2}^{-1}" into "15000"
    answer = re.sub(r'(\s\^\{-\d+\})', '', answer)
    # remove space
    answer = answer.replace(" ", "")
    # remove \n
    answer = answer.replace("\n", "").replace("\\n", "")
    # convert 3.54\times10^{10} into 3.54e10
    answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^{([+-]?\d+)}', r'\1e\2', answer)
    # convert 3.54\times10^10 into 3.54e10
    answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^([+-]?\d+)', r'\1e\2', answer)
    # convert 2^{10} into 2^10
    answer = re.sub(r'(\d+)\^{(\d+)}', r'\1^\2', answer)
    # convert 10^{-5} into 1e-5; 10^{5} into 1e5
    answer = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", answer)
    # remove comma
    answer = answer.replace(",", "")
    # lowercase
    answer = answer.lower()

    # convert 7.04e5\ into 7.04e5
    if answer.endswith("\\"):
        answer = answer[:-1]

    # convert f(x)=ax+b into ax+b; convert z=123 into 123; convert t_r=123 into 123
    func_pattern = r'^[a-zA-Z_]\w*\([a-zA-Z_]\w*\)$'
    if "=" in answer and (re.match(func_pattern, answer.split("=")[0]) or len(answer.split("=")[0]) <= 3):
        answer = answer.split("=", 1)[1]

    return answer


def round_number(answer):
    def _is_float(string):
        try:
            float(string)
            return True
        except Exception:
            return False

    if _is_float(answer) and float(answer) < 1:
        # to consider the case like 5.56e-10 (convert 5.56e-10 into 5.6e-10)
        # still return a string type
        return f"{float(answer):.2g}"

    return answer


def calculate_numbers(input_string):
    try:
        result = eval(input_string)
        return result
    except Exception:
        return None


def is_equal_after_calculation(extracted_answer, gold):
    # convert \frac{3}{2} into 3/2
    gold = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', gold)
    extracted_answer = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', extracted_answer)
    gold_result = calculate_numbers(gold)
    extracted_answer_result = calculate_numbers(extracted_answer)

    if gold_result and gold_result == extracted_answer_result:
        return True
    else:
        return False


def is_int_string(s: str) -> bool:
    try:
        int(s.strip())
        return True
    except (ValueError, AttributeError):
        return False


def score_single(pred_output: str, gold: str) -> bool:
    """
    Score one prediction against one gold answer.
    pred_output: model raw output text (string), e.g. item["output"]
    gold: gold answer (string)
    returns: True if correct else False
    """
    # extraction patterns (same as before)
    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    matches1 = pattern1_re.findall(pred_output)
    matches2 = pattern2_re.findall(pred_output)
    matches3 = pattern3_re.findall(pred_output)
    matches4 = pattern4_re.findall(pred_output)
    matches5 = pattern5_re.findall(pred_output)

    if len(matches1) >= 1:
        extracted_answer = matches1[-1]
    elif len(matches2) >= 1:
        extracted_answer = matches2[-1]
    elif len(matches3) >= 1:
        extracted_answer = matches3[-1]
    elif len(matches4) >= 1:
        extracted_answer = matches4[-1]
    elif len(matches5) >= 1:
        extracted_answer = matches5[-1]
    else:
        extracted_answer = pred_output

    if gold is None:
        return False

    if math_equal(extracted_answer, gold):
        return True
    
    extracted_answer = math_answer_cleaning(extracted_answer)
    gold = math_answer_cleaning(str(gold))
    
    if is_int_string(gold):
        return gold == extracted_answer

    if math_equal(extracted_answer, gold):
        return True
    if round_number(extracted_answer) == round_number(gold):
        return True
    if is_equal_after_calculation(extracted_answer, gold):
        return True

    return False


def evaluate_amc23_or_aime24_zeroshot(input_datapath, test_datapath):
    gold_list = []
    token_lens_list = []

    print("reading from %s" % test_datapath)
    with open(test_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            gold_list.append(str(item["answer"]))

    count_output_none = 0
    count_answer_none = 0
    correct = 0

    print("reading from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            token_lens_list.append(obj.get("token_lens", 0))

            pred_output = obj.get("output", None)
            if pred_output is None:
                count_output_none += 1
                continue

            gold = gold_list[i] if i < len(gold_list) else None
            if gold is None:
                count_answer_none += 1
                continue

            if score_single(pred_output, gold):
                correct += 1
            else:
                pass

            if re.search(r"\\boxed\{", pred_output) is None \
               and re.search(r"\*\*.*?\*\*", pred_output, re.DOTALL) is None \
               and re.search(r"\\\[\n.*?\n\\\]", pred_output, re.DOTALL) is None \
               and re.search(r'is \\\(.*?\\\)', pred_output, re.DOTALL) is None \
               and re.search(r"\\\[\\n.*?\\n\\\]", pred_output, re.DOTALL) is None:
                count_output_none += 1

    acc = correct / len(gold_list) if gold_list else 0.0
    print("count_output_none:", count_output_none)
    print("count_answer_none:", count_answer_none)
    print("accuracy:", acc)

    return acc, count_output_none, np.mean(token_lens_list) if token_lens_list else 0.0


ROOT = "/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/yachen/AceMath/eval_nemotron_v6"
DATASETS = {
    "aime24": {
        "test_path": os.path.join(ROOT, "dataset", "aime24", "test.jsonl"),
        "pred_globs": ["outputs_*/aime24.jsonl"],
        "num_questions": 30,
    },
    "aime25": {
        "test_path": os.path.join(ROOT, "dataset", "aime25", "test.jsonl"),
        "pred_globs": ["outputs_*/aime25.jsonl"],
        "num_questions": 30,
    },
    "hmmt2502": {
        "test_path": os.path.join(ROOT, "dataset", "hmmt2502", "test.jsonl"),
        "pred_globs": ["outputs_*/hmmt2502.jsonl"],
        "num_questions": 30,
    },
    "hmmt2511": {
        "test_path": os.path.join(ROOT, "dataset", "hmmt2511", "test.jsonl"),
        "pred_globs": ["outputs_*/hmmt2511.jsonl"],
        "num_questions": 30,
    },
    "smt25": {
        "test_path": os.path.join(ROOT, "dataset", "smt25", "test.jsonl"),
        "pred_globs": ["outputs_*/smt25.jsonl"],
        "num_questions": 53,
    },
    "cmimc25": {
        "test_path": os.path.join(ROOT, "dataset", "cmimc25", "test.jsonl"),
        "pred_globs": ["outputs_*/cmimc25.jsonl"],
        "num_questions": 40,
    },
}

def gather_pred_paths(model_folder, patterns):
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(os.path.join(model_folder, pat))))
    return paths

def evaluate_dataset(model_folder, dataset):
    cfg = DATASETS[dataset]
    pred_paths = gather_pred_paths(model_folder, cfg["pred_globs"])
    if not pred_paths:
        raise FileNotFoundError(f"No prediction files matching {cfg['pred_globs']} under {model_folder}")

    accs, incomplete, token_lens = [], [], []
    for pred_path in pred_paths:
        acc, count_none, avg_token_len = evaluate_amc23_or_aime24_zeroshot(pred_path, cfg["test_path"])
        accs.append(acc)
        if cfg.get("num_questions"):
            incomplete.append(count_none / cfg["num_questions"])
        token_lens.append(avg_token_len)

    acc_mean = float(np.mean(accs))
    acc_std = float(np.std(accs)) if len(accs) > 1 else 0.0
    inc_mean = float(np.mean(incomplete)) if incomplete else None
    tok_mean = float(np.mean(token_lens)) if token_lens else None
    _print_dataset_summary(
        dataset=dataset,
        accs=accs,
        acc_mean=acc_mean,
        acc_std=acc_std,
        inc_mean=inc_mean,
        tok_mean=tok_mean,
    )

    # # Optional: per-run accuracies in a table (useful when you have multiple outputs_* files).
    # if len(pred_paths) > 1:
    #     per_headers = ["run", "file", "accuracy", "accuracy(%)"]
    #     per_rows: List[List[str]] = []
    #     for i, (p, a) in enumerate(zip(pred_paths, accs), start=1):
    #         per_rows.append([str(i), os.path.basename(p), f"{a:.4f}", _fmt_pct(a, digits=2)])
    #     _print_table(per_headers, per_rows)

    results = {
        "dataset": dataset,
        "accuracy": acc_mean,
        "std": acc_std,
        "incomplete_ratio": inc_mean,
        "avg_token_len": tok_mean,
        "runs": [{"path": p, "accuracy": a} for p, a in zip(pred_paths, accs)],
    }
    out_path = os.path.join(model_folder, f"{dataset}_accuracy_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelfolder", required=True, help="Folder containing outputs_*/<dataset>.jsonl")
    ap.add_argument("--dataset", required=True, choices=DATASETS.keys(), help="Which benchmark to score")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_dataset(args.modelfolder, args.dataset)
