#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output parquet path")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Minimal schema expected by many rl/sft pipelines: prompt + ground_truth + data_source
    # Keep it super small and deterministic.
    items = [
        {"prompt": "Compute 12*13. Give only the integer.", "ground_truth": "156", "data_source": "tiny_math"},
        {"prompt": "Compute 123+456. Give only the integer.", "ground_truth": "579", "data_source": "tiny_math"},
        {"prompt": "Compute 999-123. Give only the integer.", "ground_truth": "876", "data_source": "tiny_math"},
        {"prompt": "Compute 7^5. Give only the integer.", "ground_truth": "16807", "data_source": "tiny_math"},
        {"prompt": "Compute 1001/7. Give only the integer if exact.", "ground_truth": "143", "data_source": "tiny_math"},
        {"prompt": "Compute the gcd(84, 30). Give only the integer.", "ground_truth": "6", "data_source": "tiny_math"},
        {"prompt": "Compute 2^20. Give only the integer.", "ground_truth": "1048576", "data_source": "tiny_math"},
        {"prompt": "Compute 1+2+...+50. Give only the integer.", "ground_truth": "1275", "data_source": "tiny_math"},
        {"prompt": "Compute 19*23. Give only the integer.", "ground_truth": "437", "data_source": "tiny_math"},
        {"prompt": "Compute 1000000 mod 97. Give only the integer.", "ground_truth": "47", "data_source": "tiny_math"},
    ]

    df = pd.DataFrame(items)
    df.to_parquet(out, index=False)
    print("Wrote:", str(out))
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

