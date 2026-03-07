#!/usr/bin/env python3
"""
Find the top-K experiments by best validation performance (lowest eval/regret).

Scans saved_records/{data_name}/perturb/*/  for val_logs.csv (preferred)
or falls back to parsing log.txt.

Usage:
    python rethink_exp/find_best_val.py                      # defaults: both problems, top-5
    python rethink_exp/find_best_val.py --problem knapsack    # just knapsack-gen
    python rethink_exp/find_best_val.py --problem energy      # just energy-energy
    python rethink_exp/find_best_val.py --top_k 10            # top-10
"""
import argparse
import os
import re
from pathlib import Path

import pandas as pd

SAVED_RECORDS = "saved_records"

# Map short names -> data_name subdirectory
PROBLEM_MAP = {
    "knapsack": "knapsack-gen",
    "energy": "energy-energy",
}

# Regex for parsing log.txt val lines
_LOG_PAT = re.compile(
    r"Iter\s+(\d+),\s+val\s+Objective:\s+([\d.eE+-]+),\s+"
    r"Loss:\s+([\d.eE+-]+)\s+Pred Loss:\s+([\d.eE+-]+),\s+"
    r"(?:regret|uplift):\s+([\d.eE+-]+)"
)


def best_val_from_csv(csv_path: str) -> float:
    """Return min eval (or regret) from a val_logs.csv."""
    try:
        df = pd.read_csv(csv_path)
        if "regret" in df.columns:
            return df["regret"].min()
        if "eval" in df.columns:
            return df["eval"].min()
    except Exception:
        pass
    return float("inf")


def best_val_from_log(log_path: str) -> float:
    """Return min val regret by parsing log.txt."""
    best = float("inf")
    try:
        with open(log_path) as f:
            for line in f:
                m = _LOG_PAT.search(line)
                if m:
                    val = float(m.group(5))
                    if val < best:
                        best = val
    except Exception:
        pass
    return best


def scan_experiments(data_name: str):
    """Return list of (prefix, best_val_metric) for all experiments."""
    base = os.path.join(SAVED_RECORDS, data_name, "perturb")
    if not os.path.isdir(base):
        print(f"  [WARN] Directory not found: {base}")
        return []

    results = []
    for prefix in sorted(os.listdir(base)):
        exp_dir = os.path.join(base, prefix)
        if not os.path.isdir(exp_dir):
            continue

        # Try val_logs.csv first
        csv_path = os.path.join(exp_dir, "val_logs.csv")
        if os.path.exists(csv_path):
            bv = best_val_from_csv(csv_path)
        else:
            # Fall back to log.txt
            log_path = os.path.join(exp_dir, "log.txt")
            bv = best_val_from_log(log_path)

        if bv < float("inf"):
            results.append((prefix, bv))

    return results


def find_yaml_config(prefix: str) -> str:
    """Try to find the YAML config that was used for this experiment."""
    # Check log.txt for the method_path
    for data_name in PROBLEM_MAP.values():
        log_path = os.path.join(SAVED_RECORDS, data_name, "perturb", prefix, "log.txt")
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    content = f.read(5000)  # first 5KB is enough
                m = re.search(r"method_path='([^']+)'", content)
                if m:
                    return m.group(1)
            except Exception:
                pass
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Find top-K experiments by val performance")
    parser.add_argument(
        "--problem",
        type=str,
        choices=["knapsack", "energy", "both"],
        default="both",
        help="Which problem to scan (default: both)",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top experiments to show"
    )
    parser.add_argument(
        "--show_yaml", action="store_true", help="Show YAML config path for each experiment"
    )
    args = parser.parse_args()

    problems = list(PROBLEM_MAP.keys()) if args.problem == "both" else [args.problem]

    for prob in problems:
        data_name = PROBLEM_MAP[prob]
        print(f"\n{'='*70}")
        print(f"  {prob.upper()} ({data_name})  —  Top {args.top_k} by best validation metric")
        print(f"{'='*70}")

        results = scan_experiments(data_name)
        if not results:
            print("  No completed experiments found.\n")
            continue

        # Sort by best val (lower = better)
        results.sort(key=lambda x: x[1])

        print(f"  {'Rank':<6} {'Best Val':>12}   {'Prefix'}")
        print(f"  {'-'*6} {'-'*12}   {'-'*40}")
        for rank, (prefix, bv) in enumerate(results[: args.top_k], 1):
            line = f"  {rank:<6} {bv:>12.4f}   {prefix}"
            if args.show_yaml:
                yaml_path = find_yaml_config(prefix)
                line += f"   [{yaml_path}]"
            print(line)

        print(f"\n  Total experiments scanned: {len(results)}")
        print()


if __name__ == "__main__":
    main()
