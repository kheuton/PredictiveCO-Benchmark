#!/usr/bin/env python
"""
Collect and compare results across all methods + perturb sigma variants.

Usage:
    python rethink_exp/collect_with_perturb.py
    python rethink_exp/collect_with_perturb.py --prefix bench
    python rethink_exp/collect_with_perturb.py --problem portfolio
    python rethink_exp/collect_with_perturb.py --sigmas 0.1,0.5,1.0
"""

import argparse
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Problem → (data_name used in saved_records, baseline methods)
# ---------------------------------------------------------------------------
PROBLEM_CONFIG = {
    "portfolio": {
        "data_name": "portfolio-real",
        "baselines": [
            "mse", "dfl", "blackbox", "identity", "cpLayer",
            "spo", "nce", "pointLTR", "listLTR", "pairLTR", "lodl",
        ],
    },
    "knapsack": {
        "data_name": "knapsack-gen",
        "baselines": [
            "mse", "dfl", "blackbox", "identity", "cpLayer",
            "spo", "nce", "pointLTR", "listLTR", "pairLTR", "lodl",
        ],
    },
    "knapsack_energy": {
        "data_name": "knapsack-energy",
        "baselines": [
            "mse", "dfl", "blackbox", "identity", "cpLayer",
            "spo", "nce", "pointLTR", "listLTR", "pairLTR", "lodl",
        ],
    },
    "cubic": {
        "data_name": "cubic-gen",
        "baselines": [
            "mse", "dfl", "blackbox", "identity",
            "spo", "nce", "pointLTR", "listLTR", "pairLTR", "lodl",
        ],
    },
    "budgetalloc": {
        "data_name": "budgetalloc-real",
        "baselines": [
            "mse", "dfl", "blackbox", "identity",
            "spo", "nce", "pointLTR", "listLTR", "pairLTR", "lodl",
        ],
    },
    "energy": {
        "data_name": "energy-energy",
        "baselines": [
            "mse", "dfl", "blackbox", "identity",
            "spo", "nce", "pointLTR", "pairLTR", "listLTR", "lodl",
        ],
    },
    "bipartitematching": {
        "data_name": "bipartitematching-cora",
        "baselines": [
            "bce", "dfl", "blackbox", "identity", "cpLayer",
            "spo", "nce", "pointLTR", "listLTR", "pairLTR", "lodl",
        ],
    },
    "advertising": {
        "data_name": "advertising-advertiser",
        "baselines": [
            "bce", "dfl", "identity", "blackbox",
        ],
    },
}


def get_results(data_name, model_name, prefix_name):
    """Read the last line of log.txt and extract [objective, eval, train_time, test_time]."""
    log_path = os.path.join(
        "saved_records", data_name, model_name, prefix_name, "log.txt"
    )
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        lines = f.readlines()
        if not lines:
            return None
        last_line = lines[-1].strip()
        parts = last_line.split("  ")[-4:]
    try:
        result = [float(x) for x in parts]
    except Exception:
        return None
    return np.array(result)


def collect_all_results(
    problems=None, prefix="bench", sigmas=None, records_dir="saved_records"
):
    """
    Collect results for all problems, all methods, and perturb sigma variants.

    Returns:
        dict: {problem_key: pd.DataFrame} where DataFrame has columns:
              method, objective, eval_metric, train_time, test_time
    """
    if sigmas is None:
        sigmas = ["01", "05", "10"]

    if problems is None:
        problems = list(PROBLEM_CONFIG.keys())

    all_results = {}

    for prob_key in problems:
        if prob_key not in PROBLEM_CONFIG:
            print(f"Warning: unknown problem '{prob_key}', skipping")
            continue

        cfg = PROBLEM_CONFIG[prob_key]
        data_name = cfg["data_name"]
        baselines = cfg["baselines"]

        rows = []

        # Collect baseline results
        for method in baselines:
            res = get_results(data_name, method, prefix)
            if res is not None:
                rows.append({
                    "method": method,
                    "objective": res[0],
                    "eval_metric": res[1],
                    "train_time": res[2],
                    "test_time": res[3],
                })
            else:
                rows.append({
                    "method": method,
                    "objective": np.nan,
                    "eval_metric": np.nan,
                    "train_time": np.nan,
                    "test_time": np.nan,
                })

        # Collect perturb results for each sigma
        for sigma_tag in sigmas:
            perturb_prefix = f"{prefix}_perturb_s{sigma_tag}"
            method_label = f"perturb_s{sigma_tag}"
            res = get_results(data_name, "perturb", perturb_prefix)
            if res is not None:
                rows.append({
                    "method": method_label,
                    "objective": res[0],
                    "eval_metric": res[1],
                    "train_time": res[2],
                    "test_time": res[3],
                })
            else:
                rows.append({
                    "method": method_label,
                    "objective": np.nan,
                    "eval_metric": np.nan,
                    "train_time": np.nan,
                    "test_time": np.nan,
                })

        df = pd.DataFrame(rows)
        all_results[prob_key] = df

    return all_results


def print_results_table(all_results, metric="eval_metric"):
    """Print a consolidated table showing all problems and methods."""
    # Collect all method names across problems
    all_methods = []
    for df in all_results.values():
        for m in df["method"].tolist():
            if m not in all_methods:
                all_methods.append(m)

    # Build a pivot table: rows=methods, cols=problems
    table_data = {}
    for prob_key, df in all_results.items():
        col = {}
        for _, row in df.iterrows():
            col[row["method"]] = row[metric]
        table_data[prob_key] = col

    pivot = pd.DataFrame(table_data, index=all_methods)
    pivot.index.name = "method"

    # Add ranking columns (lower eval_metric = better for regret)
    rank_df = pivot.rank(axis=0, method="min", na_option="bottom")
    rank_df.columns = [f"{c}_rank" for c in rank_df.columns]

    # Average rank
    rank_df["avg_rank"] = rank_df.mean(axis=1)

    combined = pd.concat([pivot, rank_df], axis=1)
    return pivot, rank_df, combined


def save_results(all_results, output_dir="saved_records"):
    """Save per-problem CSVs and a combined Excel workbook."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-problem CSVs
    for prob_key, df in all_results.items():
        csv_path = os.path.join(output_dir, f"{prob_key}_results.csv")
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"Saved: {csv_path}")

    # Combined Excel
    pivot, rank_df, combined = print_results_table(all_results)

    excel_path = os.path.join(output_dir, "benchmark_results_with_perturb.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pivot.to_excel(writer, sheet_name="eval_metric")
        rank_df.to_excel(writer, sheet_name="rankings")
        combined.to_excel(writer, sheet_name="combined")
        # Also objective values
        pivot_obj, _, _ = print_results_table(all_results, metric="objective")
        pivot_obj.to_excel(writer, sheet_name="objective")
    print(f"Saved: {excel_path}")

    return pivot, rank_df


def main():
    parser = argparse.ArgumentParser(description="Collect PredictiveCO benchmark results")
    parser.add_argument("--prefix", type=str, default="bench", help="Experiment prefix")
    parser.add_argument(
        "--problem", type=str, default=None,
        help="Single problem to collect (default: all)"
    )
    parser.add_argument(
        "--sigmas", type=str, default="01,05,10",
        help="Comma-separated sigma tags (e.g. '01,05,10')"
    )
    parser.add_argument(
        "--output-dir", type=str, default="saved_records",
        help="Output directory"
    )
    args = parser.parse_args()

    problems = [args.problem] if args.problem else None
    sigmas = args.sigmas.split(",")

    print("=" * 80)
    print("Collecting PredictiveCO Benchmark Results")
    print("=" * 80)

    all_results = collect_all_results(
        problems=problems, prefix=args.prefix, sigmas=sigmas
    )

    # Print summary
    for prob_key, df in all_results.items():
        print(f"\n{'=' * 60}")
        print(f"  {prob_key}")
        print(f"{'=' * 60}")
        # Only show rows with data
        has_data = df.dropna(subset=["eval_metric"])
        if len(has_data) > 0:
            print(has_data.to_string(index=False))
        else:
            print("  (no results found)")

    # Print ranking table
    pivot, rank_df, combined = print_results_table(all_results)

    print(f"\n{'=' * 80}")
    print("Eval Metric by Problem (lower = better for regret)")
    print(f"{'=' * 80}")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print(pivot.to_string())

    print(f"\n{'=' * 80}")
    print("Rankings (lower = better)")
    print(f"{'=' * 80}")
    print(rank_df.to_string())

    # Highlight best perturb
    perturb_methods = [m for m in pivot.index if m.startswith("perturb_")]
    if perturb_methods:
        print(f"\n{'=' * 80}")
        print("Best perturb sigma per problem:")
        print(f"{'=' * 80}")
        for col in pivot.columns:
            perturb_vals = pivot.loc[perturb_methods, col].dropna()
            if len(perturb_vals) > 0:
                best = perturb_vals.idxmin()
                print(f"  {col}: {best} = {perturb_vals[best]:.6f}")

    # Save
    save_results(all_results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
