"""
collect_mse_best.py
-------------------
Summarize results from the best-MSE sweep (submit_mse_best_sweep.sh).
For each problem, prints a grid of mean regret over (batch_config x lr)
and reports the best config.

Usage:
    python rethink_exp/collect_mse_best.py
    python rethink_exp/collect_mse_best.py --problem knapsack
    python rethink_exp/collect_mse_best.py --relative   # show relative regret
"""

import argparse
import os
import numpy as np

# ---- Configuration (mirrors submit_mse_best_sweep.sh) ----

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc", "cubic",
            "bipartitematching", "portfolio"]

PROB_ARG = {
    "knapsack":           "knapsack",
    "knapsack-real":      "knapsack",
    "energy":             "energy",
    "budgetalloc":        "budgetalloc",
    "cubic":              "cubic",
    "bipartitematching":  "bipartitematching",
    "portfolio":          "portfolio",
}

PROB_VERSION = {
    "knapsack":           "gen",
    "knapsack-real":      "energy",
    "energy":             "energy",
    "budgetalloc":        "real",
    "cubic":              "gen",
    "bipartitematching":  "cora",
    "portfolio":          "real",
}

LRS = ["1e-3", "5e-3", "1e-2", "5e-2", "1e-1"]
BATCH_LABELS = ["gd", "bs32", "bs64", "bs128"]

RESULTS_ROOT = "saved_records"


def results_path(prob, batch, lr):
    out_dir = f"{PROB_ARG[prob]}-{PROB_VERSION[prob]}"
    prefix = f"mse_best_{batch}_lr{lr}"
    return os.path.join(RESULTS_ROOT, out_dir, "mse", prefix, "results.npy")


def load_result(path):
    """Returns (abs_regret, rel_regret) or (None, None) if file missing."""
    if not os.path.exists(path):
        return None, None
    r = np.load(path, allow_pickle=True)
    regret = np.array(r[1], dtype=float)
    opt = np.array(r[0], dtype=float)
    abs_regret = float(np.mean(regret))
    mean_opt = float(np.mean(np.abs(opt)))
    rel_regret = abs_regret / mean_opt if mean_opt > 0 else None
    return abs_regret, rel_regret


def print_grid(prob, use_pct, baseline_regret=None):
    """Print lr × batch grid for one problem."""
    absolute = prob in USE_ABSOLUTE
    grid = {}

    for batch in BATCH_LABELS:
        for lr in LRS:
            abs_r, rel_r = load_result(results_path(prob, batch, lr))
            grid[(batch, lr)] = abs_r if absolute else rel_r

    # Find best
    completed = {k: v for k, v in grid.items() if v is not None}
    if not completed:
        print(f"  (no results yet)")
        return None, None

    best_key = min(completed, key=lambda k: completed[k])
    best_val = completed[best_key]

    n_done = len(completed)
    n_total = len(BATCH_LABELS) * len(LRS)

    metric_label = "abs regret" if absolute else "rel regret"
    if use_pct and not absolute:
        def fmt(v):
            return f"{v*100:6.2f}%" if v is not None else "  —   "
        unit = "rel %"
    else:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "  —  "
        unit = metric_label

    # Header
    col_w = 9
    lr_header = "  ".join(f"{lr:>{col_w}}" for lr in LRS)
    print(f"  {'batch':8s}  {lr_header}   ({unit}, {n_done}/{n_total} done)")
    print(f"  {'-'*8}  {'  '.join(['-'*col_w]*len(LRS))}")

    for batch in BATCH_LABELS:
        vals = []
        for lr in LRS:
            v = grid[(batch, lr)]
            s = fmt(v)
            if (batch, lr) == best_key:
                s = f"[{s.strip()}]"
                s = f"{s:>{col_w}}"
            else:
                s = f"{s:>{col_w}}"
            vals.append(s)
        print(f"  {batch:8s}  {'  '.join(vals)}")

    impr = (baseline_regret - best_val) / baseline_regret * 100 if baseline_regret else None
    print(f"\n  Best: batch={best_key[0]}, lr={best_key[1]}  →  {unit}={best_val:.4f}" +
          (f"   baseline={baseline_regret:.4f}  improvement={impr:+.1f}%"
           if baseline_regret else ""))

    return best_val, best_key


# ---- Known baselines from prior experiments ----

BASELINES = {
    # (value, description)
    "knapsack":          (0.0640, "MSE gurobi full-batch lr=5e-2"),
    "knapsack-real":     (0.0840, "MSE gurobi full-batch"),
    "energy":            (0.0210, "MSE gurobi full-batch"),
    "budgetalloc":       (0.7060, "MSE neural full-batch lr=5e-2"),
    "cubic":             (0.0010, "MSE heuristic full-batch"),
    "bipartitematching": (0.9240, "MSE cvxpy full-batch"),
    "portfolio":         (0.2531, "MSE cvxpy full-batch lr=5e-2 (surg_all_mse)"),
}

# Problems where the benchmark reports absolute regret (not relative)
USE_ABSOLUTE = {"portfolio"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default=None)
    parser.add_argument("--relative", action="store_true",
                        help="Show relative regret (regret / mean_opt) instead of raw")
    args = parser.parse_args()

    problems = [args.problem] if args.problem else PROBLEMS

    print("=" * 70)
    print("Best-MSE sweep results")
    print("=" * 70)

    summary = []
    for prob in problems:
        baseline, baseline_desc = BASELINES.get(prob, (None, ""))
        metric = "abs regret" if prob in USE_ABSOLUTE else "rel regret"
        print(f"\n{'─'*70}")
        print(f"Problem: {prob}  [{metric}]  "
              + (f"baseline={baseline:.4f} — {baseline_desc}" if baseline else ""))
        print()
        best_val, best_key = print_grid(prob, args.relative, baseline)
        if best_val is not None:
            improvement = (baseline - best_val) / baseline * 100 if baseline else None
            summary.append((prob, best_val, best_key,
                            improvement, baseline))

    # ---- Summary table ----
    if len(summary) > 1:
        print(f"\n{'='*70}")
        print("Summary — best per problem")
        print(f"{'='*70}")
        print(f"  {'Problem':20s}  {'Best':>8s}  {'Baseline':>8s}  {'Improv':>8s}  Config")
        print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")
        for prob, val, key, impr, base in summary:
            config_str = f"{key[0]}, lr={key[1]}"
            impr_str = f"{impr:+.1f}%" if impr is not None else "—"
            base_str = f"{base:.4f}" if base else "—"
            print(f"  {prob:20s}  {val:8.4f}  {base_str:>8s}  {impr_str:>8s}  {config_str}")


if __name__ == "__main__":
    main()
