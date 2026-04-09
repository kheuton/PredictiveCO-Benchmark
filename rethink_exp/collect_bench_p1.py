"""
collect_bench_p1.py
-------------------
Collect Phase 1 (LR × Batch sweep) results and output the best (lr, batch)
config per method × task.

Reads all bench_p1_* prefixes and produces:
  1. A printed grid table (rows=methods, cols=tasks) showing best regret
  2. bench_p1_best.json — consumed by submit_bench_p2.sh to run Phase 2
     with the optimal configs per task.

Usage:
    python rethink_exp/collect_bench_p1.py
    python rethink_exp/collect_bench_p1.py --problem knapsack
    python rethink_exp/collect_bench_p1.py --method mse
    python rethink_exp/collect_bench_p1.py --relative   # show relative regret %
"""

import argparse
import json
import os

import numpy as np

# ---- Configuration (mirrors submit_bench_p1.sh) ----

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio"]

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

METHODS = ["mse", "dfl", "identity", "spo", "nce", "blackbox",
           "pointLTR", "pairLTR", "listLTR", "lodl", "perturb",
           "qptl", "cpLayer"]

# Methods only valid for subset of problems
METHOD_PROBLEMS = {
    "qptl":    {"knapsack", "bipartitematching", "portfolio"},
    "cpLayer": {"knapsack", "bipartitematching", "portfolio"},
}

LRS = ["1e-2", "5e-3", "1e-3"]
BATCH_LABELS = ["default", "alt"]

# Problems reported as absolute regret (not relative)
USE_ABSOLUTE = {"portfolio"}

RESULTS_ROOT = "saved_records"
BEST_JSON_PATH = "bench_p1_best.json"


def results_path(prob, method, batch, lr):
    out_dir = f"{PROB_ARG[prob]}-{PROB_VERSION[prob]}"
    prefix = f"bench_p1_{method}_{batch}_lr{lr}"
    return os.path.join(RESULTS_ROOT, out_dir, method, prefix, "results.npy")


def load_regret(path, absolute=False):
    """Return (abs_regret, rel_regret) or (None, None)."""
    if not os.path.exists(path):
        return None, None
    try:
        r = np.load(path, allow_pickle=True)
        regret = np.array(r[1], dtype=float)
        opt    = np.array(r[0], dtype=float)
        abs_r  = float(np.mean(regret))
        mean_opt = float(np.mean(np.abs(opt)))
        rel_r  = abs_r / mean_opt if mean_opt > 0 else None
        return abs_r, rel_r
    except Exception:
        return None, None


def best_config_for(prob, method):
    """
    Return (best_val, best_batch, best_lr, n_complete, n_total).
    best_val is abs_regret for USE_ABSOLUTE problems, else rel_regret.
    """
    absolute = prob in USE_ABSOLUTE
    results = {}
    for batch in BATCH_LABELS:
        for lr in LRS:
            abs_r, rel_r = load_regret(results_path(prob, method, batch, lr), absolute)
            val = abs_r if absolute else rel_r
            results[(batch, lr)] = val

    completed = {k: v for k, v in results.items() if v is not None}
    n_total = len(results)
    n_complete = len(completed)
    if not completed:
        return None, None, None, 0, n_total

    best_key = min(completed, key=lambda k: completed[k])
    best_val = completed[best_key]
    return best_val, best_key[0], best_key[1], n_complete, n_total


def print_results(args):
    problems  = [args.problem] if args.problem else PROBLEMS
    methods   = [args.method]  if args.method  else METHODS
    use_pct   = args.relative

    # Column widths
    COL_W = 12
    METH_W = 12

    header_cells = [f"{p[:COL_W]:>{COL_W}}" for p in problems]
    print(f"\n  {'method':>{METH_W}}  " + "  ".join(header_cells))
    print("  " + "-" * (METH_W + 2 + (COL_W + 2) * len(problems)))

    # For building bench_p1_best.json
    best_json = {}

    for method in methods:
        row = []
        for prob in problems:
            # Check if method applies to this problem
            allowed = METHOD_PROBLEMS.get(method, None)
            if allowed is not None and prob not in allowed:
                row.append(f"{'N/A':>{COL_W}}")
                continue

            best_val, best_batch, best_lr, n_done, n_total = best_config_for(prob, method)

            if best_val is None:
                row.append(f"{'—':>{COL_W}}")
                continue

            absolute = prob in USE_ABSOLUTE
            if use_pct and not absolute:
                s = f"{best_val*100:.2f}%({n_done}/{n_total})"
            else:
                s = f"{best_val:.4f}({n_done}/{n_total})"
            row.append(f"{s:>{COL_W}}")

            # Record best config
            best_json.setdefault(method, {})[prob] = {
                "lr": best_lr,
                "batch": best_batch,
                "val": round(best_val, 6),
                "n_done": n_done,
                "n_total": n_total,
            }

        print(f"  {method:>{METH_W}}  " + "  ".join(row))

    print()
    return best_json


def main():
    parser = argparse.ArgumentParser(description="Collect Phase 1 sweep results")
    parser.add_argument("--problem", type=str, default=None,
                        help="Filter to one problem")
    parser.add_argument("--method", type=str, default=None,
                        help="Filter to one method")
    parser.add_argument("--relative", action="store_true",
                        help="Show relative regret (regret/opt) as %")
    parser.add_argument("--no-json", action="store_true",
                        help="Skip writing bench_p1_best.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1 results — best (lr, batch) per method × task")
    print(f"(metric: {'rel regret %' if args.relative else 'abs regret / rel regret'})")
    print("=" * 70)

    best_json = print_results(args)

    if not args.no_json and best_json:
        with open(BEST_JSON_PATH, "w") as f:
            json.dump(best_json, f, indent=2)
        print(f"Best configs written to {BEST_JSON_PATH}")
        print("Pass this to Phase 2: bash shells/slurm/submit_bench_p2.sh")

    # Print completion summary
    n_done = n_total = 0
    for prob in PROBLEMS:
        for method in METHODS:
            allowed = METHOD_PROBLEMS.get(method, None)
            if allowed is not None and prob not in allowed:
                continue
            for batch in BATCH_LABELS:
                for lr in LRS:
                    n_total += 1
                    if os.path.exists(results_path(prob, method, batch, lr)):
                        n_done += 1
    print(f"\nOverall completion: {n_done}/{n_total} jobs done")


if __name__ == "__main__":
    main()
