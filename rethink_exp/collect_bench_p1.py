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
import csv
import json
import os

import numpy as np

# ---- Configuration (mirrors submit_bench_p1.sh) ----

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio", "asurv", "cook_county",
            "speed_humps", "sp_synth", "sp_planted", "shortestpath"]

PROB_ARG = {
    "knapsack":           "knapsack",
    "knapsack-real":      "knapsack",
    "energy":             "energy",
    "budgetalloc":        "budgetalloc",
    "cubic":              "cubic",
    "bipartitematching":  "bipartitematching",
    "portfolio":          "portfolio",
    "asurv":              "asurv",
    "cook_county":        "cook_county",
    "speed_humps":        "speed_humps",
    "sp_synth":           "sp_synth",
    "sp_planted":         "sp_planted",
    "shortestpath":       "shortestpath",
}

PROB_VERSION = {
    "knapsack":           "gen",
    "knapsack-real":      "energy",
    "energy":             "energy",
    "budgetalloc":        "real",
    "cubic":              "gen",
    "bipartitematching":  "cora",
    "portfolio":          "real",
    "asurv":              "real",
    "cook_county":        "real",
    "speed_humps":        "real",
    "sp_synth":           "synth",
    "sp_planted":         "planted",
    "shortestpath":       "warcraft",
}

METHODS = ["mse", "dfl", "identity", "spo", "nce", "blackbox",
           "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg",
           "qptl", "cpLayer", "dad"]

# Methods only valid for subset of problems
METHOD_PROBLEMS = {
    "qptl":    {"knapsack", "bipartitematching", "portfolio"},
    "cpLayer": {"knapsack", "bipartitematching", "portfolio"},
    "pg":      {"knapsack", "knapsack-real", "energy", "cubic", "bipartitematching",
                "portfolio", "asurv", "cook_county", "speed_humps",
                "sp_synth", "sp_planted"},
}

LRS = ["1e-2", "5e-3", "1e-3", "5e-2", "1e-1"]
BATCH_LABELS = ["default", "alt"]

# Problems reported as absolute regret (not relative)
USE_ABSOLUTE = {"portfolio"}

RESULTS_ROOT = "saved_records"
BEST_JSON_PATH_TEST = "bench_p1_best.json"
BEST_JSON_PATH_VAL  = "bench_p1_best_val.json"


def run_dir(prob, method, batch, lr):
    out_dir = f"{PROB_ARG[prob]}-{PROB_VERSION[prob]}"
    prefix = f"bench_p1_{method}_{batch}_lr{lr}"
    return os.path.join(RESULTS_ROOT, out_dir, method, prefix)


def results_path(prob, method, batch, lr):
    return os.path.join(run_dir(prob, method, batch, lr), "results.npy")


def val_log_path(prob, method, batch, lr):
    return os.path.join(run_dir(prob, method, batch, lr), "val_logs.csv")


def train_log_txt_path(prob, method, batch, lr):
    return os.path.join(run_dir(prob, method, batch, lr), "log.txt")


def load_test_regret(path, absolute=False):
    """Return (abs_regret, rel_regret) from results.npy, or (None, None)."""
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


def load_val_regret(path):
    """
    Return (min_val_regret, best_epoch_label, source) from val_logs.csv, or
    (None, None, None). Uses the 'eval' column (per-epoch val decision regret).
    All 13 benchmark problems use regret (sense=1, lower is better).
    """
    if not os.path.exists(path):
        return None, None, None
    try:
        min_val = None
        best_epoch = None
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "eval" not in (reader.fieldnames or []):
                return None, None, None
            for row in reader:
                try:
                    v = float(row["eval"])
                except (TypeError, ValueError):
                    continue
                if min_val is None or v < min_val:
                    min_val = v
                    best_epoch = row.get("epoch")
        if min_val is None:
            return None, None, None
        return min_val, best_epoch, "val_eval"
    except Exception:
        return None, None, None


# Regex picks up lines like "Iter 42, val MSE (no solver): 5.23"
_VAL_MSE_LINE = __import__("re").compile(
    r"Iter\s+(\d+),\s*val MSE \(no solver\):\s*([0-9eE+\-\.]+)"
)


def load_val_pred_mse_from_log(path):
    """
    Fallback for runs trained with --skip_solver_eval (no val_logs.csv, no
    solver-based val eval). Parse log.txt for 'val MSE (no solver)' lines and
    return (min_val_mse, best_iter, source). These runs used val pred MSE for
    early stopping, so that is the metric that was actually used to pick the
    best checkpoint — still non-leaky.
    """
    if not os.path.exists(path):
        return None, None, None
    try:
        min_val = None
        best_iter = None
        with open(path) as f:
            for line in f:
                m = _VAL_MSE_LINE.search(line)
                if not m:
                    continue
                try:
                    v = float(m.group(2))
                except ValueError:
                    continue
                if min_val is None or v < min_val:
                    min_val = v
                    best_iter = f"Tr-{m.group(1)}"
        if min_val is None:
            return None, None, None
        return min_val, best_iter, "val_pred_mse"
    except Exception:
        return None, None, None


def best_config_for(prob, method, metric="test"):
    """
    Return (best_val, best_batch, best_lr, n_complete, n_total, extras).

    metric="test": select by test regret from results.npy (legacy, LEAKY).
      best_val is abs_regret for USE_ABSOLUTE problems, else rel_regret.
    metric="val":  select by min val regret from val_logs.csv (correct).
      best_val is absolute val regret (no rel conversion — ranking within a
      (prob, method) is invariant to val_opt scaling).
    """
    absolute = prob in USE_ABSOLUTE
    results = {}
    extras = {}  # (batch, lr) -> dict with diagnostics (e.g. best_epoch)
    for batch in BATCH_LABELS:
        for lr in LRS:
            if metric == "test":
                abs_r, rel_r = load_test_regret(results_path(prob, method, batch, lr), absolute)
                val = abs_r if absolute else rel_r
                results[(batch, lr)] = val
            elif metric == "val":
                v, best_epoch, source = load_val_regret(val_log_path(prob, method, batch, lr))
                if v is None:
                    # Fallback for --skip_solver_eval runs (e.g. energy/mse)
                    v, best_epoch, source = load_val_pred_mse_from_log(
                        train_log_txt_path(prob, method, batch, lr))
                results[(batch, lr)] = v
                extras[(batch, lr)] = {"best_epoch": best_epoch, "val_source": source}
            else:
                raise ValueError(f"unknown metric: {metric}")

    completed = {k: v for k, v in results.items() if v is not None}
    n_total = len(results)
    n_complete = len(completed)
    if not completed:
        return None, None, None, 0, n_total, {}

    best_key = min(completed, key=lambda k: completed[k])
    best_val = completed[best_key]
    return best_val, best_key[0], best_key[1], n_complete, n_total, extras.get(best_key, {})


def print_results(args):
    problems  = [args.problem] if args.problem else PROBLEMS
    methods   = [args.method]  if args.method  else METHODS
    use_pct   = args.relative
    metric    = args.metric

    # Column widths
    COL_W = 12
    METH_W = 12

    header_cells = [f"{p[:COL_W]:>{COL_W}}" for p in problems]
    print(f"\n  {'method':>{METH_W}}  " + "  ".join(header_cells))
    print("  " + "-" * (METH_W + 2 + (COL_W + 2) * len(problems)))

    # For building bench_p1_best{,_val}.json
    best_json = {}

    for method in methods:
        row = []
        for prob in problems:
            # Check if method applies to this problem
            allowed = METHOD_PROBLEMS.get(method, None)
            if allowed is not None and prob not in allowed:
                row.append(f"{'N/A':>{COL_W}}")
                continue

            best_val, best_batch, best_lr, n_done, n_total, extras = best_config_for(
                prob, method, metric=metric)

            if best_val is None:
                row.append(f"{'—':>{COL_W}}")
                continue

            absolute = prob in USE_ABSOLUTE
            # --relative only meaningful for metric=test (rel vs abs). For val
            # selection, we always display the absolute val regret.
            if metric == "test" and use_pct and not absolute:
                s = f"{best_val*100:.2f}%({n_done}/{n_total})"
            else:
                s = f"{best_val:.4f}({n_done}/{n_total})"
            row.append(f"{s:>{COL_W}}")

            # Record best config. Key "val" retained so existing readers
            # (submit_bench_p2.sh) keep working — it only reads lr/batch.
            entry = {
                "lr": best_lr,
                "batch": best_batch,
                "val": round(best_val, 6),
                "n_done": n_done,
                "n_total": n_total,
                "metric": metric,
            }
            if metric == "val":
                if extras.get("best_epoch") is not None:
                    entry["best_epoch"] = extras["best_epoch"]
                if extras.get("val_source") is not None:
                    entry["val_source"] = extras["val_source"]
            best_json.setdefault(method, {})[prob] = entry

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
    parser.add_argument("--metric", type=str, choices=["test", "val"], default="test",
                        help="Selection metric. 'test' = legacy (LEAKY: picks by "
                             "test regret from results.npy). 'val' = correct "
                             "(picks by min val regret from val_logs.csv).")
    parser.add_argument("--no-json", action="store_true",
                        help="Skip writing the best-config JSON file")
    args = parser.parse_args()

    out_path = BEST_JSON_PATH_VAL if args.metric == "val" else BEST_JSON_PATH_TEST

    print("=" * 70)
    print(f"Phase 1 results — best (lr, batch) per method × task (metric={args.metric})")
    if args.metric == "test":
        print("  ⚠ LEAKY: selecting by test regret (results.npy[1]). Use --metric val.")
    print("=" * 70)

    best_json = print_results(args)

    if not args.no_json and best_json:
        with open(out_path, "w") as f:
            json.dump(best_json, f, indent=2)
        print(f"Best configs written to {out_path}")
        if args.metric == "val":
            print("To drive Phase 2 from val-selected configs, point "
                  "submit_bench_p2.sh at this file (BEST_JSON).")
        else:
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
