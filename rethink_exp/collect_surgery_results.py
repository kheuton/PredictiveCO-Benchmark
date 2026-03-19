"""
collect_surgery_results.py
--------------------------
Collect and compare results from:
  1. surg_all_* runs (constant-sigma, full-batch, 7 problems x 5 experiments)
  2. mse_best_* runs (minibatch sweep best, 7 problems)
  3. cub_surg_* runs (cubic surgery sigma sweep)

Usage:
    python rethink_exp/collect_surgery_results.py
    python rethink_exp/collect_surgery_results.py --cubic_detail
"""

import argparse
import os
import numpy as np

RESULTS_ROOT = "saved_records"

# ---- Problem config ----

PROBLEMS = [
    ("knapsack",        "knapsack-gen",           "rel"),
    ("knapsack-real",   "knapsack-energy",         "rel"),
    ("energy",          "energy-energy",           "rel"),
    ("budgetalloc",     "budgetalloc-real",        "rel"),
    ("cubic",           "cubic-gen",               "rel"),
    ("bipartitematching","bipartitematching-cora", "rel"),
    ("portfolio",       "portfolio-real",          "abs"),
]

# Surgery sweep experiments (constant sigma, full-batch)
SURG_EXPS = [
    ("surg_all_mse",    "MSE (surg baseline)"),
    ("surg_all_plain",  "plain perturb"),
    ("surg_all_w1_lr1e2", "surgery w=1 lr=1e-2"),
    ("surg_all_w1_lr5e3", "surgery w=1 lr=5e-3"),
    ("surg_all_w0_lr1e2", "surgery w=0 lr=1e-2"),
]

# Best MSE from minibatch sweep
MSE_BEST_CONFIGS = {
    "knapsack":          ("bs32", "5e-2"),
    "knapsack-real":     ("bs32", "5e-2"),
    "energy":            ("bs32", "5e-2"),
    "budgetalloc":       ("bs32", "1e-3"),
    "cubic":             ("bs32", "5e-2"),
    "bipartitematching": ("gd",   "5e-3"),
    "portfolio":         ("gd",   "5e-3"),
}


def load_result(path):
    """Returns (abs_regret, rel_regret) or (None, None)."""
    if not os.path.exists(path):
        return None, None
    r = np.load(path, allow_pickle=True)
    regret = np.array(r[1], dtype=float)
    opt = np.array(r[0], dtype=float)
    abs_r = float(np.mean(regret))
    mean_opt = float(np.mean(np.abs(opt)))
    rel_r = abs_r / mean_opt if mean_opt > 0 else None
    return abs_r, rel_r


def get_metric(path, metric):
    abs_r, rel_r = load_result(path)
    if metric == "abs":
        return abs_r
    return rel_r


def fmt(v, metric):
    if v is None:
        return "   —   "
    if metric == "rel":
        return f"{v:.4f}"
    return f"{v:.4f}"


# ---- Main surgery sweep table ----

def print_surgery_table(args):
    print("=" * 90)
    print("Surgery sweep — constant sigma, full-batch (surg_all_*) vs best MSE")
    print("=" * 90)

    for prob, prob_dir, metric in PROBLEMS:
        print(f"\n{'─'*90}")
        print(f"  {prob}  [{metric} regret]")
        print()

        # Best MSE from minibatch sweep
        batch, lr = MSE_BEST_CONFIGS[prob]
        best_mse_path = os.path.join(
            RESULTS_ROOT, prob_dir, "mse",
            f"mse_best_{batch}_lr{lr}", "results.npy"
        )
        best_mse = get_metric(best_mse_path, metric)

        # Surgery experiments
        results = {}
        for prefix, label in SURG_EXPS:
            path = os.path.join(RESULTS_ROOT, prob_dir, "perturb" if "perturb" in prefix or "plain" in prefix or "surgery" in prefix or "w" in prefix else "mse",
                                prefix, "results.npy")
            # Determine subfolder: mse/ for surg_all_mse, perturb/ for others
            if prefix == "surg_all_mse":
                path = os.path.join(RESULTS_ROOT, prob_dir, "mse", prefix, "results.npy")
            else:
                path = os.path.join(RESULTS_ROOT, prob_dir, "perturb", prefix, "results.npy")
            results[prefix] = get_metric(path, metric)

        # Print
        col_w = 12
        header = f"  {'Experiment':30s}  {'Value':>10s}  {'vs surg-MSE':>12s}  {'vs best-MSE':>12s}"
        print(header)
        print(f"  {'-'*30}  {'-'*10}  {'-'*12}  {'-'*12}")

        surg_mse_val = results.get("surg_all_mse")

        # Print surg_all_mse first
        v = surg_mse_val
        vs_surgmse = "  (baseline)"
        vs_bestmse = f"  {(best_mse - v)/best_mse*100:+.1f}%" if (v is not None and best_mse is not None) else "  —"
        label = "MSE (surg baseline)"
        print(f"  {label:30s}  {fmt(v, metric):>10s}  {vs_surgmse:>12s}  {vs_bestmse:>12s}")

        # Print best MSE
        v = best_mse
        vs_surgmse = f"  {(surg_mse_val - v)/surg_mse_val*100:+.1f}%" if (v is not None and surg_mse_val is not None) else "  —"
        print(f"  {'MSE (best: ' + batch + ' lr=' + lr + ')':30s}  {fmt(v, metric):>10s}  {vs_surgmse:>12s}  {'  (best)':>12s}")

        # Print surgery experiments
        for prefix, label in SURG_EXPS[1:]:  # skip surg_all_mse
            v = results[prefix]
            vs_sm = f"  {(surg_mse_val - v)/surg_mse_val*100:+.1f}%" if (v is not None and surg_mse_val is not None) else "  —"
            vs_bm = f"  {(best_mse - v)/best_mse*100:+.1f}%" if (v is not None and best_mse is not None) else "  —"
            print(f"  {label:30s}  {fmt(v, metric):>10s}  {vs_sm:>12s}  {vs_bm:>12s}")


# ---- Cubic surgery sigma sweep ----

def print_cubic_detail():
    print("\n\n" + "=" * 80)
    print("Cubic surgery sigma sweep (cub_surg_s*)")
    print("=" * 80)

    prob_dir = "cubic-gen"
    metric = "rel"

    sigmas = ["s01", "s03", "s05", "s10", "s30", "s100"]
    lrs = ["lr1e-2", "lr5e-3"]
    variants = ["plain", "surgery"]

    # Header
    col_w = 10
    header_parts = []
    for sig in sigmas:
        header_parts.append(f"{sig:>{col_w}}")
    print(f"\n  {'variant':25s}  {'lr':8s}  " + "  ".join(header_parts))
    print(f"  {'-'*25}  {'-'*8}  " + "  ".join(["-"*col_w]*len(sigmas)))

    for var in variants:
        for lr in lrs:
            vals = []
            for sig in sigmas:
                prefix = f"cub_surg_{sig}_{var}_{lr}"
                path = os.path.join(RESULTS_ROOT, prob_dir, "perturb", prefix, "results.npy")
                v = get_metric(path, metric)
                vals.append(f"{v:.4f}" if v is not None else "  —  ")
            print(f"  {var:25s}  {lr:8s}  " + "  ".join(f"{v:>{col_w}}" for v in vals))

    # Also print MSE best for reference
    batch, lr_mse = MSE_BEST_CONFIGS["cubic"]
    best_mse_path = os.path.join(RESULTS_ROOT, prob_dir, "mse",
                                  f"mse_best_{batch}_lr{lr_mse}", "results.npy")
    best_mse = get_metric(best_mse_path, metric)
    print(f"\n  MSE best ({batch}, lr={lr_mse}): {best_mse:.4f}" if best_mse else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubic_detail", action="store_true",
                        help="Also print cubic surgery sigma sweep detail")
    args = parser.parse_args()

    print_surgery_table(args)
    if args.cubic_detail:
        print_cubic_detail()

    print()


if __name__ == "__main__":
    main()
