"""
collect_bench_p2.py
-------------------
Collect Phase 2 (method-specific HP sweep) results and print the final
best-per-method-per-task table.

For each method that has a HP sweep, reports:
  - The best HP value found
  - The regret at that HP
  - Comparison to Phase 1 baseline (best from LR×batch sweep)

Also reads bench_p1_best.json to get the Phase 1 baselines.

Usage:
    python rethink_exp/collect_bench_p2.py
    python rethink_exp/collect_bench_p2.py --problem knapsack
    python rethink_exp/collect_bench_p2.py --method perturb
    python rethink_exp/collect_bench_p2.py --final   # show full comparison table
"""

import argparse
import json
import os

import numpy as np

# ---- Configuration ----

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

# Methods with HP sweeps in Phase 2
P2_METHODS = ["dfl", "blackbox", "qptl", "listLTR", "lodl", "perturb", "pg", "dad"]

# Methods valid for only a subset of problems
METHOD_PROBLEMS = {
    "qptl":    {"knapsack", "bipartitematching", "portfolio"},
    "cpLayer": {"knapsack", "bipartitematching", "portfolio"},
    "pg":      {"knapsack", "knapsack-real", "energy", "cubic", "bipartitematching",
                "portfolio", "asurv", "cook_county", "speed_humps",
                "sp_synth", "sp_planted"},
}

# All methods (for final table)
ALL_METHODS = ["mse", "dfl", "identity", "spo", "nce", "blackbox",
               "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg",
               "qptl", "cpLayer", "dad"]

# HP sweep definitions per method (must match submit_bench_p2.sh)
HP_SWEEPS = {
    "dfl":     {"hp": "dflalpha",    "vals": ["0.001", "0.01", "0.1", "1.0", "10.0"],
                "tag_fn": lambda v: f"alpha{v}"},
    "blackbox":{"hp": "lambd",       "vals": ["0.01", "0.05", "0.1", "0.5", "1.0"],
                "tag_fn": lambda v: f"lam{v}"},
    "qptl":    {"hp": "tau",         "vals": ["0.1", "0.5", "1.0", "5.0", "10.0"],
                "tag_fn": lambda v: f"tau{v}"},
    "listLTR": {"hp": "tau",         "vals": ["0.1", "0.5", "1", "5", "10"],
                "tag_fn": lambda v: f"tau{v}"},
    "lodl":    {"hp": "num_samples", "vals": ["100", "250", "500", "1000", "2000"],
                "tag_fn": lambda v: f"ns{v}"},
    "pg": {
        "hp": "sigma",
        "vals": ["0.01", "0.05", "0.1", "0.5", "1.0"],
        "tag_fn": lambda v: f"s{v.replace('.', 'p')}",
    },
    "perturb_sigma": {
        "method": "perturb",
        "hp": "sigma",
        "vals": ["0.1", "0.5", "1.0", "2.0", "5.0"],
        "tag_fn": lambda v: f"s{v.replace('.', 'p')}_n10",
    },
    "perturb_n": {
        "method": "perturb",
        "hp": "n_samples",
        "vals": ["5", "10", "25", "50", "100"],
        "tag_fn": lambda v: f"s1p0_n{v}",
    },
    "dad": {
        "hp": "stein_weight",
        "vals": ["0.1", "0.5", "1.0", "2.0", "5.0"],
        "tag_fn": lambda v: f"sw{v.replace('.', 'p')}",
    },
}

USE_ABSOLUTE = {"portfolio"}
RESULTS_ROOT = "saved_records"
BEST_JSON_PATH = "bench_p1_best.json"


def out_dir(prob, method, prefix):
    parg = PROB_ARG.get(prob, prob)
    pver = PROB_VERSION.get(prob, "gen")
    return os.path.join(RESULTS_ROOT, f"{parg}-{pver}", method, prefix)


def load_regret(prob, method, prefix):
    d = out_dir(prob, method, prefix)
    rpath = os.path.join(d, "results.npy")
    if not os.path.exists(rpath):
        return None, None
    try:
        r = np.load(rpath, allow_pickle=True)
        regret = np.array(r[1], dtype=float)
        opt    = np.array(r[0], dtype=float)
        abs_r  = float(np.mean(regret))
        mean_opt = float(np.mean(np.abs(opt)))
        rel_r  = abs_r / mean_opt if mean_opt > 0 else None
        return abs_r, rel_r
    except Exception:
        return None, None


def get_metric(prob, method, prefix):
    absolute = prob in USE_ABSOLUTE
    abs_r, rel_r = load_regret(prob, method, prefix)
    return abs_r if absolute else rel_r


def p1_prefix(method, prob, best_json):
    """Get the Phase 1 best-config prefix for this method×prob."""
    cfg = best_json.get(method, {}).get(prob, {})
    lr = cfg.get("lr", "1e-2")
    batch = cfg.get("batch", "default")
    return f"bench_p1_{method}_{batch}_lr{lr}"


def best_p2_for(prob, method, sweep_key, best_json):
    """
    Sweep all HP values for (prob, method, sweep_key) using Phase 1 best config.
    Returns (best_val, best_hp_val, best_prefix, n_done, n_total).
    """
    sweep = HP_SWEEPS[sweep_key]
    actual_method = sweep.get("method", method)
    vals = sweep["vals"]
    tag_fn = sweep["tag_fn"]

    cfg = best_json.get(actual_method, {}).get(prob, {})
    lr = cfg.get("lr", "1e-2")
    batch = cfg.get("batch", "default")

    results = {}
    for v in vals:
        tag = tag_fn(v)
        prefix = f"bench_p2_{actual_method}_{tag}_{batch}_lr{lr}"
        metric = get_metric(prob, actual_method, prefix)
        results[v] = (metric, prefix)

    completed = {v: results[v] for v in results if results[v][0] is not None}
    n_total = len(vals)
    n_done = len(completed)

    if not completed:
        return None, None, None, 0, n_total

    best_v = min(completed, key=lambda v: completed[v][0])
    best_val, best_prefix = completed[best_v]
    return best_val, best_v, best_prefix, n_done, n_total


def load_p1_best(best_json, method, prob):
    return best_json.get(method, {}).get(prob, {}).get("val", None)


def print_p2_results(args, best_json):
    problems = [args.problem] if args.problem else PROBLEMS
    sweep_keys = [args.method] if args.method else list(HP_SWEEPS.keys())

    COL_W = 13
    METH_W = 16

    for sk in sweep_keys:
        if sk not in HP_SWEEPS:
            print(f"Unknown sweep key: {sk}. Options: {list(HP_SWEEPS.keys())}")
            continue

        sweep = HP_SWEEPS[sk]
        actual_method = sweep.get("method", sk)
        hp_name = sweep["hp"]

        allowed = METHOD_PROBLEMS.get(actual_method, None)

        print(f"\n{'─'*70}")
        print(f"Method: {actual_method}  HP: {hp_name}  values: {sweep['vals']}")
        print(f"{'─'*70}")
        hdr = f"  {'prob':16s}  {'p1_best':>{COL_W}}  "
        for v in sweep["vals"]:
            hdr += f"  {str(v):>{COL_W}}"
        hdr += f"  {'p2_best':>{COL_W}}  {'best_hp':>8}"
        print(hdr)
        print("  " + "-" * (18 + (COL_W + 2) * (len(sweep["vals"]) + 2) + 10))

        for prob in problems:
            if allowed is not None and prob not in allowed:
                continue

            p1_val = load_p1_best(best_json, actual_method, prob)
            p1_str = f"{p1_val:.4f}" if p1_val else "—"

            cfg = best_json.get(actual_method, {}).get(prob, {})
            lr = cfg.get("lr", "1e-2")
            batch = cfg.get("batch", "default")

            row = f"  {prob:16s}  {p1_str:>{COL_W}}  "
            vals_done = []
            for v in sweep["vals"]:
                tag = sweep["tag_fn"](v)
                prefix = f"bench_p2_{actual_method}_{tag}_{batch}_lr{lr}"
                metric = get_metric(prob, actual_method, prefix)
                if metric is not None:
                    row += f"  {metric:>{COL_W}.4f}"
                    vals_done.append((metric, v))
                else:
                    row += f"  {'—':>{COL_W}}"

            if vals_done:
                best_val, best_hp = min(vals_done, key=lambda x: x[0])
                row += f"  {best_val:>{COL_W}.4f}  {str(best_hp):>8}"
                if p1_val:
                    impr = (p1_val - best_val) / abs(p1_val) * 100
                    row += f"  ({impr:+.1f}%)"
            else:
                row += f"  {'—':>{COL_W}}  {'—':>8}"

            print(row)


def print_final_table(best_json):
    """
    Print the final comparison table: best regret per method × task,
    combining Phase 1 (methods without HP sweep) and Phase 2 (methods with HP sweep).
    """
    COL_W = 10
    METH_W = 12

    print(f"\n{'='*70}")
    print("FINAL RESULTS TABLE — best regret per method × task")
    print("(Phase 2 HP best where available, Phase 1 otherwise)")
    print(f"{'='*70}\n")

    # Header
    hdr = f"  {'method':>{METH_W}}  "
    for p in PROBLEMS:
        hdr += f"  {p[:COL_W]:>{COL_W}}"
    print(hdr)
    print("  " + "-" * (METH_W + 2 + (COL_W + 2) * len(PROBLEMS)))

    for method in ALL_METHODS:
        allowed = METHOD_PROBLEMS.get(method, None)
        row = f"  {method:>{METH_W}}  "

        # Find sweep keys for this method
        method_sweep_keys = [sk for sk, s in HP_SWEEPS.items()
                             if s.get("method", sk) == method]

        for prob in PROBLEMS:
            if allowed is not None and prob not in allowed:
                row += f"  {'N/A':>{COL_W}}"
                continue

            # Collect candidates: Phase 1 best
            candidates = []
            p1_val = load_p1_best(best_json, method, prob)
            if p1_val is not None:
                candidates.append(p1_val)

            # Phase 2 sweeps
            for sk in method_sweep_keys:
                sweep = HP_SWEEPS[sk]
                cfg = best_json.get(method, {}).get(prob, {})
                lr = cfg.get("lr", "1e-2")
                batch = cfg.get("batch", "default")
                for v in sweep["vals"]:
                    tag = sweep["tag_fn"](v)
                    prefix = f"bench_p2_{method}_{tag}_{batch}_lr{lr}"
                    metric = get_metric(prob, method, prefix)
                    if metric is not None:
                        candidates.append(metric)

            if candidates:
                best = min(candidates)
                row += f"  {best:>{COL_W}.4f}"
            else:
                row += f"  {'—':>{COL_W}}"

        print(row)


def main():
    parser = argparse.ArgumentParser(description="Collect Phase 2 sweep results")
    parser.add_argument("--problem", type=str, default=None)
    parser.add_argument("--method", type=str, default=None,
                        help="Sweep key (dfl, blackbox, qptl, listLTR, lodl, "
                             "perturb_sigma, perturb_n)")
    parser.add_argument("--final", action="store_true",
                        help="Print full final comparison table across all methods")
    parser.add_argument("--best-json", type=str, default=BEST_JSON_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.best_json):
        print(f"ERROR: {args.best_json} not found. Run collect_bench_p1.py first.")
        import sys; sys.exit(1)

    with open(args.best_json) as f:
        best_json = json.load(f)

    print("=" * 70)
    print("Phase 2 results — method-specific HP sweep")
    print("=" * 70)

    print_p2_results(args, best_json)

    if args.final:
        print_final_table(best_json)


if __name__ == "__main__":
    main()
