"""
sweep_status.py
---------------
Status dashboard for the benchmark re-run sweep (Phase 1 and Phase 2).

Reads a manifest JSON produced by submit_bench_p1.sh or submit_bench_p2.sh and
checks the on-disk state of each (problem, opt_model, prefix) tuple.

Status codes:
  ✓  = results.npy exists (complete)
  R  = checkpoint_latest.pt exists but no results.npy (in-progress / preempted)
  -- = nothing (not started or failed before first checkpoint)

Usage:
    python rethink_exp/sweep_status.py --phase 1
    python rethink_exp/sweep_status.py --phase 1 --vals
    python rethink_exp/sweep_status.py --manifest sweep_manifest_p1.json
    python rethink_exp/sweep_status.py --manifest sweep_manifest_p2.json --vals
"""

import argparse
import json
import os
import sys

import numpy as np

RESULTS_ROOT = "saved_records"


# ---- Problem metadata (mirrors submit scripts) ----

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


def out_dir(prob, opt_model, prefix):
    parg = PROB_ARG.get(prob, prob)
    pver = PROB_VERSION.get(prob, "gen")
    return os.path.join(RESULTS_ROOT, f"{parg}-{pver}", opt_model, prefix)


def cell_status(prob, opt_model, prefix):
    d = out_dir(prob, opt_model, prefix)
    results = os.path.join(d, "results.npy")
    ckpt    = os.path.join(d, "checkpoints", "checkpoint_latest.pt")
    if os.path.exists(results):
        return "done"
    if os.path.exists(ckpt):
        return "running"
    return "missing"


def load_regret(prob, opt_model, prefix):
    """Return mean test regret or None."""
    d = out_dir(prob, opt_model, prefix)
    results = os.path.join(d, "results.npy")
    if not os.path.exists(results):
        return None
    try:
        r = np.load(results, allow_pickle=True)
        regret = np.array(r[1], dtype=float)
        return float(np.mean(regret))
    except Exception:
        return None


def load_manifest(phase):
    path = f"sweep_manifest_p{phase}.json"
    if not os.path.exists(path):
        print(f"Manifest not found: {path}")
        print("Run the submit script with --dry-run first to generate the manifest.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def print_status_grid(manifest, show_vals):
    """
    manifest: list of {"prob": ..., "opt_model": ..., "prefix": ...}
    Rows = methods, Cols = problems.
    """
    problems  = []
    methods   = []
    seen_p = set()
    seen_m = set()
    for entry in manifest:
        p = entry["prob"]
        m = entry["opt_model"]
        if p not in seen_p:
            problems.append(p)
            seen_p.add(p)
        if m not in seen_m:
            methods.append(m)
            seen_m.add(m)

    # Build lookup: (prob, method) -> list of prefix entries
    table = {}
    for entry in manifest:
        key = (entry["prob"], entry["opt_model"])
        table.setdefault(key, []).append(entry["prefix"])

    col_w = 14
    prob_w = 12

    # Header
    header = f"  {'method':16s}  " + "  ".join(f"{p[:col_w]:>{col_w}}" for p in problems)
    print(header)
    print("  " + "-" * (18 + (col_w + 2) * len(problems)))

    n_done = n_run = n_miss = 0

    for method in methods:
        row_cells = []
        for prob in problems:
            prefixes = table.get((prob, method), [])
            if not prefixes:
                row_cells.append(f"{'N/A':>{col_w}}")
                continue

            statuses = [cell_status(prob, method, pfx) for pfx in prefixes]
            n_total = len(statuses)
            n_d = statuses.count("done")
            n_r = statuses.count("running")

            n_done += n_d
            n_run  += n_r
            n_miss += statuses.count("missing")

            if show_vals and n_d > 0:
                # Show best regret among completed entries
                regrets = [load_regret(prob, method, pfx) for pfx in prefixes]
                regrets = [r for r in regrets if r is not None]
                best = min(regrets)
                tag = f"{best:.4f}({n_d}/{n_total})"
            else:
                if n_d == n_total:
                    tag = f"✓({n_d})"
                elif n_d > 0:
                    tag = f"✓{n_d}/R{n_r}/{n_total}"
                elif n_r > 0:
                    tag = f"R({n_r}/{n_total})"
                else:
                    tag = f"--({n_total})"

            row_cells.append(f"{tag:>{col_w}}")

        print(f"  {method:16s}  " + "  ".join(row_cells))

    print()
    n_total_all = n_done + n_run + n_miss
    print(f"  Total: {n_total_all}  ✓ done={n_done}  R in-progress={n_run}  -- missing={n_miss}")


def main():
    parser = argparse.ArgumentParser(description="Sweep status dashboard")
    parser.add_argument("--phase", type=int, default=None,
                        help="Phase number (1 or 2); loads sweep_manifest_p{N}.json")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to manifest JSON (overrides --phase)")
    parser.add_argument("--vals", action="store_true",
                        help="Show best test regret per cell instead of status codes")
    args = parser.parse_args()

    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        label = os.path.basename(args.manifest)
    elif args.phase:
        manifest = load_manifest(args.phase)
        label = f"Phase {args.phase}"
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\n=== Sweep status: {label}  ({'regret' if args.vals else 'status codes'}) ===\n")
    print_status_grid(manifest, args.vals)


if __name__ == "__main__":
    main()
