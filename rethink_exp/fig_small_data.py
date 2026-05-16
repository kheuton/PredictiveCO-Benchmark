"""
fig_small_data.py
-----------------
Small-data sweep figure (Committee Plan #4).

For each problem, plots test regret vs. train size with one line per method.
The point of the plot: do decision-aware methods (perturb, lodl, dad) degrade
faster than MSE as train size shrinks?

Inputs: saved_records/.../small_n<size>_<method>_<batch>_lr<lr>/results.npy
  - Prefixes are written by shells/slurm/submit_small_data.sh.
  - Best (lr, batch) per (method, problem) comes from bench_p1_best.json.

Output:
  results/fig_small_data.{png,pdf}  — 3-panel grid (one per problem)

Usage:
    python rethink_exp/fig_small_data.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

# ---- Config ----
RESULTS_ROOT = "saved_records"
BEST_JSON    = "bench_p1_best.json"
OUT_DIR      = "results"

PROBLEMS = ["knapsack", "sp_synth", "cook_county"]

PROB_ARG = {
    "knapsack": "knapsack", "sp_synth": "sp_synth", "cook_county": "cook_county",
}
PROB_VERSION = {
    "knapsack": "gen", "sp_synth": "synth", "cook_county": "real",
}
PROB_DISPLAY = {
    "knapsack":    "Knapsack (synthetic, linear CO)",
    "sp_synth":    "SP-synth (linear CO, 1-layer linear head, mis-spec)",
    "cook_county": "Cook County (real, TopK, time-series)",
}
PROB_XLABEL = {
    "knapsack":    "Train size (instances)",
    "sp_synth":    "Train size (instances)",
    "cook_county": "Train size (years)",
}

SIZES = {
    "knapsack":    [50, 100, 200, 320],
    "sp_synth":    [50, 100, 200, 320],
    "cook_county": [1, 2, 3, 4],
}

METHODS = ["mse", "dfl", "perturb", "lodl", "dad"]

METHOD_STYLE = {
    "mse":     {"label": "MSE",     "color": "#1f77b4", "marker": "o", "lw": 2.0},
    "dfl":     {"label": "DFL",     "color": "#2ca02c", "marker": "s", "lw": 1.5},
    "perturb": {"label": "Perturb", "color": "#d62728", "marker": "h", "lw": 1.5},
    "lodl":    {"label": "LODL",    "color": "#9467bd", "marker": "D", "lw": 1.5},
    "dad":     {"label": "DAD",     "color": "#e67e22", "marker": "^", "lw": 1.5},
}

USE_ABSOLUTE = set()  # none of our 3 problems are absolute-regret


def load_best_json():
    if not os.path.exists(BEST_JSON):
        raise FileNotFoundError(f"{BEST_JSON} not found; run collect_bench_p1.py first.")
    with open(BEST_JSON) as f:
        return json.load(f)


def run_dir(prob, method, prefix):
    return os.path.join(RESULTS_ROOT,
                        f"{PROB_ARG[prob]}-{PROB_VERSION[prob]}",
                        method, prefix)


def read_test_regret(prob, method, prefix):
    d = run_dir(prob, method, prefix)
    p = os.path.join(d, "results.npy")
    if not os.path.exists(p):
        return np.nan
    try:
        r = np.load(p, allow_pickle=True)
        regret = float(np.mean(np.array(r[1], dtype=float)))
        opt    = float(np.mean(np.abs(np.array(r[0], dtype=float))))
        if prob in USE_ABSOLUTE:
            return regret
        return regret / opt if opt > 0 else np.nan
    except Exception as e:
        print(f"  [warn] {prob}/{method}/{prefix}: {e}")
        return np.nan


def main():
    best = load_best_json()
    os.makedirs(OUT_DIR, exist_ok=True)

    # regrets[prob][method] = array of length len(SIZES[prob])
    regrets = {p: {m: np.full(len(SIZES[p]), np.nan) for m in METHODS}
               for p in PROBLEMS}

    for prob in PROBLEMS:
        for method in METHODS:
            cfg = best.get(method, {}).get(prob)
            if cfg is None:
                continue
            lr    = cfg["lr"]
            batch = cfg["batch"]
            for i, n in enumerate(SIZES[prob]):
                prefix = f"small_n{n}_{method}_{batch}_lr{lr}"
                regrets[prob][method][i] = read_test_regret(prob, method, prefix)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for col, prob in enumerate(PROBLEMS):
        ax = axes[col]
        sizes = SIZES[prob]
        for method in METHODS:
            ys = regrets[prob][method]
            finite = np.isfinite(ys)
            if not finite.any():
                continue
            st = METHOD_STYLE[method]
            ax.plot(np.array(sizes)[finite], ys[finite],
                    color=st["color"], marker=st["marker"],
                    linewidth=st["lw"], markersize=7,
                    label=st["label"], markeredgecolor="white",
                    markeredgewidth=0.6)
            # mark missing with an "×" at the size's x-coord
            for n, y, f in zip(sizes, ys, finite):
                if not f:
                    ax.plot(n, 0, "x", color=st["color"], markersize=8,
                            alpha=0.5, transform=ax.get_xaxis_transform(),
                            clip_on=False)

        ax.set_xscale("log" if prob != "cook_county" else "linear")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel(PROB_XLABEL[prob], fontsize=10)
        if col == 0:
            ax.set_ylabel("Test regret  (relative)", fontsize=10)
        ax.set_title(PROB_DISPLAY[prob], fontsize=10.5, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    fig.suptitle("Small-data sweep: test regret vs. train size "
                 "(Phase-1-best LR/batch per method, seed=2023)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"fig_small_data.{ext}")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)

    # ---- Summary printout ----
    print("\n=== Summary table (relative test regret) ===")
    for prob in PROBLEMS:
        print(f"\n-- {prob} --")
        hdr = "method   " + " ".join(f"n={n:>4}" for n in SIZES[prob])
        print(hdr)
        for method in METHODS:
            ys = regrets[prob][method]
            row = METHOD_STYLE[method]["label"].ljust(8)
            for y in ys:
                row += f" {('   nan' if not np.isfinite(y) else f'{y:.4f}'):>7}"
            print(row)


if __name__ == "__main__":
    main()
