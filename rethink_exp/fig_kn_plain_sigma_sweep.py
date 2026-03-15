"""
Step 0 figure: plain perturbed sigma sweep on knapsack (benchmark scale).

Reads results from:
  saved_records/knapsack-gen/perturb/kn_plain_sigma_sweep_{sigma}_lr{lr}/

Produces two panels:
  Panel 1: val regret (normalised) vs epoch, one curve per sigma × LR combination
  Panel 2: best test regret vs sigma, one line per LR — heatmap-style

Also prints a summary table of test regrets.

Usage:
  python rethink_exp/fig_kn_plain_sigma_sweep.py
  python rethink_exp/fig_kn_plain_sigma_sweep.py --out my_fig.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Config ----
BASE_DIR = "saved_records/knapsack-gen/perturb"
OUT_PATH = os.path.join(BASE_DIR, "fig_kn_plain_sigma_sweep.png")

SIGMA_LABELS = ["s01", "s05", "s1", "s5", "s10"]
SIGMA_VALS   = [0.1,  0.5,  1.0, 5.0, 10.0]
LRS          = ["5e-2", "1e-2", "5e-3"]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # one per LR
MARKERS = ["o", "s", "^", "D", "v"]          # one per sigma

MSE_BASELINE = 0.0640  # normalized test regret


def load_val_regret(prefix):
    """Load normalised val regret per epoch from val_logs.csv."""
    path = os.path.join(BASE_DIR, prefix, "val_logs.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    # eval = absolute regret, obj = achieved objective
    # normalized regret = eval / (obj + eval) since opt = obj + eval
    abs_reg = df["eval"].values
    achieved = df["obj"].values
    opt_approx = achieved + abs_reg
    norm_reg = abs_reg / (opt_approx + 1e-8)
    epochs = np.arange(1, len(norm_reg) + 1)
    return epochs, norm_reg


def load_test_regret(prefix):
    """Load normalised test regret from results.npy."""
    path = os.path.join(BASE_DIR, prefix, "results.npy")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    opt, ev = d[0], d[1]
    return float(np.mean(ev / (opt + 1e-8)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    # ---- Collect data ----
    # test_regret[sigma_idx][lr_idx]
    test_regret = np.full((len(SIGMA_LABELS), len(LRS)), np.nan)

    print("=" * 60)
    print(f"{'Prefix':<45}  {'Test regret':>12}")
    print("=" * 60)

    for j, lr in enumerate(LRS):
        for i, sigma_label in enumerate(SIGMA_LABELS):
            prefix = f"kn_plain_sigma_sweep_{sigma_label}_lr{lr}"
            r = load_test_regret(prefix)
            if r is not None:
                test_regret[i, j] = r
                print(f"  {prefix:<43}  {r:.4f}")
            else:
                print(f"  {prefix:<43}  MISSING")
    print("=" * 60)

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Knapsack: plain perturbed sigma sweep (benchmark scale)\n"
        "Dense model, DP solver, n_samples=100, 300 epochs, no MSE pretraining",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: val regret vs epoch (best LR per sigma for clarity)
    ax1.set_title("Val regret vs epoch (best LR per sigma)", fontsize=11)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Normalised val regret", fontsize=11)
    ax1.axhline(MSE_BASELINE, color="k", ls="--", lw=1.5, label=f"MSE baseline ({MSE_BASELINE:.3f})")

    for i, (sigma_label, sigma_val) in enumerate(zip(SIGMA_LABELS, SIGMA_VALS)):
        # Find best LR for this sigma (lowest min test regret)
        best_lr_idx = 0
        best_r = np.inf
        for j, lr in enumerate(LRS):
            r = test_regret[i, j]
            if np.isfinite(r) and r < best_r:
                best_r = r
                best_lr_idx = j

        lr = LRS[best_lr_idx]
        prefix = f"kn_plain_sigma_sweep_{sigma_label}_lr{lr}"
        epochs, norm_reg = load_val_regret(prefix)
        if epochs is not None:
            ax1.plot(epochs, norm_reg, color=COLORS[best_lr_idx],
                     label=f"σ={sigma_val} (lr={lr})", alpha=0.85, lw=1.5)

    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Panel 2: test regret vs sigma, one line per LR
    ax2.set_title("Best test regret vs sigma (by LR)", fontsize=11)
    ax2.set_xlabel("Sigma", fontsize=11)
    ax2.set_ylabel("Normalised test regret", fontsize=11)
    ax2.axhline(MSE_BASELINE, color="k", ls="--", lw=1.5, label=f"MSE baseline ({MSE_BASELINE:.3f})")

    for j, lr in enumerate(LRS):
        vals = test_regret[:, j]
        ax2.plot(SIGMA_VALS, vals, color=COLORS[j], marker=MARKERS[j],
                 label=f"lr={lr}", lw=2, ms=7, alpha=0.9)

    ax2.set_xscale("log")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Print best overall
    best_idx = np.unravel_index(np.nanargmin(test_regret), test_regret.shape)
    best_sigma = SIGMA_VALS[best_idx[0]]
    best_lr = LRS[best_idx[1]]
    best_r = test_regret[best_idx]
    print(f"\nBest: sigma={best_sigma}  lr={best_lr}  test_regret={best_r:.4f}")
    print(f"  → use YAML: openpto/config/models/perturb_kn_s{SIGMA_LABELS[best_idx[0]]}_n100.yaml")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {args.out}")


if __name__ == "__main__":
    main()
