"""
Exp 4 figure: LR schedule ablation (step decay vs constant).

Reads results from:
  saved_records/knapsack-gen/perturb/kn_plain_lrsched_{nosched,sched}_lr{lr}/

Produces a 2×2 figure:
  Row 0 (no-sched): val regret vs epoch (col 0) + bar chart by LR (col 1)
  Row 1 (sched):    val regret vs epoch (col 0) + bar chart by LR (col 1)

Also prints summary table.

Usage:
  python rethink_exp/fig_kn_lrsched_sweep.py
  python rethink_exp/fig_kn_lrsched_sweep.py --out my_fig.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = "saved_records/knapsack-gen/perturb"
OUT_PATH = os.path.join(BASE_DIR, "fig_kn_lrsched_sweep.png")

LRS    = ["5e-2", "1e-2", "5e-3"]
SCHEDS = [("nosched", "Constant LR"), ("sched", "Step decay (ep100, ep200)")]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

MSE_BASELINE  = 0.0640
COLD_BASELINE = 0.085


def load_val_regret(prefix):
    path = os.path.join(BASE_DIR, prefix, "val_logs.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    abs_reg = df["eval"].values
    achieved = df["obj"].values
    norm_reg = abs_reg / (achieved + abs_reg + 1e-8)
    epochs = np.arange(1, len(norm_reg) + 1)
    return epochs, norm_reg


def load_test_regret(prefix):
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

    # ---- Collect ----
    # test_regret[sched_idx][lr_idx]
    test_regret = np.full((len(SCHEDS), len(LRS)), np.nan)

    print("=" * 65)
    print(f"{'Prefix':<52}  {'Test regret':>12}")
    print("=" * 65)
    for s_idx, (sched_tag, _) in enumerate(SCHEDS):
        for j, lr in enumerate(LRS):
            prefix = f"kn_plain_lrsched_{sched_tag}_lr{lr}"
            r = load_test_regret(prefix)
            if r is not None:
                test_regret[s_idx, j] = r
                print(f"  {prefix:<50}  {r:.4f}")
            else:
                print(f"  {prefix:<50}  MISSING")
    print("=" * 65)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Knapsack: LR schedule ablation (constant vs step decay)\n"
        "Dense model, DP solver, n_samples=100, 300 epochs",
        fontsize=12, fontweight="bold",
    )

    for s_idx, (sched_tag, sched_name) in enumerate(SCHEDS):
        ax_curve = axes[s_idx, 0]
        ax_bar   = axes[s_idx, 1]

        # Left panel: val regret vs epoch
        ax_curve.set_title(f"{sched_name} — val regret vs epoch", fontsize=10)
        ax_curve.set_xlabel("Epoch", fontsize=10)
        ax_curve.set_ylabel("Normalised val regret", fontsize=10)
        ax_curve.axhline(MSE_BASELINE, color="k", ls="--", lw=1.5,
                         label=f"MSE baseline ({MSE_BASELINE:.3f})")
        ax_curve.axhline(COLD_BASELINE, color="gray", ls=":", lw=1.2,
                         label=f"Cold-start (~{COLD_BASELINE:.3f})")

        for j, lr in enumerate(LRS):
            prefix = f"kn_plain_lrsched_{sched_tag}_lr{lr}"
            epochs, norm_reg = load_val_regret(prefix)
            if epochs is not None:
                ax_curve.plot(epochs, norm_reg, color=COLORS[j],
                              label=f"lr={lr}", lw=1.5, alpha=0.9)

        ax_curve.legend(fontsize=8, loc="upper right")
        ax_curve.set_yscale("log")
        ax_curve.grid(True, alpha=0.3)

        # Right panel: bar chart of test regret by LR
        ax_bar.set_title(f"{sched_name} — test regret by LR", fontsize=10)
        ax_bar.set_xlabel("Learning rate", fontsize=10)
        ax_bar.set_ylabel("Normalised test regret", fontsize=10)
        ax_bar.axhline(MSE_BASELINE, color="k", ls="--", lw=1.5,
                       label=f"MSE baseline ({MSE_BASELINE:.3f})")
        ax_bar.axhline(COLD_BASELINE, color="gray", ls=":", lw=1.2,
                       label=f"Cold-start (~{COLD_BASELINE:.3f})")

        vals = test_regret[s_idx, :]
        bars = ax_bar.bar(LRS, vals, color=COLORS, alpha=0.8, edgecolor="k")
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax_bar.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax_bar.legend(fontsize=8)
        ax_bar.grid(True, axis="y", alpha=0.3)

    # Summary
    for s_idx, (_, sched_name) in enumerate(SCHEDS):
        best_j = np.nanargmin(test_regret[s_idx])
        print(f"  {sched_name}: best test_regret={test_regret[s_idx, best_j]:.4f} "
              f"at lr={LRS[best_j]}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {args.out}")


if __name__ == "__main__":
    main()
