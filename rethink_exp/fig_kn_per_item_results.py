"""
Visualize per-item sigma sweep results for knapsack.

3-panel figure:
  1. Test regret vs hamming target (4 variant curves + MSE baseline)
  2. Per-item sigma bar chart (final sigma_vec/sigma_mat values per item)
  3. Sigma_item_std / sigma_item_mean ratio over training (stability metric)

Variants:
  A: --per_item               sigma_min=1e-4   stem=pitem_t
  B: --per_item --no_sigma_min sigma_min=None  stem=pitem_t
  C: --per_inst_item           sigma_min=1e-4  stem=piitem_t
  D: --per_inst_item --no_sigma_min             stem=piitem_t

Usage:
    python rethink_exp/fig_kn_per_item_results.py
    python rethink_exp/fig_kn_per_item_results.py --lr 1e-2
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

# ---- Configuration ----
BASE_DIR = "saved_records/knapsack-gen/perturb_adaptive_sweep"
OUT_DIR  = "saved_records/knapsack-gen/oracle_ceiling"
OUT_PATH = os.path.join(OUT_DIR, "fig_kn_per_item_results.png")

MSE_BASELINE = 0.0640

VARIANTS = [
    # (label, prefix_template, stem, style)
    ("A: per-item σ_min=1e-4",   "kn_per_item_A_lr{lr}", "pitem_t",  "-",  "C0"),
    ("B: per-item σ_min=None",   "kn_per_item_B_lr{lr}", "pitem_t",  "--", "C1"),
    ("C: per-inst-item σ_min=1e-4", "kn_per_item_C_lr{lr}", "piitem_t", "-",  "C2"),
    ("D: per-inst-item σ_min=None", "kn_per_item_D_lr{lr}", "piitem_t", "--", "C3"),
]

TARGETS  = [0.05, 0.10, 0.20]
N_ITEMS  = 20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr",  type=str, default="1e-2",
                   help="LR to plot (default 1e-2, best from prior sweeps).")
    p.add_argument("--out", type=str, default=OUT_PATH)
    return p.parse_args()


def load_npz(base_dir, prefix, stem, target):
    """Load npz for a given (prefix, stem, target). Returns dict or None."""
    path = os.path.join(base_dir, prefix, f"{stem}{target:.3f}_s1_dense.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Per-item σ sweep — knapsack  (LR={args.lr}, DP n=100, 300 epochs)\n"
        f"MSE baseline = {MSE_BASELINE:.4f}",
        fontsize=11, fontweight="bold",
    )

    # ---- Panel 1: test regret vs target ----
    ax = axes[0]
    for label, prefix_tmpl, stem, ls, color in VARIANTS:
        prefix  = prefix_tmpl.format(lr=args.lr)
        regrets = []
        valid_targets = []
        for t in TARGETS:
            data = load_npz(BASE_DIR, prefix, stem, t)
            if data is None:
                continue
            regrets.append(float(data["test_regret"]))
            valid_targets.append(t)
        if regrets:
            ax.plot(valid_targets, regrets, ls=ls, color=color, marker="o", label=label)
        else:
            print(f"[MISSING] {prefix}/{stem}*.npz — skipping {label}")

    ax.axhline(MSE_BASELINE, color="black", ls=":", lw=1.5, label=f"MSE baseline ({MSE_BASELINE:.4f})")
    ax.set_xlabel("Hamming target", fontsize=10)
    ax.set_ylabel("Test regret", fontsize=10)
    ax.set_title("Test regret vs target", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: per-item sigma bar chart at best target ----
    ax = axes[1]
    bar_width = 0.18
    item_idx  = np.arange(N_ITEMS)

    for vi, (label, prefix_tmpl, stem, ls, color) in enumerate(VARIANTS):
        prefix = prefix_tmpl.format(lr=args.lr)
        # Find best target (min test_regret)
        best_data, best_t = None, None
        for t in TARGETS:
            data = load_npz(BASE_DIR, prefix, stem, t)
            if data is None:
                continue
            if best_data is None or float(data["test_regret"]) < float(best_data["test_regret"]):
                best_data, best_t = data, t

        if best_data is None:
            continue

        # Extract final sigma_vec values (last epoch where it's non-nan)
        sigma_mean = best_data.get("sigma_item_mean", None)
        if sigma_mean is None:
            continue

        # sigma_item_mean is a scalar per epoch; need per-item from the npz
        # Per-item sigma is stored indirectly — check if sigma_vec_item is present
        # (For now we use sigma_item_mean as a proxy; future runs may store full vec)
        # If sigma_vec_item not stored, skip the bar chart
        if "sigma_vec_item" in best_data:
            # (n_epochs, D) array
            sv = best_data["sigma_vec_item"][-1]  # last epoch
        else:
            print(f"  [{label}] sigma_vec_item not stored in npz — skipping bar chart")
            continue

        offset = (vi - 1.5) * bar_width
        ax.bar(item_idx + offset, sv, width=bar_width, color=color, alpha=0.75,
               label=f"{label} (t={best_t:.2f})")

    ax.set_xlabel("Item index", fontsize=10)
    ax.set_ylabel("σ (final)", fontsize=10)
    ax.set_title("Per-item σ at best target (final epoch)", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # ---- Panel 3: sigma_item_std / mean ratio over training ----
    ax = axes[2]
    for label, prefix_tmpl, stem, ls, color in VARIANTS:
        prefix = prefix_tmpl.format(lr=args.lr)
        # Pick best target
        best_data, best_t = None, None
        for t in TARGETS:
            data = load_npz(BASE_DIR, prefix, stem, t)
            if data is None:
                continue
            if best_data is None or float(data["test_regret"]) < float(best_data["test_regret"]):
                best_data, best_t = data, t

        if best_data is None:
            continue

        sig_mean = best_data.get("sigma_item_mean")
        sig_std  = best_data.get("sigma_item_std")
        if sig_mean is None or sig_std is None:
            continue

        # Avoid division by zero; mask epochs where mean~0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sig_mean > 1e-10, sig_std / sig_mean, np.nan)

        epochs = best_data.get("epoch", np.arange(1, len(ratio) + 1))
        ax.plot(epochs, ratio, ls=ls, color=color, lw=1.5,
                label=f"{label} (t={best_t:.2f})")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("σ_item_std / σ_item_mean", fontsize=10)
    ax.set_title("Item-σ spread ratio over training\n(0 = uniform, >0 = differentiated)", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {args.out}")


if __name__ == "__main__":
    main()
