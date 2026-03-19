"""
Training dynamics for per-item sigma controllers.

4-panel figure comparing per-item variants (A: per-item, C: per-inst-item)
across Hamming targets, with a per-instance reference curve on every panel.

Panels:
  1. Val regret over training  (vs per-instance reference)
  2. Sigma trajectory: mean ± (min, max) band — shows ballooning
  3. Per-item Hamming rate: mean ± std band, with target reference line
  4. Sigma spread ratio (std/mean) + coeff_norm on right axis

All data loaded from existing .npz files — no re-run required.

Usage:
    python rethink_exp/fig_kn_per_item_dynamics.py
    python rethink_exp/fig_kn_per_item_dynamics.py --lr 5e-2 --out my.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

# ---- Paths ----
BASE   = "saved_records/knapsack-gen/perturb_adaptive_sweep"
OUT_DIR = "saved_records/knapsack-gen/oracle_ceiling"
OUT_PATH = os.path.join(OUT_DIR, "fig_kn_per_item_dynamics.png")

MSE_BASELINE  = 0.0640
ORACLE_REGRET = 0.0519


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr",  default="1e-2")
    p.add_argument("--out", default=OUT_PATH)
    return p.parse_args()


def load(prefix, fname):
    path = os.path.join(BASE, prefix, fname)
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    lr = args.lr

    # ---- Define runs ----
    TARGETS = ["0.050", "0.100", "0.200"]
    TARGET_COLORS = {"0.050": "C0", "0.100": "C1", "0.200": "C2"}

    VARIANTS = [
        # (label, prefix_template, stem, linestyle)
        ("A: per-item",          f"kn_per_item_A_lr{lr}", "pitem_t",  "-"),
        ("C: per-inst-item",     f"kn_per_item_C_lr{lr}", "piitem_t", "--"),
    ]

    # Per-instance Hamming reference (best: t=0.10, same lr)
    ref_data = load(f"kn_bench_hamming_lr{lr}", "ham_t0.100_s1_dense.npz")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Per-item σ controller — training dynamics  (lr={lr}, DP n=100, 300 epochs)\n"
        f"Comparing variants A (per-item) and C (per-inst-item) across Hamming targets",
        fontsize=12, fontweight="bold",
    )
    ax_regret, ax_sigma, ax_hamming, ax_spread = axes.flat

    # ---- Panel 1: Val regret ----
    ax = ax_regret
    if ref_data is not None:
        ax.plot(ref_data["epoch"], ref_data["val_regret"],
                color="black", lw=2, ls=":", label="Per-instance ref (t=0.10)")

    for var_label, prefix, stem, var_ls in VARIANTS:
        for tstr in TARGETS:
            d = load(prefix, f"{stem}{tstr}_s1_dense.npz")
            if d is None:
                continue
            color = TARGET_COLORS[tstr]
            t_float = float(tstr)
            ax.plot(d["epoch"], d["val_regret"],
                    color=color, ls=var_ls, lw=1.5,
                    label=f"{var_label}  t={t_float:.2f}")

    ax.axhline(MSE_BASELINE,  color="gray",    ls=":", lw=1.2, label=f"MSE ({MSE_BASELINE:.3f})")
    ax.axhline(ORACLE_REGRET, color="#333333", ls=":", lw=1.2, label=f"Oracle ({ORACLE_REGRET:.3f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val regret")
    ax.set_title("Val regret over training")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Sigma trajectory (mean + min/max band) ----
    ax = ax_sigma
    if ref_data is not None:
        sig = ref_data["adaptive_sigma"]
        ax.plot(ref_data["epoch"], sig,
                color="black", lw=2, ls=":", label="Per-instance ref (σ scalar)")

    for var_label, prefix, stem, var_ls in VARIANTS:
        for tstr in TARGETS:
            d = load(prefix, f"{stem}{tstr}_s1_dense.npz")
            if d is None or "sigma_item_mean" not in d:
                continue
            color = TARGET_COLORS[tstr]
            ep  = d["epoch"]
            mu  = d["sigma_item_mean"]
            lo  = d["sigma_item_min"]
            hi  = d["sigma_item_max"]
            t_float = float(tstr)
            ax.fill_between(ep, lo, hi, color=color, alpha=0.12)
            ax.plot(ep, mu,  color=color, ls=var_ls, lw=1.5,
                    label=f"{var_label}  t={t_float:.2f}  (mean)")
            ax.plot(ep, hi,  color=color, ls=var_ls, lw=0.7, alpha=0.6)
            ax.plot(ep, lo,  color=color, ls=var_ls, lw=0.7, alpha=0.6)

    ax.axhline(1e-4, color="red",   ls=":", lw=1, alpha=0.7, label="σ_min floor (1e-4)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("σ (item)")
    ax.set_title("Sigma trajectory\n(band = item min/max; line = item mean)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Per-item Hamming rate (mean ± std) ----
    ax = ax_hamming
    if ref_data is not None:
        ax.plot(ref_data["epoch"], ref_data["hamming"],
                color="black", lw=2, ls=":", label="Per-instance ref (global)")

    plotted_targets = set()
    for var_label, prefix, stem, var_ls in VARIANTS:
        for tstr in TARGETS:
            d = load(prefix, f"{stem}{tstr}_s1_dense.npz")
            if d is None or "hamming_item_mean" not in d:
                continue
            color = TARGET_COLORS[tstr]
            ep   = d["epoch"]
            mu   = d["hamming_item_mean"]
            std  = d["hamming_item_std"]
            t_float = float(tstr)

            ax.fill_between(ep, (mu - std).clip(0), mu + std,
                            color=color, alpha=0.12)
            ax.plot(ep, mu, color=color, ls=var_ls, lw=1.5,
                    label=f"{var_label}  t={t_float:.2f}")

            # Draw target line once per target value
            if tstr not in plotted_targets:
                ax.axhline(t_float, color=color, ls=":", lw=1.0, alpha=0.8)
                plotted_targets.add(tstr)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Hamming rate (per-item mean)")
    ax.set_title("Per-item Hamming rate over training\n(band = mean ± std; dotted = target)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ---- Panel 4: Sigma spread ratio + coeff_norm ----
    ax  = ax_spread
    ax2 = ax.twinx()

    # coeff_norm for one representative run (same across variants, same training data)
    coeff_plotted = False
    for var_label, prefix, stem, var_ls in VARIANTS:
        for tstr in TARGETS:
            d = load(prefix, f"{stem}{tstr}_s1_dense.npz")
            if d is None or "sigma_item_mean" not in d:
                continue
            color  = TARGET_COLORS[tstr]
            ep     = d["epoch"]
            mu     = d["sigma_item_mean"]
            std    = d["sigma_item_std"]
            t_float = float(tstr)

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(mu > 1e-10, std / mu, np.nan)

            ax.plot(ep, ratio, color=color, ls=var_ls, lw=1.5,
                    label=f"{var_label}  t={t_float:.2f}")

            # Plot coeff_norm on right axis once (all runs share the same predictor init)
            if not coeff_plotted and "coeff_norm" in d:
                ax2.plot(ep, d["coeff_norm"], color="gray", lw=1.2, ls="-",
                         alpha=0.6, label="coeff_norm (right)")
                coeff_plotted = True

    if ref_data is not None and "coeff_norm" in ref_data:
        ax2.plot(ref_data["epoch"], ref_data["coeff_norm"],
                 color="black", lw=1.2, ls=":", alpha=0.6,
                 label="coeff_norm ref (right)")

    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("σ_item_std / σ_item_mean  (spread ratio)", color="black")
    ax2.set_ylabel("coeff_norm", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.set_title("Sigma spread ratio over training\n"
                 "(0 = uniform σ across items;  growing = divergence)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {args.out}")

    # ---- Print summary table ----
    print("\n--- Summary: final sigma spread and val_regret ---")
    print(f"{'Run':<38} {'σ_mean':>7} {'σ_max':>7} {'σ_min':>8} {'spread':>7} {'val_regret':>11}")
    print("-" * 82)
    if ref_data is not None:
        sig = ref_data["adaptive_sigma"]
        print(f"{'Per-instance ref  t=0.10':<38} "
              f"{sig[-1]:>7.3f} {sig.max():>7.3f} {'—':>8} {'—':>7} "
              f"{ref_data['val_regret'][-1]:>11.4f}")
    for var_label, prefix, stem, _ in VARIANTS:
        for tstr in TARGETS:
            d = load(prefix, f"{stem}{tstr}_s1_dense.npz")
            if d is None or "sigma_item_mean" not in d:
                continue
            mu  = d["sigma_item_mean"][-1]
            hi  = d["sigma_item_max"][-1]
            lo  = d["sigma_item_min"][-1]
            std = d["sigma_item_std"][-1]
            spread = std / mu if mu > 0 else float("nan")
            print(f"{var_label+' t='+tstr:<38} "
                  f"{mu:>7.3f} {hi:>7.3f} {lo:>8.5f} {spread:>7.3f} "
                  f"{d['val_regret'][-1]:>11.4f}")


if __name__ == "__main__":
    main()
