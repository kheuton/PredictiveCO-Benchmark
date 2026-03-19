"""
Per-item sigma heatmap — epoch × item grid.

Loads sigma_vec_item (n_epochs, D) and hamming_vec_item (n_epochs, D) from
the diagnostic re-run npz files (kn_per_item_diag_*) and plots:

  Row 1: sigma heatmap  (log10 color)  — variant A (left) vs C (right)
  Row 2: hamming heatmap (linear color) — variant A (left) vs C (right)
  Row 3: summary — sigma_item_max over training (left) | val_regret (right)

Each column = one variant.  Figure is generated per (lr, target).

Usage:
    python rethink_exp/fig_kn_per_item_heatmap.py
    python rethink_exp/fig_kn_per_item_heatmap.py --lr 1e-2 --target 0.100
    python rethink_exp/fig_kn_per_item_heatmap.py --all_targets
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
BASE    = "saved_records/knapsack-gen/perturb_adaptive_sweep"
OUT_DIR = "saved_records/knapsack-gen/oracle_ceiling"

MSE_BASELINE  = 0.0640
ORACLE_REGRET = 0.0519


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr",         default="1e-2")
    p.add_argument("--target",     default="0.100",
                   help="Single hamming target to plot (e.g. 0.100).")
    p.add_argument("--all_targets", action="store_true",
                   help="Generate one figure per target (0.050, 0.100, 0.200).")
    p.add_argument("--prefix_A",   default=None,
                   help="Override prefix for variant A (default: kn_per_item_diag_A_lr{lr}).")
    p.add_argument("--prefix_C",   default=None,
                   help="Override prefix for variant C (default: kn_per_item_diag_C_lr{lr}).")
    p.add_argument("--out",        default=None,
                   help="Output path (default: <OUT_DIR>/fig_kn_per_item_heatmap_t{target}.png).")
    return p.parse_args()


def load(prefix, fname):
    path = os.path.join(BASE, prefix, fname)
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def make_figure(lr, tstr, prefix_A, prefix_C, out_path):
    t_float = float(tstr)

    d_A = load(prefix_A, f"pitem_t{tstr}_s1_dense.npz")
    d_C = load(prefix_C, f"piitem_t{tstr}_s1_dense.npz")

    if d_A is None and d_C is None:
        print(f"  No data for t={tstr} (A prefix={prefix_A}, C prefix={prefix_C}) — skipping.")
        return

    # ---- Figure ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(
        f"Per-item σ heatmap  (lr={lr}, target={t_float:.2f}, DP n=100)\n"
        f"Left: A (per-item σ_vec)   Right: C (per-inst-item σ_mat, mean over instances)",
        fontsize=11, fontweight="bold",
    )

    VARIANTS = [("A: per-item", d_A), ("C: per-inst-item", d_C)]

    # ---- Rows 1 & 2: sigma and hamming heatmaps ----
    for col, (var_label, d) in enumerate(VARIANTS):
        ax_sig = axes[0][col]
        ax_ham = axes[1][col]

        if d is None:
            ax_sig.set_visible(False)
            ax_ham.set_visible(False)
            continue

        epochs = d["epoch"]          # (E,)
        n_epochs = len(epochs)

        # ---- Row 1: sigma heatmap ----
        if "sigma_vec_item" in d:
            sv = d["sigma_vec_item"]   # (E, D)
            D  = sv.shape[1]
            # log10 clipped to reasonable range
            log_sv = np.log10(np.clip(sv, 1e-5, 1e5))
            im = ax_sig.imshow(
                log_sv.T,  # (D, E)
                aspect="auto",
                origin="lower",
                extent=[epochs[0], epochs[-1], -0.5, D - 0.5],
                cmap="RdYlBu_r",
            )
            plt.colorbar(im, ax=ax_sig, label="log₁₀(σ_item)")
            ax_sig.set_xlabel("Epoch")
            ax_sig.set_ylabel("Item index")
            ax_sig.set_title(f"{var_label} — σ per item (log₁₀)\nt={t_float:.2f}")
            ax_sig.set_yticks(range(0, D, max(1, D // 10)))
        else:
            ax_sig.text(0.5, 0.5, "sigma_vec_item not in npz\n(old run — re-run needed)",
                        ha="center", va="center", transform=ax_sig.transAxes)
            ax_sig.set_title(f"{var_label} — σ heatmap (no data)")

        # ---- Row 2: hamming heatmap ----
        if "hamming_vec_item" in d:
            hv = d["hamming_vec_item"]   # (E, D)
            D  = hv.shape[1]
            im2 = ax_ham.imshow(
                hv.T,  # (D, E)
                aspect="auto",
                origin="lower",
                extent=[epochs[0], epochs[-1], -0.5, D - 0.5],
                cmap="viridis",
                vmin=0, vmax=min(0.5, hv.max() + 0.05),
            )
            plt.colorbar(im2, ax=ax_ham, label="Hamming rate")
            # Target line annotation
            ax_ham.set_xlabel("Epoch")
            ax_ham.set_ylabel("Item index")
            ax_ham.set_title(f"{var_label} — Hamming per item\ntarget={t_float:.2f} (dotted lines)")
            ax_ham.set_yticks(range(0, D, max(1, D // 10)))
            # Overlay items that are above/below target
        else:
            ax_ham.text(0.5, 0.5, "hamming_vec_item not in npz\n(old run — re-run needed)",
                        ha="center", va="center", transform=ax_ham.transAxes)
            ax_ham.set_title(f"{var_label} — Hamming heatmap (no data)")

    # ---- Row 3: sigma_item_max + val_regret ----
    ax_max  = axes[2][0]
    ax_reg  = axes[2][1]

    for var_label, d, color, ls in [
        ("A: per-item",      d_A, "C0", "-"),
        ("C: per-inst-item", d_C, "C1", "--"),
    ]:
        if d is None:
            continue
        ep = d["epoch"]
        # sigma_item_max (scalar per epoch)
        if "sigma_item_max" in d:
            ax_max.plot(ep, d["sigma_item_max"], color=color, ls=ls, lw=1.8,
                        label=f"{var_label} σ_max")
        if "sigma_item_mean" in d:
            ax_max.plot(ep, d["sigma_item_mean"], color=color, ls=ls, lw=1.0,
                        alpha=0.5, label=f"{var_label} σ_mean")
        ax_reg.plot(ep, d["val_regret"], color=color, ls=ls, lw=1.8, label=var_label)

    ax_max.set_yscale("log")
    ax_max.set_xlabel("Epoch")
    ax_max.set_ylabel("σ_item (log scale)")
    ax_max.set_title("σ_item_max and σ_item_mean over training\n(ballooning diagnostic)")
    ax_max.legend(fontsize=8)
    ax_max.grid(True, alpha=0.3)
    ax_max.axhline(1e-4, color="red", ls=":", lw=1, alpha=0.7, label="σ_min floor")

    ax_reg.axhline(MSE_BASELINE,  color="gray",    ls=":", lw=1.2, label=f"MSE ({MSE_BASELINE:.3f})")
    ax_reg.axhline(ORACLE_REGRET, color="#333333", ls=":", lw=1.2, label=f"Oracle ({ORACLE_REGRET:.3f})")
    ax_reg.set_xlabel("Epoch")
    ax_reg.set_ylabel("Val regret")
    ax_reg.set_title("Val regret over training")
    ax_reg.legend(fontsize=8)
    ax_reg.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # ---- Print per-item summary at final epoch ----
    print(f"\n--- Per-item sigma summary at final epoch (t={tstr}) ---")
    for var_label, d in VARIANTS:
        if d is None or "sigma_vec_item" not in d:
            continue
        sv_final = d["sigma_vec_item"][-1]   # (D,)
        hv_final = d["hamming_vec_item"][-1] if "hamming_vec_item" in d else None
        print(f"\n{var_label}:")
        print(f"  σ_item: mean={sv_final.mean():.4f}  std={sv_final.std():.4f}  "
              f"min={sv_final.min():.5f}  max={sv_final.max():.3f}")
        if hv_final is not None:
            print(f"  ham:    mean={hv_final.mean():.4f}  std={hv_final.std():.4f}  "
                  f"min={hv_final.min():.4f}  max={hv_final.max():.4f}")
        # Top-5 highest-sigma items
        top5 = np.argsort(sv_final)[::-1][:5]
        print(f"  Top-5 σ items: " + ", ".join(f"item{i}={sv_final[i]:.3f}" for i in top5))
        if hv_final is not None:
            bot5 = np.argsort(hv_final)[:5]
            print(f"  Bottom-5 hamming items: " + ", ".join(
                f"item{i}={hv_final[i]:.4f}" for i in bot5))


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    lr = args.lr

    prefix_A = args.prefix_A or f"kn_per_item_diag_A_lr{lr}"
    prefix_C = args.prefix_C or f"kn_per_item_diag_C_lr{lr}"

    targets = ["0.050", "0.100", "0.200"] if args.all_targets else [args.target]

    for tstr in targets:
        out = args.out or os.path.join(
            OUT_DIR, f"fig_kn_per_item_heatmap_t{tstr}_lr{lr}.png"
        )
        make_figure(lr, tstr, prefix_A, prefix_C, out)


if __name__ == "__main__":
    main()
