"""
Training dynamics for the capped Hamming-Star sweep.

Layout: 2 figures (dense, poly), each with
  rows = 3 LRs (5e-2, 1e-2, 5e-3)
  cols = val_regret | sigma (mean) | hamming (mean) | frac_improving
  lines = 4 cap values, coloured

Fixed-target Hamming baseline (ham_t{cap}_s1 at same LR) added as a dashed
line on the val_regret panel where available, for direct comparison.

Saved to saved_records/knapsack-gen/perturb_adaptive_sweep/fig_kn_hstar_dynamics_{model}.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Config ---------------------------------------------------------------
BASE  = "saved_records/knapsack-gen/perturb_adaptive_sweep"
LRS   = ["5e-2", "1e-2", "5e-3"]
CAPS  = ["0.050", "0.100", "0.150", "0.200"]
MODELS = ["dense", "poly"]

CAP_COLORS = {
    "0.050": "#1f77b4",
    "0.100": "#ff7f0e",
    "0.150": "#2ca02c",
    "0.200": "#d62728",
}

MSE_BEST = 0.0640   # benchmark reference

# ---- Helpers --------------------------------------------------------------

def load(lr, cap, model):
    path = f"{BASE}/kn_hstar_lr{lr}/hams_s{cap}_s1_{model}.npz"
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)

def load_fixed(lr, cap, model):
    """Fixed-target Hamming baseline at matching (lr, cap) if it exists."""
    path = f"{BASE}/kn_bench_hamming_lr{lr}/ham_t{cap}_s1_{model}.npz"
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)

# ---- Plot -----------------------------------------------------------------

COLS = [
    ("val_regret",    "Val regret"),
    ("adaptive_sigma","Sigma (mean)"),
    ("hamming",       "Hamming rate (mean)"),
    ("frac_improving","Frac improving"),
]

for model in MODELS:
    fig, axes = plt.subplots(
        len(LRS), len(COLS),
        figsize=(16, 3.5 * len(LRS)),
        sharex=True,
    )
    fig.suptitle(
        f"Hamming-Star (capped) — {model} — training dynamics",
        fontsize=13, y=1.01,
    )

    for row, lr in enumerate(LRS):
        for col, (key, ylabel) in enumerate(COLS):
            ax = axes[row, col]

            for cap in CAPS:
                d = load(lr, cap, model)
                if d is None:
                    continue
                epochs = d["epoch"]
                ax.plot(
                    epochs, d[key],
                    color=CAP_COLORS[cap],
                    lw=1.5,
                    label=f"cap={cap}",
                )

                # Dashed fixed-target baseline on val_regret panel
                if key == "val_regret":
                    df = load_fixed(lr, cap, model)
                    if df is not None:
                        ax.plot(
                            df["epoch"], df["val_regret"],
                            color=CAP_COLORS[cap],
                            lw=1.0, ls="--", alpha=0.5,
                        )

            # MSE reference line on val_regret
            if key == "val_regret":
                ax.axhline(MSE_BEST, color="k", lw=1.0, ls=":", alpha=0.6,
                           label="MSE baseline")

            if row == 0:
                ax.set_title(ylabel, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"LR={lr}", fontsize=9)
            if row == len(LRS) - 1:
                ax.set_xlabel("Epoch")

            ax.grid(True, alpha=0.3)
            if key == "val_regret":
                ax.set_ylim(bottom=0)

    # Legend from last axis
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Add fixed-target legend entry manually
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="gray", lw=1.0, ls="--", alpha=0.6))
    labels.append("fixed-target Hamming (dashed)")
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8,
               bbox_to_anchor=(1.0, 1.0))

    out = f"{BASE}/fig_kn_hstar_dynamics_{model}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

print("Done.")
