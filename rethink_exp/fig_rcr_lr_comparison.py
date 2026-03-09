"""
Figure: RCR target sweep across learning rates.

Replicates fig_rcr_target_sweep.py but overlays three LR conditions:
  lr=5e-2  (adaptive_run1  — existing)
  lr=1e-2  (adaptive_lr1e-2)
  lr=5e-3  (adaptive_lr5e-3)

Two rows (dense, poly), three panels each:
  1  Val regret vs RCR target        (log scale)
  2  Final sigma vs RCR target       (log scale)
  3  Controller tracking accuracy    (achieved RCR vs target)

Saves to saved_records/cubic-gen/perturb_adaptive_sweep/fig_rcr_lr_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob, os

BASE     = "saved_records/cubic-gen/perturb_adaptive_sweep"
DIAG_BASE = "saved_records/cubic-gen/perturb_sigma_sweep"

# ---- LR conditions: (label, adaptive_dir, fixed_sigma_dir, color) ----
LR_CONDITIONS = [
    ("5e-2", f"{BASE}/adaptive_run1",   f"{DIAG_BASE}/diag_run1",    "#2166ac"),  # blue
    ("1e-2", f"{BASE}/adaptive_lr1e-2", f"{DIAG_BASE}/diag_lr1e-2",  "#f4a11d"),  # orange
    ("5e-3", f"{BASE}/adaptive_lr5e-3", f"{DIAG_BASE}/diag_lr5e-3",  "#4dac26"),  # green
]

STUCK = 0.40   # val regret threshold for "stuck" / failed run


# ---- Loaders ----

def load_prop(run_dir, model):
    rows = []
    for f in sorted(glob.glob(f"{run_dir}/prop_*_{model}.npz")):
        d = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace(".npz", "")
        target = float(name.split("_")[1][1:])   # prop_tX.XXX_sY → X.XXX
        rows.append(dict(
            target=target,
            val=float(d["val_regret"][-20:].mean()),
            mean_rcr=float(d["rank_change_rate"].mean()),
            final_sigma=float(d["adaptive_sigma"][-1]),
        ))
    rows.sort(key=lambda r: r["target"])
    return rows


def fixed_best(diag_dir, model):
    vals = [np.load(f, allow_pickle=True)["val_regret"][-20:].mean()
            for f in glob.glob(f"{diag_dir}/*_{model}.npz")]
    return min(vals) if vals else float("nan")


# ---- Figure: 2 rows × 3 cols ----

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex="col")
fig.suptitle(
    "Proportional adaptive sigma — RCR target sweep, three learning rates (cubic)",
    fontsize=13, fontweight="bold",
)

row_labels = ["dense", "poly"]
col_labels = ["Val regret vs RCR target", "Converged sigma vs RCR target",
              "Controller tracking accuracy"]

for row_idx, model in enumerate(row_labels):
    ax_val, ax_sigma, ax_rcr = axes[row_idx]

    # ---- LR conditions ----
    for lr_label, run_dir, diag_dir, color in LR_CONDITIONS:
        rows = load_prop(run_dir, model)
        if not rows:
            print(f"  WARNING: no data for {model} in {run_dir}")
            continue

        targets = np.array([r["target"] for r in rows])
        vals    = np.array([r["val"]    for r in rows])
        sigmas  = np.array([r["final_sigma"] for r in rows])
        rcrs    = np.array([r["mean_rcr"]    for r in rows])
        stuck   = vals >= STUCK

        # Fixed-sigma baseline for this LR
        fixed = fixed_best(diag_dir, model)
        ax_val.axhline(fixed, color=color, ls=":", lw=1.2, alpha=0.5,
                       label=f"fixed-σ best lr={lr_label} ({fixed:.3f})")

        # Panel 1: val regret (log scale)
        ax_val.plot(targets[~stuck], vals[~stuck], "o-",
                    color=color, lw=1.8, ms=5, label=f"adaptive lr={lr_label}")
        if stuck.any():
            ax_val.scatter(targets[stuck], vals[stuck],
                           marker="x", color=color, s=60, lw=1.8, zorder=5)

        # Panel 2: final sigma (log scale)
        conv_mask = ~stuck
        if conv_mask.any():
            ax_sigma.semilogy(targets[conv_mask], sigmas[conv_mask], "o-",
                              color=color, lw=1.8, ms=5, label=f"lr={lr_label}")
        if stuck.any():
            ax_sigma.scatter(targets[stuck], sigmas[stuck],
                             marker="x", color=color, s=60, lw=1.8, zorder=5)

        # Panel 3: tracking accuracy (achieved RCR vs target)
        ax_rcr.plot(targets, rcrs, "o-",
                    color=color, lw=1.8, ms=5, label=f"lr={lr_label}")

    # Perfect-tracking diagonal
    ax_rcr.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="perfect")

    # Axis formatting
    ax_val.set_yscale("log")
    ax_val.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax_val.set_ylabel(f"{model}\nFinal val regret (lower = better)", fontsize=10)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(fontsize=7.5, loc="upper right")

    ax_sigma.set_ylabel("Final sigma (log scale)", fontsize=10)
    ax_sigma.grid(True, alpha=0.3)
    ax_sigma.legend(fontsize=7.5)

    ax_rcr.set_ylabel("Mean achieved RCR", fontsize=10)
    ax_rcr.set_xlim(0, 1); ax_rcr.set_ylim(0, 1)
    ax_rcr.grid(True, alpha=0.3)
    ax_rcr.legend(fontsize=7.5)

# ---- Column titles (top row only) ----
for ax, title in zip(axes[0], col_labels):
    ax.set_title(title, fontsize=11)

# ---- X labels (bottom row only) ----
for ax in axes[1]:
    ax.set_xlabel("RCR target", fontsize=10)

plt.tight_layout()
out = f"{BASE}/fig_rcr_lr_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
