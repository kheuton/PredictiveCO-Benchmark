"""
Figure: Val regret vs fixed sigma, across learning rates.

Two panels (dense, poly), three lines per panel (lr=5e-2, 1e-2, 5e-3).

Saves to saved_records/cubic-gen/perturb_sigma_sweep/fig_sigma_lr_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob

DIAG_BASE = "saved_records/cubic-gen/perturb_sigma_sweep"

LR_CONDITIONS = [
    ("5e-2", f"{DIAG_BASE}/diag_run1",   "#2166ac"),
    ("1e-2", f"{DIAG_BASE}/diag_lr1e-2", "#f4a11d"),
    ("5e-3", f"{DIAG_BASE}/diag_lr5e-3", "#4dac26"),
]


def load_fixed(diag_dir, model):
    rows = []
    for f in sorted(glob.glob(f"{diag_dir}/*_{model}.npz")):
        d = np.load(f, allow_pickle=True)
        # filename: sigma_X.XXX_dense.npz  or  X.XXX_dense.npz
        import os
        name = os.path.basename(f).replace(".npz", "")
        # sigma is the second-to-last underscore token (before model name)
        parts = name.split("_")
        sigma = float(parts[-2])
        rows.append(dict(
            sigma=sigma,
            val=float(d["val_regret"][-20:].mean()),
        ))
    rows.sort(key=lambda r: r["sigma"])
    return rows


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
fig.suptitle("Fixed sigma: val regret vs σ, three learning rates (cubic)",
             fontsize=12, fontweight="bold")

for ax, model in zip(axes, ["dense", "poly"]):
    for lr_label, diag_dir, color in LR_CONDITIONS:
        rows = load_fixed(diag_dir, model)
        if not rows:
            print(f"WARNING: no data for {model} in {diag_dir}")
            continue
        sigmas = np.array([r["sigma"] for r in rows])
        vals   = np.array([r["val"]   for r in rows])
        ax.plot(sigmas, vals, "o-", color=color, lw=1.8, ms=6, label=f"lr={lr_label}")
        best_idx = np.argmin(vals)
        ax.scatter([sigmas[best_idx]], [vals[best_idx]],
                   marker="*", color=color, s=150, zorder=5,
                   edgecolors="k", linewidths=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_xlabel("Sigma (log scale)", fontsize=11)
    ax.set_ylabel("Final val regret (lower = better)", fontsize=11)
    ax.set_title(f"{model} model", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f"{DIAG_BASE}/fig_sigma_lr_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
