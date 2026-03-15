"""
MSE regularization sweep summary for knapsack — 3-panel figure.

Panel 1: Test regret vs λ (all lr=5e-3 checkpoints; λ=0.1/lr=1e-2 marked confounded)
Panel 2: Gradient scale ratio (λ·L_mse / L_perturb) over training epochs
         — only available for new runs that have the updated perturbed.py logging
Panel 3: Checkpoint MSE loss vs λ (distance from oracle in MSE space)
         — loaded from oracle_interpolation.npz

Usage:
    python rethink_exp/fig_kn_mse_reg_sweep.py
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

# ---- Constants ----
RESULTS_BASE = "saved_records/knapsack-gen/perturb"
OUT_DIR = "saved_records/knapsack-gen/oracle_ceiling"
OUT_PATH = os.path.join(OUT_DIR, "fig_kn_mse_reg_sweep.png")
MSE_BASELINE = 0.0640
ORACLE_CEILING = 0.0519

# All runs at lr=5e-3; has_grad_log=True means the updated perturbed.py was used
RUNS = [
    # (lambda,  run_dir,                              test_regret,  has_grad_log)
    (0,      "kn_plain_sigma_sweep_s05_lr5e-3",       0.0743, False),
    (0.01,   "kn_plain_mse_reg_w001_lr5e-3",          0.0730, False),
    (0.033,  "kn_plain_mse_reg_w0033_lr5e-3",         0.0759, True),
    (0.05,   "kn_plain_mse_reg_w005_lr5e-3",          0.0707, False),
    (0.5,    "kn_plain_mse_reg_w05_lr5e-3",           0.0706, False),
    (1,      "kn_plain_mse_reg_w1_lr5e-3",            0.0698, False),
    (5,      "kn_plain_mse_reg_w5_lr5e-3",            0.0657, False),
    (10,     "kn_plain_mse_reg_w10_lr5e-3",           0.0614, False),
    (20,     "kn_plain_mse_reg_w20_lr5e-3",           0.0618, True),
    (50,     "kn_plain_mse_reg_w50_lr5e-3",           0.0650, True),
    (100,    "kn_plain_mse_reg_w100_lr5e-3",          0.0646, True),
]
# λ=0.1 at lr=1e-2 is confounded (different lr than all others)
CONFOUNDED = (0.1, 0.0868)

GRAD_PAT = re.compile(
    r"epoch (\d+).*L_perturb = ([\d.]+).*lambda\*L_mse = ([\d.]+).*ratio\(lambda\*mse/perturb\) = ([\d.]+)"
)

# Interpolation npz labels (must match ENDPOINTS in fig_kn_oracle_interpolation.py)
NPZ_LABELS = {
    0:      "Plain (λ=0)",
    0.033:  "λ=0.033",
    0.05:   "λ=0.05",
    0.5:    "λ=0.5",
    1:      "λ=1",
    5:      "λ=5",
    10:     "λ=10",
    20:     "λ=20",
    50:     "λ=50",
    100:    "λ=100",
    np.inf: "MSE (λ=∞)",
}


def load_grad_log(run_dir):
    path = os.path.join(RESULTS_BASE, run_dir, "log.txt")
    epochs, ratios, lam_mse, perturb_l = [], [], [], []
    if not os.path.exists(path):
        return np.array([]), np.array([]), np.array([]), np.array([])
    for line in open(path):
        m = GRAD_PAT.search(line)
        if m:
            epochs.append(int(m.group(1)))
            perturb_l.append(float(m.group(2)))
            lam_mse.append(float(m.group(3)))
            ratios.append(float(m.group(4)))
    return np.array(epochs), np.array(ratios), np.array(lam_mse), np.array(perturb_l)


# ---- Colour scale (log10 of lambda) ----
lam_vals = [r[0] for r in RUNS if r[0] > 0]
lam_log = np.log10(lam_vals)
norm = plt.Normalize(lam_log.min(), lam_log.max())
cmap = matplotlib.colormaps["plasma"]


def lam_color(lam):
    if lam == 0:
        return "grey"
    return cmap(norm(np.log10(float(lam))))


# ---- Figure ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.subplots_adjust(wspace=0.38)

# ---- Panel 1: Test regret vs lambda ----
ax = axes[0]
for lam, run_dir, regret, _ in RUNS:
    x = lam if lam > 0 else 0.005
    ax.scatter(x, regret, color=lam_color(lam), s=60, zorder=5)

# Connect non-zero points
nz = sorted([(r[0], r[2]) for r in RUNS if r[0] > 0])
ax.plot([x[0] for x in nz], [x[1] for x in nz], color="k", lw=1.2, alpha=0.4, zorder=3)

# λ=0 marker
ax.scatter(0.005, RUNS[0][2], color="grey", s=60, marker="D", zorder=5,
           label="λ=0 (plain, lr=5e-3)")
ax.scatter(CONFOUNDED[0], CONFOUNDED[1], color="red", s=70, marker="x", zorder=6,
           label="λ=0.1  lr=1e-2 (confounded)")
ax.axhline(MSE_BASELINE, color="steelblue", ls="--", lw=1.5,
           label="MSE baseline (%.4f)" % MSE_BASELINE)
ax.axhline(ORACLE_CEILING, color="green", ls="--", lw=1.5,
           label="Oracle ceiling (%.4f)" % ORACLE_CEILING)

ax.set_xscale("log")
ax.set_xlim(0.003, 200)
ax.set_xlabel("λ (MSE reg weight)")
ax.set_ylabel("Test regret (normalised)")
ax.set_title("Test regret vs λ\n(all lr=5e-3, σ=0.5, n=100)")
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, alpha=0.25)

# ---- Panel 2: Gradient scale ratio over training ----
ax = axes[1]
for lam, run_dir, regret, has_log in RUNS:
    if not has_log:
        continue
    epochs, ratios, _, _ = load_grad_log(run_dir)
    if len(epochs) == 0:
        continue
    ax.plot(epochs, ratios, color=lam_color(lam), lw=2.0,
            label="λ=%.3g (r=%.3f)" % (lam, regret))

ax.axhline(1.0, color="k", ls=":", lw=1.5, label="Balance (ratio=1)")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("λ·L_mse / L_perturb")
ax.set_title("Gradient scale ratio over training\n(logged for new runs only)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.25)

# Annotate balance line
ax.text(280, 1.1, "balance", fontsize=8, ha="right", color="k", style="italic")

# ---- Panel 3: Checkpoint MSE at endpoint (from interpolation npz) ----
ax = axes[2]
npz_path = os.path.join(OUT_DIR, "oracle_interpolation.npz")
if os.path.exists(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    alphas = data["alphas"]
    idx0 = np.argmin(np.abs(alphas - 0.0))
    idx1 = np.argmin(np.abs(alphas - 1.0))

    ep_lams, ep_mse = [], []
    for lam, _, regret, _ in RUNS:
        label = NPZ_LABELS.get(lam)
        key = "mse_" + label if label else None
        if key and key in data:
            x = lam if lam > 0 else 0.005
            y = float(data[key][idx1])
            ep_lams.append(x)
            ep_mse.append(y)
            ax.scatter(x, y, color=lam_color(lam), s=60, zorder=5)

    # MSE baseline endpoint
    mse_label = NPZ_LABELS[np.inf]
    mse_key = "mse_" + mse_label
    if mse_key in data:
        mse_val = float(data[mse_key][idx1])
        ax.scatter(150, mse_val, color=lam_color(100), s=60, marker="*", zorder=5)
        ax.axhline(mse_val, color="steelblue", ls="--", lw=1.2,
                   label="MSE baseline (%.3f)" % mse_val)

    # Connect
    nz2 = [(l, m) for l, m in zip(ep_lams, ep_mse) if l > 0]
    nz2.sort()
    ax.plot([x[0] for x in nz2], [x[1] for x in nz2], color="k", lw=1.2, alpha=0.4)

    # Oracle MSE (at alpha=0, same for all curves)
    first_key = [k for k in data.files if k.startswith("mse_")][0]
    oracle_mse = float(data[first_key][idx0])
    ax.axhline(oracle_mse, color="green", ls="--", lw=1.5,
               label="Oracle MSE (%.3f)" % oracle_mse)

    ax.set_xscale("log")
    ax.set_xlim(0.003, 200)
    ax.set_xlabel("λ (MSE reg weight)")
    ax.set_ylabel("MSE loss at trained checkpoint")
    ax.set_title("Checkpoint MSE loss vs λ\n(α=1 along oracle→checkpoint path)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
else:
    ax.text(0.5, 0.5, "oracle_interpolation.npz not found\n(run fig_kn_oracle_interpolation.py first)",
            ha="center", va="center", transform=ax.transAxes, fontsize=9)

# ---- Shared colorbar ----
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="vertical", fraction=0.015, pad=0.01)
cbar.set_label("log₁₀(λ)", rotation=270, labelpad=14)

fig.suptitle(
    "Knapsack — MSE regularization sweep (σ=0.5, n_samples=100, lr=5e-3, 300 epochs)",
    fontsize=11, y=1.01
)

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print("Saved:", OUT_PATH)

# ---- Summary table ----
print("\n{'λ':<8}  {'test_regret':<14}  {'ratio_end (est)'}")
for lam, run_dir, regret, has_log in RUNS:
    ratio_str = ""
    if has_log:
        epochs, ratios, _, _ = load_grad_log(run_dir)
        if len(ratios):
            ratio_str = "%.3f" % ratios[-1]
    print("  %-8s  %-14.4f  %s" % (("%.3g" % lam if lam > 0 else "0"), regret, ratio_str))
print("  %-8s  %-14.4f" % ("∞ (MSE)", MSE_BASELINE))
print("  %-8s  %-14.4f  (oracle)" % ("—", ORACLE_CEILING))
