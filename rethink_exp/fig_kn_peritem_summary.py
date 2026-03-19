"""
Summary visualization for per-item sigma + warm-start experiments.

3-panel figure:
  1. Test regret bar chart — all approaches grouped by family
  2. Sigma ballooning trajectories — sigma_item_max over training for
     per-item Hamming, coeff-relative, and per-instance reference
  3. Val regret training curves — warm-start vs cold-start for lambda=10

Usage:
    python rethink_exp/fig_kn_peritem_summary.py
    python rethink_exp/fig_kn_peritem_summary.py --out my_fig.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# ---- Paths ----
BASE_SWEEP = "saved_records/knapsack-gen/perturb_adaptive_sweep"
BASE_PERTURB = "saved_records/knapsack-gen/perturb"
BASE_MSE = "saved_records/knapsack-gen/mse"
OUT_DIR = "saved_records/knapsack-gen/oracle_ceiling"
OUT_PATH = os.path.join(OUT_DIR, "fig_kn_peritem_summary.png")

OPT_OBJ = 36.315   # test-set optimal for knapsack-small benchmark


# ---- Helpers ----
def regret_from_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = f.readlines()
    for line in reversed(lines):
        line = line.strip()
        if "[Optimal Obj]:" in line and "[regret]:" in line:
            opt = float(line.split("[Optimal Obj]:")[1].split()[0])
            reg = float(line.split("[regret]:")[1].split()[0])
            return reg / opt if opt > 0 else None
    return None


def best_npz_regret(folder, glob_prefix):
    """Return (best_test_regret, best_npz_data) over matching .npz files."""
    if not os.path.isdir(folder):
        return None, None
    best_r, best_d = None, None
    for fn in os.listdir(folder):
        if fn.startswith(glob_prefix) and fn.endswith("_dense.npz"):
            d = dict(np.load(os.path.join(folder, fn), allow_pickle=True))
            r = float(d["test_regret"])
            if best_r is None or r < best_r:
                best_r, best_d = r, d
    return best_r, best_d


# ====================================================================
# Panel 1 — Test regret bar chart
# ====================================================================

BAR_GROUPS = [
    # (group_label, [(bar_label, regret_or_None, color, hatch)])
    ("Baselines", [
        ("Oracle ceiling", 0.052, "#333333", ""),
        ("MSE (Gurobi)", 0.0640, "#555555", ""),
    ]),
    ("Plain perturb", [
        ("σ=0.5, lr=5e-3", None, "C0", ""),  # loaded from log
    ]),
    ("MSE-reg cold-start", [
        ("λ=1",  None, "C1", ""),
        ("λ=5",  None, "C1", "/"),
        ("λ=10", None, "C1", "x"),
        ("λ=20", None, "C1", "o"),
    ]),
    ("MSE-reg warm-start", [
        ("λ=10, lr=5e-3", None, "C4", ""),
        ("λ=10, lr=1e-2", None, "C4", "/"),
    ]),
    ("Per-instance adaptive", [
        ("OCV_Y (best)", 0.0838, "C2", ""),
        ("Hamming const (best)", 0.0869, "C2", "/"),
    ]),
    ("Per-item Hamming", [
        ("A–D best (lr=5e-2)", None, "C3", ""),
    ]),
    ("Coeff-relative", [
        ("best (lr=1e-2)", None, "C5", ""),
    ]),
]

LOG_FILLS = {
    "σ=0.5, lr=5e-3":     f"{BASE_PERTURB}/kn_plain_sigma_sweep_s05_lr5e-3/log.txt",
    "λ=1":                 f"{BASE_PERTURB}/kn_plain_mse_reg_w1_lr5e-3/log.txt",
    "λ=5":                 f"{BASE_PERTURB}/kn_plain_mse_reg_w5_lr5e-3/log.txt",
    "λ=10":                f"{BASE_PERTURB}/kn_plain_mse_reg_w10_lr5e-3/log.txt",
    "λ=20":                f"{BASE_PERTURB}/kn_plain_mse_reg_w20_lr5e-3/log.txt",
    "λ=10, lr=5e-3":       f"{BASE_PERTURB}/kn_plain_mse_reg_w10_warmstart_lr5e-3/log.txt",
    "λ=10, lr=1e-2":       f"{BASE_PERTURB}/kn_plain_mse_reg_w10_warmstart_lr1e-2/log.txt",
    "A–D best (lr=5e-2)":  None,   # computed below from npz
    "best (lr=1e-2)":      None,   # computed below from npz
}


def fill_regrets(groups):
    # Fill per-item and coeff-rel from npz
    pi_best = None
    for var in "ABCD":
        for lr in ["5e-2", "1e-2", "5e-3"]:
            stem = "pitem_t" if var in "AB" else "piitem_t"
            folder = f"{BASE_SWEEP}/kn_per_item_{var}_lr{lr}"
            r, _ = best_npz_regret(folder, stem)
            if r is not None and (pi_best is None or r < pi_best):
                pi_best = r

    cr_best = None
    for lr in ["5e-2", "1e-2", "5e-3"]:
        folder = f"{BASE_SWEEP}/kn_coeff_rel_lr{lr}"
        r, _ = best_npz_regret(folder, "cr_a")
        if r is not None and (cr_best is None or r < cr_best):
            cr_best = r

    filled = []
    for group_label, bars in groups:
        new_bars = []
        for label, regret, color, hatch in bars:
            if regret is None:
                if label in LOG_FILLS and LOG_FILLS[label]:
                    regret = regret_from_log(LOG_FILLS[label])
                elif label == "A–D best (lr=5e-2)":
                    regret = pi_best
                elif label == "best (lr=1e-2)":
                    regret = cr_best
            new_bars.append((label, regret, color, hatch))
        filled.append((group_label, new_bars))
    return filled


# ====================================================================
# Panel 2 — Sigma ballooning
# ====================================================================

SIGMA_TRACES = [
    # (label, npz_path, key, color, ls)
    (
        "Per-item Hamming A lr=5e-2 t=0.05\n(sigma_item_max)",
        f"{BASE_SWEEP}/kn_per_item_A_lr5e-2/pitem_t0.050_s1_dense.npz",
        "sigma_item_max", "C3", "-",
    ),
    (
        "Per-item Hamming C lr=5e-2 t=0.05\n(sigma_item_max)",
        f"{BASE_SWEEP}/kn_per_item_C_lr5e-2/piitem_t0.050_s1_dense.npz",
        "sigma_item_max", "C3", "--",
    ),
    (
        "Coeff-relative lr=5e-2 α=2\n(adaptive_sigma)",
        f"{BASE_SWEEP}/kn_coeff_rel_lr5e-2/cr_a2_s1_dense.npz",
        "adaptive_sigma", "C5", "-",
    ),
    (
        "Coeff-relative lr=1e-2 α=1\n(adaptive_sigma)",
        f"{BASE_SWEEP}/kn_coeff_rel_lr1e-2/cr_a1_s1_dense.npz",
        "adaptive_sigma", "C5", "--",
    ),
    (
        "Per-inst Hamming const lr=1e-2 t=0.10\n(adaptive_sigma, reference)",
        f"{BASE_SWEEP}/kn_bench_hamming_lr1e-2/ham_t0.100_s1_dense.npz",
        "adaptive_sigma", "C2", "-.",
    ),
]


# ====================================================================
# Panel 3 — Val regret training curves
# ====================================================================

TRAINING_CURVES = [
    (
        "Cold-start λ=10, lr=5e-3",
        f"{BASE_PERTURB}/kn_plain_mse_reg_w10_lr5e-3/val_logs.csv",
        "C1", "-",
    ),
    (
        "Warm-start λ=10, lr=5e-3",
        f"{BASE_PERTURB}/kn_plain_mse_reg_w10_warmstart_lr5e-3/val_logs.csv",
        "C4", "-",
    ),
    (
        "Warm-start λ=10, lr=1e-2",
        f"{BASE_PERTURB}/kn_plain_mse_reg_w10_warmstart_lr1e-2/val_logs.csv",
        "C4", "--",
    ),
]


# ====================================================================
# Main
# ====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Knapsack: Per-item σ sweep + warm-start results\n"
        "(DP solver, n_samples=100, 300 epochs, dense model)",
        fontsize=12, fontweight="bold",
    )

    # ---- Panel 1: bar chart ----
    ax = axes[0]
    groups = fill_regrets(BAR_GROUPS)

    all_bars = []   # (x_pos, regret, color, hatch, label)
    group_ticks = []
    group_labels = []
    x = 0
    sep = 0.4

    for group_label, bars in groups:
        group_start = x
        for label, regret, color, hatch in bars:
            if regret is not None:
                ax.bar(x, regret, width=0.6, color=color, hatch=hatch,
                       alpha=0.85, edgecolor="black", linewidth=0.5)
                ax.text(x, regret + 0.003, f"{regret:.4f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)
            else:
                ax.bar(x, 0, width=0.6, color="lightgray", edgecolor="black",
                       linewidth=0.5, linestyle="--")
                ax.text(x, 0.005, "N/A", ha="center", va="bottom",
                        fontsize=6.5, color="gray")
            all_bars.append((x, label))
            x += 0.8
        group_center = (group_start + x - 0.8) / 2
        group_ticks.append(group_center)
        group_labels.append(group_label)
        x += sep

    ax.set_xticks(group_ticks)
    ax.set_xticklabels(group_labels, fontsize=8, rotation=20, ha="right")
    ax.axhline(0.052, color="#333333", lw=1, ls=":", alpha=0.7, label="Oracle (0.052)")
    ax.axhline(0.0640, color="#555555", lw=1, ls=":", alpha=0.7, label="MSE (0.064)")
    ax.set_ylabel("Test regret (normalized)", fontsize=10)
    ax.set_title("Test regret by approach", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 0.42)
    ax.grid(True, alpha=0.3, axis="y")

    # ---- Panel 2: sigma ballooning ----
    ax = axes[1]
    for label, path, key, color, ls in SIGMA_TRACES:
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            continue
        d = np.load(path, allow_pickle=True)
        if key not in d:
            print(f"  [KEY MISSING] {key} in {path}")
            continue
        vals = d[key]
        epochs = d.get("epoch", np.arange(1, len(vals) + 1))
        ax.plot(epochs, vals, color=color, ls=ls, lw=1.5, label=label)

    ax.axhline(1.0, color="gray", ls=":", lw=1, label="σ=1 init")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("σ (max over items)", fontsize=10)
    ax.set_title("Sigma ballooning over training\n(per-item Hamming & coeff-relative)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # ---- Panel 3: training curves ----
    ax = axes[2]
    for label, csv_path, color, ls in TRAINING_CURVES:
        if not os.path.exists(csv_path):
            print(f"  [MISSING] {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # "eval" column = absolute val regret; normalize by OPT_OBJ
        val_regret = df["eval"].values / OPT_OBJ
        epochs = np.arange(1, len(val_regret) + 1)
        ax.plot(epochs, val_regret, color=color, ls=ls, lw=1.5, label=label)

    ax.axhline(0.052, color="#333333", lw=1, ls=":", alpha=0.7, label="Oracle (0.052)")
    ax.axhline(0.0640, color="#555555", lw=1, ls=":", alpha=0.7, label="MSE (0.064)")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Val regret (normalized)", fontsize=10)
    ax.set_title("Val regret: warm-start vs cold-start\n(λ=10 MSE-reg)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {args.out}")


if __name__ == "__main__":
    main()
