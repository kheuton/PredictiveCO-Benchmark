"""
Compare warm-start (MSE init) vs cold-start (random init) for the perturbed method.

Both use: per-instance OCV_Y, lr=1e-2, t=0.066, DP n=100, 300 epochs.

Cold-start: kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense.npz
Warm-start: kn_bench_warmstart/pi_t0.066_s1_dense.npz

Panels: val_regret | sigma | OCV_Y | FracImproving

Usage:
    conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_warmstart.py
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

ADAPTIVE_BASE = "saved_records/knapsack-gen/perturb_adaptive_sweep"
MSE_BASELINE = 0.0640

COLD_PATH = os.path.join(
    ADAPTIVE_BASE, "kn_bench_dp_per_instance_lr1e-2", "pi_t0.066_s1_dense.npz"
)
WARM_PATH = os.path.join(
    ADAPTIVE_BASE, "kn_bench_warmstart", "pi_t0.066_s1_dense.npz"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runs = {}
    for tag, path in [("cold (random init)", COLD_PATH), ("warm (MSE init)", WARM_PATH)]:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        runs[tag] = dict(np.load(path, allow_pickle=True))

    if not runs:
        print("No results found. Run submit_kn_warmstart_sweep.sh first.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    ax_val, ax_sig, ax_ocv, ax_frac = axes

    fig.suptitle(
        "Warm-start (MSE init) vs cold-start (random init) — perturbed method\n"
        "Per-instance OCV_Y, lr=1e-2, t=0.066, DP n=100, 300ep",
        fontsize=11, fontweight="bold",
    )

    colors = {"cold (random init)": "#e66101", "warm (MSE init)": "#5e3c99"}
    for tag, d in runs.items():
        color = colors.get(tag, "gray")
        epochs = np.arange(1, len(d["val_regret"]) + 1)
        test_r = float(d["test_regret"]) if "test_regret" in d else float("nan")
        label = f"{tag}  (test={test_r:.4f})"

        ax_val.plot(epochs, d["val_regret"],    color=color, lw=2, label=label)
        ax_sig.plot(epochs, d["adaptive_sigma"], color=color, lw=2, label=label)
        ax_ocv.plot(epochs, d["ocv_y"],          color=color, lw=2, label=label)
        ax_frac.plot(epochs, d["frac_improving"], color=color, lw=2, label=label)

    ax_val.axhline(MSE_BASELINE, color="gray", lw=1.5, ls=":", label=f"MSE ({MSE_BASELINE:.4f})")

    titles = ["Val regret", "Sigma", "OCV_Y", "FracImproving"]
    ylabels = ["Val regret (norm.)", "Sigma", "OCV_Y", "FracImproving"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_val.set_yscale("log")
    ax_sig.set_yscale("log")
    ax_frac.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out = args.out or os.path.join(ADAPTIVE_BASE, "fig_kn_warmstart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")

    print("\n--- Summary ---")
    for tag, d in runs.items():
        test_r = float(d["test_regret"]) if "test_regret" in d else float("nan")
        print(f"  {tag}: best_val={d['val_regret'].min():.4f}  test={test_r:.4f}")
    print(f"  MSE baseline:        test=0.0640")


if __name__ == "__main__":
    main()
