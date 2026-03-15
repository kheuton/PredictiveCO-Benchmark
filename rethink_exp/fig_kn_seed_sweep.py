"""
Exp 3 figure: Random seed sweep — distribution of cold-start test regret.

Reads results from:
  saved_records/knapsack-gen/perturb/kn_plain_seed_{0..24}/results.npy

Produces:
  Panel 1: Violin + box plot of test regret across 25 seeds
  Panel 2: Test regret per seed sorted (ranked)

Prints mean, std, min, max, and fraction of runs > MSE baseline.

Usage:
  python rethink_exp/fig_kn_seed_sweep.py
  python rethink_exp/fig_kn_seed_sweep.py --n_seeds 25 --out my_fig.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "saved_records/knapsack-gen/perturb"
OUT_PATH = os.path.join(BASE_DIR, "fig_kn_seed_sweep.png")

MSE_BASELINE  = 0.0640
WARM_BASELINE = 0.0640  # warm-start perturbed matches MSE
COLD_LOCAL_MIN = 0.085  # approximate cold-start local minimum


def load_test_regret(seed):
    prefix = f"kn_plain_seed_{seed}"
    path = os.path.join(BASE_DIR, prefix, "results.npy")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    opt, ev = d[0], d[1]
    return float(np.mean(ev / (opt + 1e-8)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=25)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    # ---- Load results ----
    regrets = []
    missing = []
    for seed in range(args.n_seeds):
        r = load_test_regret(seed)
        if r is not None:
            regrets.append(r)
        else:
            missing.append(seed)

    if missing:
        print(f"WARNING: {len(missing)} seeds missing: {missing}")
    if not regrets:
        print("No results found. Run submit_kn_seed_sweep.sh first.")
        return

    regrets = np.array(regrets)
    print("=" * 50)
    print(f"Seeds completed: {len(regrets)} / {args.n_seeds}")
    print(f"  Mean:    {regrets.mean():.4f}")
    print(f"  Std:     {regrets.std():.4f}")
    print(f"  Min:     {regrets.min():.4f}")
    print(f"  Max:     {regrets.max():.4f}")
    print(f"  Median:  {np.median(regrets):.4f}")
    print(f"  > MSE baseline ({MSE_BASELINE:.4f}): "
          f"{(regrets > MSE_BASELINE).sum()}/{len(regrets)} "
          f"({100*(regrets > MSE_BASELINE).mean():.0f}%)")
    print("=" * 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Knapsack: cold-start perturbed optimizer — 25 random seeds\n"
        "Dense model, DP solver, n_samples=100, 300 epochs",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: violin + box
    ax1.set_title(f"Test regret distribution (n={len(regrets)})", fontsize=11)
    ax1.set_ylabel("Normalised test regret", fontsize=11)
    ax1.set_xticks([])

    vp = ax1.violinplot([regrets], positions=[0], showmedians=True,
                        showextrema=True)
    for pc in vp["bodies"]:
        pc.set_facecolor("#4dac26")
        pc.set_alpha(0.6)
    ax1.scatter([0] * len(regrets), regrets, color="#4dac26", s=20,
                alpha=0.5, zorder=5)
    ax1.axhline(MSE_BASELINE, color="k", ls="--", lw=2,
                label=f"MSE baseline ({MSE_BASELINE:.3f})")
    ax1.axhline(WARM_BASELINE, color="#1f77b4", ls="-.", lw=1.5,
                label=f"Warm-start perturbed ({WARM_BASELINE:.3f})")
    ax1.axhline(COLD_LOCAL_MIN, color="#d73027", ls=":", lw=1.5,
                label=f"Local min estimate ({COLD_LOCAL_MIN:.3f})")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel 2: sorted regrets
    sorted_r = np.sort(regrets)
    ax2.set_title("Sorted test regret per seed", fontsize=11)
    ax2.set_xlabel("Rank (0=best)", fontsize=11)
    ax2.set_ylabel("Normalised test regret", fontsize=11)
    ax2.bar(range(len(sorted_r)), sorted_r, color="#4dac26", alpha=0.7)
    ax2.axhline(MSE_BASELINE, color="k", ls="--", lw=2,
                label=f"MSE baseline ({MSE_BASELINE:.3f})")
    ax2.axhline(COLD_LOCAL_MIN, color="#d73027", ls=":", lw=1.5,
                label=f"Local min ({COLD_LOCAL_MIN:.3f})")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {args.out}")


if __name__ == "__main__":
    main()
