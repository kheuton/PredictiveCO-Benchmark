"""
fig_pi_sigma_vs_margin_epochs.py

Per-instance sigma vs Y-margin at several training snapshots.
Shows whether the small-margin → small-sigma pattern emerges early
(structural) or only after convergence.

Usage:
    python rethink_exp/fig_pi_sigma_vs_margin_epochs.py \
        --npz saved_records/cubic-gen/perturb_adaptive_sweep/pi_lr1e-2/pi_t0.206_s1_dense.npz \
        --problem cubic
"""

import argparse
import os
import types

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from openpto.config import load_conf, setup_seed
from openpto.problems.wrapper_prob import problem_wrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True)
    p.add_argument("--problem", type=str, default="cubic")
    p.add_argument("--snapshots", type=int, nargs="+", default=[5, 20, 50, 150, 300],
                   help="Epochs to plot (uses closest available).")
    p.add_argument("--config_path", type=str, default="")
    p.add_argument("--instances", type=int, default=250)
    p.add_argument("--testinstances", type=int, default=400)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--solver", type=str, default="heuristic")
    p.add_argument("--loadnew", action="store_true")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    data = np.load(args.npz, allow_pickle=True)
    epochs       = data["epoch"]           # (T,)
    val_regret   = data["val_regret"]      # (T,)
    sigma_vec_traj = data["sigma_vec"]     # (T, N_train)
    T, N_train = sigma_vec_traj.shape

    # ---- Load problem ----
    ns = types.SimpleNamespace(
        problem=args.problem, solver=args.solver,
        config_path=args.config_path, loadnew=args.loadnew,
        instances=args.instances, testinstances=args.testinstances,
        val_frac=args.val_frac, seed=args.seed,
        data_dir=os.path.join("./openpto/data/", args.problem),
    )
    conf = load_conf(args.config_path, "openpto/config/models/perturb_s01_n5.yaml", args.problem)
    problem = problem_wrapper(ns, conf)
    _, Y_train, _ = problem.get_train_data()
    K = getattr(problem, "budget", 5)

    Y_np = Y_train.numpy()
    if Y_np.ndim == 3:
        Y_np = Y_np.reshape(Y_np.shape[0], -1)
    Y_np = Y_np[:N_train]

    # Per-instance margin: gap between K-th and (K+1)-th largest Y
    Y_sorted = np.sort(Y_np, axis=1)[:, ::-1]
    margin = Y_sorted[:, K - 1] - Y_sorted[:, K]   # (N_train,)

    # ---- Resolve snapshot indices ----
    snapshots = []
    for ep in args.snapshots:
        idx = np.argmin(np.abs(epochs - ep))
        snapshots.append((int(epochs[idx]), idx))

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Per-instance sigma vs decision margin — early vs late\n{os.path.basename(args.npz)}",
        fontsize=10,
    )

    for ax, (ep, t_idx) in zip(axes, snapshots):
        sigma = sigma_vec_traj[t_idx]          # (N_train,)
        vr    = val_regret[t_idx]

        ax.scatter(margin, sigma, s=8, alpha=0.5, color="steelblue")

        # Spearman correlation
        r, p = scipy.stats.spearmanr(margin, sigma)
        ax.set_title(f"Epoch {ep}  |  val_regret={vr:.4f}\nSpearman r={r:.2f}  p={p:.3f}",
                     fontsize=8)
        ax.set_xlabel(f"Y margin (Y[{K}] − Y[{K+1}])", fontsize=8)
        ax.set_ylabel("Sigma", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Trend line
        m_fit, b_fit = np.polyfit(margin, sigma, 1)
        x_line = np.linspace(margin.min(), margin.max(), 100)
        ax.plot(x_line, m_fit * x_line + b_fit, color="tomato", lw=1.5, ls="--")

    plt.tight_layout()

    out = args.out
    if not out:
        stem = os.path.splitext(os.path.basename(args.npz))[0]
        out = os.path.join(os.path.dirname(args.npz), f"fig_sigma_margin_epochs_{stem}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
