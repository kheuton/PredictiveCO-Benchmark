"""
fig_per_instance_sigma.py — Visualise per-instance sigma results.

4-panel figure:
  Panel 1: Final per-instance sigma vs X feature value (scatter).
           Expects elevated sigma near ±sqrt(0.65/3) ≈ ±0.465 (cubic decision boundary).
  Panel 2: Sigma evolution — mean ± std band + faint individual traces (alpha=0.05).
  Panel 3: Val regret over training.
  Panel 4: FracImproving over training.

Usage:
    python rethink_exp/fig_per_instance_sigma.py \
        --npz saved_records/cubic-gen/perturb_adaptive_sweep/pi_test/pi_t0.066_s1.0_dense.npz \
        --problem cubic \
        --model_type dense \
        --out saved_records/cubic-gen/perturb_adaptive_sweep/pi_test/fig_per_instance_sigma.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from openpto.config import load_conf, setup_seed
from openpto.problems.wrapper_prob import problem_wrapper
import types


# ---- Argument parsing ----

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True,
                   help="Path to .npz file from perturb_adaptive_sweep.py --per_instance")
    p.add_argument("--problem", type=str, default="cubic")
    p.add_argument("--model_type", type=str, default="dense")
    p.add_argument("--config_path", type=str, default="")
    p.add_argument("--instances", type=int, default=250)
    p.add_argument("--testinstances", type=int, default=400)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--solver", type=str, default="heuristic")
    p.add_argument("--loadnew", action="store_true")
    p.add_argument("--out", type=str, default="",
                   help="Output path for figure. Defaults to same directory as --npz.")
    return p.parse_args()


# ---- Main ----

def main():
    args = parse_args()
    setup_seed(args.seed)

    # Load results
    data = np.load(args.npz, allow_pickle=True)
    print(f"Keys: {list(data.keys())}")

    epochs     = data["epoch"]
    val_regret = data["val_regret"]
    frac_impr  = data["frac_improving"]
    sigma_vec_traj = data.get("sigma_vec", None)  # (T, N_train) or None

    if sigma_vec_traj is None:
        print("ERROR: No 'sigma_vec' in npz. Was --per_instance used?")
        sys.exit(1)

    T, N_train = sigma_vec_traj.shape
    print(f"sigma_vec_traj shape: {T} epochs × {N_train} instances")

    # ---- Load problem to get X_train, Y_train ----
    ns = types.SimpleNamespace(
        problem=args.problem,
        solver=args.solver,
        config_path=args.config_path,
        loadnew=args.loadnew,
        instances=args.instances,
        testinstances=args.testinstances,
        val_frac=args.val_frac,
        seed=args.seed,
        data_dir=os.path.join("./openpto/data/", args.problem),
    )
    conf = load_conf(args.config_path, "openpto/config/models/perturb_s01_n5.yaml", args.problem)
    problem = problem_wrapper(ns, conf)
    X_train, Y_train, _ = problem.get_train_data()

    # K from problem config (default 5 for cubic)
    K = getattr(problem, "budget", 5)

    # ---- Per-instance difficulty metrics ----
    # Y_train: (N, n_items) — true costs
    Y_np = Y_train.numpy()
    if Y_np.ndim == 3:
        Y_np = Y_np.reshape(Y_np.shape[0], -1)
    Y_np = Y_np[:N_train]

    # Margin: gap between K-th and (K+1)-th largest Y per instance
    Y_sorted = np.sort(Y_np, axis=1)[:, ::-1]   # descending
    margin = Y_sorted[:, K - 1] - Y_sorted[:, K]  # (N_train,)

    # Count of items in the non-monotonic region of Y = 10(x^3 - 0.65x): x in [-0.465, +0.465]
    X_np = X_train.numpy()
    if X_np.ndim == 3:
        X_np = X_np.reshape(X_np.shape[0], -1)
    X_np = X_np[:N_train]
    boundary = np.sqrt(0.65 / 3.0)  # ≈ 0.465
    n_ambiguous = ((X_np > -boundary) & (X_np < boundary)).sum(axis=1).astype(float)  # (N_train,)

    final_sigma = sigma_vec_traj[-1]   # (N_train,)

    # ---- Figure ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"Per-instance sigma — {args.problem} / {args.model_type}\n{os.path.basename(args.npz)}",
        fontsize=10,
    )

    # Panel 1: final sigma vs Y-margin (coloured by n_ambiguous items)
    ax = axes[0]
    sc = ax.scatter(margin, final_sigma, c=n_ambiguous, cmap="plasma", s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="# items in non-monotonic region")
    ax.set_xlabel(f"Y margin (Y[{K}] − Y[{K+1}])", fontsize=9)
    ax.set_ylabel("Final sigma")
    ax.set_title("Final sigma vs decision margin\n(colour = # ambiguous items)", fontsize=9)

    # Panel 2: sigma evolution (mean ± std + faint traces)
    ax = axes[1]
    sigma_mean = sigma_vec_traj.mean(axis=1)   # (T,)
    sigma_std  = sigma_vec_traj.std(axis=1)    # (T,)
    # Faint individual traces (subsample for readability)
    n_traces = min(50, N_train)
    idx_sample = np.linspace(0, N_train - 1, n_traces, dtype=int)
    for i in idx_sample:
        ax.plot(epochs[:T], sigma_vec_traj[:, i], color="steelblue", alpha=0.05, lw=0.5)
    ax.plot(epochs[:T], sigma_mean, color="steelblue", lw=2, label="mean")
    ax.fill_between(
        epochs[:T],
        sigma_mean - sigma_std,
        sigma_mean + sigma_std,
        alpha=0.25, color="steelblue", label="±1 std"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sigma")
    ax.set_title("Sigma evolution")
    ax.legend(fontsize=8)

    # Panel 3: val regret
    ax = axes[2]
    ax.plot(epochs, val_regret, color="darkorange", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val regret")
    ax.set_title("Val regret")
    ax.set_yscale("log" if val_regret.min() > 0 else "linear")

    # Panel 4: FracImproving
    ax = axes[3]
    ax.plot(epochs, frac_impr, color="green", lw=1.5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FracImproving")
    ax.set_title("FracImproving (→0 = converged)")
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()

    out_path = args.out
    if not out_path:
        out_dir = os.path.dirname(args.npz)
        stem = os.path.splitext(os.path.basename(args.npz))[0]
        out_path = os.path.join(out_dir, f"fig_per_instance_{stem}.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
