"""
Interpolation plot: MSE weights → each best perturbed checkpoint.

For each perturbed checkpoint, computes decision regret at
θ(α) = (1−α)·θ_MSE + α·θ_perturb  for α ∈ [0, 1]

All curves share the same MSE endpoint (α=0). The right endpoint (α=1)
is a different model for each curve.

Checkpoints included:
  - Plain perturbed best     (σ=0.5, lr=5e-3, test=0.074)
  - MSE reg w=0.05  best     (σ=0.5, lr=5e-3, test=0.071)
  - MSE reg w=1.0   best     (σ=0.5, lr=5e-3, test=0.070)
  - Adaptive per-instance    (prev. best, test=0.084)

Usage:
  conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_multi_interpolation.py
  conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_multi_interpolation.py --n_grid 50
"""

import argparse
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")

MSE_CKPT = (
    "saved_records/knapsack-gen/mse/"
    "kn_bench_mse_dense_lr5e-2/checkpoints/tr_pred_best.pt"
)
OUT_PATH = "saved_records/knapsack-gen/perturb/fig_kn_multi_interpolation.png"

CHECKPOINTS = [
    {
        "label": "Plain perturbed\n(σ=0.5, lr=5e-3, test=0.074)",
        "short": "Plain (0.074)",
        "path": "saved_records/knapsack-gen/perturb/"
                "kn_plain_sigma_sweep_s05_lr5e-3/checkpoints/tr_pred_best.pt",
        "test_regret": 0.0743,
        "color": "#d73027",
    },
    {
        "label": "MSE reg λ=0.05\n(σ=0.5, lr=5e-3, test=0.071)",
        "short": "MSE reg λ=0.05 (0.071)",
        "path": "saved_records/knapsack-gen/perturb/"
                "kn_plain_mse_reg_w005_lr5e-3/checkpoints/tr_pred_best.pt",
        "test_regret": 0.0707,
        "color": "#f4a582",
    },
    {
        "label": "MSE reg λ=1.0\n(σ=0.5, lr=5e-3, test=0.070)",
        "short": "MSE reg λ=1.0 (0.070)",
        "path": "saved_records/knapsack-gen/perturb/"
                "kn_plain_mse_reg_w1_lr5e-3/checkpoints/tr_pred_best.pt",
        "test_regret": 0.0698,
        "color": "#b2182b",
    },
    {
        "label": "Adaptive per-instance\n(prev. best, test=0.084)",
        "short": "Adaptive pi (0.084)",
        "path": "saved_records/knapsack-gen/perturb_adaptive_sweep/"
                "kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense_best_pred.pt",
        "test_regret": 0.0852,
        "color": "#4393c3",
    },
]

MSE_TEST_REGRET  = 0.0640
MSE_TEST_REGRET_DP = 0.0658


def interpolate_state_dicts(sd_a, sd_b, alpha):
    return {k: (1.0 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
            for k in sd_a}


@torch.no_grad()
def compute_perturbed_regret(pred_model, problem, ptoSolver, X, Y, Y_aux, opt_obj,
                             sigma, n_samples=20, seed=0):
    """Normalised regret using soft decisions at the given sigma.
    z̃_i = (1/S) Σ_s z*(ĉ_i + σ·ε_s),  regret = mean((opt_i - Y_i·z̃_i) / opt_i)"""
    pred_model.eval()
    rng = np.random.default_rng(seed)
    coeff_hat = pred_model(torch.FloatTensor(X)).numpy()
    true_objs = np.zeros(len(X))
    for _ in range(n_samples):
        noise = rng.standard_normal(coeff_hat.shape) * sigma
        pc = coeff_hat + noise
        for i in range(len(X)):
            sol, _, _ = ptoSolver.solve(pc[i])
            true_objs[i] += float(problem.get_objective(
                Y[i:i+1], np.array(sol)[np.newaxis],
                Y_aux[i:i+1] if Y_aux is not None else None,
            ).mean())
    true_objs /= n_samples
    return float(np.mean((opt_obj - true_objs) / (opt_obj + 1e-8)))


@torch.no_grad()
def compute_regret(pred_model, problem, ptoSolver, X, Y, Y_aux, opt_obj):
    pred_model.eval()
    X_t = torch.FloatTensor(X)
    coeff_hat = pred_model(X_t).numpy()
    total = 0.0
    for i in range(len(X)):
        sol, _, _ = ptoSolver.solve(coeff_hat[i])
        true_obj = float(problem.get_objective(
            Y[i:i+1], np.array(sol)[np.newaxis],
            Y_aux[i:i+1] if Y_aux is not None else None,
        ).mean())
        total += (opt_obj[i] - true_obj) / (opt_obj[i] + 1e-8)
    return total / len(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_grid", type=int, default=30)
    parser.add_argument("--alpha_min", type=float, default=-0.5)
    parser.add_argument("--alpha_max", type=float, default=1.5)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    # ---- Load problem + solver ----
    from openpto.config import load_conf, setup_seed
    from openpto.method.Predicts.dense import MLP
    from openpto.method.Solvers.wrapper_solver import solver_wrapper
    from openpto.problems.wrapper_prob import problem_wrapper

    setup_seed(2023)
    wrapper_args = types.SimpleNamespace(
        problem="knapsack", solver="heuristic",
        config_path="openpto/config/probs/knapsack_small.yaml",
        loadnew=False, instances=400, testinstances=200,
        val_frac=0.2, seed=2023, data_dir="./openpto/data/knapsack",
    )
    conf = load_conf(
        "openpto/config/probs/knapsack_small.yaml",
        "openpto/config/models/perturb_kn_s05_n100.yaml",
        "knapsack",
    )
    problem = problem_wrapper(wrapper_args, conf)
    ptoSolver = solver_wrapper(wrapper_args, conf, problem)

    X_val, Y_val, Y_aux_val = problem.get_val_data()
    X_train, Y_train, Y_aux_train = problem.get_train_data()
    print(f"Val set: {X_val.shape}  Train set: {X_train.shape}")

    # ---- Optimal objectives (val + train) ----
    print("Precomputing optimal val objectives...")
    opt_obj = np.array([
        float(problem.get_objective(
            Y_val[i:i+1],
            problem.get_decision(Y_val[i:i+1], {}, ptoSolver)[0],
            Y_aux_val[i:i+1] if Y_aux_val is not None else None,
        ).mean())
        for i in range(len(X_val))
    ])

    print("Precomputing optimal train objectives...")
    opt_obj_train = np.array([
        float(problem.get_objective(
            Y_train[i:i+1],
            problem.get_decision(Y_train[i:i+1], {}, ptoSolver)[0],
            Y_aux_train[i:i+1] if Y_aux_train is not None else None,
        ).mean())
        for i in range(len(X_train))
    ])

    # ---- Build model ----
    in_dim, out_dim = problem.get_model_shape()
    pred_model = MLP(
        num_features=in_dim, num_targets=out_dim,
        num_layers=2, intermediate_size=32,
        activation="relu", output_activation="identity",
    )

    sd_mse = torch.load(MSE_CKPT, map_location="cpu")
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_grid)

    # ---- Compute interpolation for each checkpoint ----
    all_regrets = {}
    all_perturb_obj = {}   # perturbed mean obj at sigma=0.01

    for m_idx, ckpt_info in enumerate(CHECKPOINTS):
        if not os.path.exists(ckpt_info["path"]):
            print(f"MISSING: {ckpt_info['path']}")
            continue

        sd_p = torch.load(ckpt_info["path"], map_location="cpu")
        regrets      = np.zeros(args.n_grid)
        perturb_vals = np.zeros(args.n_grid)

        print(f"\nInterpolating: {ckpt_info['short']}")
        for idx, alpha in enumerate(alphas):
            if idx % 10 == 0:
                print(f"  α={alpha:.2f} ({idx+1}/{args.n_grid})")
            pred_model.load_state_dict(
                interpolate_state_dicts(sd_mse, sd_p, alpha)
            )
            regrets[idx] = compute_regret(
                pred_model, problem, ptoSolver,
                X_val, Y_val, Y_aux_val, opt_obj,
            )
            perturb_vals[idx] = compute_perturbed_regret(
                pred_model, problem, ptoSolver,
                X_train, Y_train, Y_aux_train, opt_obj_train, sigma=0.01,
                n_samples=20, seed=m_idx * 1000 + idx,
            )

        key = ckpt_info["short"]
        all_regrets[key]     = regrets
        all_perturb_obj[key] = perturb_vals

        idx0 = np.argmin(np.abs(alphas - 0.0))
        idx1 = np.argmin(np.abs(alphas - 1.0))
        print(f"  α≈0 (MSE): hard={regrets[idx0]:.4f}  soft={perturb_vals[idx0]:.4f}"
              f"  |  α≈1: hard={regrets[idx1]:.4f}  soft={perturb_vals[idx1]:.4f}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    fig.suptitle(
        "Loss landscape extrapolation: θ(α) = (1−α)·θ_MSE + α·θ_perturb\n"
        f"Knapsack benchmark, val set ({len(X_val)} instances), DP solver, "
        f"α ∈ [{args.alpha_min}, {args.alpha_max}], {args.n_grid} grid points",
        fontsize=12, fontweight="bold",
    )

    ax_top, ax_bot = axes
    idx0_ref = np.argmin(np.abs(alphas - 0.0))

    # ---- Top subplot: decision regret ----
    ax_top.axhline(MSE_TEST_REGRET_DP, color="k", ls="--", lw=1.5,
                   label=f"MSE baseline, DP eval ({MSE_TEST_REGRET_DP:.4f})")

    for ckpt_info in CHECKPOINTS:
        key = ckpt_info["short"]
        if key not in all_regrets:
            continue
        regrets = all_regrets[key]
        ax_top.plot(alphas, regrets, color=ckpt_info["color"],
                    lw=2, label=ckpt_info["label"])
        idx0 = np.argmin(np.abs(alphas - 0.0))
        idx1 = np.argmin(np.abs(alphas - 1.0))
        ax_top.scatter([alphas[idx0]], [regrets[idx0]], color=ckpt_info["color"],
                       s=60, zorder=5, marker="o")
        ax_top.scatter([alphas[idx1]], [regrets[idx1]], color=ckpt_info["color"],
                       s=60, zorder=5, marker="s")

    ax_top.axvspan(0.0, 1.0, color="lightyellow", alpha=0.5, zorder=0)
    ax_top.axvline(0.0, color="green", ls=":", lw=1.5, alpha=0.8)
    ax_top.axvline(1.0, color="gray",  ls=":", lw=1.5, alpha=0.8)
    ax_top.set_ylim(0, 0.1)
    ax_top.set_ylabel("Normalised val regret", fontsize=11)
    ax_top.legend(fontsize=8, loc="upper left", ncol=2)
    ax_top.grid(True, alpha=0.3)

    # ---- Bottom subplot: absolute perturbed mean obj at σ=0.01 ----
    for ckpt_info in CHECKPOINTS:
        key = ckpt_info["short"]
        if key not in all_perturb_obj:
            continue
        perv = all_perturb_obj[key]
        ax_bot.plot(alphas, perv, color=ckpt_info["color"], lw=2,
                    label=ckpt_info["label"])
        idx0 = np.argmin(np.abs(alphas - 0.0))
        idx1 = np.argmin(np.abs(alphas - 1.0))
        ax_bot.scatter([alphas[idx0]], [perv[idx0]], color=ckpt_info["color"],
                       s=60, zorder=5, marker="o")
        ax_bot.scatter([alphas[idx1]], [perv[idx1]], color=ckpt_info["color"],
                       s=60, zorder=5, marker="s")

    ax_bot.axvspan(0.0, 1.0, color="lightyellow", alpha=0.5, zorder=0)
    ax_bot.axvline(0.0, color="green", ls=":", lw=1.5, alpha=0.8)
    ax_bot.axvline(1.0, color="gray",  ls=":", lw=1.5, alpha=0.8)
    ax_bot.set_ylabel("Normalised train regret  (soft decision, σ=0.01)", fontsize=11)
    ax_bot.set_xlabel("α  (0 = MSE weights, 1 = perturbed weights)", fontsize=11)
    ax_bot.legend(fontsize=8, loc="lower left", ncol=2)
    ax_bot.grid(True, alpha=0.3)

    for ax in axes:
        ax.text(0.01, ax.get_ylim()[1] * 0.97, "← MSE", fontsize=8,
                color="green", va="top", transform=ax.transData)
        ax.text(1.01, ax.get_ylim()[1] * 0.97, "perturb →", fontsize=8,
                color="gray", va="top")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {args.out}")


if __name__ == "__main__":
    main()
