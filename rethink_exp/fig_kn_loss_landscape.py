"""
Loss landscape interpolation: MSE weights ↔ perturbed weights.

Linearly interpolates between the two best checkpoints on a fine grid:
    θ(α) = (1 - α) * θ_mse  +  α * θ_perturb    α ∈ [0, 1]

At each grid point computes:
    1. Decision regret   — normalized regret under the CO solver
    2. MSE loss          — mean squared error vs true costs
    3. Perturbed objective — at several sigma values

Plots: x-axis = α, two panels side by side:
    Left:  decision regret (left y-axis, blue) + MSE (right y-axis, orange)
    Right: perturbed objective at each sigma (one line per sigma)

Usage:
    conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_loss_landscape.py
    conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_loss_landscape.py \\
        --n_grid 200 --sigmas 0.01 0.1 1.0 5.0 20.0
"""

import argparse
import os
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---- Paths ----
PERTURB_CKPT = (
    "saved_records/knapsack-gen/perturb_adaptive_sweep/"
    "kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense_best_pred.pt"
)
MSE_CKPT = (
    "saved_records/knapsack-gen/mse/"
    "kn_bench_mse_dense_lr5e-2/checkpoints/tr_pred_best.pt"
)
ADAPTIVE_BASE = "saved_records/knapsack-gen/perturb_adaptive_sweep"

DEFAULT_SIGMAS = [0.01, 0.1, 0.5, 1.0, 5.0, 20.0]


# ---- Regret / MSE / perturbed-obj computation ----

@torch.no_grad()
def compute_regret(pred_model, problem, ptoSolver, X, Y, Y_aux, opt_obj, device):
    """Normalized regret for a given pred_model on dataset (X, Y)."""
    pred_model.eval()
    X_t = torch.FloatTensor(X).to(device)
    coeff_hat = pred_model(X_t).detach().cpu().numpy()

    total_regret = 0.0
    n = len(X)
    for i in range(n):
        sol, obj, _ = ptoSolver.solve(coeff_hat[i])
        true_obj = float(problem.get_objective(
            Y[i:i+1], np.array(sol)[np.newaxis],
            Y_aux[i:i+1] if Y_aux is not None else None,
        ).mean())
        total_regret += (opt_obj[i] - true_obj) / (opt_obj[i] + 1e-8)
    return total_regret / n


@torch.no_grad()
def compute_mse(pred_model, X, Y, device):
    pred_model.eval()
    X_t = torch.FloatTensor(X).to(device)
    Y_t = torch.FloatTensor(Y).to(device)
    coeff_hat = pred_model(X_t)
    return F.mse_loss(coeff_hat, Y_t).item()


def compute_perturbed_obj(pred_model, problem, ptoSolver, X, Y, Y_aux,
                          opt_obj, sigma, n_samples, device):
    """Mean normalised regret under Gaussian-perturbed predicted costs."""
    pred_model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        coeff_hat = pred_model(X_t).detach().cpu().numpy()

    n = len(X)
    total = 0.0
    for i in range(n):
        c = coeff_hat[i]
        objs = []
        for _ in range(n_samples):
            noise = np.random.randn(*c.shape) * sigma
            sol, _, _ = ptoSolver.solve(c + noise)
            true_obj = float(problem.get_objective(
                Y[i:i+1], np.array(sol)[np.newaxis],
                Y_aux[i:i+1] if Y_aux is not None else None,
            ).mean())
            objs.append((opt_obj[i] - true_obj) / (opt_obj[i] + 1e-8))
        total += float(np.mean(objs))
    return total / n


def interpolate_state_dicts(sd_a, sd_b, alpha):
    return {k: (1.0 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
            for k in sd_a}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_grid", type=int, default=200)
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--n_samples_perturb", type=int, default=20)
    parser.add_argument("--perturb_ckpt", default=PERTURB_CKPT)
    parser.add_argument("--mse_ckpt", default=MSE_CKPT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Check checkpoints ----
    for path, name in [(args.perturb_ckpt, "perturbed"), (args.mse_ckpt, "MSE")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} checkpoint not found: {path}")
            sys.exit(1)

    # ---- Load problem + solver (same pattern as perturb_adaptive_sweep.py) ----
    from openpto.config import load_conf, setup_seed
    from openpto.method.Predicts.dense import MLP
    from openpto.method.Solvers.wrapper_solver import solver_wrapper
    from openpto.problems.wrapper_prob import problem_wrapper

    setup_seed(42)

    wrapper_args = types.SimpleNamespace(
        problem="knapsack",
        solver="heuristic",
        config_path="openpto/config/probs/knapsack_small.yaml",
        loadnew=False,
        instances=400,
        testinstances=200,
        val_frac=0.2,
        seed=42,
        data_dir="./openpto/data/knapsack",
    )
    conf = load_conf(
        "openpto/config/probs/knapsack_small.yaml",
        "openpto/config/models/perturb_s01_n5.yaml",
        "knapsack",
    )

    print("Loading problem...")
    problem = problem_wrapper(wrapper_args, conf)
    ptoSolver = solver_wrapper(wrapper_args, conf, problem)

    X_val, Y_val, Y_aux_val = problem.get_val_data()
    print(f"Val set: {X_val.shape}")

    # ---- Precompute optimal objectives ----
    print("Precomputing optimal objectives on val set...")
    opt_obj = np.array([
        float(problem.get_objective(
            Y_val[i:i+1],
            problem.get_decision(Y_val[i:i+1], {}, ptoSolver)[0],
            Y_aux_val[i:i+1] if Y_aux_val is not None else None,
        ).mean())
        for i in range(len(X_val))
    ])

    # ---- Load checkpoints ----
    sd_mse = torch.load(args.mse_ckpt, map_location="cpu")
    sd_perturb = torch.load(args.perturb_ckpt, map_location="cpu")
    print(f"Loaded MSE ckpt:     {args.mse_ckpt}")
    print(f"Loaded perturb ckpt: {args.perturb_ckpt}")

    # ---- Build model from problem shape ----
    in_dim, out_dim = problem.get_model_shape()
    pred_model = MLP(
        num_features=in_dim, num_targets=out_dim,
        num_layers=2, intermediate_size=32,
        activation="relu", output_activation="identity",
    ).to(device)

    # ---- Interpolation grid ----
    alphas = np.linspace(0.0, 1.0, args.n_grid)
    regrets = np.zeros(args.n_grid)
    mses = np.zeros(args.n_grid)
    perturbed_objs = {sigma: np.zeros(args.n_grid) for sigma in args.sigmas}

    print(f"\nInterpolating on {args.n_grid}-point grid "
          f"({len(args.sigmas)} sigmas × {args.n_samples_perturb} samples)...")
    for idx, alpha in enumerate(alphas):
        if idx % 20 == 0:
            print(f"  α={alpha:.3f}  ({idx}/{args.n_grid})")

        pred_model.load_state_dict(interpolate_state_dicts(sd_mse, sd_perturb, alpha))

        regrets[idx] = compute_regret(
            pred_model, problem, ptoSolver,
            X_val, Y_val, Y_aux_val, opt_obj, device,
        )
        mses[idx] = compute_mse(pred_model, X_val, Y_val, device)

        for sigma in args.sigmas:
            perturbed_objs[sigma][idx] = compute_perturbed_obj(
                pred_model, problem, ptoSolver,
                X_val, Y_val, Y_aux_val, opt_obj,
                sigma=sigma, n_samples=args.n_samples_perturb, device=device,
            )

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Loss landscape interpolation: θ(α) = (1−α)·θ_MSE + α·θ_perturb\n"
        f"Knapsack benchmark, val set ({len(X_val)} instances)",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: decision regret + MSE
    color_reg = "#2166ac"
    color_mse = "#d73027"

    ax1.plot(alphas, regrets, color=color_reg, lw=2, label="Decision regret (norm.)")
    ax1.set_xlabel("α  (0=MSE weights, 1=perturbed weights)", fontsize=11)
    ax1.set_ylabel("Decision regret (norm.)", color=color_reg, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_reg)
    ax1.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax1.axvline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax1r = ax1.twinx()
    ax1r.plot(alphas, mses, color=color_mse, lw=2, ls="--", label="MSE loss")
    ax1r.set_ylabel("MSE loss", color=color_mse, fontsize=11)
    ax1r.tick_params(axis="y", labelcolor=color_mse)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper center")
    ax1.set_title("Decision regret vs MSE along interpolation", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: perturbed objective at each sigma
    sigma_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(args.sigmas)))
    for sigma, color in zip(args.sigmas, sigma_colors):
        ax2.plot(alphas, perturbed_objs[sigma], color=color, lw=1.8, label=f"σ={sigma}")
    ax2.set_xlabel("α  (0=MSE weights, 1=perturbed weights)", fontsize=11)
    ax2.set_ylabel("Perturbed objective (mean norm. regret)", fontsize=11)
    ax2.set_title("Perturbed objective along interpolation (several σ)", fontsize=10)
    ax2.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax2.axvline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = args.out or os.path.join(ADAPTIVE_BASE, "fig_kn_loss_landscape.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()

    # ---- Summary ----
    print(f"\n--- Interpolation summary ---")
    print(f"α=0.00 (MSE):     regret={regrets[0]:.4f}  mse={mses[0]:.4f}")
    print(f"α=1.00 (perturb): regret={regrets[-1]:.4f}  mse={mses[-1]:.4f}")
    best_reg_idx = int(np.argmin(regrets))
    print(f"Best regret:      {regrets[best_reg_idx]:.4f} at α={alphas[best_reg_idx]:.3f}")
    for sigma in args.sigmas:
        best_idx = int(np.argmin(perturbed_objs[sigma]))
        print(f"  Perturbed obj σ={sigma}: min={perturbed_objs[sigma][best_idx]:.4f} "
              f"at α={alphas[best_idx]:.3f}")


if __name__ == "__main__":
    main()
