"""
Exp 1: Gradient projection along the MSE → perturbed interpolation path.

At 30 points θ(α) = (1−α)·θ_MSE + α·θ_perturb, compute:
  1. Decision regret (val set, no grad)
  2. G(α) = ∇_θ L_perturb(θ(α)) via loss.backward()
  3. v = (θ_MSE − θ_perturb) / ‖...‖  — unit vector pointing MSE-ward
  4. proj(α) = G(α) · v   (positive → gradient pushes toward MSE)
  5. ‖G(α)‖               (gradient norm)

3-panel figure:
  Panel 1: decision regret vs α
  Panel 2: proj(α) vs α   — key diagnostic
  Panel 3: ‖G(α)‖ vs α

Interpretation:
  If proj(α=1) < 0, the gradient at the perturbed minimum actively pushes
  *away* from the MSE solution — confirming the local minimum picture.
  If proj(α=1) ≈ 0, the gradient is orthogonal — saddle or flat.
  If proj(α=1) > 0, gradient points toward MSE — GD should escape (but doesn't?).

Output: saved_records/knapsack-gen/perturb_adaptive_sweep/fig_kn_gradient_projection.png

Usage (run on login node, ~5-10 min):
  conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_gradient_projection.py
  python rethink_exp/fig_kn_gradient_projection.py --n_grid 30 --n_samples 50
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

# ---- Paths ----
PERTURB_CKPT = (
    "saved_records/knapsack-gen/perturb_adaptive_sweep/"
    "kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense_best_pred.pt"
)
PERTURB_NPZ = (
    "saved_records/knapsack-gen/perturb_adaptive_sweep/"
    "kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense.npz"
)
MSE_CKPT = (
    "saved_records/knapsack-gen/mse/"
    "kn_bench_mse_dense_lr5e-2/checkpoints/tr_pred_best.pt"
)
OUT_DIR = "saved_records/knapsack-gen/perturb_adaptive_sweep"

sys.path.insert(0, ".")


# ---- Helpers ----

def interpolate_state_dicts(sd_a, sd_b, alpha):
    """θ(α) = (1-α)·sd_a + α·sd_b"""
    return {k: (1.0 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
            for k in sd_a}


def flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def flat_grad(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros_like(p).flatten())
    return torch.cat(grads)


@torch.no_grad()
def compute_regret(pred_model, problem, ptoSolver, X, Y, Y_aux, opt_obj, device):
    pred_model.eval()
    X_t = torch.FloatTensor(X).to(device)
    coeff_hat = pred_model(X_t).detach().cpu().numpy()
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
    parser.add_argument("--n_grid", type=int, default=30,
                        help="Number of interpolation points")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Perturbation samples for gradient computation")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Sigma override (default: read from NPZ final epoch)")
    parser.add_argument("--perturb_ckpt", default=PERTURB_CKPT)
    parser.add_argument("--mse_ckpt", default=MSE_CKPT)
    parser.add_argument("--perturb_npz", default=PERTURB_NPZ)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Check checkpoints ----
    for path, name in [(args.perturb_ckpt, "perturbed"), (args.mse_ckpt, "MSE")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} checkpoint not found: {path}")
            sys.exit(1)

    # ---- Load sigma ----
    if args.sigma is not None:
        sigma = args.sigma
    elif os.path.exists(args.perturb_npz):
        npz = np.load(args.perturb_npz)
        sigma = float(npz["adaptive_sigma"][-1])
        print(f"Loaded sigma={sigma:.4f} from NPZ (final epoch mean)")
    else:
        sigma = 1.0
        print(f"NPZ not found, using default sigma={sigma}")

    # ---- Load problem + solver ----
    from openpto.config import load_conf, setup_seed
    from openpto.method.Models.perturbed import perturbed as PerturbModel
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
        "openpto/config/models/perturb_kn_s1_n100.yaml",
        "knapsack",
    )

    print("Loading problem...")
    problem = problem_wrapper(wrapper_args, conf)
    ptoSolver = solver_wrapper(wrapper_args, conf, problem)

    X_val, Y_val, Y_aux_val = problem.get_val_data()
    X_train, Y_train, Y_aux_train = problem.get_train_data()
    print(f"Val set: {X_val.shape}  Train set: {X_train.shape}")

    # ---- Precompute optimal objectives on val set ----
    print("Precomputing optimal objectives on val set...")
    opt_obj_val = np.array([
        float(problem.get_objective(
            Y_val[i:i+1],
            problem.get_decision(Y_val[i:i+1], {}, ptoSolver)[0],
            Y_aux_val[i:i+1] if Y_aux_val is not None else None,
        ).mean())
        for i in range(len(X_val))
    ])
    print(f"  opt_obj_val mean: {opt_obj_val.mean():.3f}")

    # ---- Build perturbed loss function ----
    loss_fn = PerturbModel(
        ptoSolver=ptoSolver,
        n_samples=args.n_samples,
        sigma=sigma,
        noise="normal",
        seed=42,
    )
    loss_fn.eval()

    # ---- Load checkpoints ----
    sd_mse = torch.load(args.mse_ckpt, map_location="cpu")
    sd_perturb = torch.load(args.perturb_ckpt, map_location="cpu")
    print(f"Loaded MSE ckpt:     {args.mse_ckpt}")
    print(f"Loaded perturb ckpt: {args.perturb_ckpt}")

    # ---- Build model ----
    in_dim, out_dim = problem.get_model_shape()
    pred_model = MLP(
        num_features=in_dim, num_targets=out_dim,
        num_layers=2, intermediate_size=32,
        activation="relu", output_activation="identity",
    ).to(device)

    # ---- Compute unit vector v = (θ_MSE − θ_perturb) / ‖...‖ ----
    pred_model.load_state_dict(sd_mse)
    theta_mse = flat_params(pred_model)
    pred_model.load_state_dict(sd_perturb)
    theta_perturb = flat_params(pred_model)

    v_flat = (theta_mse - theta_perturb)
    v_norm = v_flat.norm().item()
    v_flat = v_flat / (v_norm + 1e-10)
    print(f"\n‖θ_MSE − θ_perturb‖ = {v_norm:.4f}")

    # ---- Interpolation grid ----
    alphas = np.linspace(0.0, 1.0, args.n_grid)
    regrets = np.zeros(args.n_grid)
    projs   = np.zeros(args.n_grid)
    gnorms  = np.zeros(args.n_grid)

    # Prepare training data tensors for gradient computation
    X_train_t   = torch.FloatTensor(X_train).to(device)
    Y_train_t   = torch.FloatTensor(Y_train).to(device)
    Y_aux_train_t = Y_aux_train  # kept as numpy for solver calls inside loss_fn

    print(f"\nGradient projection grid: {args.n_grid} points, "
          f"sigma={sigma:.4f}, n_samples={args.n_samples}")
    print("(gradient at each point uses full training set)")

    for idx, alpha in enumerate(alphas):
        if idx % 5 == 0:
            print(f"  α={alpha:.3f}  ({idx+1}/{args.n_grid})")

        # Set model weights to interpolated point
        pred_model.load_state_dict(
            interpolate_state_dicts(sd_mse, sd_perturb, alpha)
        )

        # Compute decision regret on val set (no grad)
        regrets[idx] = compute_regret(
            pred_model, problem, ptoSolver,
            X_val, Y_val, Y_aux_val, opt_obj_val, device,
        )

        # Compute perturbed gradient on training set
        pred_model.train()
        for p in pred_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        coeff_hat = pred_model(X_train_t)
        loss = loss_fn(
            problem,
            coeff_hat,
            params=Y_aux_train_t,
            coeff_true=Y_train_t,
            reduction="mean",
        )
        loss.backward()

        G = flat_grad(pred_model)
        projs[idx]  = torch.dot(G, v_flat).item()
        gnorms[idx] = G.norm().item()
        pred_model.eval()

    # ---- Derived quantities ----
    # GD update direction is -G; project onto v (toward MSE)
    # gd_proj > 0  →  GD step moves toward MSE
    # gd_proj < 0  →  GD step moves away from MSE
    gd_projs = -np.array(projs)           # (-G) · v
    frac_pct  = np.abs(gd_projs) / (np.array(gnorms) + 1e-12) * 100  # % of ‖G‖

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Gradient projection: θ(α) = (1−α)·θ_MSE + α·θ_perturb\n"
        f"Knapsack benchmark, σ={sigma:.3f}, n_samples={args.n_samples}, "
        f"n_grid={args.n_grid}",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: decision regret vs α
    ax = axes[0]
    ax.plot(alphas, regrets, color="#2166ac", lw=2)
    ax.axvline(0.0, color="green",  ls="--", lw=1, label="MSE weights (α=0)")
    ax.axvline(1.0, color="#d73027", ls="--", lw=1, label="Perturb weights (α=1)")
    ax.set_xlabel("α  (0=MSE, 1=perturbed)", fontsize=11)
    ax.set_ylabel("Normalised val regret", fontsize=11)
    ax.set_title("Decision regret", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: GD projection vs α + ±‖G‖ envelope to show scale
    # gd_proj = (-G)·v: positive = GD step moves toward MSE
    ax = axes[1]
    # Grey envelope shows ±‖G‖ so the projection can be judged against total gradient scale
    ax.fill_between(alphas, -gnorms, gnorms, color="gray", alpha=0.12,
                    label="±‖G‖  (total gradient scale)")
    ax.axhline(0.0, color="k", ls="-", lw=0.8, alpha=0.5)
    ax.plot(alphas, gd_projs, color="#d73027", lw=2, label="(−G)·v  (GD toward MSE component)")
    ax.fill_between(alphas, gd_projs, 0, where=(gd_projs > 0),
                    color="green", alpha=0.3, label="GD moves toward MSE")
    ax.fill_between(alphas, gd_projs, 0, where=(gd_projs < 0),
                    color="red", alpha=0.3, label="GD moves away from MSE")
    ax.axvline(0.0, color="green",  ls="--", lw=1)
    ax.axvline(1.0, color="#d73027", ls="--", lw=1)
    # Annotate the α=1 value
    ax.annotate(
        f"α=1: {gd_projs[-1]:+.3f}",
        xy=(alphas[-1], gd_projs[-1]),
        xytext=(alphas[-1] - 0.25, gd_projs[-1] + gnorms[-1] * 0.15),
        fontsize=9, color="#d73027",
        arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.2),
    )
    ax.set_xlabel("α  (0=MSE, 1=perturbed)", fontsize=11)
    ax.set_ylabel("(−G)·v", fontsize=11)
    ax.set_title("GD update component toward MSE\n"
                 "Grey band = ±‖G‖ (total gradient scale)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 3: fraction of gradient in MSE direction (%)
    ax = axes[2]
    ax.plot(alphas, frac_pct, color="#4dac26", lw=2)
    ax.axhline(100.0, color="gray", ls=":", lw=1, alpha=0.6, label="100% (all gradient toward MSE)")
    ax.axvline(0.0, color="green",  ls="--", lw=1, label="MSE weights")
    ax.axvline(1.0, color="#d73027", ls="--", lw=1, label="Perturb weights")
    # Annotate α=1
    ax.annotate(
        f"α=1: {frac_pct[-1]:.1f}%",
        xy=(alphas[-1], frac_pct[-1]),
        xytext=(alphas[-1] - 0.30, frac_pct[-1] + 2),
        fontsize=9, color="#d73027",
        arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.2),
    )
    ax.set_xlabel("α  (0=MSE, 1=perturbed)", fontsize=11)
    ax.set_ylabel("|(−G)·v| / ‖G‖  (%)", fontsize=11)
    ax.set_title("Fraction of gradient\naligned with MSE direction", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Print key values
    print("\n=== Key values ===")
    print(f"  GD proj at α=0 (MSE weights):     {gd_projs[0]:+.4f}  ({frac_pct[0]:.1f}% of ‖G‖)")
    print(f"  GD proj at α=1 (perturb weights): {gd_projs[-1]:+.4f}  ({frac_pct[-1]:.1f}% of ‖G‖)")
    print(f"  regret at α=0:                    {regrets[0]:.4f}")
    print(f"  regret at α=1:                    {regrets[-1]:.4f}")
    print(f"  ‖G‖ at α=0:                       {gnorms[0]:.4f}")
    print(f"  ‖G‖ at α=1:                       {gnorms[-1]:.4f}")
    sign_str = "toward" if gd_projs[-1] > 0 else "away from"
    print(f"\n  → At perturbed weights: GD nudges {sign_str} MSE, "
          f"but only {frac_pct[-1]:.1f}% of gradient is in that direction.")

    out_path = args.out or os.path.join(OUT_DIR, "fig_kn_gradient_projection.png")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")

    # Save raw data
    npz_path = out_path.replace(".png", "_data.npz")
    np.savez(npz_path, alphas=alphas, regrets=regrets, projs=projs, gnorms=gnorms,
             sigma=sigma, n_samples=args.n_samples, v_norm=v_norm)
    print(f"Data saved to:   {npz_path}")


if __name__ == "__main__":
    main()
