"""
fig_realdata_interpolation.py
-----------------------------
Loss landscape interpolation on a real problem (Cook County, TopK).

θ(α) = (1−α)·θ_MSE + α·θ_Perturb   for α ∈ [alpha_min, alpha_max]

At each α, evaluates on train/val/test:
    (1) Relative decision regret   (TopK solver on Y_pred, scored with Y)
    (2) Prediction MSE

Output: 6-panel figure (3 regret + 3 pred MSE).

Context (Committee Plan #6): On the synthetic knapsack, we showed the MSE
and Perturb endpoints are connected by a monotone descent path in regret.
Does that structure survive on a real, time-series task where data is
scarce and features are noisy?

Endpoints used (from bench_p1_best / bench_p2):
    MSE     : saved_records/cook_county-real/mse/bench_p1_mse_default_lr1e-3/
              checkpoints/tr_pred_best.pt               (test_regret=0.188)
    Perturb : saved_records/cook_county-real/perturb/
              bench_p2_perturb_s1p0_n50_default_lr1e-3/
              checkpoints/tr_pred_best.pt               (test_regret=0.211)

Usage:
    conda run -n pco_bench_rhel7 python rethink_exp/fig_realdata_interpolation.py
    python rethink_exp/fig_realdata_interpolation.py --n_grid 40
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
    "saved_records/cook_county-real/mse/"
    "bench_p1_mse_default_lr1e-3/checkpoints/tr_pred_best.pt"
)
PERTURB_CKPT = (
    "saved_records/cook_county-real/perturb/"
    "bench_p2_perturb_s1p0_n50_default_lr1e-3/checkpoints/tr_pred_best.pt"
)
MSE_TEST_REGRET     = 0.1880
PERTURB_TEST_REGRET = 0.2107

OUT_PATH = "results/fig_realdata_interpolation.png"


def interpolate_state_dicts(sd_a, sd_b, alpha):
    return {k: (1.0 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
            for k in sd_a}


@torch.no_grad()
def eval_split(pred_model, problem, ptoSolver, X, Y, budget):
    """Return (relative_regret, pred_mse) for one split.

    X: (T, S, F) tensor    Y: (T, S, 1) tensor
    - Pred MSE : mean((model(X) − Y)²) over (T, S, 1)
    - Regret   : per-timestep (opt − achieved)/opt, averaged over T.
    """
    pred_model.eval()
    X_t = X if isinstance(X, torch.Tensor) else torch.FloatTensor(X)
    Y_t = Y if isinstance(Y, torch.Tensor) else torch.FloatTensor(Y)
    Y_hat = pred_model(X_t.float())

    pred_mse = float(((Y_hat - Y_t) ** 2).mean().item())

    Z_opt    = ptoSolver.solve(Y_t,   budget)     # (T, S)
    Z_pred   = ptoSolver.solve(Y_hat, budget)     # (T, S)
    opt_obj  = problem.get_objective(Y_t, Z_opt).cpu().numpy()
    ach_obj  = problem.get_objective(Y_t, Z_pred).cpu().numpy()
    rel_reg  = float(np.mean((opt_obj - ach_obj) / (opt_obj + 1e-8)))
    return rel_reg, pred_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_grid", type=int, default=30)
    parser.add_argument("--alpha_min", type=float, default=-0.5)
    parser.add_argument("--alpha_max", type=float, default=1.5)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    from openpto.config import load_conf, setup_seed
    from openpto.method.Predicts.dense import MLP
    from openpto.method.Solvers.wrapper_solver import solver_wrapper
    from openpto.problems.wrapper_prob import problem_wrapper

    setup_seed(2023)

    wrapper_args = types.SimpleNamespace(
        problem="cook_county", solver="heuristic",
        config_path="openpto/config/probs/cook_county.yaml",
        loadnew=False, instances=400, testinstances=200,
        val_frac=0.2, seed=2023,
        data_dir="./openpto/data/cook_county",
    )
    conf = load_conf(
        "openpto/config/probs/cook_county.yaml",
        "openpto/config/models/default.yaml",
        "cook_county",
    )
    problem   = problem_wrapper(wrapper_args, conf)
    ptoSolver = solver_wrapper(wrapper_args, conf, problem)
    budget    = problem.budget

    X_train, Y_train, _ = problem.get_train_data()
    X_val,   Y_val,   _ = problem.get_val_data()
    X_test,  Y_test,  _ = problem.get_test_data()
    print(f"train: X={tuple(X_train.shape)}  Y={tuple(Y_train.shape)}")
    print(f"val  : X={tuple(X_val.shape)}    Y={tuple(Y_val.shape)}")
    print(f"test : X={tuple(X_test.shape)}   Y={tuple(Y_test.shape)}")

    in_dim, out_dim = problem.get_model_shape()
    pred_model = MLP(
        num_features=in_dim, num_targets=out_dim,
        num_layers=2, intermediate_size=32,
        activation="relu", output_activation="none",
    )

    sd_mse     = torch.load(MSE_CKPT, map_location="cpu")
    sd_perturb = torch.load(PERTURB_CKPT, map_location="cpu")

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_grid)
    splits = [
        ("train", X_train, Y_train),
        ("val",   X_val,   Y_val),
        ("test",  X_test,  Y_test),
    ]
    regrets = {name: np.zeros(args.n_grid) for name, _, _ in splits}
    mses    = {name: np.zeros(args.n_grid) for name, _, _ in splits}

    print(f"\nInterpolating {args.n_grid} grid points, α ∈ "
          f"[{args.alpha_min}, {args.alpha_max}]")
    for idx, alpha in enumerate(alphas):
        pred_model.load_state_dict(
            interpolate_state_dicts(sd_mse, sd_perturb, alpha)
        )
        for name, X, Y in splits:
            rel_reg, pred_mse = eval_split(pred_model, problem, ptoSolver,
                                           X, Y, budget)
            regrets[name][idx] = rel_reg
            mses[name][idx]    = pred_mse
        if idx % 5 == 0 or idx == args.n_grid - 1:
            print(f"  α={alpha:+.2f}  "
                  f"regret(tr/va/te)="
                  f"{regrets['train'][idx]:.4f}/"
                  f"{regrets['val'][idx]:.4f}/"
                  f"{regrets['test'][idx]:.4f}  "
                  f"mse(tr/va/te)="
                  f"{mses['train'][idx]:.3f}/"
                  f"{mses['val'][idx]:.3f}/"
                  f"{mses['test'][idx]:.3f}")

    # ---- Plot: 2 rows × 3 cols ----
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), sharex=True)
    split_names = ["train", "val", "test"]
    split_colors = {"train": "#1b7837", "val": "#762a83", "test": "#b35806"}

    for col, name in enumerate(split_names):
        ax_r = axes[0, col]
        ax_r.plot(alphas, regrets[name], color=split_colors[name], lw=2.0)
        ax_r.axvspan(0.0, 1.0, color="lightyellow", alpha=0.6, zorder=0)
        ax_r.axvline(0.0, color="green", ls=":", lw=1.4, alpha=0.9)
        ax_r.axvline(1.0, color="gray",  ls=":", lw=1.4, alpha=0.9)
        idx0 = np.argmin(np.abs(alphas - 0.0))
        idx1 = np.argmin(np.abs(alphas - 1.0))
        ax_r.scatter([alphas[idx0]], [regrets[name][idx0]], color="green",
                     s=55, zorder=5, marker="o", label=f"MSE α=0: {regrets[name][idx0]:.3f}")
        ax_r.scatter([alphas[idx1]], [regrets[name][idx1]], color="gray",
                     s=55, zorder=5, marker="s", label=f"Perturb α=1: {regrets[name][idx1]:.3f}")
        ax_r.set_title(f"{name}: relative regret", fontsize=11, fontweight="bold")
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(fontsize=8, loc="best")
        if col == 0:
            ax_r.set_ylabel("Relative regret\n(opt − achieved) / opt", fontsize=10)

        ax_m = axes[1, col]
        ax_m.plot(alphas, mses[name], color=split_colors[name], lw=2.0)
        ax_m.axvspan(0.0, 1.0, color="lightyellow", alpha=0.6, zorder=0)
        ax_m.axvline(0.0, color="green", ls=":", lw=1.4, alpha=0.9)
        ax_m.axvline(1.0, color="gray",  ls=":", lw=1.4, alpha=0.9)
        ax_m.scatter([alphas[idx0]], [mses[name][idx0]], color="green",
                     s=55, zorder=5, marker="o", label=f"MSE α=0: {mses[name][idx0]:.3f}")
        ax_m.scatter([alphas[idx1]], [mses[name][idx1]], color="gray",
                     s=55, zorder=5, marker="s", label=f"Perturb α=1: {mses[name][idx1]:.3f}")
        ax_m.set_title(f"{name}: prediction MSE", fontsize=11, fontweight="bold")
        ax_m.set_xlabel("α   (0 = MSE weights, 1 = Perturb weights)", fontsize=10)
        ax_m.grid(True, alpha=0.3)
        ax_m.legend(fontsize=8, loc="best")
        if col == 0:
            ax_m.set_ylabel("Prediction MSE  (Y_pred vs Y)", fontsize=10)

    fig.suptitle(
        "Cook County (real, TopK, time-series): loss landscape between "
        "MSE and Perturb endpoints\n"
        f"θ(α) = (1−α)·θ_MSE + α·θ_Perturb, "
        f"α ∈ [{args.alpha_min:g}, {args.alpha_max:g}], "
        f"{args.n_grid} grid points",
        fontsize=12.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        p = args.out.rsplit(".", 1)[0] + "." + ext
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)

    # ---- npz cache ----
    npz_path = args.out.rsplit(".", 1)[0] + ".npz"
    np.savez(npz_path, alphas=alphas,
             **{f"regret_{n}": regrets[n] for n in split_names},
             **{f"mse_{n}":    mses[n]    for n in split_names})
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
