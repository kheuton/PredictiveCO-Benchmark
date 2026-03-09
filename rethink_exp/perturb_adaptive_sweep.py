"""
Adaptive sigma sweep for the perturbed optimizer.

Tests AdaptiveSigmaPerturb with two controller modes:

  proportional  — sigma *= clip(ocv_target / ocv_y, 1/max_step, max_step)
                  Sweep: ocv_target values over a log-spaced range, fixed sigma_init=1.0.

  feedback      — sigma *= exp(alpha * (ocv_y - ocv_target))
                  Sweep: 5 ocv_target values × 2 sigma_init values.

Both dense and poly prediction models are tested.
Logs the same per-epoch diagnostics as perturb_sigma_sweep.py, plus
`adaptive_sigma`, `ocv_y`, and `frac_improving` (which now evolve each epoch).

Usage:
    python rethink_exp/perturb_adaptive_sweep.py \\
        --mode proportional \\
        --prefix adapt_prop_run1

    python rethink_exp/perturb_adaptive_sweep.py \\
        --mode feedback \\
        --prefix adapt_fb_run1
"""

import argparse
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

from openpto.config import load_conf, setup_seed
from openpto.diagnostics.perturb_metrics import (
    fd_gradient,
    interiority_metrics,
    rank_change_rate,
)
from openpto.method.Models.perturb_diag import AdaptiveSigmaPerturb
from openpto.method.Predicts.dense import MLP
from openpto.method.Predicts.poly_model import PolyPredModel
from openpto.method.Solvers.wrapper_solver import solver_wrapper
from openpto.method.utils_method import to_array, to_tensor
from openpto.problems.wrapper_prob import problem_wrapper


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Adaptive sigma sweep for perturbed optimizer")

    # Controller
    p.add_argument("--mode", type=str, default="proportional",
                   choices=["proportional", "feedback"],
                   help="Adaptive sigma controller mode.")

    # Proportional sweep params
    p.add_argument("--prop_targets", type=float, nargs="+", default=None,
                   help="Explicit list of ocv_target values. Overrides --n_targets/--target_min/--target_max.")
    p.add_argument("--n_targets", type=int, default=15,
                   help="Number of ocv_target values (proportional mode).")
    p.add_argument("--target_min", type=float, default=0.01)
    p.add_argument("--target_max", type=float, default=2.0)
    p.add_argument("--prop_sigma_init", type=float, default=1.0,
                   help="Starting sigma for proportional mode.")
    p.add_argument("--max_step", type=float, default=10.0,
                   help="Max multiplicative sigma change per epoch (proportional).")

    # Feedback sweep params
    p.add_argument(
        "--fb_targets", type=float, nargs="+",
        default=[0.10, 0.20, 0.30, 0.50, 0.80],
        help="ocv_target values to sweep (feedback mode).",
    )
    p.add_argument(
        "--fb_sigma_inits", type=float, nargs="+",
        default=[0.1, 1.0],
        help="Starting sigma values to sweep (feedback mode).",
    )
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Controller gain (feedback mode).")

    # Shared controller params
    p.add_argument("--sigma_min", type=float, default=1e-4)
    p.add_argument("--sigma_max", type=float, default=1e3)

    # Problem
    p.add_argument("--problem", type=str, default="cubic")
    p.add_argument("--solver", type=str, default="heuristic")
    p.add_argument("--config_path", type=str, default="")
    p.add_argument("--loadnew", action="store_true")

    # Prediction models
    p.add_argument(
        "--pred_models", type=str, nargs="+", default=["dense", "poly"],
    )
    p.add_argument("--poly_deg", type=int, default=3)
    p.add_argument("--poly_offset", type=float, default=0.0)

    # Training
    p.add_argument("--n_epochs", type=int, default=300)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--noise", type=str, default="normal", choices=["normal", "gumbel"])
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_hidden", type=int, default=32)

    # FD gradient
    p.add_argument("--fd_epsilon", type=float, default=1e-3)
    p.add_argument("--fd_freq", type=int, default=1)
    p.add_argument("--fd_max_instances", type=int, default=32)

    # Dataset
    p.add_argument("--instances", type=int, default=250)
    p.add_argument("--testinstances", type=int, default=400)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2023)

    # Output
    p.add_argument("--prefix", type=str, default="default")
    p.add_argument("--gpu", type=str, default="-1")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers (identical to perturb_sigma_sweep.py)
# ---------------------------------------------------------------------------

def make_args_namespace(args):
    ns = types.SimpleNamespace()
    ns.problem = args.problem
    ns.solver = args.solver
    ns.config_path = args.config_path
    ns.loadnew = args.loadnew
    ns.instances = args.instances
    ns.testinstances = args.testinstances
    ns.val_frac = args.val_frac
    ns.seed = args.seed
    ns.data_dir = os.path.join("./openpto/data/", args.problem)
    return ns


def build_pred_model(model_type, ipdim, opdim, args):
    if model_type == "dense":
        return MLP(
            num_features=ipdim,
            num_targets=opdim,
            num_layers=args.n_layers,
            intermediate_size=args.n_hidden,
            activation="relu",
            output_activation="identity",
        )
    elif model_type == "poly":
        return PolyPredModel(
            num_features=ipdim,
            num_targets=opdim,
            poly_deg=args.poly_deg,
            offset=args.poly_offset,
        )
    else:
        raise ValueError(f"Unknown pred_model type: {model_type}")


def compute_regret(problem, ptoSolver, pred_model, X, Y, Y_aux, opt_obj, device):
    pred_model.eval()
    with torch.no_grad():
        preds = pred_model(X.to(device)).detach().cpu()
    z, _ = problem.get_decision(preds, Y_aux, ptoSolver, **problem.init_API())
    z = to_tensor(z)
    obj = problem.get_objective(Y, z, Y_aux)
    obj = to_tensor(obj).float()
    opt = to_tensor(opt_obj).float()
    regret = (opt - obj).abs() / (opt.abs() + 1e-8)
    pred_model.train()
    return regret.mean().item()


def precompute_opt(problem, ptoSolver, Y, Y_aux):
    z_star, _ = problem.get_decision(Y, Y_aux, ptoSolver, **problem.init_API())
    z_star = to_tensor(z_star)
    obj_star = problem.get_objective(Y, z_star, Y_aux)
    return to_tensor(obj_star).float()


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------

def run_one(
    ocv_target, sigma_init, controller_mode, model_type,
    problem, ptoSolver, args, device,
    X_train, Y_train, Y_aux_train,
    X_val,   Y_val,   Y_aux_val,
    opt_obj_train, opt_obj_val,
):
    """
    Train a pred model from random init with AdaptiveSigmaPerturb,
    logging per-epoch diagnostics. Returns a dict of per-epoch arrays.
    """
    ipdim, opdim = problem.get_model_shape()
    pred_model = build_pred_model(model_type, ipdim, opdim, args).to(device)
    pred_model.train()

    loss_fn = AdaptiveSigmaPerturb(
        ptoSolver,
        n_samples=args.n_samples,
        sigma=sigma_init,
        noise=args.noise,
        ocv_target=ocv_target,
        mode=controller_mode,
        max_step=args.max_step,
        alpha=args.alpha,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        reduction="mean",
    )

    optimizer = torch.optim.Adam(pred_model.parameters(), lr=args.lr)
    model_args = {"reduction": "mean"}

    logs = {k: [] for k in [
        "epoch", "loss", "train_regret", "val_regret",
        "coeff_norm", "adaptive_sigma",
        "grad_norm", "fd_grad_norm", "cosine_sim",
        "softness", "dist_binary", "entropy",
        "rank_change_rate", "ocv_y", "frac_improving",
    ]}

    for epoch in range(1, args.n_epochs + 1):
        pred_model.train()

        preds = pred_model(X_train.to(device))

        # Capture gradient w.r.t. preds via hook
        captured_grad = {}
        def _hook(g):
            captured_grad["g"] = g.detach().cpu()
        preds.register_hook(_hook)

        loss = loss_fn(
            problem,
            coeff_hat=preds,
            coeff_true=Y_train.to(device),
            params=Y_aux_train,
            **model_args,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Adaptive sigma update (computes RCR from accumulated batch data) ----
        current_sigma = loss_fn.step(epoch)

        # ---- Diagnostics (no_grad) ----
        with torch.no_grad():
            coeff_hat_cpu = preds.detach().cpu()
            coeff_norm = coeff_hat_cpu.abs().mean().item()

            # Interiority of soft decision
            z_bar = loss_fn.last_z_bar
            intr = interiority_metrics(z_bar) if z_bar is not None else \
                   {"softness": float("nan"), "dist_binary": float("nan"), "entropy": float("nan")}

            # Rank change rate (recompute from this epoch's accumulated data, already cleared
            # by step() — use last_perturbed_solutions from the final batch as a proxy)
            perturbed_sols = loss_fn.last_perturbed_solutions
            if perturbed_sols is not None:
                z0, _ = problem.get_decision(
                    coeff_hat_cpu, Y_aux_train, ptoSolver, **problem.init_API()
                )
                z0 = to_tensor(z0).float()
                rcr = rank_change_rate(
                    perturbed_sols.float(), z0,
                    K=getattr(problem, "budget", 1),
                )
            else:
                rcr = float("nan")

            # Backprop gradient norm
            bp_grad = captured_grad.get("g")
            grad_norm = bp_grad.norm().item() if bp_grad is not None else float("nan")

            # FD gradient (every fd_freq epochs)
            if epoch % args.fd_freq == 0:
                fd_grad = fd_gradient(
                    coeff_hat_cpu,
                    Y_train.cpu(),
                    problem, ptoSolver, Y_aux_train,
                    epsilon=args.fd_epsilon,
                    opt_obj=opt_obj_train,
                    max_instances=args.fd_max_instances,
                )
                fd_grad_norm = fd_grad.norm().item()
                if bp_grad is not None:
                    bp_sub = bp_grad[:args.fd_max_instances].reshape(-1)
                    fd_sub = fd_grad.reshape(-1).to(bp_sub.dtype)
                    cos_sim = F.cosine_similarity(
                        bp_sub.unsqueeze(0), fd_sub.unsqueeze(0)
                    ).item()
                else:
                    cos_sim = float("nan")
            else:
                fd_grad_norm = float("nan")
                cos_sim = float("nan")

            # Regrets
            train_regret = compute_regret(
                problem, ptoSolver, pred_model,
                X_train, Y_train, Y_aux_train, opt_obj_train, device,
            )
            val_regret = compute_regret(
                problem, ptoSolver, pred_model,
                X_val, Y_val, Y_aux_val, opt_obj_val, device,
            )

        logs["epoch"].append(epoch)
        logs["loss"].append(loss.item())
        logs["train_regret"].append(train_regret)
        logs["val_regret"].append(val_regret)
        logs["coeff_norm"].append(coeff_norm)
        logs["adaptive_sigma"].append(current_sigma)
        logs["grad_norm"].append(grad_norm)
        logs["fd_grad_norm"].append(fd_grad_norm)
        logs["cosine_sim"].append(cos_sim)
        logs["softness"].append(intr["softness"])
        logs["dist_binary"].append(intr["dist_binary"])
        logs["entropy"].append(intr["entropy"])
        logs["rank_change_rate"].append(rcr)
        logs["ocv_y"].append(loss_fn.last_ocv_y)
        logs["frac_improving"].append(loss_fn.last_frac_improving)

        if epoch % 10 == 0:
            print(
                f"  epoch {epoch:3d} | loss {loss.item():.4f} | "
                f"val_regret {val_regret:.4f} | "
                f"ocv_y {loss_fn.last_ocv_y:.4f} | sigma {current_sigma:.4f}"
            )

    return {k: np.array(v) for k, v in logs.items()}


# ---------------------------------------------------------------------------
# Build sweep grid
# ---------------------------------------------------------------------------

def build_sweep_grid(args):
    """
    Returns a list of (ocv_target, sigma_init, label) tuples.
    label is used as the filename stem.
    """
    if args.mode == "proportional":
        if args.prop_targets is not None:
            targets = args.prop_targets
        else:
            targets = np.logspace(
                np.log10(args.target_min), np.log10(args.target_max), args.n_targets
            ).tolist()
        return [
            (t, args.prop_sigma_init, f"prop_t{t:.3f}_s{args.prop_sigma_init:.3g}")
            for t in targets
        ]
    else:  # feedback
        return [
            (t, s, f"fb_t{t:.3f}_s{s:.3g}")
            for t in args.fb_targets
            for s in args.fb_sigma_inits
        ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu"
    )
    print(f"Device: {device}")

    # ---- Load problem + solver ----
    wrapper_args = make_args_namespace(args)
    conf = load_conf(args.config_path, "openpto/config/models/perturb_s01_n5.yaml", args.problem)
    problem = problem_wrapper(wrapper_args, conf)
    ptoSolver = solver_wrapper(wrapper_args, conf, problem)

    X_train, Y_train, Y_aux_train = problem.get_train_data()
    X_val,   Y_val,   Y_aux_val   = problem.get_val_data()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # ---- Precompute optimal objectives ----
    print("Precomputing optimal objectives...")
    opt_obj_train = precompute_opt(problem, ptoSolver, Y_train, Y_aux_train)
    opt_obj_val   = precompute_opt(problem, ptoSolver, Y_val,   Y_aux_val)

    # ---- Sweep grid ----
    sweep = build_sweep_grid(args)
    print(f"\nMode: {args.mode} | {len(sweep)} ocv_target configs × {len(args.pred_models)} models "
          f"= {len(sweep) * len(args.pred_models)} runs")
    for t, s, label in sweep:
        print(f"  {label} (ocv_target={t:.4f})")

    # ---- Output directory ----
    out_dir = os.path.join(
        "saved_records",
        f"{args.problem}-{conf['dataset']['prob_version']}",
        "perturb_adaptive_sweep",
        args.prefix,
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving results to {out_dir}/")

    np.save(
        os.path.join(out_dir, "run_config.npy"),
        {
            "mode": args.mode,
            "sweep": [(t, s, l) for t, s, l in sweep],
            "pred_models": args.pred_models,
            "n_epochs": args.n_epochs,
            "n_samples": args.n_samples,
            "alpha": args.alpha,
            "max_step": args.max_step,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "lr": args.lr,
            "seed": args.seed,
        },
        allow_pickle=True,
    )

    # ---- Main sweep ----
    for model_type in args.pred_models:
        print(f"\n{'='*60}")
        print(f"Model type: {model_type}")
        print(f"{'='*60}")

        for ocv_target, sigma_init, label in sweep:
            out_path = os.path.join(out_dir, f"{label}_{model_type}.npz")

            if os.path.exists(out_path):
                print(f"\n[{model_type}] {label} — already exists, skipping.")
                continue

            print(f"\n[{model_type}] {label}  ({args.n_epochs} epochs)...")

            logs = run_one(
                ocv_target=ocv_target,
                sigma_init=sigma_init,
                controller_mode=args.mode,
                model_type=model_type,
                problem=problem,
                ptoSolver=ptoSolver,
                args=args,
                device=device,
                X_train=X_train, Y_train=Y_train, Y_aux_train=Y_aux_train,
                X_val=X_val,     Y_val=Y_val,     Y_aux_val=Y_aux_val,
                opt_obj_train=opt_obj_train,
                opt_obj_val=opt_obj_val,
            )

            np.savez(out_path, **logs)
            print(f"  Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
