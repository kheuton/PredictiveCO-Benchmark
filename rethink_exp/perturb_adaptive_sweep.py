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
from openpto.method.Models.perturb_diag import (
    AdaptiveSigmaPerturb, PerInstanceAdaptiveSigma, PerInstanceHammingPerturb,
    PerInstanceHammingStarPerturb, PerItemHammingPerturb, PerInstItemHammingPerturb,
    CoeffRelativeSigmaPerturb, FlipMarginSigmaPerturb, HammingProportionalSigmaPerturb,
)
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

    # Per-instance sigma
    p.add_argument("--per_instance", action="store_true",
                   help="Use PerInstanceAdaptiveSigma (per-instance sigma vector).")
    p.add_argument("--sigma_ema", type=float, default=0.7,
                   help="EMA smoothing for per-instance sigma update.")

    # Hamming-rate controller
    p.add_argument("--hamming", action="store_true",
                   help="Use PerInstanceHammingPerturb (Hamming-rate control target).")
    p.add_argument("--hamming_targets", type=float, nargs="+", default=None,
                   help="Explicit list of hamming_target values (overrides --prop_targets).")
    p.add_argument("--hamming_warmup_epochs", type=int, default=0,
                   help="Epochs to hold Hamming target constant before decaying.")
    p.add_argument("--hamming_decay_epochs", type=int, default=0,
                   help="Epochs over which to linearly decay Hamming target to 0 after warmup.")
    p.add_argument("--hamming_star", action="store_true",
                   help="Use PerInstanceHammingStarPerturb: dynamic target = hamming(z*,z0)/D per instance.")

    # Per-item sigma
    p.add_argument("--per_item", action="store_true",
                   help="Use PerItemHammingPerturb (per-item sigma vector, shape (D,)).")
    p.add_argument("--per_inst_item", action="store_true",
                   help="Use PerInstItemHammingPerturb (per-instance×per-item sigma matrix, shape (N,D)).")
    p.add_argument("--no_sigma_min", action="store_true",
                   help="Disable sigma_min clamping (sigma_min=None) for per-item controllers.")

    # Coefficient-relative sigma
    p.add_argument("--coeff_relative", action="store_true",
                   help="Use CoeffRelativeSigmaPerturb: sigma_{b,i} = alpha * |coeff_hat_{b,i}|.")
    p.add_argument("--cr_alphas", type=float, nargs="+", default=None,
                   help="Alpha values to sweep for coeff_relative mode.")

    # Flip-margin controller
    p.add_argument("--flip_margin", action="store_true",
                   help="Use FlipMarginSigmaPerturb: per-(instance,item) sigma guided by z* vs z0 agreement.")
    p.add_argument("--fm_error_target", type=float, default=0.20,
                   help="Hamming target for items where the model is wrong (flip_margin mode).")
    p.add_argument("--fm_correct_target", type=float, default=0.01,
                   help="Hamming target for items where the model is correct (flip_margin mode).")
    p.add_argument("--fm_regret_threshold", type=float, default=0.01,
                   help="Regret above which the error/correct split has full weight (flip_margin).")
    p.add_argument("--fm_sigma_coeff_max", type=float, default=5.0,
                   help="Hard ceiling sigma_{b,i} <= fm_sigma_coeff_max * |coeff_hat_{b,i}|. "
                        "Set 0 to disable.")

    # Hamming-proportional controller (no setpoint)
    p.add_argument("--hamming_proportional", action="store_true",
                   help="Use HammingProportionalSigmaPerturb: sigma_i ∝ hamming_i^alpha, no setpoint.")
    p.add_argument("--hp_alpha", type=float, default=0.5,
                   help="Exponent alpha in sigma ∝ hamming^alpha (hamming_proportional mode).")
    p.add_argument("--hp_scale", type=float, default=1.0,
                   help="Overall sigma_scale for hamming_proportional; swept via --prop_targets.")
    p.add_argument("--hp_warmup", type=int, default=10,
                   help="Warm-up epochs before first sigma assignment (hamming_proportional).")
    p.add_argument("--hp_update_freq", type=int, default=20,
                   help="Re-estimate sigma every N epochs (hamming_proportional).")

    # Gradient surgery
    p.add_argument("--grad_surgery", action="store_true",
                   help="Enable gradient surgery: remove MSE-misaligned component of perturbed "
                        "gradient and inject clean MSE gradient direction.")
    p.add_argument("--surgery_weight", type=float, default=1.0,
                   help="Weight for injected MSE gradient in surgery: "
                        "g_update = (g_p - proj) + surgery_weight * g_m.")

    # Warm-start
    p.add_argument("--warmstart_ckpt", type=str, default=None,
                   help="Path to a .pt checkpoint to warm-start the pred model from.")

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
    X_test=None, Y_test=None, Y_aux_test=None, opt_obj_test=None,
    ckpt_path=None,
):
    """
    Train a pred model from random init with AdaptiveSigmaPerturb (or
    PerInstanceAdaptiveSigma if args.per_instance), logging per-epoch
    diagnostics. Returns a dict of per-epoch arrays.
    If test data is provided, evaluates test_regret at the best-val epoch.
    """
    ipdim, opdim = problem.get_model_shape()
    pred_model = build_pred_model(model_type, ipdim, opdim, args).to(device)

    # Optionally warm-start from a saved checkpoint
    warmstart_ckpt = getattr(args, "warmstart_ckpt", None)
    if warmstart_ckpt:
        sd = torch.load(warmstart_ckpt, map_location=device)
        pred_model.load_state_dict(sd)
        print(f"  Warm-started from: {warmstart_ckpt}")

    pred_model.train()

    no_sigma_min = getattr(args, "no_sigma_min", False)
    item_sigma_min = None if no_sigma_min else args.sigma_min

    common_kwargs = dict(
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
    # common_kwargs minus sigma_min, for classes that take sigma_min explicitly
    common_kwargs_no_smin = {k: v for k, v in common_kwargs.items() if k != "sigma_min"}

    if getattr(args, "hamming_proportional", False):
        loss_fn = HammingProportionalSigmaPerturb(
            ptoSolver,
            hp_alpha=getattr(args, "hp_alpha", 0.5),
            sigma_scale=ocv_target,   # ocv_target reused as sigma_scale in sweep grid
            warmup_epochs=getattr(args, "hp_warmup", 10),
            update_freq=getattr(args, "hp_update_freq", 20),
            sigma_ema=args.sigma_ema,
            sigma_min=item_sigma_min,
            **common_kwargs_no_smin,  # includes sigma_max=args.sigma_max
        )
    elif getattr(args, "coeff_relative", False):
        loss_fn = CoeffRelativeSigmaPerturb(
            ptoSolver,
            alpha=ocv_target,   # ocv_target reused as alpha in the sweep grid
            n_samples=args.n_samples,
            sigma=sigma_init,
            noise=args.noise,
            reduction="mean",
        )
    elif getattr(args, "flip_margin", False):
        fm_sigma_coeff_max = getattr(args, "fm_sigma_coeff_max", 5.0)
        loss_fn = FlipMarginSigmaPerturb(
            ptoSolver,
            hamming_target=ocv_target,          # base fallback target before cache warm
            error_target=args.fm_error_target,
            correct_target=args.fm_correct_target,
            regret_threshold=args.fm_regret_threshold,
            sigma_coeff_max=fm_sigma_coeff_max if fm_sigma_coeff_max > 0 else None,
            sigma_ema=args.sigma_ema,
            sigma_min=item_sigma_min,
            **common_kwargs_no_smin,
        )
    elif getattr(args, "per_inst_item", False):
        loss_fn = PerInstItemHammingPerturb(
            ptoSolver,
            hamming_target=ocv_target,
            sigma_ema=args.sigma_ema,
            sigma_min=item_sigma_min,
            **common_kwargs_no_smin,
        )
    elif getattr(args, "per_item", False):
        loss_fn = PerItemHammingPerturb(
            ptoSolver,
            hamming_target=ocv_target,
            sigma_ema=args.sigma_ema,
            sigma_min=item_sigma_min,
            **common_kwargs_no_smin,
        )
    elif getattr(args, "hamming_star", False):
        loss_fn = PerInstanceHammingStarPerturb(
            ptoSolver,
            hamming_target=ocv_target,   # fallback for epoch 0 before cache is warm
            sigma_ema=args.sigma_ema,
            **common_kwargs,
        )
    elif getattr(args, "hamming", False):
        loss_fn = PerInstanceHammingPerturb(
            ptoSolver,
            hamming_target=ocv_target,
            hamming_warmup_epochs=getattr(args, "hamming_warmup_epochs", 0),
            hamming_decay_epochs=getattr(args, "hamming_decay_epochs", 0),
            sigma_ema=args.sigma_ema,
            **common_kwargs,
        )
    elif args.per_instance:
        loss_fn = PerInstanceAdaptiveSigma(ptoSolver, sigma_ema=args.sigma_ema, **common_kwargs)
    else:
        loss_fn = AdaptiveSigmaPerturb(ptoSolver, **common_kwargs)

    optimizer = torch.optim.Adam(pred_model.parameters(), lr=args.lr)
    model_args = {"reduction": "mean"}

    logs = {k: [] for k in [
        "epoch", "loss", "train_regret", "val_regret",
        "coeff_norm", "adaptive_sigma",
        "grad_norm", "fd_grad_norm", "cosine_sim",
        "softness", "dist_binary", "entropy",
        "rank_change_rate", "ocv_y", "frac_improving", "hamming",
        "sigma_item_mean", "sigma_item_std", "sigma_item_min", "sigma_item_max",
        "hamming_item_mean", "hamming_item_std",
        "surgery_cos_sim", "surgery_proj_frac",
    ]}
    if args.per_instance:
        logs["sigma_vec"] = []  # list of (N_train,) arrays, one per epoch
    _log_item_vecs = (
        getattr(args, "per_item", False)
        or getattr(args, "per_inst_item", False)
        or getattr(args, "flip_margin", False)
        or getattr(args, "hamming_proportional", False)
    )
    if _log_item_vecs:
        logs["sigma_vec_item"]  = []  # list of (D,) arrays → saved as (n_epochs, D)
        logs["hamming_vec_item"] = []  # list of (D,) arrays → saved as (n_epochs, D)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.n_epochs + 1):
        pred_model.train()

        preds = pred_model(X_train.to(device))

        # Capture gradient w.r.t. preds via hook (fires on first backward through preds)
        captured_grad = {}
        hook_fired = [False]
        def _hook(g):
            if not hook_fired[0]:
                captured_grad["g"] = g.detach().cpu()
                hook_fired[0] = True
        hook_handle = preds.register_hook(_hook)

        loss = loss_fn(
            problem,
            coeff_hat=preds,
            coeff_true=Y_train.to(device),
            params=Y_aux_train,
            **model_args,
        )

        surgery_cos_sim   = float("nan")
        surgery_proj_frac = float("nan")

        optimizer.zero_grad()
        if getattr(args, "grad_surgery", False):
            # --- Two-pass gradient surgery ---
            # Pass 1: perturbed gradient w.r.t. preds
            g_p = torch.autograd.grad(loss, preds, retain_graph=True)[0]  # (B, D)
            captured_grad["g"] = g_p.detach().cpu()  # for diagnostics
            hook_handle.remove()

            # Pass 2: MSE gradient w.r.t. preds (cheap, no solver)
            Y_dev = Y_train.to(device)
            mse_loss = F.mse_loss(preds, Y_dev)
            g_m = torch.autograd.grad(mse_loss, preds)[0]  # (B, D)

            # Surgery: remove MSE-misaligned component, inject clean MSE direction
            gp_flat = g_p.reshape(-1).float()
            gm_flat = g_m.reshape(-1).float()
            dot     = (gp_flat * gm_flat).sum()
            norm_sq = (gm_flat * gm_flat).sum() + 1e-12
            g_proj_flat   = (dot / norm_sq) * gm_flat
            g_update_flat = (gp_flat - g_proj_flat) + args.surgery_weight * gm_flat
            g_update      = g_update_flat.reshape(g_p.shape)

            # Diagnostics
            surgery_cos_sim   = F.cosine_similarity(
                gp_flat.unsqueeze(0), gm_flat.unsqueeze(0)
            ).item()
            surgery_proj_frac = g_proj_flat.norm().item() / (gp_flat.norm().item() + 1e-12)

            # Inject modified gradient into pred_model parameters
            preds.backward(g_update)
        else:
            hook_handle.remove()
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
                _K = getattr(problem, "budget", None)
                if _K is not None:
                    rcr = rank_change_rate(perturbed_sols.float(), z0, K=_K)
                else:
                    rcr = float("nan")
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
        logs["hamming"].append(getattr(loss_fn, "last_hamming", float("nan")))
        if args.per_instance and loss_fn.sigma_vec is not None:
            logs["sigma_vec"].append(loss_fn.sigma_vec.detach().numpy().copy())

        # Per-item sigma stats
        _is_per_item = (
            getattr(args, "per_item", False)
            or getattr(args, "hamming_proportional", False)
        )
        _is_per_inst_item = getattr(args, "per_inst_item", False) or getattr(args, "flip_margin", False)
        if _is_per_item and hasattr(loss_fn, "sigma_vec") and loss_fn.sigma_vec is not None:
            sv = loss_fn.sigma_vec.detach()
            logs["sigma_item_mean"].append(sv.mean().item())
            logs["sigma_item_std"].append(sv.std().item())
            logs["sigma_item_min"].append(sv.min().item())
            logs["sigma_item_max"].append(sv.max().item())
        elif _is_per_inst_item and hasattr(loss_fn, "sigma_mat") and loss_fn.sigma_mat is not None:
            sm = loss_fn.sigma_mat.detach()
            logs["sigma_item_mean"].append(sm.mean().item())
            logs["sigma_item_std"].append(sm.std().item())
            logs["sigma_item_min"].append(sm.min().item())
            logs["sigma_item_max"].append(sm.max().item())
        else:
            logs["sigma_item_mean"].append(float("nan"))
            logs["sigma_item_std"].append(float("nan"))
            logs["sigma_item_min"].append(float("nan"))
            logs["sigma_item_max"].append(float("nan"))

        hv = getattr(loss_fn, "last_hamming_vec", None)
        if hv is not None:
            logs["hamming_item_mean"].append(hv.mean().item())
            logs["hamming_item_std"].append(hv.std().item())
        else:
            logs["hamming_item_mean"].append(float("nan"))
            logs["hamming_item_std"].append(float("nan"))

        logs["surgery_cos_sim"].append(surgery_cos_sim)
        logs["surgery_proj_frac"].append(surgery_proj_frac)

        # Full per-item vectors (D,) — only when tracking is enabled
        if _log_item_vecs:
            sv_item = getattr(loss_fn, "last_sigma_vec_item", None)
            hv_item = getattr(loss_fn, "last_hamming_vec", None)
            logs["sigma_vec_item"].append(
                sv_item.detach().numpy().copy() if sv_item is not None
                else np.full(problem.get_model_shape()[1], float("nan"))
            )
            logs["hamming_vec_item"].append(
                hv_item.detach().numpy().copy() if hv_item is not None
                else np.full(problem.get_model_shape()[1], float("nan"))
            )

        if val_regret < best_val:
            best_val = val_regret
            best_state = {k: v.clone() for k, v in pred_model.state_dict().items()}

        if epoch % 10 == 0:
            if getattr(args, "flip_margin", False):
                wrong_frac = getattr(loss_fn, "last_hamming_vec", None)
                wf = wrong_frac.mean().item() if wrong_frac is not None else float("nan")
                ctrl_str = f"hamming {loss_fn.last_hamming:.4f} wrong_frac {wf:.4f}"
            elif getattr(args, "hamming_star", False) or getattr(args, "hamming", False):
                ctrl_str = f"hamming {loss_fn.last_hamming:.4f}"
            else:
                ctrl_str = f"ocv_y {loss_fn.last_ocv_y:.4f}"
            print(
                f"  epoch {epoch:3d} | loss {loss.item():.4f} | "
                f"val_regret {val_regret:.4f} | "
                f"{ctrl_str} | sigma {current_sigma:.4f}"
            )

    # ---- Test regret at best-val epoch ----
    if X_test is not None and best_state is not None:
        pred_model.load_state_dict(best_state)
        test_regret = compute_regret(
            problem, ptoSolver, pred_model,
            X_test, Y_test, Y_aux_test, opt_obj_test, device,
        )
        print(f"  test_regret (at best val epoch): {test_regret:.4f}")
    else:
        test_regret = float("nan")

    result = {}
    for k, v in logs.items():
        if k in ("sigma_vec", "sigma_vec_item", "hamming_vec_item") and v:
            result[k] = np.stack(v, axis=0)  # (n_epochs, N_train) or (n_epochs, D)
        else:
            result[k] = np.array(v)
    result["test_regret"] = np.float64(test_regret)

    if ckpt_path is not None and best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"  Checkpoint → {ckpt_path}")

    return result


# ---------------------------------------------------------------------------
# Build sweep grid
# ---------------------------------------------------------------------------

def build_sweep_grid(args):
    """
    Returns a list of (ocv_target, sigma_init, label) tuples.
    label is used as the filename stem.
    """
    if args.mode == "proportional":
        if getattr(args, "hamming_proportional", False):
            targets = args.prop_targets or [0.5, 1.0, 2.0]
            alpha   = getattr(args, "hp_alpha", 0.5)
            stem    = f"hprop_a{alpha:.2g}_s"
            return [
                (t, args.prop_sigma_init, f"{stem}{t:.3g}_s{args.prop_sigma_init:.3g}")
                for t in targets
            ]
        elif getattr(args, "coeff_relative", False):
            targets = args.cr_alphas or [0.1, 0.5, 1.0, 2.0, 5.0]
            stem = "cr_a"
            return [
                (t, args.prop_sigma_init, f"{stem}{t:.3g}_s{args.prop_sigma_init:.3g}")
                for t in targets
            ]
        elif getattr(args, "flip_margin", False):
            # ocv_target = base hamming_target (fallback before z* cache warm)
            targets = args.hamming_targets or [0.05, 0.10, 0.20]
            stem = "fm_t"
        elif getattr(args, "per_inst_item", False):
            targets = args.hamming_targets or [0.05, 0.10, 0.20]
            stem = "piitem_t"
        elif getattr(args, "per_item", False):
            targets = args.hamming_targets or [0.05, 0.10, 0.20]
            stem = "pitem_t"
        elif getattr(args, "hamming_star", False):
            # hamming_star: no target sweep needed — target is dynamic.
            # The ocv_target value here serves only as epoch-0 fallback sigma.
            targets = args.hamming_targets or [args.prop_sigma_init]
            stem = "hams_s"  # hams = hamming-star
        elif getattr(args, "hamming", False):
            targets = args.hamming_targets or [0.05, 0.10, 0.15, 0.20, 0.30]
            decay = getattr(args, "hamming_decay_epochs", 0)
            stem = "hamd_t" if decay > 0 else "ham_t"
        elif args.prop_targets is not None:
            targets = args.prop_targets
            stem = "pi_t" if getattr(args, "per_instance", False) else "prop_t"
        else:
            targets = np.logspace(
                np.log10(args.target_min), np.log10(args.target_max), args.n_targets
            ).tolist()
            stem = "pi_t" if getattr(args, "per_instance", False) else "prop_t"
        return [
            (t, args.prop_sigma_init, f"{stem}{t:.3f}_s{args.prop_sigma_init:.3g}")
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
    X_test,  Y_test,  Y_aux_test  = problem.get_test_data()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ---- Precompute optimal objectives ----
    print("Precomputing optimal objectives...")
    opt_obj_train = precompute_opt(problem, ptoSolver, Y_train, Y_aux_train)
    opt_obj_val   = precompute_opt(problem, ptoSolver, Y_val,   Y_aux_val)
    opt_obj_test  = precompute_opt(problem, ptoSolver, Y_test,  Y_aux_test)

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

            ckpt_path = out_path.replace(".npz", "_best_pred.pt")
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
                X_test=X_test,   Y_test=Y_test,   Y_aux_test=Y_aux_test,
                opt_obj_test=opt_obj_test,
                ckpt_path=ckpt_path,
            )

            np.savez(out_path, **logs)
            print(f"  Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
