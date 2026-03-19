"""
Interpolation from oracle → λ=10 MSE-reg endpoint.

Evaluates 4 loss functions along the linear path
    θ(α) = (1−α)·θ_oracle + α·θ_{λ=10}

Losses:
  1. Hard regret    = (opt_obj - Y·z*(Ŷ)) / opt_obj  (argmax / DP solve on Ŷ)
  2. Soft regret    = normalised perturbed loss at σ=0.5
  3. MSE            = mean((Ŷ - Y_noisy)²)
  4. Reg loss       = soft_regret + 10·MSE  (the training objective)

MSE and soft regret are loaded from the existing oracle_interpolation.npz cache
(computed by fig_kn_oracle_interpolation.py).  Hard regret is computed fresh
and also cached to oracle_lam10_interpolation.npz.

Usage:
    python rethink_exp/fig_kn_lam10_interpolation.py
    python rethink_exp/fig_kn_lam10_interpolation.py --n_grid 40
    python rethink_exp/fig_kn_lam10_interpolation.py --from_npz  # replot only
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
sys.path.insert(0, ".")

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from openpto.method.Predicts.dense import MLP
from openpto.method.Solvers.heuristic.dp import DPSolver
from openpto.problems.Knapsack import Knapsack

# ---- Paths ----
OUT_DIR        = "saved_records/knapsack-gen/oracle_ceiling"
ORACLE_SD      = os.path.join(OUT_DIR, "oracle_state_dict.pt")
EXISTING_NPZ   = os.path.join(OUT_DIR, "oracle_interpolation.npz")
NEW_NPZ        = os.path.join(OUT_DIR, "oracle_lam10_interpolation.npz")
OUT_PATH       = os.path.join(OUT_DIR, "fig_kn_lam10_interpolation.png")

LAM10_CKPT     = ("saved_records/knapsack-gen/perturb/"
                  "kn_plain_mse_reg_w10_lr5e-3/checkpoints/tr_pred_best.pt")
PROBLEM_CACHE  = "saved_problems/Knapsack/Knapsack_7.pkl"

SIGMA  = 0.5
LAM    = 10.0
LABEL  = "λ=10"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_grid",    type=int,   default=30)
    p.add_argument("--alpha_min", type=float, default=-0.2)
    p.add_argument("--alpha_max", type=float, default=1.2)
    p.add_argument("--n_samples", type=int,   default=100,
                   help="Samples for soft-regret (ignored if loading from cache)")
    p.add_argument("--from_npz",  action="store_true",
                   help="Load all data from NEW_NPZ cache and skip to plotting.")
    p.add_argument("--out", default=OUT_PATH)
    return p.parse_args()


ORACLE_N      = 10_000
ORACLE_EPOCHS = 400
ORACLE_LR     = 5e-2
ORACLE_BATCH  = 512
ORACLE_SEED   = 2023


def build_model():
    return MLP(num_features=5, num_targets=20, num_layers=2,
               intermediate_size=32, activation="relu",
               output_activation="identity")


def get_oracle_sd():
    """Load oracle state-dict from cache, or train and save it."""
    if os.path.exists(ORACLE_SD):
        print(f"Loading cached oracle from {ORACLE_SD}")
        return torch.load(ORACLE_SD, map_location="cpu")

    print(f"Training oracle (noiseless, n={ORACLE_N}, {ORACLE_EPOCHS} epochs)...")
    torch.manual_seed(0); np.random.seed(0)
    _, feats, profits = Knapsack.genKPData(
        num_instances=ORACLE_N, num_features=5, num_items=20,
        mean=0, var=1, dim=1, poly_deg=4, noise_width=0.0,
        distr="normal", seed=ORACLE_SEED,
    )
    X, Y = feats.float(), profits.float()
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=ORACLE_LR)
    dl  = DataLoader(TensorDataset(X, Y), batch_size=ORACLE_BATCH, shuffle=True)
    for epoch in range(1, ORACLE_EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                mse = nn.functional.mse_loss(model(X), Y).item()
            print(f"  epoch {epoch}  train_mse={mse:.5f}")
    sd = model.state_dict()
    torch.save(sd, ORACLE_SD)
    print(f"Cached to {ORACLE_SD}")
    return sd


def load_test_set():
    with open(PROBLEM_CACHE, "rb") as f:
        prob = pickle.load(f)
    return prob.Xs_test, prob.Ys_test, prob.weights, prob.capacity


@torch.no_grad()
def eval_mse(model, Xs, Ys):
    model.eval()
    return nn.functional.mse_loss(model(Xs.float()), Ys.float()).item()


@torch.no_grad()
def eval_hard_regret(model, Xs, Ys, opt_obj, solver):
    """Argmax decisions from Ŷ; no perturbation."""
    model.eval()
    Yhat = model(Xs.float()).numpy()
    Y_np = Ys.numpy()
    N = len(Yhat)
    objs = np.zeros(N)
    for i in range(N):
        sol, _, _ = solver.solve(Yhat[i])
        objs[i] = float(np.dot(Y_np[i], sol))
    return float(np.mean((opt_obj - objs) / (opt_obj + 1e-8)))


@torch.no_grad()
def eval_soft_regret(model, Xs, Ys, opt_obj, solver, n_samples, seed=0):
    """Soft regret: E_ε[Y·z*(Ŷ + σε)] averaged over test set."""
    model.eval()
    rng  = np.random.default_rng(seed)
    Yhat = model(Xs.float()).numpy()
    Y_np = Ys.numpy()
    N = len(Yhat)
    mean_obj = np.zeros(N)
    for _ in range(n_samples):
        noise = rng.standard_normal(Yhat.shape) * SIGMA
        for i in range(N):
            sol, _, _ = solver.solve(Yhat[i] + noise[i])
            mean_obj[i] += float(np.dot(Y_np[i], sol))
    mean_obj /= n_samples
    return float(np.mean((opt_obj - mean_obj) / (opt_obj + 1e-8)))


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Load or compute ----
    if args.from_npz and os.path.exists(NEW_NPZ):
        print(f"Loading from cache: {NEW_NPZ}")
        npz = np.load(NEW_NPZ, allow_pickle=True)
        alphas     = npz["alphas"]
        hard       = npz["hard_regret"]
        soft       = npz["soft_regret"]
        mse_vals   = npz["mse"]
        reg_vals   = npz["reg_loss"]
    else:
        # Load MSE and soft regret from existing cache if available
        pre_mse = pre_soft = None
        if os.path.exists(EXISTING_NPZ):
            old = np.load(EXISTING_NPZ, allow_pickle=True)
            if f"mse_{LABEL}" in old and f"perturb_{LABEL}" in old:
                pre_mse   = old[f"mse_{LABEL}"]
                pre_soft  = old[f"perturb_{LABEL}"]
                alphas    = old["alphas"]
                print(f"Loaded MSE + soft regret from existing cache ({len(alphas)} points)")

        if pre_mse is None:
            alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_grid)

        # Always compute hard regret fresh
        print("Loading test set and oracle/endpoint checkpoints...")
        Xs_test, Ys_test, weights, capacity = load_test_set()
        solver  = DPSolver(weights=weights, capacity=capacity, modelSense=1)
        opt_obj = np.array([solver.solve(Ys_test[i].numpy())[1]
                            for i in range(len(Ys_test))])

        sd_oracle   = get_oracle_sd()
        sd_endpoint = torch.load(LAM10_CKPT, map_location="cpu")
        model       = build_model()

        n = len(alphas)
        hard       = np.zeros(n)
        soft_fresh = np.zeros(n)
        mse_fresh  = np.zeros(n)

        print(f"Sweeping {n} alpha values...")
        for idx, alpha in enumerate(alphas):
            sd = {k: (1 - alpha) * sd_oracle[k].float() + alpha * sd_endpoint[k].float()
                  for k in sd_oracle}
            model.load_state_dict(sd)

            hard[idx]       = eval_hard_regret(model, Xs_test, Ys_test, opt_obj, solver)
            mse_fresh[idx]  = eval_mse(model, Xs_test, Ys_test)

            if pre_soft is None:
                soft_fresh[idx] = eval_soft_regret(
                    model, Xs_test, Ys_test, opt_obj, solver, args.n_samples, seed=idx)

            if idx % 5 == 0:
                print(f"  α={alpha:.2f}  hard={hard[idx]:.4f}  "
                      f"mse={mse_fresh[idx]:.4f}")

        mse_vals = pre_mse  if pre_mse  is not None else mse_fresh
        soft     = pre_soft if pre_soft is not None else soft_fresh
        reg_vals = soft + LAM * mse_vals

        np.savez(NEW_NPZ, alphas=alphas, hard_regret=hard,
                 soft_regret=soft, mse=mse_vals, reg_loss=reg_vals)
        print(f"Saved cache: {NEW_NPZ}")

    # ---- Plot ----
    idx0 = int(np.argmin(np.abs(alphas)))        # α≈0 (oracle)
    idx1 = int(np.argmin(np.abs(alphas - 1.0)))  # α≈1 (endpoint)

    # Print endpoint table
    print("\n--- Values at α=0 (oracle) and α=1 (λ=10 endpoint) ---")
    for name, arr in [("Hard regret", hard), ("Soft regret (σ=0.5)", soft),
                      ("MSE loss", mse_vals), ("Reg loss (soft+10·MSE)", reg_vals)]:
        print(f"  {name:<28} α=0: {arr[idx0]:.4f}   α=1: {arr[idx1]:.4f}")

    TRACES = [
        ("Hard regret",            hard,     "C3", "-",  2.0),
        ("Soft regret (σ=0.5)",    soft,     "C0", "-",  2.0),
        ("MSE loss",               mse_vals, "C2", "--", 1.8),
        ("Reg loss (soft+10·MSE)", reg_vals, "C1", ":",  1.8),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Interpolation: oracle  →  λ=10 MSE-reg endpoint\n"
        f"θ(α) = (1−α)·θ_oracle + α·θ_{{λ=10}}  |  "
        f"benchmark test set (200 instances)",
        fontsize=11, fontweight="bold",
    )

    # ---- Left panel: hard + soft regret (absolute, task-relevant scale) ----
    ax = axes[0]
    for label, arr, color, ls, lw in TRACES[:2]:   # hard + soft only
        ax.plot(alphas, arr, color=color, ls=ls, lw=lw, label=label)
        ax.scatter([alphas[idx1]], [arr[idx1]], color=color, s=50, zorder=5)
    ax.scatter([alphas[idx0]], [hard[idx0]], color="black",
               s=80, marker="*", zorder=6, label="Oracle (α=0)")
    ax.axhline(0.052, color="#333333", ls=":", lw=1, label="Oracle ceiling (0.052)")
    ax.axhline(0.0640, color="#555555", ls=":", lw=1, label="MSE baseline (0.064)")
    ax.axvspan(0, 1, color="lightyellow", alpha=0.4, zorder=0)
    ax.axvline(0, color="gray", ls=":", lw=1.0)
    ax.axvline(1, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("α   (0 = oracle,  1 = λ=10 endpoint)", fontsize=10)
    ax.set_ylabel("Test regret (normalized)", fontsize=10)
    ax.set_title("Task regret along path", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Right panel: all 4 losses normalised by oracle value ----
    # reg_loss ≈ 10·MSE (soft term is tiny), so their shapes are nearly identical
    # when normalised.  Use a twin y-axis: hard/soft on left, MSE/reg on right,
    # so all 4 curves are legible.
    ax  = axes[1]
    ax2 = ax.twinx()

    for label, arr, color, ls, lw in TRACES[:2]:   # hard + soft → left axis
        norm = arr / (arr[idx0] + 1e-12)
        ax.plot(alphas, norm, color=color, ls=ls, lw=lw, label=label)
        ax.scatter([alphas[idx1]], [norm[idx1]], color=color, s=50, zorder=5)

    for label, arr, color, ls, lw in TRACES[2:]:   # MSE + reg → right axis
        norm = arr / (arr[idx0] + 1e-12)
        ax2.plot(alphas, norm, color=color, ls=ls, lw=lw, label=label)
        ax2.scatter([alphas[idx1]], [norm[idx1]], color=color, s=50, zorder=5)

    ax.axhline(1.0, color="black", ls=":", lw=1.0, alpha=0.4)
    ax2.axhline(1.0, color="black", ls=":", lw=1.0, alpha=0.4)
    ax.scatter([alphas[idx0]], [1.0], color="black", s=80, marker="*", zorder=6,
               label="Oracle (α=0)")
    ax.axvspan(0, 1, color="lightyellow", alpha=0.4, zorder=0)
    ax.axvline(0, color="gray", ls=":", lw=1.0)
    ax.axvline(1, color="gray", ls=":", lw=1.0)

    ax.set_xlabel("α   (0 = oracle,  1 = λ=10 endpoint)", fontsize=10)
    ax.set_ylabel("Hard / soft regret  (÷ oracle value)", fontsize=10, color="black")
    ax2.set_ylabel("MSE / reg loss  (÷ oracle value)", fontsize=10, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.set_title("All 4 losses — normalised by oracle value\n"
                 "(left axis: regret;  right axis: MSE/reg)", fontsize=10)

    # Combined legend from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {args.out}")


if __name__ == "__main__":
    main()
