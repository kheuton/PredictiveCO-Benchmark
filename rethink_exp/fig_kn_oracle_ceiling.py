"""
Oracle performance ceiling for knapsack perturbed method.

Generates a noise-free dataset using the true DGP (same B matrix / weights as the
benchmark, noise_width=0), trains an MLP with MSE loss until convergence, then
evaluates regret on the standard benchmark test set (Knapsack_7.pkl, 200 instances).

The resulting test regret is the irreducible floor:  the minimum regret achievable
by any model that can only see X and not the noise realization epsilon.

Usage:
    python rethink_exp/fig_kn_oracle_ceiling.py
    python rethink_exp/fig_kn_oracle_ceiling.py --n_oracle 20000 --n_epochs 2000
"""

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# make openpto importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openpto.method.Predicts.dense import MLP
from openpto.method.Solvers.heuristic.dp import DPSolver
from openpto.problems.Knapsack import Knapsack

# ---- Config defaults ----
ORACLE_N        = 10_000   # number of noise-free training instances
N_EPOCHS        = 2_000
LR              = 5e-2
BATCH_SIZE      = 512
N_LAYERS        = 2
N_HIDDEN        = 32
SEED            = 2023     # MUST match benchmark seed → same B, same weights
PROBLEM_CACHE   = "saved_problems/Knapsack/Knapsack_7.pkl"
OUT_DIR         = "saved_records/knapsack-gen/oracle_ceiling"
DEVICE          = "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_oracle", type=int, default=ORACLE_N)
    p.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    p.add_argument("--lr",       type=float, default=LR)
    p.add_argument("--n_layers", type=int, default=N_LAYERS)
    p.add_argument("--n_hidden", type=int, default=N_HIDDEN)
    p.add_argument("--out_dir",  type=str, default=OUT_DIR)
    return p.parse_args()


# ---- Generate oracle dataset (noise_width=0, same seed → same B/weights) ----
def gen_oracle_data(n, seed=SEED):
    """Returns (feats, profits, weights) with noise_width=0."""
    weights, feats, profits = Knapsack.genKPData(
        num_instances=n,
        num_features=5,
        num_items=20,
        mean=0,
        var=1,
        dim=1,
        poly_deg=4,
        noise_width=0.0,   # ← key: no noise
        distr="normal",
        seed=seed,
    )
    return feats, profits, weights   # (n,5), (n,20), (20,)


# ---- Load benchmark test set from cached problem ----
def load_benchmark_test(cache_path):
    with open(cache_path, "rb") as f:
        prob = pickle.load(f)
    Xs_test   = prob.Xs_test    # (200, 5)
    Ys_test   = prob.Ys_test    # (200, 20)  — noisy
    weights   = prob.weights    # (20,)
    capacity  = prob.capacity   # scalar
    return Xs_test, Ys_test, weights, capacity


# ---- Train MLP with MSE loss ----
def train_oracle_model(feats, profits, args):
    X = feats.float().to(DEVICE)
    Y = profits.float().to(DEVICE)

    model = MLP(
        num_features=X.shape[1],
        num_targets=Y.shape[1],
        num_layers=args.n_layers,
        intermediate_size=args.n_hidden,
        activation="relu",
        output_activation="identity",
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ds  = TensorDataset(X, Y)
    dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    losses = []
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X)
        losses.append(epoch_loss)

        if epoch % 200 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  train_mse={epoch_loss:.6f}")

    return model, losses


# ---- Evaluate regret on test set ----
def eval_regret(model, Xs_test, Ys_test, weights, capacity):
    """
    Returns per-instance regret array and normalized regret scalar.
    Uses DP solver for both optimal (Y_true) and oracle (Y_hat) decisions.
    """
    model.eval()
    with torch.no_grad():
        Y_hat = model(Xs_test.float().to(DEVICE)).cpu()   # (200, 20)

    solver = DPSolver(weights=weights, capacity=capacity, modelSense=1)  # 1 = GRB.MAXIMIZE

    Y_true_np = Ys_test.numpy()   # (200, 20), noisy — this IS the realized profit
    Y_hat_np  = Y_hat.numpy()

    opt_objs, oracle_objs = [], []
    for i in range(len(Y_true_np)):
        # optimal: solve on true (noisy) profits — the best achievable for this instance
        _, obj_opt, _ = solver.solve(Y_true_np[i])
        opt_objs.append(obj_opt)

        # oracle: solve on predicted (noiseless) profits, evaluate on true profits
        z_oracle, _, _ = solver.solve(Y_hat_np[i])
        obj_oracle = float(np.dot(Y_true_np[i], z_oracle))
        oracle_objs.append(obj_oracle)

    opt_objs    = np.array(opt_objs)
    oracle_objs = np.array(oracle_objs)
    regrets     = (opt_objs - oracle_objs) / (opt_objs + 1e-8)   # per-instance

    norm_regret = regrets.mean()
    return regrets, norm_regret, opt_objs, oracle_objs


# ---- Also evaluate MSE on benchmark test set (to see prediction error) ----
def eval_mse_on_test(model, Xs_test, Ys_test):
    model.eval()
    with torch.no_grad():
        Y_hat = model(Xs_test.float().to(DEVICE)).cpu()
    mse = nn.functional.mse_loss(Y_hat, Ys_test.float()).item()
    return mse


# ---- Figure ----
def make_figure(losses, regrets, norm_regret, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Training loss curve
    ax = axes[0]
    ax.semilogy(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Oracle model: training MSE")
    ax.grid(True, alpha=0.3)

    # Regret histogram
    ax = axes[1]
    ax.hist(regrets, bins=30, edgecolor="k", alpha=0.7, color="steelblue")
    ax.axvline(norm_regret, color="red", linestyle="--", linewidth=1.5,
               label=f"mean={norm_regret:.4f}")
    ax.set_xlabel("Normalized regret (per instance)")
    ax.set_ylabel("Count")
    ax.set_title("Oracle regret on benchmark test set")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Regret CDF
    ax = axes[2]
    sorted_r = np.sort(regrets)
    cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    ax.plot(sorted_r, cdf, linewidth=2)
    ax.axvline(0.064, color="orange", linestyle="--", linewidth=1.5, label="MSE best (0.064)")
    ax.axvline(norm_regret, color="red", linestyle="--", linewidth=1.5,
               label=f"Oracle ({norm_regret:.4f})")
    ax.set_xlabel("Normalized regret")
    ax.set_ylabel("CDF")
    ax.set_title("Regret CDF: oracle vs benchmarks")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig_kn_oracle_ceiling.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")
    plt.close()


# ---- Main ----
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    # 1. Generate oracle training data (same B/weights as benchmark via seed=SEED)
    print(f"Generating {args.n_oracle} oracle (noise-free) training instances...")
    feats, profits, weights_oracle = gen_oracle_data(args.n_oracle, seed=SEED)
    print(f"  feats: {feats.shape},  profits: {profits.shape}")
    print(f"  profit range: [{profits.min():.1f}, {profits.max():.1f}]")

    # 2. Train
    print(f"\nTraining MLP({args.n_layers} layers, {args.n_hidden} hidden) "
          f"for {args.n_epochs} epochs, lr={args.lr}...")
    model, losses = train_oracle_model(feats, profits, args)
    final_mse = losses[-1]
    print(f"\nFinal oracle train MSE: {final_mse:.6f}")

    # 3. Load benchmark test set
    print(f"\nLoading benchmark test set from {PROBLEM_CACHE}...")
    Xs_test, Ys_test, weights_bench, capacity = load_benchmark_test(PROBLEM_CACHE)
    print(f"  Xs_test: {Xs_test.shape},  Ys_test: {Ys_test.shape}")
    print(f"  capacity: {capacity}")

    # Sanity check: weights should match
    w_match = torch.allclose(weights_oracle.float(), weights_bench.float(), atol=1e-4)
    print(f"  weights match oracle vs benchmark: {w_match}")
    if not w_match:
        print("  WARNING: weights differ! Oracle B/weights may not match benchmark.")

    # 4. Evaluate MSE on test
    test_mse = eval_mse_on_test(model, Xs_test, Ys_test)
    print(f"\nOracle model test MSE on noisy benchmark Y: {test_mse:.4f}")
    print("  (non-zero expected: oracle predicts E[Y|X], test Y includes noise)")

    # 5. Evaluate regret
    print("\nEvaluating regret on benchmark test set (200 instances, DP solver)...")
    regrets, norm_regret, opt_objs, oracle_objs = eval_regret(
        model, Xs_test, Ys_test, weights_bench, capacity
    )
    print(f"\n{'='*50}")
    print(f"Oracle ceiling test regret: {norm_regret:.4f}")
    print(f"MSE best (benchmark):       0.0640")
    print(f"Best perturbed (warm-start): 0.0640")
    print(f"Best perturbed (cold-start): 0.0743")
    print(f"{'='*50}")
    print(f"\nPer-instance stats:")
    print(f"  mean regret:   {regrets.mean():.4f}")
    print(f"  median regret: {np.median(regrets):.4f}")
    print(f"  std regret:    {regrets.std():.4f}")
    print(f"  max regret:    {regrets.max():.4f}")
    print(f"  % zero-regret: {(regrets < 1e-6).mean() * 100:.1f}%")

    # 6. Save results
    results_path = os.path.join(args.out_dir, "oracle_results.npz")
    np.savez(results_path,
             losses=np.array(losses),
             regrets=regrets,
             norm_regret=np.array(norm_regret),
             opt_objs=opt_objs,
             oracle_objs=oracle_objs,
             final_train_mse=np.array(final_mse),
             test_mse=np.array(test_mse))
    print(f"Results saved: {results_path}")

    # 7. Figure
    make_figure(losses, regrets, norm_regret, args.out_dir)


if __name__ == "__main__":
    main()
