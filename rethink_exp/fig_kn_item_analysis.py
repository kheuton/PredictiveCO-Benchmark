"""
Per-item prediction bias and error diagnostic for knapsack.

Loads MSE and plain-perturb checkpoints, computes per-item stats across all
training instances, and produces a 4-panel figure.

Panels:
  1. Item weights (bar) + z* selection rate (line) — natural boundary structure
  2. Prediction bias per item (perturb vs MSE) — signed mean(ĉ_i - y_i)
  3. Error rate per item — fraction of instances where z0_i ≠ z*_i
  4. Scatter of (weight, error_rate) with z*_rate colour — connects item
     properties to sigma behaviour (ballooning vs collapse markers)

Ballooning items (2, 9, 13, 19) and collapsing items (1, 3, 5, 8, 12, 14, 17)
are highlighted with distinct markers (0-indexed: subtract 1 from plan indices).

Usage:
    python rethink_exp/fig_kn_item_analysis.py
    python rethink_exp/fig_kn_item_analysis.py --out_dir /tmp/figs
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openpto.method.Predicts.dense import MLP
from openpto.method.Solvers.heuristic.dp import DPSolver
from gurobipy import GRB  # for modelSense constant

# ---- Defaults ----
PROBLEM_CACHE  = "saved_problems/Knapsack/Knapsack_7.pkl"
PERTURB_CKPT   = ("saved_records/knapsack-gen/perturb/"
                   "kn_plain_sigma_sweep_s05_lr5e-3/checkpoints/tr_pred_best.pt")
MSE_CKPT       = ("saved_records/knapsack-gen/mse/"
                   "kn_bench_mse_dense_lr5e-2/checkpoints/tr_pred_best.pt")
OUT_DIR        = "saved_records/knapsack-gen/item_analysis"

N_LAYERS = 2
N_HIDDEN = 32

# Items identified as ballooning/collapsing in heatmap analysis (1-indexed → 0-indexed)
BALLOONING_ITEMS  = [i - 1 for i in [2, 9, 13, 19]]   # heavy, rigid items
COLLAPSING_ITEMS  = [i - 1 for i in [1, 3, 5, 8, 12, 14, 17]]  # light, easy items


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--problem_cache", type=str, default=PROBLEM_CACHE)
    p.add_argument("--perturb_ckpt",  type=str, default=PERTURB_CKPT)
    p.add_argument("--mse_ckpt",      type=str, default=MSE_CKPT)
    p.add_argument("--out_dir",       type=str, default=OUT_DIR)
    p.add_argument("--n_layers",      type=int, default=N_LAYERS)
    p.add_argument("--n_hidden",      type=int, default=N_HIDDEN)
    p.add_argument("--n_dp_samples",  type=int, default=100,
                   help="Number of perturbation samples for DP solver (unused here).")
    return p.parse_args()


# ---- Load problem ----
def load_problem(cache_path):
    with open(cache_path, "rb") as f:
        prob = pickle.load(f)
    return prob


# ---- Build MLP and load checkpoint ----
def load_model(ckpt_path, in_dim, out_dim, n_layers, n_hidden):
    model = MLP(
        num_features=in_dim,
        num_targets=out_dim,
        num_layers=n_layers,
        intermediate_size=n_hidden,
        activation="relu",
        output_activation="identity",
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


# ---- DP solver call to get z* ----
def get_z_star(Y_train, weights, capacity):
    """Run DP solver on costs (rows of Y_train) to get optimal decisions.

    Returns z_star: (B, D) numpy array of {0,1}.
    """
    w_np = weights.numpy() if torch.is_tensor(weights) else np.array(weights)
    cap  = float(capacity)
    solver = DPSolver(w_np, cap, GRB.MAXIMIZE)
    z_list = []
    for i in range(Y_train.shape[0]):
        y_i = Y_train[i].numpy() if torch.is_tensor(Y_train) else np.array(Y_train[i])
        z_i, _, _ = solver.solve(y_i)
        z_list.append(np.array(z_i, dtype=float))
    return np.stack(z_list, axis=0)  # (B, D)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load problem ----
    print(f"Loading problem from {args.problem_cache}...")
    prob = load_problem(args.problem_cache)

    # Extract data attributes (handle both tensor and numpy)
    def to_tensor(x):
        if torch.is_tensor(x):
            return x.float()
        return torch.as_tensor(np.array(x), dtype=torch.float32)

    X_train = to_tensor(prob.Xs_train)  # (N, F)
    Y_train = to_tensor(prob.Ys_train)  # (N, D)
    weights  = to_tensor(prob.weights)  # (D,)
    capacity = float(prob.capacity)

    N, D = Y_train.shape
    F    = X_train.shape[1]
    print(f"Train: {N} instances, {D} items, {F} features. Capacity={capacity:.1f}")
    print(f"Item weights: {weights.numpy()}")

    # ---- Load models ----
    print(f"Loading perturb model from {args.perturb_ckpt}...")
    model_perturb = load_model(args.perturb_ckpt, F, D, args.n_layers, args.n_hidden)

    print(f"Loading MSE model from {args.mse_ckpt}...")
    model_mse = load_model(args.mse_ckpt, F, D, args.n_layers, args.n_hidden)

    # ---- Predictions ----
    with torch.no_grad():
        c_perturb = model_perturb(X_train).numpy()  # (N, D)
        c_mse     = model_mse(X_train).numpy()       # (N, D)
    y_true = Y_train.numpy()  # (N, D)

    # ---- Oracle z* via DP ----
    print("Computing oracle z* for training set (DP)...")
    try:
        z_star = get_z_star(Y_train, weights, capacity)   # (N, D)
    except Exception as e:
        print(f"  WARNING: DP solver failed ({e}). Falling back to greedy approximation.")
        # Greedy fallback: sort items by profit/weight, fill capacity
        z_star = np.zeros((N, D), dtype=float)
        w_np = weights.numpy()
        for i in range(N):
            order = np.argsort(-y_true[i] / (w_np + 1e-8))
            cap_left = capacity
            for j in order:
                if w_np[j] <= cap_left:
                    z_star[i, j] = 1.0
                    cap_left -= w_np[j]

    # z0 for perturb model (greedy DP on predicted costs)
    print("Computing z0 for perturb model (DP on predicted costs)...")
    try:
        z0_perturb = get_z_star(
            torch.as_tensor(c_perturb, dtype=torch.float32), weights, capacity
        )
    except Exception as e:
        print(f"  WARNING: DP solver failed for perturb ({e}). Using greedy.")
        z0_perturb = np.zeros((N, D), dtype=float)
        w_np = weights.numpy()
        for i in range(N):
            order = np.argsort(-c_perturb[i] / (w_np + 1e-8))
            cap_left = capacity
            for j in order:
                if w_np[j] <= cap_left:
                    z0_perturb[i, j] = 1.0
                    cap_left -= w_np[j]

    print("Computing z0 for MSE model (DP on predicted costs)...")
    try:
        z0_mse = get_z_star(
            torch.as_tensor(c_mse, dtype=torch.float32), weights, capacity
        )
    except Exception as e:
        print(f"  WARNING: DP solver failed for MSE ({e}). Using greedy.")
        z0_mse = np.zeros((N, D), dtype=float)
        w_np = weights.numpy()
        for i in range(N):
            order = np.argsort(-c_mse[i] / (w_np + 1e-8))
            cap_left = capacity
            for j in order:
                if w_np[j] <= cap_left:
                    z0_mse[i, j] = 1.0
                    cap_left -= w_np[j]

    # ---- Per-item stats ----
    items     = np.arange(D)
    weight_i  = weights.numpy()                          # (D,)
    mean_y_i  = y_true.mean(axis=0)                     # (D,)
    bias_p_i  = (c_perturb - y_true).mean(axis=0)       # (D,) signed bias
    bias_m_i  = (c_mse     - y_true).mean(axis=0)       # (D,)
    zstar_rate_i  = z_star.mean(axis=0)                 # (D,) fraction of instances where item in z*
    z0p_rate_i    = z0_perturb.mean(axis=0)             # (D,)
    z0m_rate_i    = z0_mse.mean(axis=0)                 # (D,)
    error_rate_p  = (z0_perturb != z_star).mean(axis=0) # (D,) perturb error rate
    error_rate_m  = (z0_mse     != z_star).mean(axis=0) # (D,) MSE error rate

    print(f"\nPer-item summary:")
    print(f"  weight range:      [{weight_i.min():.2f}, {weight_i.max():.2f}]")
    print(f"  z* rate range:     [{zstar_rate_i.min():.3f}, {zstar_rate_i.max():.3f}]")
    print(f"  bias_perturb range: [{bias_p_i.min():.4f}, {bias_p_i.max():.4f}]")
    print(f"  error_rate_p range: [{error_rate_p.min():.3f}, {error_rate_p.max():.3f}]")

    # ---- Item category markers ----
    def make_marker_arrays(vals):
        colors = np.full(D, "steelblue", dtype=object)
        markers = np.full(D, "o", dtype=object)
        for i in BALLOONING_ITEMS:
            colors[i] = "red"
            markers[i] = "^"
        for i in COLLAPSING_ITEMS:
            colors[i] = "green"
            markers[i] = "s"
        return colors, markers

    colors, markers = make_marker_arrays(items)

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-item knapsack diagnostics (train set)", fontsize=13)

    x = items + 1  # 1-indexed for display

    # ---- Panel 1: weights + z* rate ----
    ax = axes[0, 0]
    bar_colors = [colors[i] for i in items]
    ax.bar(x, weight_i, color=bar_colors, alpha=0.7, label="Weight")
    ax2 = ax.twinx()
    ax2.plot(x, zstar_rate_i, "k--o", linewidth=1.5, markersize=5, label="z* selection rate")
    ax.set_xlabel("Item")
    ax.set_ylabel("Weight")
    ax2.set_ylabel("z* selection rate")
    ax2.set_ylim(0, 1)
    ax.set_title("Panel 1: Item weights + oracle selection rate")
    ax.set_xticks(x)
    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(color="red",       label="Ballooning σ (items 2,9,13,19)"),
        Patch(color="green",     label="Collapsing σ (items 1,3,5,8,12,14,17)"),
        Patch(color="steelblue", label="Other"),
    ]
    ax.legend(handles=legend_elems, fontsize=7, loc="upper right")

    # ---- Panel 2: prediction bias per item ----
    ax = axes[0, 1]
    width = 0.35
    xi = np.arange(D)
    ax.bar(xi + 1 - width/2, bias_p_i, width=width, color="orange", alpha=0.8, label="Perturb")
    ax.bar(xi + 1 + width/2, bias_m_i, width=width, color="steelblue", alpha=0.8, label="MSE")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    for i in BALLOONING_ITEMS:
        ax.axvline(i + 1, color="red", linewidth=0.8, alpha=0.5, linestyle=":")
    for i in COLLAPSING_ITEMS:
        ax.axvline(i + 1, color="green", linewidth=0.8, alpha=0.5, linestyle=":")
    ax.set_xlabel("Item")
    ax.set_ylabel("Mean(ĉ − y)")
    ax.set_title("Panel 2: Prediction bias per item")
    ax.set_xticks(xi + 1)
    ax.legend(fontsize=9)

    # ---- Panel 3: error rate per item ----
    ax = axes[1, 0]
    ax.bar(xi + 1 - width/2, error_rate_p, width=width, color="orange", alpha=0.8, label="Perturb")
    ax.bar(xi + 1 + width/2, error_rate_m, width=width, color="steelblue", alpha=0.8, label="MSE")
    for i in BALLOONING_ITEMS:
        ax.axvline(i + 1, color="red", linewidth=0.8, alpha=0.5, linestyle=":")
    for i in COLLAPSING_ITEMS:
        ax.axvline(i + 1, color="green", linewidth=0.8, alpha=0.5, linestyle=":")
    ax.set_xlabel("Item")
    ax.set_ylabel("Fraction of instances with z0_i ≠ z*_i")
    ax.set_title("Panel 3: Per-item error rate (z0 ≠ z*)")
    ax.set_xticks(xi + 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

    # ---- Panel 4: scatter (weight, z*_rate, error_rate) ----
    ax = axes[1, 1]
    sc = ax.scatter(
        weight_i, error_rate_p,
        c=zstar_rate_i, cmap="viridis",
        s=80, zorder=5, edgecolors="k", linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="z* selection rate")
    # Mark ballooning and collapsing with special markers
    if BALLOONING_ITEMS:
        ax.scatter(
            weight_i[BALLOONING_ITEMS], error_rate_p[BALLOONING_ITEMS],
            s=180, marker="^", c="red", zorder=10, label="Ballooning σ", edgecolors="k"
        )
    if COLLAPSING_ITEMS:
        ax.scatter(
            weight_i[COLLAPSING_ITEMS], error_rate_p[COLLAPSING_ITEMS],
            s=180, marker="s", c="green", zorder=10, label="Collapsing σ", edgecolors="k"
        )
    # Annotate item numbers
    for i in range(D):
        ax.annotate(str(i + 1), (weight_i[i], error_rate_p[i]),
                    fontsize=6, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_xlabel("Item weight")
    ax.set_ylabel("Error rate (perturb model)")
    ax.set_title("Panel 4: Weight vs error rate (colour = z* rate)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "fig_kn_item_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")

    # Save per-item data for follow-up analysis
    npz_path = os.path.join(args.out_dir, "kn_item_analysis.npz")
    np.savez(
        npz_path,
        items=items,
        weight_i=weight_i,
        mean_y_i=mean_y_i,
        bias_perturb_i=bias_p_i,
        bias_mse_i=bias_m_i,
        zstar_rate_i=zstar_rate_i,
        z0p_rate_i=z0p_rate_i,
        z0m_rate_i=z0m_rate_i,
        error_rate_perturb=error_rate_p,
        error_rate_mse=error_rate_m,
    )
    print(f"Data saved → {npz_path}")

    # ---- Print summary table ----
    print(f"\n{'Item':>4} {'Weight':>6} {'z*Rate':>7} {'BiasP':>8} {'BiasM':>8} "
          f"{'ErrP':>6} {'ErrM':>6} {'Category':>12}")
    print("-" * 70)
    for i in range(D):
        cat = ("BALLOON" if i in BALLOONING_ITEMS
               else "COLLAPSE" if i in COLLAPSING_ITEMS else "")
        print(f"{i+1:>4} {weight_i[i]:>6.2f} {zstar_rate_i[i]:>7.3f} "
              f"{bias_p_i[i]:>8.4f} {bias_m_i[i]:>8.4f} "
              f"{error_rate_p[i]:>6.3f} {error_rate_m[i]:>6.3f} {cat:>12}")


if __name__ == "__main__":
    main()
