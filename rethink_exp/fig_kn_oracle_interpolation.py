"""
Interpolation landscape anchored at the oracle model.

For each right-endpoint checkpoint (plain perturbed, MSE-reg λ=0.1/0.5/1/5/10, MSE),
draws the linear path:  θ(α) = (1−α)·θ_oracle + α·θ_endpoint

Along each path we evaluate on the benchmark test set:
  • MSE loss        = mean((f_θ(X) - Y_noisy)²)
  • Perturbed loss  = normalised soft regret at σ=0.5
      = mean((opt_obj - E_ε[Y·z*(f_θ(X)+σε)]) / opt_obj)

Layout: 1 row × 2 cols   (MSE loss | Perturbed loss)
Curves are coloured by λ (plain perturbed → MSE as a spectrum).
Reference lines: plain perturbed (λ=0) and pure MSE as dashed bookends.

Usage:
    python rethink_exp/fig_kn_oracle_interpolation.py
    python rethink_exp/fig_kn_oracle_interpolation.py --n_grid 40 --n_samples 50
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from openpto.method.Predicts.dense import MLP
from openpto.method.Solvers.heuristic.dp import DPSolver
from openpto.problems.Knapsack import Knapsack

# ---- Checkpoints ----
RESULTS_BASE = "saved_records/knapsack-gen/perturb"
ADAPTIVE_BASE = "saved_records/knapsack-gen/perturb_adaptive_sweep"
MSE_CKPT = ("saved_records/knapsack-gen/mse/"
             "kn_bench_mse_dense_lr5e-2/checkpoints/tr_pred_best.pt")
ORACLE_SD_CACHE = os.path.join(OUT_DIR, "oracle_state_dict.pt")

ENDPOINTS = [
    # (label, lambda_val, checkpoint_path, test_regret)
    ("Plain (λ=0)",  0,     f"{RESULTS_BASE}/kn_plain_sigma_sweep_s05_lr5e-3/checkpoints/tr_pred_best.pt",    0.0743),
    ("λ=0.01",      0.01,  f"{RESULTS_BASE}/kn_plain_mse_reg_w001_lr5e-3/checkpoints/tr_pred_best.pt",       0.0730),
    ("λ=0.033",     0.033, f"{RESULTS_BASE}/kn_plain_mse_reg_w0033_lr5e-3/checkpoints/tr_pred_best.pt",      0.0759),
    ("λ=0.05",      0.05,  f"{RESULTS_BASE}/kn_plain_mse_reg_w005_lr5e-3/checkpoints/tr_pred_best.pt",       0.0707),
    ("λ=0.1",       0.1,   f"{RESULTS_BASE}/kn_plain_mse_reg_w01_lr1e-2/checkpoints/tr_pred_best.pt",        0.0868),
    ("λ=0.5",       0.5,   f"{RESULTS_BASE}/kn_plain_mse_reg_w05_lr5e-3/checkpoints/tr_pred_best.pt",        0.0706),
    ("λ=1",         1,     f"{RESULTS_BASE}/kn_plain_mse_reg_w1_lr5e-3/checkpoints/tr_pred_best.pt",         0.0698),
    ("λ=5",         5,     f"{RESULTS_BASE}/kn_plain_mse_reg_w5_lr5e-3/checkpoints/tr_pred_best.pt",         0.0657),
    ("λ=10",        10,    f"{RESULTS_BASE}/kn_plain_mse_reg_w10_lr5e-3/checkpoints/tr_pred_best.pt",        0.0614),
    ("λ=20",        20,    f"{RESULTS_BASE}/kn_plain_mse_reg_w20_lr5e-3/checkpoints/tr_pred_best.pt",        0.0618),
    ("λ=50",        50,    f"{RESULTS_BASE}/kn_plain_mse_reg_w50_lr5e-3/checkpoints/tr_pred_best.pt",        0.0650),
    ("λ=100",       100,   f"{RESULTS_BASE}/kn_plain_mse_reg_w100_lr5e-3/checkpoints/tr_pred_best.pt",       0.0646),
    ("MSE (λ=∞)",   np.inf, MSE_CKPT,                                                                         0.0640),
    # Per-instance adaptive checkpoints
    ("pi-OCV_Y",    None,  f"{ADAPTIVE_BASE}/kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense_best_pred.pt", 0.0844),
    ("warm-start",  None,  f"{ADAPTIVE_BASE}/kn_bench_warmstart/pi_t0.066_s1_dense_best_pred.pt",               0.0640),
    ("hstar",       None,  f"{ADAPTIVE_BASE}/kn_hstar_lr1e-2/hams_s0.100_s1_dense_best_pred.pt",                0.0970),
]

PROBLEM_CACHE = "saved_problems/Knapsack/Knapsack_7.pkl"
OUT_DIR  = "saved_records/knapsack-gen/oracle_ceiling"
OUT_PATH = os.path.join(OUT_DIR, "fig_kn_oracle_interpolation.png")

SIGMA         = 0.5
ORACLE_N      = 10_000
ORACLE_EPOCHS = 400
ORACLE_LR     = 5e-2
ORACLE_BATCH  = 512
ORACLE_SEED   = 2023


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_grid",    type=int,   default=30)
    p.add_argument("--alpha_min", type=float, default=-0.2)
    p.add_argument("--alpha_max", type=float, default=1.2)
    p.add_argument("--n_samples", type=int,   default=100)
    p.add_argument("--out", default=OUT_PATH)
    p.add_argument("--from_npz", default=os.path.join(OUT_DIR, "oracle_interpolation.npz"),
                   help="Load pre-computed sweep from npz and skip to plotting.")
    return p.parse_args()


def build_model():
    return MLP(num_features=5, num_targets=20, num_layers=2,
               intermediate_size=32, activation="relu",
               output_activation="identity")


def train_oracle_model():
    """Train oracle model, using cached state-dict if available."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(ORACLE_SD_CACHE):
        print(f"Loading cached oracle model from {ORACLE_SD_CACHE}")
        return torch.load(ORACLE_SD_CACHE, map_location="cpu")

    print("Training oracle model (noise_width=0, seed=2023)...")
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
    torch.save(sd, ORACLE_SD_CACHE)
    print(f"Oracle state-dict cached to {ORACLE_SD_CACHE}")
    return sd


def load_test_set():
    import pickle
    with open(PROBLEM_CACHE, "rb") as f:
        prob = pickle.load(f)
    return prob.Xs_test, prob.Ys_test, prob.weights, prob.capacity


def precompute_opt(Ys_test, solver):
    return np.array([solver.solve(Ys_test[i].numpy())[1]
                     for i in range(len(Ys_test))])


@torch.no_grad()
def eval_mse(model, Xs_test, Ys_test):
    model.eval()
    return nn.functional.mse_loss(
        model(Xs_test.float()), Ys_test.float()
    ).item()


@torch.no_grad()
def eval_perturb_loss(model, Xs_test, Ys_test, opt_obj, solver, n_samples, seed):
    model.eval()
    rng  = np.random.default_rng(seed)
    Yhat = model(Xs_test.float()).numpy()
    Y_np = Ys_test.numpy()
    N    = len(Yhat)
    mean_obj = np.zeros(N)
    for _ in range(n_samples):
        noise = rng.standard_normal(Yhat.shape) * SIGMA
        for i in range(N):
            sol, _, _ = solver.solve(Yhat[i] + noise[i])
            mean_obj[i] += float(np.dot(Y_np[i], sol))
    mean_obj /= n_samples
    return float(np.mean((opt_obj - mean_obj) / (opt_obj + 1e-8)))


def sweep(sd_oracle, sd_endpoint, alphas, model, Xs, Ys, opt_obj, solver, n_samples, tag):
    mse_vals, perturb_vals = np.zeros(len(alphas)), np.zeros(len(alphas))
    for idx, alpha in enumerate(alphas):
        sd = {k: (1 - alpha) * sd_oracle[k].float() + alpha * sd_endpoint[k].float()
              for k in sd_oracle}
        model.load_state_dict(sd)
        mse_vals[idx]     = eval_mse(model, Xs, Ys)
        perturb_vals[idx] = eval_perturb_loss(model, Xs, Ys, opt_obj, solver, n_samples, seed=idx)
        if idx % 10 == 0:
            print(f"  [{tag}] α={alpha:.2f}  mse={mse_vals[idx]:.3f}  perturb={perturb_vals[idx]:.4f}")
    return mse_vals, perturb_vals


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Load cached curves; run sweep only for any that are missing ----
    all_mse, all_perturb = {}, {}
    if args.from_npz and os.path.exists(args.from_npz):
        print(f"Loading cached sweep from {args.from_npz}")
        npz = np.load(args.from_npz, allow_pickle=True)
        alphas = npz["alphas"]
        for label, *_ in ENDPOINTS:
            if f"mse_{label}" in npz:
                all_mse[label]     = npz[f"mse_{label}"]
                all_perturb[label] = npz[f"perturb_{label}"]
        print(f"  cached: {list(all_mse.keys())}")
    else:
        alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_grid)

    missing = [ep for ep in ENDPOINTS if ep[0] not in all_mse]
    if missing:
        sd_oracle = train_oracle_model()
        Xs_test, Ys_test, weights, capacity = load_test_set()
        solver = DPSolver(weights=weights, capacity=capacity, modelSense=1)
        print("\nPrecomputing optimal objectives...")
        opt_obj = precompute_opt(Ys_test, solver)
        model   = build_model()
        if not args.from_npz or not os.path.exists(args.from_npz):
            alphas = np.linspace(args.alpha_min, args.alpha_max, args.n_grid)

        for label, lam, ckpt_path, test_r in missing:
            if not os.path.exists(ckpt_path):
                print(f"MISSING: {ckpt_path} — skipping {label}")
                continue
            sd_ep = torch.load(ckpt_path, map_location="cpu")
            print(f"\n=== {label} (test_regret={test_r:.4f}) ===")
            mse_v, perturb_v = sweep(
                sd_oracle, sd_ep, alphas, model,
                Xs_test, Ys_test, opt_obj, solver, args.n_samples, label,
            )
            all_mse[label]     = mse_v
            all_perturb[label] = perturb_v

        # save merged results
        np.savez(os.path.join(OUT_DIR, "oracle_interpolation.npz"),
                 alphas=alphas, **{f"mse_{k}": v for k, v in all_mse.items()},
                 **{f"perturb_{k}": v for k, v in all_perturb.items()})
        print(f"\nSaved merged npz ({len(all_mse)} curves)")

    idx0 = np.argmin(np.abs(alphas))
    idx1 = np.argmin(np.abs(alphas - 1.0))

    # ---- Print endpoint summary ----
    print("\n--- Endpoint values (α=0 oracle, α=1 endpoint) ---")
    print("  %-15s  %7s  %7s  %10s  %10s" % ("label", "mse@0", "mse@1", "perturb@0", "perturb@1"))
    for label, _, _, _ in ENDPOINTS:
        if label not in all_mse: continue
        m = all_mse[label]; p = all_perturb[label]
        print("  %-15s  %7.3f  %7.3f  %10.4f  %10.4f" % (label, m[idx0], m[idx1], p[idx0], p[idx1]))

    # ---- Colour map: plain perturb (λ=0) → MSE (λ=∞) ----
    # Map finite λ values to [0,1] on a log scale for colouring.
    # None λ (per-instance adaptive) → use tab10 palette with distinct colours.
    finite_lams = [lam for _, lam, _, _ in ENDPOINTS if lam is not None and np.isfinite(lam)]
    log_min = np.log10(max(min(finite_lams), 1e-3))
    log_max = np.log10(max(finite_lams))
    cmap = matplotlib.colormaps["plasma"]
    _tab10 = matplotlib.colormaps["tab10"]
    _none_lam_labels = [label for label, lam, _, _ in ENDPOINTS if lam is None]
    _none_lam_idx = {lbl: i for i, lbl in enumerate(_none_lam_labels)}

    def get_color(lam, label=None):
        if lam is None:
            # Distinct tab10 colour for per-instance adaptive endpoints
            return _tab10(_none_lam_idx.get(label, 0) / max(len(_none_lam_labels), 1))
        if lam == 0:
            return cmap(0.0)
        if not np.isfinite(lam):
            return cmap(1.0)
        t = (np.log10(lam) - log_min) / (log_max - log_min)
        return cmap(np.clip(t, 0, 1))

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Interpolation from oracle  →  each endpoint\n"
        f"θ(α) = (1−α)·θ_oracle + α·θ_endpoint  |  "
        f"benchmark test set (200 instances)  |  σ={SIGMA}",
        fontsize=11, fontweight="bold",
    )

    # Left panel: MSE loss
    ax = axes[0]
    for label, lam, _, test_r in ENDPOINTS:
        if label not in all_mse: continue
        vals  = all_mse[label]
        color = get_color(lam, label)
        ls    = "--" if (lam is None or lam == 0 or (lam is not None and not np.isfinite(lam))) else "-"
        lw    = 2.0 if (lam is None or lam == 0 or (lam is not None and not np.isfinite(lam))) else 1.5
        ax.plot(alphas, vals, color=color, ls=ls, lw=lw,
                label=f"{label}  [test={test_r:.4f}]")
        ax.scatter([alphas[idx1]], [vals[idx1]], color=color, s=50, zorder=5)
    first_label = next(l for l, *_ in ENDPOINTS if l in all_mse)
    ax.scatter([alphas[idx0]], [all_mse[first_label][idx0]], color="black",
               s=80, zorder=6, marker="*", label="Oracle (α=0)")
    ax.axvspan(0, 1, color="lightyellow", alpha=0.35, zorder=0)
    ax.axvline(0, color="gray", ls=":", lw=1.2)
    ax.axvline(1, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("α   (0 = oracle,  1 = endpoint)", fontsize=10)
    ax.set_ylabel("MSE loss  (Ŷ vs Y_noisy)", fontsize=10)
    ax.set_title("MSE loss", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right panel: regularized objective  L_λ(θ) = perturb(θ) + λ·MSE(θ)
    # Normalise each curve by its value at α=0 (oracle) so all start at 1.0
    ax = axes[1]
    for label, lam, _, test_r in ENDPOINTS:
        if label not in all_mse: continue
        if lam is None:
            # Per-instance adaptive: show perturbed loss alone (λ=0 interpretation)
            raw = all_perturb[label]
            lam_str = "adaptive"
        elif not np.isfinite(lam):
            # Pure MSE model: regularized obj = MSE alone (λ→∞ limit, show normalised MSE)
            raw = all_mse[label]
            lam_str = "∞ (MSE)"
        else:
            raw = all_perturb[label] + lam * all_mse[label]
            lam_str = str(lam)
        oracle_val = raw[idx0]
        vals  = raw / oracle_val          # normalised so oracle = 1.0
        color = get_color(lam, label)
        ls    = "--" if (lam is None or lam == 0 or (lam is not None and not np.isfinite(lam))) else "-"
        lw    = 2.0 if (lam is None or lam == 0 or (lam is not None and not np.isfinite(lam))) else 1.5
        ax.plot(alphas, vals, color=color, ls=ls, lw=lw,
                label=f"λ={lam_str}  [test={test_r:.4f}]")
        ax.scatter([alphas[idx1]], [vals[idx1]], color=color, s=50, zorder=5)
    ax.axhline(1.0, color="black", ls=":", lw=1.0, alpha=0.5)   # oracle level
    ax.scatter([alphas[idx0]], [1.0], color="black", s=80, zorder=6,
               marker="*", label="Oracle (α=0, normalised=1)")
    ax.axvspan(0, 1, color="lightyellow", alpha=0.35, zorder=0)
    ax.axvline(0, color="gray", ls=":", lw=1.2)
    ax.axvline(1, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("α   (0 = oracle,  1 = endpoint)", fontsize=10)
    ax.set_ylabel("Regularised objective  (normalised by oracle value)", fontsize=10)
    ax.set_title(f"L_λ = perturb + λ·MSE  (normalised)", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {args.out}")


if __name__ == "__main__":
    main()
