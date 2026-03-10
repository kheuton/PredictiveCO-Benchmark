"""
Gradient decomposition at the perturbed local minimum (test≈0.085).

Two central questions:
  Q1. Do high-regret instances provide coherent or diverse perturbed gradient signal?
  Q2. Are instances making the same swap mistake (coherent small-margin class)?

For each training instance computes:
  - z0  : predicted decision under current model
  - z*  : optimal decision under true costs
  - per-instance regret
  - per-instance MSE gradient vector (flattened model params)
  - per-instance perturbed gradient vector (n_samples_diag perturbations)

Then produces two figures:

Figure 1 — Gradient coherence (2×3):
  Row 0: MSE gradient
  Row 1: Perturbed gradient
  Cols: (a) gradient norm vs regret
        (b) cosine-sim(g_i, G_total) vs regret
        (c) projection of g_i onto G_total direction vs regret
  Plus annotated total-gradient alignment: cos(G_MSE, G_perturb).

Figure 2 — Swap pattern (2×2):
  [0,0] Histogram of hamming(z0, z*) per instance
  [0,1] Per-item "should-add" frequency (in z* but not z0)
  [1,0] Per-item "should-remove" frequency (in z0 but not z*)
  [1,1] Distribution of pairwise cosine-similarity of swap vectors
        (concentrated near 1 → coherent mistakes; near 0 → diverse)
"""

import os, sys, types
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ---- paths -----------------------------------------------------------------
CKPT  = ("saved_records/knapsack-gen/perturb_adaptive_sweep"
         "/kn_bench_dp_per_instance_lr1e-2/pi_t0.066_s1_dense_best_pred.pt")
OUT_DIR = "saved_records/knapsack-gen/perturb_adaptive_sweep"

# ---- problem setup ---------------------------------------------------------
sys.path.insert(0, ".")
from openpto.config import load_conf, setup_seed
from openpto.method.Predicts.dense import MLP
from openpto.method.Solvers.wrapper_solver import solver_wrapper
from openpto.problems.wrapper_prob import problem_wrapper
from openpto.method.Models.perturb_diag import perturbedSoftDecisionDiag
from openpto.method.Models.perturbed import sample_noise_with_gradients

N_SAMPLES_DIAG = 20     # perturbation samples per instance for gradient estimate
SIGMA_DIAG     = 1.0    # sigma for gradient estimate
SEED           = 2023

setup_seed(SEED)

wrapper_args = types.SimpleNamespace(
    problem="knapsack", solver="heuristic",
    config_path="openpto/config/probs/knapsack_small.yaml",
    loadnew=False, instances=400, testinstances=200,
    val_frac=0.2, seed=SEED,
    data_dir="./openpto/data/knapsack",
)
conf    = load_conf("openpto/config/probs/knapsack_small.yaml",
                    "openpto/config/models/perturb_s01_n5.yaml", "knapsack")
problem = problem_wrapper(wrapper_args, conf)
solver  = solver_wrapper(wrapper_args, conf, problem)

X_train, Y_train, Y_aux_train = problem.get_train_data()
N_train = X_train.shape[0]
print(f"Train instances: {N_train}")

# ---- load model ------------------------------------------------------------
ipdim, opdim = problem.get_model_shape()
model = MLP(ipdim, opdim, num_layers=2, intermediate_size=32,
            activation="relu", output_activation="identity")
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()
print(f"Loaded checkpoint: {CKPT}")

# ---- helper: flatten params gradient ---------------------------------------
def flat_grad(model):
    return torch.cat([p.grad.detach().flatten()
                      for p in model.parameters() if p.grad is not None])

def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

# ---- precompute z0 and z* per instance ------------------------------------
print("Computing z0 and z* for all training instances...")
init_api = problem.init_API()

with torch.no_grad():
    y_hat = model(X_train)   # (N, D)

z0_list, zs_list = [], []
for i in range(N_train):
    yh_i = y_hat[i].unsqueeze(0)
    z0_np, _ = problem.get_decision(yh_i, Y_aux_train, solver, **init_api)
    z0_list.append(torch.as_tensor(z0_np, dtype=torch.float32).squeeze(0))

    yt_i = Y_train[i].unsqueeze(0)
    zs_np, _ = problem.get_decision(yt_i, Y_aux_train, solver, **init_api)
    zs_list.append(torch.as_tensor(zs_np, dtype=torch.float32).squeeze(0))

z0   = torch.stack(z0_list)   # (N, D)
zstar = torch.stack(zs_list)  # (N, D)

# Per-instance true objective and regret
obj_z0   = (Y_train * z0  ).sum(dim=-1)   # (N,)
obj_zstar= (Y_train * zstar).sum(dim=-1)  # (N,)
regret   = (obj_zstar - obj_z0).abs() / (obj_zstar.abs() + 1e-8)  # (N,)

n_wrong = (regret > 1e-4).sum().item()
print(f"Instances with z0 ≠ z* (regret>1e-4): {n_wrong}/{N_train}")
print(f"Regret: mean={regret.mean():.4f}  max={regret.max():.4f}")

# ---- per-instance MSE gradients -------------------------------------------
print("Computing per-instance MSE gradients...")
mse_grads = []
for i in range(N_train):
    zero_grad(model)
    yh_i = model(X_train[i].unsqueeze(0))
    loss_i = F.mse_loss(yh_i, Y_train[i].unsqueeze(0))
    loss_i.backward()
    mse_grads.append(flat_grad(model).clone())

mse_grads = torch.stack(mse_grads)   # (N, P)
G_mse     = mse_grads.mean(dim=0)    # (P,)

# ---- per-instance perturbed gradients -------------------------------------
print(f"Computing per-instance perturbed gradients (n_samples={N_SAMPLES_DIAG})...")
perturb_grads = []
for i in range(N_train):
    zero_grad(model)
    yh_i = model(X_train[i].unsqueeze(0))   # (1, D), requires_grad

    # regret loss for single instance
    yt_i  = Y_train[i].unsqueeze(0)
    zs_i  = zstar[i].unsqueeze(0)

    z_bar, _ = perturbedSoftDecisionDiag.apply(
        yh_i, solver, problem, Y_aux_train,
        N_SAMPLES_DIAG, SIGMA_DIAG, "normal",
    )
    obj    = (yt_i * z_bar).sum(dim=-1)
    obj_s  = (yt_i * zs_i ).sum(dim=-1).detach()
    loss_i = (obj - obj_s).abs().mean()
    loss_i.backward()

    pg = flat_grad(model)
    perturb_grads.append(pg.clone() if pg.numel() > 0 else torch.zeros_like(G_mse))

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_train}")

perturb_grads = torch.stack(perturb_grads)  # (N, P)
G_perturb     = perturb_grads.mean(dim=0)   # (P,)

# ---- summary stats ---------------------------------------------------------
cos_total = F.cosine_similarity(G_mse.unsqueeze(0), G_perturb.unsqueeze(0)).item()
print(f"\nCosine similarity (G_MSE, G_perturb): {cos_total:.4f}")

# Per-instance cosine sims with their respective total gradients
def cos_with_total(grads, G):
    norms_i = grads.norm(dim=1).clamp(min=1e-12)
    norm_G  = G.norm().clamp(min=1e-12)
    dots    = (grads @ G) / (norms_i * norm_G)
    return dots  # (N,)

cos_mse_i     = cos_with_total(mse_grads,     G_mse)
cos_perturb_i = cos_with_total(perturb_grads, G_perturb)
proj_mse_i     = (mse_grads     @ G_mse)     / G_mse.norm().clamp(min=1e-12)
proj_perturb_i = (perturb_grads @ G_perturb) / G_perturb.norm().clamp(min=1e-12)

reg = regret.numpy()
wrong = reg > 1e-4

# ---- swap analysis ---------------------------------------------------------
swap = (zstar - z0).numpy()          # (N, D)  in {-1, 0, 1}
should_add    = (swap >  0.5).astype(float)   # in z* not in z0
should_remove = (swap < -0.5).astype(float)   # in z0 not in z*
hamming_dist  = np.abs(swap).sum(axis=1)      # (N,)  items to change

# Pairwise cosine sim of swap vectors (wrong instances only)
swap_wrong = swap[wrong]
norms_sw   = np.linalg.norm(swap_wrong, axis=1, keepdims=True).clip(min=1e-12)
swap_norm  = swap_wrong / norms_sw
cos_pairs  = (swap_norm @ swap_norm.T).ravel()  # all pairs incl. diagonal

# ---- FIGURE 1: gradient coherence -----------------------------------------
fig1, axes = plt.subplots(2, 3, figsize=(15, 8))
fig1.suptitle(
    f"Gradient decomposition at perturbed local minimum (test≈0.085)\n"
    f"cos(G_MSE, G_perturb) = {cos_total:.3f}  |  "
    f"wrong instances: {n_wrong}/{N_train}",
    fontsize=11,
)

labels = ["MSE gradient", f"Perturbed gradient (n={N_SAMPLES_DIAG}, σ={SIGMA_DIAG})"]
grad_sets = [
    (mse_grads,     G_mse,     cos_mse_i,     proj_mse_i),
    (perturb_grads, G_perturb, cos_perturb_i, proj_perturb_i),
]

for row, (grads, G, cos_i, proj_i) in enumerate(grad_sets):
    norms_i = grads.norm(dim=1).numpy()

    # (a) gradient norm vs regret
    ax = axes[row, 0]
    ax.scatter(reg[~wrong], norms_i[~wrong], s=15, alpha=0.5,
               color="steelblue", label="z0=z*")
    ax.scatter(reg[wrong],  norms_i[wrong],  s=15, alpha=0.7,
               color="tomato",    label="z0≠z*")
    ax.set_xlabel("per-instance regret")
    ax.set_ylabel("gradient norm")
    ax.set_title(f"{labels[row]}\n(a) grad norm vs regret")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) cosine sim with total gradient vs regret
    ax = axes[row, 1]
    ax.scatter(reg[~wrong], cos_i.numpy()[~wrong], s=15, alpha=0.5, color="steelblue")
    ax.scatter(reg[wrong],  cos_i.numpy()[wrong],  s=15, alpha=0.7, color="tomato")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("per-instance regret")
    ax.set_ylabel("cos(g_i, G_total)")
    ax.set_title(f"(b) gradient alignment with total")
    ax.grid(True, alpha=0.3)

    # (c) projection onto total gradient direction vs regret
    ax = axes[row, 2]
    ax.scatter(reg[~wrong], proj_i.numpy()[~wrong], s=15, alpha=0.5, color="steelblue")
    ax.scatter(reg[wrong],  proj_i.numpy()[wrong],  s=15, alpha=0.7, color="tomato")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("per-instance regret")
    ax.set_ylabel("projection onto G_total")
    ax.set_title(f"(c) signed contribution to total gradient")
    ax.grid(True, alpha=0.3)

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "fig_gradient_decomp_coherence.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved {out1}")

# ---- FIGURE 2: swap patterns -----------------------------------------------
fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle("Swap pattern analysis: what does the model get wrong?", fontsize=11)
D = z0.shape[1]

# [0,0] Hamming distance distribution
ax = axes[0, 0]
ax.hist(hamming_dist, bins=np.arange(0, D+2)-0.5, color="steelblue", edgecolor="white")
ax.set_xlabel("hamming(z0, z*)  [items to swap]")
ax.set_ylabel("# instances")
ax.set_title("(a) Decision error per instance")
ax.axvline(0, color="k", lw=1, ls="--")
ax.grid(True, alpha=0.3)

# [0,1] Per-item "should add" frequency
ax = axes[0, 1]
freq_add = should_add.sum(axis=0)
ax.bar(np.arange(D), freq_add, color="green", alpha=0.7)
ax.set_xlabel("item index")
ax.set_ylabel("# instances where item should be ADDED")
ax.set_title("(b) Should-add frequency per item\n(in z* but not z0)")
ax.grid(True, alpha=0.3, axis="y")

# [1,0] Per-item "should remove" frequency
ax = axes[1, 0]
freq_rem = should_remove.sum(axis=0)
ax.bar(np.arange(D), freq_rem, color="tomato", alpha=0.7)
ax.set_xlabel("item index")
ax.set_ylabel("# instances where item should be REMOVED")
ax.set_title("(c) Should-remove frequency per item\n(in z0 but not z*)")
ax.grid(True, alpha=0.3, axis="y")

# [1,1] Pairwise cosine sim of swap vectors
ax = axes[1, 1]
ax.hist(cos_pairs, bins=50, color="purple", alpha=0.7, edgecolor="white")
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_xlabel("cosine similarity between swap vectors")
ax.set_ylabel("count (all pairs of wrong instances)")
ax.set_title("(d) Swap coherence across instances\n(near 1.0 = same mistake; near 0 = diverse)")
ax.grid(True, alpha=0.3)

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "fig_gradient_decomp_swaps.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved {out2}")

print("\nDone.")
