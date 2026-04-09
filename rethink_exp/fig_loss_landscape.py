#!/usr/bin/env python
"""
fig_loss_landscape.py

1-D surrogate loss landscape for a predict-then-optimize presentation slide.
All curves are derived from the SAME discrete optimisation problem.

Problem: MINIMIZE c^T z  over  Z = all {0,1}^d vectors with exactly k ones.
         d=7, k=3  →  C(7,3)=35 feasible solutions.

Parameterisation:  c(theta) = y + theta * dir_vec
  • theta = 0  →  c = y  →  z_hat = z*  →  regret = 0  (optimal prediction)
  • |theta| > 0  →  prediction drifts from truth  →  regret rises in steps

Losses plotted:
  1. Decision regret  (piecewise-constant staircase)
        regret(theta) = y^T z_hat(c(theta)) - V(y)           [>= 0]

  2. SPO+  (convex upper bound, always >= regret)
        SPO+(c, y) = -V(2c - y) + 2*c^T z* - V(y)
        = 0 at theta=0, convex in c.

  3. PGB  (backward-FD upper bound, centered, always >= regret)
        PGB_c(c) = [V(c) - V(c - h*y)] / h  -  V(y)
        Proof: (V(c) - V(c-hy))/h >= y^T z_hat  (concavity of V)
               => PGB_c >= y^T z_hat - V(y) = regret

  4. DBB  (forward-FD lower bound, centered, always <= regret)
        DBB_c(c) = [V(c + h*y) - V(c)] / h  -  V(y)
        Proof: (V(c+hy) - V(c))/h <= y^T z_hat  (concavity of V)
               => DBB_c <= y^T z_hat - V(y) = regret

  5. Perturb / DPO  (Berthet et al. 2020 — smooth, no hard bound)
        DPO(c) = E_{eps~N(0,I)}[ y^T z_min(c + sigma*eps) ] - V(y)
        Approximated by Monte Carlo (n_samples draws per theta).

  6. LODL  (learned surrogate, no structural guarantee)
        Approximated as Gaussian-smoothed staircase + oscillations.

Outputs:
  loss_landscape.png           — figure (12x6.5 in, 180 dpi, dark background)
  loss_landscape_data.json     — all curve data for JS re-creation
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from itertools import combinations

# ── Feasible set ──────────────────────────────────────────────────────────────
def make_Z(d, k):
    """All {0,1}^d binary vectors with exactly k ones → shape [C(d,k), d]."""
    rows = []
    for idx in combinations(range(d), k):
        z = np.zeros(d)
        z[list(idx)] = 1.0
        rows.append(z)
    return np.array(rows)

d, k = 7, 3
Z = make_Z(d, k)   # [35, 7]

# ── Optimisation primitives (MINIMIZE c^T z over Z) ───────────────────────────
def V(c):
    """V(c) = min_{z in Z} c^T z  (concave function of c)."""
    return float(np.min(Z @ c))

def z_min(c):
    """argmin_{z in Z} c^T z."""
    return Z[np.argmin(Z @ c)]

def V_batch(C):
    """V for each row of C: shape [n] → [n]. Vectorised."""
    return (C @ Z.T).min(axis=1)   # [n, |Z|].min(1)

# ── Seed search ───────────────────────────────────────────────────────────────
# Criteria:
#   - narrow zero-regret flat region (<= 18% of theta range)
#   - at least 3 transitions on the positive side (theta > 0)
#   - at least 6 total transitions
#   - at least 5 distinct non-zero regret levels
theta_search = np.linspace(-1.2, 1.2, 800)
mid = len(theta_search) // 2   # index of theta=0

def analyse_seed(y_t, dv_t):
    z_s  = z_min(y_t)
    V_y  = y_t @ z_s
    sols = np.array([z_min(y_t + t * dv_t) for t in theta_search])
    regs = np.array([float(y_t @ sols[i]) - V_y for i in range(len(theta_search))])

    trans_mask  = np.any(np.diff(sols, axis=0) != 0, axis=1)
    ntrans      = int(trans_mask.sum())
    n_pos_trans = int(trans_mask[mid:].sum())

    zero_frac  = float((regs < 1e-8).mean())
    levels_nz  = sorted(set(round(r, 5) for r in regs if r > 1e-8))
    nlevels_nz = len(levels_nz)
    # smallest non-zero regret level — must be large enough for DPO to be well-behaved
    min_nz     = min(levels_nz) if levels_nz else 0.0

    return ntrans, n_pos_trans, zero_frac, nlevels_nz, min_nz

best_score  = -1
best_params = None

for seed in range(1400):
    rng = np.random.default_rng(seed * 17 + 5)
    y_t  = rng.uniform(0.5, 3.0, d)
    dv_t = rng.standard_normal(d)
    dv_t /= np.linalg.norm(dv_t)

    ntrans, n_pos, zfrac, nlevels_nz, min_nz = analyse_seed(y_t, dv_t)

    # Hard constraints
    if ntrans < 5 or n_pos < 2 or zfrac > 0.22 or min_nz < 0.05:
        continue

    # Score: transitions, positive-side coverage, wide first margin, narrow zero region
    score = ntrans * 2 + n_pos * 3 + nlevels_nz + min_nz * 10 - zfrac * 20
    if score > best_score:
        best_score  = score
        best_params = (y_t.copy(), dv_t.copy(), seed, ntrans, n_pos, zfrac, nlevels_nz, min_nz)

y, dir_vec, chosen_seed, ntrans, n_pos, zfrac, nlevels_nz, min_nz = best_params
print(
    f"Seed {chosen_seed}: {ntrans} total transitions, {n_pos} on positive side, "
    f"zero-fraction={zfrac:.2%}, {nlevels_nz} non-zero levels, min-margin={min_nz:.4f}",
    file=sys.stderr,
)

# ── Compute grid ──────────────────────────────────────────────────────────────
n_theta   = 1200
theta_val = np.linspace(-1.2, 1.2, n_theta)

V_y    = V(y)
z_star = z_min(y)

h          = 0.55    # finite-difference step for PGB / DBB
sigma_dpo  = 0.06    # DPO perturbation std — small enough that DPO(0) ≈ 0
n_dpo      = 1200    # MC samples per theta for DPO (more to reduce MC noise at small sigma)

regret   = np.zeros(n_theta)
spo_plus = np.zeros(n_theta)
pgb_c    = np.zeros(n_theta)
dbb_c    = np.zeros(n_theta)

for i, theta in enumerate(theta_val):
    c_hat = y + theta * dir_vec
    z_hat = z_min(c_hat)

    regret[i]   = float(y @ z_hat) - V_y
    spo_plus[i] = -V(2.0 * c_hat - y) + 2.0 * float(c_hat @ z_star) - V_y
    pgb_c[i]    = (V(c_hat) - V(c_hat - h * y)) / h - V_y
    dbb_c[i]    = (V(c_hat + h * y) - V(c_hat)) / h - V_y

# ── DPO (perturbed optimizer, Berthet et al. 2020) ─────────────────────────
# DPO(c) = E_{eps}[y^T argmin_{z}(c + sigma*eps)^T z] - V(y)
# Vectorised: for each theta, sample n_dpo perturbations at once.
rng_dpo = np.random.default_rng(2024)
eps_mat = rng_dpo.standard_normal((n_dpo, d))   # fixed noise realisations

dpo = np.zeros(n_theta)
for i, theta in enumerate(theta_val):
    c_hat      = y + theta * dir_vec             # [d]
    c_perturb  = c_hat[None, :] + sigma_dpo * eps_mat   # [n_dpo, d]
    # argmin for each perturbed cost (vectorised over Z)
    obj_mat    = c_perturb @ Z.T                 # [n_dpo, |Z|]
    z_idx      = obj_mat.argmin(axis=1)          # [n_dpo]
    z_sols     = Z[z_idx]                        # [n_dpo, d]
    dpo[i]     = float((z_sols @ y).mean()) - V_y

# ── Assert hard bounds ────────────────────────────────────────────────────────
spo_margin = float((spo_plus - regret).min())
pgb_margin = float((pgb_c   - regret).min())
dbb_margin = float((regret  - dbb_c ).min())
assert spo_margin >= -1e-8, f"SPO+ violated: margin={spo_margin:.2e}"
assert pgb_margin >= -1e-8, f"PGB violated:  margin={pgb_margin:.2e}"
assert dbb_margin >= -1e-8, f"DBB violated:  margin={dbb_margin:.2e}"
print(
    f"Bounds OK  SPO+>= min margin={spo_margin:.4f} | "
    f"PGB>= min margin={pgb_margin:.4f} | "
    f"regret-DBB>= min={dbb_margin:.4f}",
    file=sys.stderr,
)

# ── LODL: smooth staircase + small oscillation ────────────────────────────────
sigma_sm = 28
lodl = (  gaussian_filter1d(regret, sigma=sigma_sm)
        + 0.038 * np.sin(8.5 * theta_val)
          * gaussian_filter1d(np.maximum(regret, 0.04), sigma=sigma_sm // 3))

# ── Clip for display ──────────────────────────────────────────────────────────
reg_max   = regret.max()
disp_ymax = min(max(reg_max * 2.8, pgb_c.max() * 1.05, spo_plus.max() * 0.55 + 0.3),
                reg_max * 4.5)
spo_disp  = np.clip(spo_plus, -0.05, disp_ymax)

# ── Export JSON ───────────────────────────────────────────────────────────────
json_data = {
    "theta":    [round(float(v), 6) for v in theta_val],
    "regret":   [round(float(v), 6) for v in regret],
    "spo_plus": [round(float(v), 6) for v in spo_disp],
    "pgb":      [round(float(v), 6) for v in pgb_c],
    "dbb":      [round(float(v), 6) for v in dbb_c],
    "dpo":      [round(float(v), 6) for v in dpo],
    "lodl":     [round(float(v), 6) for v in lodl],
    "metadata": {
        "problem":        "minimize_c_z_k_subset",
        "d":              d,
        "k":              k,
        "n_solutions":    int(len(Z)),
        "h_finite_diff":  h,
        "dpo_sigma":      sigma_dpo,
        "dpo_n_samples":  n_dpo,
        "disp_ymax":      round(disp_ymax, 6),
        "n_transitions":  ntrans,
        "n_pos_transitions": n_pos,
        "zero_frac":      round(zfrac, 4),
        "n_nonzero_regret_levels": nlevels_nz,
        "seed":           chosen_seed,
        "colors": {
            "background": "#1E2761",
            "regret":     "#FFFFFF",
            "spo_plus":   "#FF6B6B",
            "pgb":        "#FFD93D",
            "dbb":        "#6BCB77",
            "dpo":        "#FF9F45",
            "lodl":       "#C77DFF",
        }
    }
}
json_path = "loss_landscape_data.json"
with open(json_path, "w") as fh:
    json.dump(json_data, fh, indent=2)
print(f"Saved {json_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
BG         = "#1E2761"
COL_REGRET = "#FFFFFF"
COL_SPO    = "#FF6B6B"    # coral-red
COL_PGB    = "#FFD93D"    # amber
COL_DBB    = "#6BCB77"    # mint-green
COL_DPO    = "#FF9F45"    # orange  (perturb / DPO)
COL_LODL   = "#C77DFF"    # violet

fig, ax = plt.subplots(figsize=(12, 6.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

lw = 2.2

# ── Curves (back to front) ────────────────────────────────────────────────────
ax.plot(theta_val, lodl,
        color=COL_LODL, linewidth=lw, linestyle=(0, (4, 1, 1, 1)),
        label="LODL  (learned, no bound)", zorder=3)

ax.plot(theta_val, dbb_c,
        color=COL_DBB, linewidth=lw, linestyle="-.",
        label=r"DBB  (forward FD, always $\leq\ell$)", zorder=4)

ax.plot(theta_val, dpo,
        color=COL_DPO, linewidth=lw, linestyle=(0, (6, 1)),
        label=r"Perturb / DPO  (smooth, no bound)", zorder=4)

ax.plot(theta_val, pgb_c,
        color=COL_PGB, linewidth=lw, linestyle="--",
        label=r"PGB  (backward FD, always $\geq\ell$)", zorder=5)

ax.plot(theta_val, spo_disp,
        color=COL_SPO, linewidth=lw,
        label=r"SPO+  (convex UB, always $\geq\ell$)", zorder=5)

# Regret staircase on top
ax.step(theta_val, regret, where="post",
        color=COL_REGRET, linewidth=lw + 0.6,
        label=r"Decision regret  (true $\ell$)", zorder=7)

# Reference lines
ax.axhline(0, color="white", alpha=0.18, linewidth=0.8, zorder=1)
ax.axvline(0, color="white", alpha=0.13, linewidth=0.8, linestyle=":", zorder=1)

# ── Axis styling ──────────────────────────────────────────────────────────────
ax.set_xlabel(r"$\theta$  (model parameter)", color="white", fontsize=15, labelpad=8)
ax.set_ylabel("Loss / Regret", color="white", fontsize=15, labelpad=8)
ax.tick_params(colors="white", labelsize=12)
for sp in ("bottom", "left"):
    ax.spines[sp].set_color("#9999bb")
    ax.spines[sp].set_linewidth(0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlim(theta_val[0], theta_val[-1])
ax.set_ylim(-0.06, disp_ymax + 0.1)

# ── Legend (3 columns, 2 rows) ────────────────────────────────────────────────
ax.legend(
    framealpha=0.28,
    facecolor="#1a2260",
    edgecolor="#6666aa",
    labelcolor="white",
    fontsize=11.5,
    loc="upper center",
    ncol=3,
    columnspacing=1.0,
    handlelength=2.4,
    borderpad=0.8,
)

plt.tight_layout()
png_path = "loss_landscape.png"
plt.savefig(png_path, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved {png_path}")
