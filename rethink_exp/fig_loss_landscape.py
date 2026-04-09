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
        = 0 at theta=0 (tight at optimum), convex in c.

  3. PGB  (backward-FD upper bound, centered, always >= regret)
        PGB_c(c) = [V(c) - V(c - h*y)] / h  -  V(y)
        Proof: (V(c) - V(c-hy))/h >= y^T z_hat  (concavity of V)
               => PGB_c >= y^T z_hat - V(y) = regret
        PGB_c = 0 at theta=0 (since V(y) = V(y - hy + hy)).

  4. DBB  (forward-FD lower bound, centered, always <= regret)
        DBB_c(c) = [V(c + h*y) - V(c)] / h  -  V(y)
        Proof: (V(c+hy) - V(c))/h <= y^T z_hat  (concavity of V)
               => DBB_c <= y^T z_hat - V(y) = regret
        DBB_c = 0 at theta=0.

  5. LODL  (learned surrogate, no structural guarantee)
        Approximated as Gaussian-smoothed staircase + oscillations.

Output: loss_landscape.png  (12x6.5 in, 180 dpi, dark background #1E2761)
"""

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
Z = make_Z(d, k)   # 35 feasible solutions

# ── Optimisation primitives (MINIMIZE c^T z over Z) ───────────────────────────
def V(c):
    """V(c) = min_{z in Z} c^T z  (concave function of c)."""
    return float(np.min(Z @ c))

def z_min(c):
    """argmin_{z in Z} c^T z."""
    return Z[np.argmin(Z @ c)]

# ── Seed search: find (y, dir) with 5-8 staircase transitions in [-1, 1] ──────
# We fix c0 = y so that c(0) = y and regret(0) = 0.
theta_search = np.linspace(-1.2, 1.2, 800)

def count_transitions_from_y(y, dv):
    """Count z-transitions along c(t)=y+t*dv for t in theta_search."""
    sols = [z_min(y + t * dv) for t in theta_search]
    return int(sum(not np.allclose(sols[i], sols[i-1]) for i in range(1, len(sols))))

def unique_levels_from_y(y, dv):
    V_y = V(y)
    z_star = z_min(y)
    regrets = set()
    for t in theta_search:
        zh = z_min(y + t * dv)
        regrets.add(round(float(y @ zh) - V_y, 5))
    return len(regrets)

best_score  = -1
best_params = None

for seed in range(500):
    rng = np.random.default_rng(seed * 13 + 3)
    # y: positive, moderate spread so V(y) is not too small or too large
    y_t  = rng.uniform(0.5, 3.0, d)
    dv_t = rng.standard_normal(d)
    dv_t /= np.linalg.norm(dv_t)

    ntrans  = count_transitions_from_y(y_t, dv_t)
    nlevels = unique_levels_from_y(y_t, dv_t)

    if ntrans < 4:
        continue

    # Prefer 5-6 transitions and many distinct levels (including 0)
    score = ntrans * 2 + nlevels - abs(ntrans - 6) * 2
    if score > best_score:
        best_score  = score
        best_params = (y_t.copy(), dv_t.copy(), seed, ntrans, nlevels)
    if ntrans >= 5 and nlevels >= 5:
        break

y, dir_vec, chosen_seed, ntrans, nlevels = best_params
print(
    f"Seed {chosen_seed}: {ntrans} transitions, {nlevels} regret levels",
    file=sys.stderr,
)

# ── Grid ──────────────────────────────────────────────────────────────────────
n_theta   = 1200
theta_val = np.linspace(-1.2, 1.2, n_theta)

V_y    = V(y)
z_star = z_min(y)   # fixed optimal decision under y

h = 0.55   # finite-difference step (same for PGB and DBB)

regret    = np.zeros(n_theta)
spo_plus  = np.zeros(n_theta)
pgb_c     = np.zeros(n_theta)   # PGB centered: (V(c) - V(c-hy))/h - V(y)
dbb_c     = np.zeros(n_theta)   # DBB centered: (V(c+hy) - V(c))/h - V(y)

for i, theta in enumerate(theta_val):
    c_hat = y + theta * dir_vec
    z_hat = z_min(c_hat)

    # Decision regret
    regret[i] = float(y @ z_hat) - V_y

    # SPO+ (code formula, MINIMIZE):  -V(2c - y) + 2*c^T z* - V(y)
    spo_plus[i] = -V(2.0 * c_hat - y) + 2.0 * float(c_hat @ z_star) - V_y

    # PGB centered (backward FD minus V(y)):  proven >= regret
    raw_pgb  = (V(c_hat) - V(c_hat - h * y)) / h
    pgb_c[i] = raw_pgb - V_y

    # DBB centered (forward FD minus V(y)):  proven <= regret
    raw_dbb  = (V(c_hat + h * y) - V(c_hat)) / h
    dbb_c[i] = raw_dbb - V_y

# ── Assert bounds ─────────────────────────────────────────────────────────────
spo_margin = float((spo_plus - regret).min())
pgb_margin = float((pgb_c   - regret).min())
dbb_margin = float((regret  - dbb_c ).min())   # dbb_c <= regret  → regret - dbb_c >= 0
assert spo_margin >= -1e-8, f"SPO+ violated regret bound: min margin = {spo_margin:.2e}"
assert pgb_margin >= -1e-8, f"PGB_c violated regret bound: min margin = {pgb_margin:.2e}"
assert dbb_margin >= -1e-8, f"DBB_c violated upper bound: min(regret-dbb_c) = {dbb_margin:.2e}"
print(
    f"Bounds OK:  SPO+ min margin = {spo_margin:.4f}  |  "
    f"PGB_c min margin = {pgb_margin:.4f}  |  "
    f"regret - DBB_c min = {dbb_margin:.4f}",
    file=sys.stderr,
)

# ── LODL: smooth staircase + small residual oscillation ───────────────────────
sigma_sm = 30
lodl = (  gaussian_filter1d(regret, sigma=sigma_sm)
        + 0.04 * np.sin(8.5 * theta_val)
          * gaussian_filter1d(np.maximum(regret, 0.05), sigma=sigma_sm // 3))

# ── Clip for display (SPO+ grows large far from theta=0) ──────────────────────
reg_max   = regret.max()
disp_ymax = max(reg_max * 2.8, pgb_c.max() * 1.05, spo_plus.max() * 0.5 + 0.5)
disp_ymax = min(disp_ymax, reg_max * 4.5)      # hard cap
spo_disp  = np.clip(spo_plus, -0.05, disp_ymax)

# ── Plot ──────────────────────────────────────────────────────────────────────
BG         = "#1E2761"
COL_REGRET = "#FFFFFF"    # white
COL_SPO    = "#FF6B6B"    # coral-red
COL_PGB    = "#FFD93D"    # amber
COL_DBB    = "#6BCB77"    # mint-green
COL_LODL   = "#C77DFF"    # violet

fig, ax = plt.subplots(figsize=(12, 6.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

lw = 2.3

# Decision regret (piecewise-constant staircase)
ax.step(theta_val, regret, where="post",
        color=COL_REGRET, linewidth=lw + 0.5,
        label=r"Decision regret  (true $\ell$)", zorder=7)

# LODL (below SPO+ but plotted first so it sits behind)
ax.plot(theta_val, lodl,
        color=COL_LODL, linewidth=lw, linestyle=(0, (4, 1, 1, 1)),
        label="LODL  (learned surrogate, no bound)", zorder=3)

# DBB centered (lower bound on regret)
ax.plot(theta_val, dbb_c,
        color=COL_DBB, linewidth=lw, linestyle="-.",
        label=r"DBB  (forward FD, always $\leq\ell$)", zorder=4)

# PGB centered (upper bound on regret)
ax.plot(theta_val, pgb_c,
        color=COL_PGB, linewidth=lw, linestyle="--",
        label=r"PGB  (backward FD, always $\geq\ell$)", zorder=5)

# SPO+ (upper bound, convex bowl)
ax.plot(theta_val, spo_disp,
        color=COL_SPO, linewidth=lw,
        label=r"SPO+  (convex UB, always $\geq\ell$)", zorder=5)

# Zero-regret reference line
ax.axhline(0, color="white", alpha=0.20, linewidth=0.9, zorder=1)

# Vertical line at theta=0 (optimal prediction)
ax.axvline(0, color="white", alpha=0.15, linewidth=0.8, linestyle=":", zorder=1)

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

# ── Legend ───────────────────────────────────────────────────────────────────
ax.legend(
    framealpha=0.28,
    facecolor="#1a2260",
    edgecolor="#6666aa",
    labelcolor="white",
    fontsize=12,
    loc="upper center",
    ncol=2,
    columnspacing=1.2,
    handlelength=2.5,
    borderpad=0.8,
)

plt.tight_layout()
out_path = "loss_landscape.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved {out_path}")
