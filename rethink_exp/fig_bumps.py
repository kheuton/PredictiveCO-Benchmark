
# =====================================================================
# PLOT D: Arrow chart with regret as % of best (log scale, fixed cap)
# Combines arrow style from Plot B2 with regret data
# Y-axis top is always 100% (best); bottom is capped at 1000%
# If MSE (tuned) is <100%, that experiment is rescaled so MSE maps to 100%
# Avg column: shows average rank (1 = best)
# =====================================================================
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize

# -----------------------------
# 1) Paste / edit your data here
# -----------------------------
experiments = [
    "Knapsack (Gen)",
    "Knapsack (Energy)",
    "Scheduling (Energy)",
    "Budget Allocation",
    "TopK (Cubic)",
    "Bipartite Matching",
    "Portfolio",
]

# Baseline methods (11) — from your table
baseline_methods = [
    "Two-stage", "DFL", "Blackbox", "Identity", "CPLayer", "SPO",
    "NCE", "point-LTR", "pair-LTR", "list-LTR", "LODL"
]


# Abbreviation map
abbr = {
    "Two-stage": "MSE", "DFL": "DFL", "Blackbox": "DBB",
    "Identity": "Id", "CPLayer": "CPL", "SPO": "SPO+",
    "NCE": "NCE", "point-LTR": "pt-LTR", "pair-LTR": "pr-LTR",
    "list-LTR": "l-LTR", "LODL": "LODL",
    "DPO (naive)": "DPO-n", "DPO (tuned)": "DPO-t",
    "MSE (tuned)": "MSE-t",
}

exp_short = {
    "Knapsack (Gen)": "Knapsack\n(Gen)",
    "Knapsack (Energy)": "Knapsack\n(Energy)",
    "Scheduling (Energy)": "Scheduling\n(Energy)",
    "Budget Allocation": "Budget\nAlloc.",
    "TopK (Cubic)": "TopK\n(Cubic)",
    "Bipartite Matching": "Bipartite\nMatch.",
    "Portfolio": "Portfolio",
}

all_methods = baseline_methods + ["DPO (naive)", "DPO (tuned)", "MSE (tuned)"]
n_exp = len(experiments)

# Build regret matrix
regret_mat = pd.DataFrame(index=all_methods, columns=experiments, dtype=float)
for exp in experiments:
    for m in baseline_methods:
        regret_mat.loc[m, exp] = regret_data[exp].get(m, np.nan)

# Compute percentage of best (100% = best, higher = worse)
pct_of_best = regret_mat.copy()
for exp in experiments:
    col = regret_mat[exp]
    min_regret = col.min()
    pct_of_best[exp] = 100 * col / min_regret

# MSE tuned percentages vs best baseline per experiment
mse_pct_by_exp = {}
for exp in experiments:
    mse_pct_by_exp[exp] = pct_of_best.loc["MSE (tuned)", exp]

# Per-experiment rescaling: if MSE < 100, map it to 100 and scale others accordingly
scale_by_exp = {}
for exp in experiments:
    mse_pct = mse_pct_by_exp.get(exp, np.nan)
    if np.isfinite(mse_pct) and mse_pct > 0 and mse_pct < 100:
        scale_by_exp[exp] = 100.0 / mse_pct
    else:
        scale_by_exp[exp] = 1.0

# Compute ranks per experiment
rank_mat = regret_mat.copy()
for exp in experiments:
    rank_mat[exp] = regret_mat[exp].rank(method="min", ascending=True)

# --- Inherit rank for tuned if missing ---
for exp in experiments:
    tuned_rank = rank_mat.loc["DPO (tuned)", exp]
    naive_rank = rank_mat.loc["DPO (naive)", exp]
    if not np.isfinite(tuned_rank) and np.isfinite(naive_rank):
        rank_mat.loc["DPO (tuned)", exp] = naive_rank

# Compute average rank across experiments
avg_rank = {}
for m in all_methods:
    ranks = [rank_mat.loc[m, e] for e in experiments if np.isfinite(rank_mat.loc[m, e])]
    avg_rank[m] = np.mean(ranks) if ranks else np.nan

max_avg_rank = max(v for v in avg_rank.values() if np.isfinite(v))

# ---------- Identify best-avg-rank baseline (excluding DPO, SPO, and MSE-tuned) ----------
best_baseline = min(
    (m for m in baseline_methods if m != "SPO" and np.isfinite(avg_rank.get(m, np.nan))),
    key=lambda m: avg_rank[m],
)
print(f"Best avg-rank baseline (excl. SPO): {best_baseline}  (avg rank {avg_rank[best_baseline]:.2f})")

# ---------- Per-method marker / color helpers ----------
DARK_GRAY   = "#444444"   # SPO+ square & best-baseline triangle
BASELINE_C  = "#888888"   # other baselines
MSE_C       = "#1f77b4"   # tuned MSE star

def baseline_style(m):
    """Return (marker, color, size, alpha) for a baseline method."""
    if m == "SPO":
        return "s", DARK_GRAY, 48, 0.85          # dark gray square
    if m == best_baseline:
        return "^", DARK_GRAY, 52, 0.85          # darker gray triangle
    return "o", BASELINE_C, 40, 0.7               # default gray circle

# Experiments where naive & tuned should be horizontally separated
SPREAD_EXPS = {"Knapsack (Gen)", "Scheduling (Energy)"}
SPREAD_DX = 0.20  # half-width of horizontal offset
MSE_X = 0.0       # centered x-position for tuned MSE stars

# Fixed y-axis limits for all columns (log scale): top best at 100, bottom capped at 1000
Y_MIN = 100
Y_MAX = 1000

# ---- Style ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": False,
})

DPO_NAIVE_C = "#c0392b"
DPO_TUNED_C = "#27ae60"
ARROW_C     = "#2c3e50"

# Figure setup
fig = plt.figure(figsize=(7.2, 2.8))
gs = GridSpec(1, n_exp + 1, figure=fig,
              width_ratios=[1]*n_exp + [1.1], wspace=0.10)

axes_exp = [fig.add_subplot(gs[0, j]) for j in range(n_exp)]
ax_avg = fig.add_subplot(gs[0, n_exp])

# Random jitter for baselines (fixed seed for reproducibility)
np.random.seed(42)
jitter_x = {m: np.random.uniform(-0.25, 0.25) for m in baseline_methods}

def style_ax(ax, show_ylabel=False, show_ticks=False):
    ax.set_yscale("log")
    ax.set_ylim(Y_MAX, Y_MIN)  # Inverted: 100% (best) at top
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([])

    # Set tick locations for log scale - only major ticks, no minor
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    if show_ticks:
        ax.set_yticks([100, 200, 500, 1000])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/100:.0f}x"))
        ax.tick_params(axis="y", labelsize=6.5, length=0, pad=2)
    else:
        ax.set_yticks([])

    # Add horizontal reference lines at 200% and 500%
    ax.axhline(200, color="#cccccc", linewidth=0.5, linestyle="-", zorder=1)
    ax.axhline(500, color="#cccccc", linewidth=0.5, linestyle="-", zorder=1)

    if show_ylabel:
        ax.set_ylabel("Regret relative to best\n(1x = best)", fontsize=7, fontweight="bold", labelpad=2)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("#cccccc")

# ---- Per-experiment columns ----
for j, exp in enumerate(experiments):
    ax = axes_exp[j]
    style_ax(ax, show_ylabel=(j == 0), show_ticks=(j == 0))
    spread = exp in SPREAD_EXPS
    scale = scale_by_exp.get(exp, 1.0)

    # Draw baseline methods with per-method style
    for m in baseline_methods:
        raw_y = pct_of_best.loc[m, exp]
        if np.isnan(raw_y):
            continue
        yval = min(raw_y * scale, Y_MAX)
        xpos = jitter_x[m]
        mkr, col, sz, alp = baseline_style(m)
        ax.scatter(xpos, yval, s=sz, color=col, edgecolors="white",
                   linewidths=0.5, zorder=3, marker=mkr, alpha=alp)

    y_naive_raw = pct_of_best.loc["DPO (naive)", exp]
    y_tuned_raw = pct_of_best.loc["DPO (tuned)", exp]
    y_naive = min(y_naive_raw * scale, Y_MAX) if np.isfinite(y_naive_raw) else np.nan
    y_tuned = min(y_tuned_raw * scale, Y_MAX) if np.isfinite(y_tuned_raw) else np.nan

    # Determine x positions for naive / tuned
    x_naive = -SPREAD_DX if spread else 0.0
    x_tuned = SPREAD_DX if spread else 0.0

    if np.isfinite(y_naive) and np.isfinite(y_tuned):
        # Arrow from naive to tuned — thinner for spread panels
        arrow_lw = 1.2 if spread else 2.2
        ax.annotate(
            "", xy=(x_tuned, y_tuned), xytext=(x_naive, y_naive),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=arrow_lw,
                            shrinkA=6, shrinkB=6, mutation_scale=10),
            zorder=4,
        )

    if np.isfinite(y_naive):
        ax.scatter(x_naive, y_naive, s=70, color=DPO_NAIVE_C, edgecolors="white",
                   linewidths=1.2, zorder=6, marker="o")
        label = f">{Y_MAX:.0f}%" if (y_naive_raw * scale) > Y_MAX else f"{y_naive_raw * scale:.0f}%"
        text_x = x_naive - 0.18 if not spread else x_naive - 0.12
        ax.text(text_x, y_naive, label, va="center", ha="right",
                fontsize=6, color=DPO_NAIVE_C, fontweight="bold", zorder=7)

    if np.isfinite(y_tuned):
        ax.scatter(x_tuned, y_tuned, s=70, color=DPO_TUNED_C, edgecolors="white",
                   linewidths=1.2, zorder=6, marker="D")
        label = f">{Y_MAX:.0f}%" if (y_tuned_raw * scale) > Y_MAX else f"{y_tuned_raw * scale:.0f}%"
        text_x = x_tuned - 0.18 if not spread else x_tuned + 0.12
        ha = "right" if not spread else "left"
        ax.text(text_x, y_tuned, label, va="center", ha=ha,
                fontsize=6, color=DPO_TUNED_C, fontweight="bold", zorder=7)

    # MSE tuned marker + percentage label (centered)
    mse_pct_raw = mse_pct_by_exp.get(exp, np.nan)
    if np.isfinite(mse_pct_raw):
        mse_scaled = mse_pct_raw * scale
        mse_plot = min(mse_scaled, Y_MAX)
        ax.scatter(MSE_X, mse_plot, s=95, color=MSE_C, edgecolors="white",
                   linewidths=1.0, zorder=6, marker="*")
        mse_label = f">{Y_MAX:.0f}%" if mse_scaled > Y_MAX else f"{mse_scaled:.0f}%"
        ax.text(MSE_X + 0.07, mse_plot, mse_label, va="center", ha="left",
                fontsize=6, color=MSE_C, fontweight="bold", zorder=7)

    ax.set_xlabel(exp_short.get(exp, exp), fontsize=6.5, fontweight="bold", labelpad=3)

# ---- Average Rank column ----
methods_by_avg = sorted(all_methods, key=lambda m: avg_rank.get(m, 9999))

# Style avg column for rank (linear scale, 1 to max)
ax_avg.set_ylim(max_avg_rank + 0.5, 0.5)  # Inverted: 1 (best) at top
ax_avg.set_xlim(-0.45, 0.45)
ax_avg.set_xticks([])
ax_avg.tick_params(axis="y", labelsize=6.5, length=0, pad=2)
ax_avg.spines["left"].set_linewidth(0.6)
ax_avg.spines["left"].set_color("#cccccc")

for m in methods_by_avg:
    ar = avg_rank[m]
    if np.isnan(ar):
        continue

    if m == "MSE":
        ax_avg.scatter(0, ar, s=95, marker="*", color=MSE_C,
                       edgecolors="white", linewidths=1.0, zorder=6)
        ax_avg.text(0.14, ar, f"{abbr[m]} ({ar:.1f})", va="center", ha="left",
                    fontsize=6.5, color=MSE_C, fontweight="bold", zorder=7)
        continue

    is_dpo = "DPO" in m
    if is_dpo:
        c = DPO_NAIVE_C if m == "DPO (naive)" else DPO_TUNED_C
        mkr = "o" if m == "DPO (naive)" else "D"
        ax_avg.scatter(0, ar, s=70, marker=mkr, color=c,
                       edgecolors="white", linewidths=1.2, zorder=6)
        ax_avg.text(0.14, ar, f"{abbr[m]} ({ar:.1f})", va="center", ha="left",
                    fontsize=6.5, color=c, fontweight="bold", zorder=7)
    else:
        xpos = jitter_x.get(m, 0)
        mkr, col, sz, alp = baseline_style(m)
        ax_avg.scatter(xpos, ar, s=sz, color=col, edgecolors="white",
                       linewidths=0.5, zorder=3, marker=mkr, alpha=alp)

ax_avg.set_xlabel("Avg\nRank", fontsize=6.5, fontweight="bold", labelpad=3)

# Legend — now includes SPO+ and best baseline
legend_els = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=DPO_NAIVE_C,
           markeredgecolor="white", markersize=7, markeredgewidth=1.0,
           label="DPO (naive)"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor=DPO_TUNED_C,
           markeredgecolor="white", markersize=6, markeredgewidth=1.0,
           label="DPO (tuned)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor=MSE_C,
           markeredgecolor="white", markersize=9, markeredgewidth=0.8,
           label="MSE (Decision-Blind)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=DARK_GRAY,
           markeredgecolor="white", markersize=6, markeredgewidth=0.5,
           alpha=0.85, label="SPO+"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor=DARK_GRAY,
           markeredgecolor="white", markersize=6, markeredgewidth=0.5,
           alpha=0.85, label=f"{abbr.get(best_baseline, best_baseline)} (best baseline)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=BASELINE_C,
           markeredgecolor="white", markersize=6, markeredgewidth=0.5,
           alpha=0.7, label="Other baselines"),
]
fig.legend(handles=legend_els, loc="upper right", bbox_to_anchor=(.835, 0.55),
           ncol=1, frameon=True, framealpha=0.85, edgecolor="#cccccc",
           fontsize=7, handletextpad=0.3, borderpad=0.4, labelspacing=0.3)

plt.subplots_adjust(top=0.95, bottom=0.18, left=0.08, right=0.97)
plt.savefig("dpo_pct_of_best_arrow_mseasy.pdf", bbox_inches="tight", dpi=300)
print("Saved: dpo_pct_of_best_arrow_mseasy.pdf")
plt.show()
