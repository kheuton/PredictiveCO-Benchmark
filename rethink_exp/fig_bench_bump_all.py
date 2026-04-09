"""
fig_bench_bump_all.py
---------------------
Single-figure bump chart of all 11 methods × 7 tasks from hardcoded regret data.
Y-axis: relative regret % (log scale, inverted — lower = better = top).

Output:
  results/fig_bench_bump_all.{png,pdf}
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Hardcoded regret data (relative regret %) ----
TASKS = [
    "Knapsack (Gen)", "Knapsack (Energy)", "Scheduling (Energy)",
    "Budget Allocation", "TopK (Cubic)", "Bipartite Matching", "Portfolio",
]

METHODS = [
    "Two-stage", "DFL", "Blackbox", "Identity", "CPLayer",
    "SPO", "NCE", "point-LTR", "pair-LTR", "list-LTR", "LODL",
]

regret_data = {
    "Knapsack (Gen)": {
        "Two-stage": 6.595, "DFL": 11.744, "Blackbox": 24.274, "Identity": 31.874,
        "CPLayer": 24.769, "SPO": 6.223, "NCE": 13.438, "point-LTR": 6.402,
        "pair-LTR": 7.820, "list-LTR": 6.031, "LODL": 6.044,
    },
    "Knapsack (Energy)": {
        "Two-stage": 8.745, "DFL": 8.353, "Blackbox": 35.705, "Identity": 17.156,
        "CPLayer": 36.402, "SPO": 8.407, "NCE": 11.932, "point-LTR": 8.236,
        "pair-LTR": 9.022, "list-LTR": 8.083, "LODL": 9.567,
    },
    "Scheduling (Energy)": {
        "Two-stage": 1.793, "DFL": 6.272, "Blackbox": 6.503, "Identity": 5.690,
        "CPLayer": np.nan, "SPO": 1.505, "NCE": 1.663, "point-LTR": 4.548,
        "pair-LTR": 1.540, "list-LTR": 1.551, "LODL": 1.786,
    },
    "Budget Allocation": {
        "Two-stage": 20.332, "DFL": 35.970, "Blackbox": 26.905, "Identity": 14.799,
        "CPLayer": np.nan, "SPO": 5.559, "NCE": 9.979, "point-LTR": 69.663,
        "pair-LTR": 5.958, "list-LTR": 5.742, "LODL": 25.700,
    },
    "TopK (Cubic)": {
        "Two-stage": 0.110, "DFL": 1.974, "Blackbox": 13.944, "Identity": 13.944,
        "CPLayer": np.nan, "SPO": 160.408, "NCE": 160.408, "point-LTR": 1.149,
        "pair-LTR": 5.072, "list-LTR": 0.193, "LODL": 0.172,
    },
    "Bipartite Matching": {
        "Two-stage": 92.963, "DFL": 91.364, "Blackbox": 91.988, "Identity": 91.868,
        "CPLayer": 92.007, "SPO": 93.327, "NCE": 92.622, "point-LTR": 91.035,
        "pair-LTR": 92.285, "list-LTR": 91.831, "LODL": 91.113,
    },
    "Portfolio": {
        "Two-stage": 0.243, "DFL": 0.380, "Blackbox": 0.286, "Identity": 0.280,
        "CPLayer": 0.309, "SPO": 0.245, "NCE": 0.367, "point-LTR": 0.214,
        "pair-LTR": 0.255, "list-LTR": 0.249, "LODL": 0.160,
    },
}

# ---- Colors and markers ----
MSE_C  = "#1f77b4"   # blue  (Two-stage)
LTRL_C = "#e67e22"   # orange (list-LTR)
G1 = "#bbbbbb"
G2 = "#888888"
G3 = "#444444"
G4 = "#222222"
G5 = "#666666"

# (marker, color, size, alpha)
METHOD_STYLE = {
    "Two-stage":  ("*",  MSE_C,  80,  1.0),
    # Surrogate Gradient → hexagon
    "DFL":        ("h",  G1,     50,  0.90),
    "Blackbox":   ("h",  G2,     50,  0.90),
    "Identity":   ("h",  G3,     50,  0.90),
    "SPO":        ("h",  G5,     50,  0.90),
    # Trained Surrogate → circle
    "LODL":       ("o",  G2,     44,  0.85),
    # Continuous → triangle
    "CPLayer":    ("^",  G1,     50,  0.90),
    # Statistical → square (list-LTR orange)
    "NCE":        ("s",  G1,     44,  0.85),
    "point-LTR":  ("s",  G2,     44,  0.85),
    "pair-LTR":   ("s",  G3,     44,  0.85),
    "list-LTR":   ("s",  LTRL_C, 50,  0.90),
}

# ---- Jitter (fixed per method) ----
np.random.seed(42)
jitter_x = {m: np.random.uniform(-0.28, 0.28) for m in METHODS}
jitter_x["Two-stage"] = -0.15
jitter_x["list-LTR"]  =  0.0

# ---- Plotting ----
plt.rcParams.update({
    "font.family":         "sans-serif",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.bottom":  False,
})


def _choose_5_ticks(v_min, v_max):
    lo = np.log10(max(v_min, 1e-9))
    hi = np.log10(max(v_max, lo + 1e-6))
    return 10 ** np.linspace(lo, hi, 5)


def _fmt_val(x, _, is_absolute=False):
    if is_absolute:
        return f"{x:.3f}"
    if x < 1.0:
        return f"{x:.3f}%"
    if x < 10.0:
        return f"{x:.2f}%"
    return f"{x:.1f}%"


ABSOLUTE_TASKS = {"Portfolio"}


# ---- Build figure ----
n_tasks = len(TASKS)
# Task columns + avg rank column
n_cols = n_tasks + 1
width_ratios = [1] * n_tasks + [1.1]
fig = plt.figure(figsize=(12.5, 3.2))
gs = GridSpec(1, n_cols, figure=fig, width_ratios=width_ratios, wspace=0.10)
axes = [fig.add_subplot(gs[0, j]) for j in range(n_tasks)]

for j, (ax, task) in enumerate(zip(axes, TASKS)):
    vals = [regret_data[task][m] for m in METHODS
            if np.isfinite(regret_data[task].get(m, np.nan))]
    if not vals:
        continue

    log_range = np.log10(max(vals)) - np.log10(max(min(vals), 1e-9))
    pad = log_range * 0.15 + 0.005
    v_min = max(10 ** (np.log10(max(min(vals), 1e-9)) - pad), 1e-4)
    v_max = 10 ** (np.log10(max(vals)) + pad)
    ticks = _choose_5_ticks(v_min, v_max)

    ax.set_yscale("log")
    ax.set_ylim(v_max, v_min)   # inverted
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks(ticks)
    is_abs = task in ABSOLUTE_TASKS
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos, _abs=is_abs: _fmt_val(x, pos, _abs)))
    ax.tick_params(axis="y", labelsize=6.5, length=0, pad=-3)

    if j == 0:
        ax.set_ylabel("Regret (log scale)", fontsize=7,
                       fontweight="bold", labelpad=24)

    for y in ticks:
        ax.axhline(y, color="#dddddd", linewidth=0.5, zorder=0)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["left"].set_zorder(1)

    # Plot each method
    for m in METHODS:
        yval = regret_data[task].get(m, np.nan)
        if not np.isfinite(yval):
            continue
        mkr, col, sz, alp = METHOD_STYLE[m]
        ax.scatter(jitter_x[m], yval, s=sz, color=col, edgecolors="white",
                   linewidths=0.5, zorder=3, marker=mkr, alpha=alp)

    # Wrap task name for display
    label = task.replace(" (", "\n(")
    ax.set_xlabel(label, fontsize=6.5, fontweight="bold", labelpad=3)

# ---- Average Rank column ----
ax_avg = fig.add_subplot(gs[0, n_tasks])
avg_rank = {}
for m in METHODS:
    ranks = []
    for task in TASKS:
        col_vals = {mm: regret_data[task].get(mm, np.nan) for mm in METHODS
                    if np.isfinite(regret_data[task].get(mm, np.nan))}
        if m not in col_vals:
            continue
        sorted_m = sorted(col_vals, key=col_vals.__getitem__)
        ranks.append(sorted_m.index(m) + 1)
    avg_rank[m] = float(np.mean(ranks)) if ranks else np.nan

valid_ranks = [v for v in avg_rank.values() if np.isfinite(v)]
max_rank = max(valid_ranks) if valid_ranks else len(METHODS)

ax_avg.set_ylim(max_rank + 0.5, 0.5)
ax_avg.set_xlim(-0.45, 0.45)
ax_avg.set_xticks([])
labeled = [r for r in range(1, int(max_rank) + 1) if r % 2 == 1]
ax_avg.set_yticks(labeled)
ax_avg.set_yticklabels([str(r) for r in labeled])
ax_avg.tick_params(axis="y", labelsize=6.5, length=0, pad=2)
ax_avg.spines["left"].set_linewidth(0.6)
ax_avg.spines["left"].set_color("#cccccc")
ax_avg.spines["top"].set_visible(False)
ax_avg.spines["right"].set_visible(False)
ax_avg.spines["bottom"].set_visible(False)

for r in range(1, int(max_rank) + 1):
    ax_avg.axhline(r, color="#dddddd", linewidth=0.5, zorder=0)

for m in METHODS:
    ar = avg_rank.get(m, np.nan)
    if not np.isfinite(ar):
        continue
    mkr, col, sz, alp = METHOD_STYLE[m]
    ax_avg.scatter(jitter_x[m], ar, s=sz, color=col, edgecolors="white",
                   linewidths=0.5, zorder=3, marker=mkr, alpha=alp)

ax_avg.set_xlabel("Avg\nRank", fontsize=6.5, fontweight="bold", labelpad=3)

# ---- Legend ----
def _le(marker, color, label, ms=7):
    return Line2D([0], [0], marker=marker, color="w",
                  markerfacecolor=color, markeredgecolor="white",
                  markersize=ms, markeredgewidth=0.5, label=label)

def _hdr(text):
    return Line2D([0], [0], color="none",
                  label=r"$\bf{" + text.replace(" ", r"\ ") + r"}$")

legend_els = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor=MSE_C,
           markeredgecolor="white", markersize=10, markeredgewidth=0.8,
           label="Two-stage  (Decision-Unaware)"),
    _hdr("Surrogate Gradient"),
    _le("h", G1, "DFL"),
    _le("h", G2, "Blackbox"),
    _le("h", G3, "Identity"),
    _le("h", G5, "SPO"),
    _hdr("Trained Surrogate"),
    _le("o", G2, "LODL"),
    _hdr("Continuous"),
    _le("^", G1, "CPLayer"),
    _hdr("Statistical"),
    _le("s", G1, "NCE"),
    _le("s", G2, "point-LTR"),
    _le("s", G3, "pair-LTR"),
    _le("s", LTRL_C, "list-LTR"),
]

fig.legend(handles=legend_els, loc="upper left",
           bbox_to_anchor=(0.86, 0.92), ncol=1,
           frameon=True, framealpha=0.88, edgecolor="#cccccc",
           fontsize=7, handletextpad=0.3, borderpad=0.4, labelspacing=0.15)

plt.subplots_adjust(top=0.88, bottom=0.22, left=0.06, right=0.85)
fig.suptitle("Benchmark — Relative regret % (best ↑)", fontsize=8,
             fontweight="bold", y=1.01)

# ---- White-bbox tick labels (occludes spine) ----
all_axes = axes + [ax_avg]
for ax in all_axes:
    ax.spines["left"].set_zorder(2)
    fig.canvas.draw()
    tick_positions = ax.get_yticks()
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    ax.set_yticklabels([""] * len(tick_positions))
    for yval, txt in zip(tick_positions, tick_labels):
        if not txt:
            continue
        ax.text(0.0, yval, txt, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.5,
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="square,pad=0.15"),
                zorder=10, clip_on=False)

# ---- Save ----
out_png = os.path.join(OUT_DIR, "fig_bench_bump_all.png")
out_pdf = os.path.join(OUT_DIR, "fig_bench_bump_all.pdf")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved {out_png}  +  {out_pdf}")

# ---- Print summary ----
print(f"\n{'Method':>12}", end="")
for t in TASKS:
    print(f"  {t[:14]:>14}", end="")
print(f"  {'Avg Rank':>8}")
print("-" * (12 + 16 * len(TASKS) + 10))
for m in METHODS:
    print(f"{m:>12}", end="")
    for t in TASKS:
        v = regret_data[t].get(m, np.nan)
        if np.isfinite(v):
            print(f"  {v:>14.3f}", end="")
        else:
            print(f"  {'—':>14}", end="")
    ar = avg_rank.get(m, np.nan)
    print(f"  {ar:>8.2f}" if np.isfinite(ar) else f"  {'—':>8}")

# ---- Export JSON for JS visualization ----
import json

export = {
    "tasks": TASKS,
    "methods": METHODS,
    "absolute_tasks": list(ABSOLUTE_TASKS),
    "regret": {
        task: {m: (None if np.isnan(v) else v)
               for m, v in regret_data[task].items()}
        for task in TASKS
    },
    "avg_rank": {m: (None if np.isnan(v) else round(v, 4))
                 for m, v in avg_rank.items()},
    "style": {
        m: {"marker": s[0], "color": s[1], "size": s[2], "alpha": s[3]}
        for m, s in METHOD_STYLE.items()
    },
    "method_groups": {
        "Decision-Unaware": ["Two-stage"],
        "Surrogate Gradient": ["DFL", "Blackbox", "Identity", "SPO"],
        "Trained Surrogate": ["LODL"],
        "Continuous": ["CPLayer"],
        "Statistical": ["NCE", "point-LTR", "pair-LTR", "list-LTR"],
    },
}

json_path = os.path.join(OUT_DIR, "bench_bump_all_data.json")
with open(json_path, "w") as f:
    json.dump(export, f, indent=2)
print(f"Exported {json_path}")
