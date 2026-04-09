"""
Benchmark vs. re-run comparison figures.

Outputs:
  results/fig_tables_paper.png        -- paper benchmark table (top-3 highlighted)
  results/fig_tables_rerun.png        -- re-run table (top-3 highlighted)
  results/fig_tables_sidebyside.png   -- both tables side by side
  results/fig_scatter_comparison.png  -- per-task scatter: original vs re-run
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ---- Config ----
CSV_PATH = "results/comparable_methods_tables.csv"
OUT_DIR  = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Task ordering + display names ----
TASK_ORDER = ["knapsack", "kn-real", "energy", "budgalloc", "cubic", "bipart", "portfolio"]
TASK_DISPLAY = {
    "knapsack":  "Knapsack (Gen)",
    "kn-real":   "Knapsack (Real)",
    "energy":    "Scheduling (Energy)",
    "budgalloc": "Budget Alloc.",
    "cubic":     "TopK (Cubic)",
    "bipart":    "Bipartite Match.",
    "portfolio": "Portfolio",
}

# ---- Method ordering + display names ----
METHOD_ORDER = ["mse", "spo", "dfl", "blackbox", "identity",
                "nce", "pointLTR", "pairLTR", "listLTR", "lodl", "cpLayer", "perturb"]
METHOD_DISPLAY = {
    "mse":      "MSE",
    "spo":      "SPO+",
    "dfl":      "DFL",
    "blackbox": "Blackbox",
    "identity": "Identity",
    "nce":      "NCE",
    "pointLTR": "PointLTR",
    "pairLTR":  "PairLTR",
    "listLTR":  "ListLTR",
    "lodl":     "LODL",
    "cpLayer":  "cpLayer",
    "perturb":  "Perturb",
}

# ---- Load CSV ----
df = pd.read_csv(CSV_PATH)
df.columns = ["source", "problem", "variant", "metric", "method", "value"]

# ---- Helper: parse value string to float ----
def _parse(v):
    if isinstance(v, float):
        return v
    s = str(v).strip()
    if s in ("—", "N/A", "-", ""):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ============================================================
# 1. Summary matrix (re-run) — already in [0,1] range
#    multiply ×100 for all except portfolio
# ============================================================
summary_raw = df[(df["source"] == "Summary matrix") & (df["metric"] == "Value")].copy()
summary_raw["val_f"] = summary_raw["value"].apply(_parse)

def scale_summary(row):
    v = row["val_f"]
    if np.isnan(v):
        return np.nan
    if row["variant"] == "portfolio":
        return v          # already in [0,1] raw regret
    return v * 100        # convert to %

summary_raw["regret"] = summary_raw.apply(scale_summary, axis=1)

# Map method "mse" stays "mse"; others stay as-is
# Build pivot: task × method
rerun_pivot = summary_raw.pivot_table(
    index="variant", columns="method", values="regret", aggfunc="first"
)
rerun_pivot = rerun_pivot.reindex(index=TASK_ORDER, columns=METHOD_ORDER)


# ============================================================
# 2. Benchmark matrix (paper)
#    Metric "Regret (%)" — already in %
#    Portfolio metric "Regret" — already in [0,1] raw regret
#    Map (Problem, Variant) → canonical task name
#    Map method "two-stage" → "mse"
# ============================================================
def get_bench_task(row):
    prob = str(row["problem"]).strip()
    var  = str(row["variant"]).strip()
    if prob == "Knapsack" and var == "Gen":
        return "knapsack"
    if prob == "Knapsack" and var == "Energy":
        return "kn-real"
    if prob == "Scheduling" and var == "Energy":
        return "energy"
    if prob == "Budget Allocation":
        return "budgalloc"
    if prob == "TopK" and var == "Cubic":
        return "cubic"
    if prob == "Bipartite Matching":
        return "bipart"
    if prob == "Portfolio":
        return "portfolio"
    return None

bench_raw = df[
    (df["source"] == "Benchmark matrix") &
    (df["metric"].str.contains("Regret"))
].copy()
bench_raw["task"] = bench_raw.apply(get_bench_task, axis=1)
bench_raw = bench_raw[bench_raw["task"].notna()].copy()
bench_raw["val_f"] = bench_raw["value"].apply(_parse)
bench_raw["method_can"] = bench_raw["method"].replace({"two-stage": "mse"})

paper_pivot = bench_raw.pivot_table(
    index="task", columns="method_can", values="val_f", aggfunc="first"
)
paper_pivot = paper_pivot.reindex(index=TASK_ORDER, columns=METHOD_ORDER)


# ============================================================
# Helper: build display table (string + float array)
# ============================================================
def fmt(v, task):
    if np.isnan(v):
        return "—"
    if task == "portfolio":
        return f"{v:.3f}"
    return f"{v:.1f}"

def make_cell_text(pivot):
    rows = []
    for task in TASK_ORDER:
        row = []
        for method in METHOD_ORDER:
            v = pivot.loc[task, method] if (task in pivot.index and method in pivot.columns) else np.nan
            if not isinstance(v, float):
                v = float(v) if v is not None else np.nan
            row.append(fmt(v, task))
        rows.append(row)
    return rows


# ============================================================
# Helper: compute top-3 mask per row
# ============================================================
def top3_mask(pivot):
    """Returns rank array (1=best, 2=second, 3=third, 0=other, -1=nan)."""
    mask = np.full(pivot.shape, 0, dtype=int)
    for i, task in enumerate(TASK_ORDER):
        vals = np.array([
            pivot.loc[task, m] if (task in pivot.index and m in pivot.columns) else np.nan
            for m in METHOD_ORDER
        ], dtype=float)
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            continue
        order = np.argsort(vals[valid])   # ascending (lower regret = better)
        ranked = np.where(valid)[0][order]
        for rank, col_idx in enumerate(ranked[:3], start=1):
            mask[i, col_idx] = rank
        mask[i, ~valid] = -1
    return mask


# ============================================================
# Top-3 colors
# ============================================================
RANK_COLORS = {
    1: "#2ecc71",   # gold-ish green
    2: "#a8e6cf",   # medium green
    3: "#d4f1e4",   # light green
    0: "white",
    -1: "#f0f0f0",  # grey for missing
}

def rank_to_color(r):
    return RANK_COLORS.get(r, "white")


# ============================================================
# Figure: single table (for paper or rerun)
# ============================================================
def draw_table(ax, pivot, title, font_size=8):
    cell_text = make_cell_text(pivot)
    mask = top3_mask(pivot)

    col_labels = [METHOD_DISPLAY[m] for m in METHOD_ORDER]
    row_labels  = [TASK_DISPLAY[t]  for t in TASK_ORDER]

    # Build color array
    cell_colors = [[rank_to_color(mask[i, j])
                    for j in range(len(METHOD_ORDER))]
                   for i in range(len(TASK_ORDER))]

    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.6)
    ax.set_title(title, fontsize=font_size + 2, fontweight="bold", pad=8)

    # Bold top-1 cells
    for i in range(len(TASK_ORDER)):
        for j in range(len(METHOD_ORDER)):
            cell = tbl[i + 1, j]   # +1 for header row
            if mask[i, j] == 1:
                cell.set_text_props(fontweight="bold")


# ============================================================
# Figure 1a/1b: separate table PNGs
# ============================================================
for label, pivot, fname in [
    ("Original Benchmark (NeurIPS 2024) — Regret (lower is better)",
     paper_pivot, "fig_tables_paper.png"),
    ("Re-run (LR × Batch + Method HP tuning) — Regret (lower is better)",
     rerun_pivot, "fig_tables_rerun.png"),
]:
    fig, ax = plt.subplots(figsize=(16, 4))
    draw_table(ax, pivot, label)
    legend_patches = [
        Patch(facecolor=RANK_COLORS[1], label="Best"),
        Patch(facecolor=RANK_COLORS[2], label="2nd"),
        Patch(facecolor=RANK_COLORS[3], label="3rd"),
        Patch(facecolor=RANK_COLORS[-1], label="N/A"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ============================================================
# Figure 1c: side-by-side
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(18, 9))
draw_table(axes[0], paper_pivot,
           "Original Benchmark (NeurIPS 2024) — Regret (lower is better)", font_size=7)
draw_table(axes[1], rerun_pivot,
           "Re-run (LR × Batch + Method HP Tuning) — Regret (lower is better)", font_size=7)

legend_patches = [
    Patch(facecolor=RANK_COLORS[1], label="Best per task"),
    Patch(facecolor=RANK_COLORS[2], label="2nd"),
    Patch(facecolor=RANK_COLORS[3], label="3rd"),
    Patch(facecolor=RANK_COLORS[-1], label="N/A"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.00))
plt.tight_layout(rect=[0, 0.03, 1, 1])
out = os.path.join(OUT_DIR, "fig_tables_sidebyside.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")


# ============================================================
# Figure 2: per-task scatter (paper vs re-run)
# ============================================================
# Build aligned data: for each (task, method) pair, paper value + rerun value
# Only include methods present in both

n_tasks = len(TASK_ORDER)
n_methods = len(METHOD_ORDER)
x_base = np.arange(n_methods)
JITTER = 0.14   # horizontal offset between paper dot and rerun dot

fig, axes = plt.subplots(n_tasks, 1, figsize=(14, 2.8 * n_tasks),
                         sharex=True)

for ax_i, (ax, task) in enumerate(zip(axes, TASK_ORDER)):
    paper_vals = np.array([
        paper_pivot.loc[task, m] if (task in paper_pivot.index and m in paper_pivot.columns) else np.nan
        for m in METHOD_ORDER
    ], dtype=float)
    rerun_vals = np.array([
        rerun_pivot.loc[task, m] if (task in rerun_pivot.index and m in rerun_pivot.columns) else np.nan
        for m in METHOD_ORDER
    ], dtype=float)

    # For methods where BOTH are valid, draw a thin connector line first
    for j in range(n_methods):
        pv, rv = paper_vals[j], rerun_vals[j]
        if np.isnan(pv) or np.isnan(rv):
            continue
        color = "#2ecc71" if rv <= pv else "#e74c3c"
        ax.plot([x_base[j] - JITTER, x_base[j] + JITTER], [pv, rv],
                color=color, linewidth=1.0, alpha=0.5, zorder=1)

    # Paper dots (circle)
    valid_p = ~np.isnan(paper_vals)
    ax.scatter(x_base[valid_p] - JITTER, paper_vals[valid_p],
               marker="o", s=55, color="#2c3e50", zorder=3,
               label="Original" if ax_i == 0 else None)

    # Re-run dots (triangle), colored by improvement
    for j in range(n_methods):
        rv = rerun_vals[j]
        pv = paper_vals[j]
        if np.isnan(rv):
            continue
        color = "#2ecc71" if (not np.isnan(pv) and rv <= pv) else "#e74c3c"
        ax.scatter(x_base[j] + JITTER, rv, marker="^", s=55,
                   color=color, zorder=3,
                   label=("Re-run (↓ better)" if (ax_i == 0 and j == 0) else
                          ("Re-run (↑ worse)" if (ax_i == 0 and j == 1) else None)))

    ax.set_ylabel(TASK_DISPLAY[task], fontsize=8, rotation=0,
                  ha="right", va="center", labelpad=4)
    ax.yaxis.set_label_coords(-0.01, 0.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add y-axis label (regret unit)
    unit = "Regret" if task == "portfolio" else "Regret (%)"
    ax.set_title(f"{TASK_DISPLAY[task]}  [{unit}]", fontsize=8,
                 loc="left", pad=2)

# Shared x-axis labels
axes[-1].set_xticks(x_base)
axes[-1].set_xticklabels([METHOD_DISPLAY[m] for m in METHOD_ORDER],
                         rotation=30, ha="right", fontsize=8)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2c3e50",
           markersize=7, label="Original (NeurIPS 2024)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ecc71",
           markersize=7, label="Re-run (lower = improved)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#e74c3c",
           markersize=7, label="Re-run (higher = regressed)"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=3,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.005))
fig.suptitle("", y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, "fig_scatter_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
