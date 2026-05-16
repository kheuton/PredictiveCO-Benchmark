"""
fig_rank_transition_2col.py
---------------------------
Two-column rank bump charts (subset of fig_rank_transition.py):

  Col 1: Original paper numbers
  Col 2: Rerun numbers, same methods/tasks
         [arrows from Col 1 -> Col 2: HP-tuning effect]

cpLayer is dropped, leaving 10 methods.

Two figures emitted:
  fig_rank_transition_2col_7tasks.{png,pdf}  -- original 7 tasks
  fig_rank_transition_2col_5tasks.{png,pdf}  -- original 7 minus
                                                budgetalloc and portfolio
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

CSV_PATH   = "results/comparable_methods_tables.csv"
RERUN_JSON = "results/bench_bump_rerun_data.json"
OUT_DIR    = "results"
os.makedirs(OUT_DIR, exist_ok=True)

ORIGINAL_7 = ["knapsack", "knapsack-real", "energy", "budgetalloc",
              "cubic", "bipartitematching", "portfolio"]
TASKS_5    = ["knapsack", "knapsack-real", "energy",
              "cubic", "bipartitematching"]
ALL_13 = ORIGINAL_7 + ["asurv", "cook_county", "speed_humps",
                       "sp_synth", "sp_planted", "shortestpath"]

# 10 methods (cpLayer dropped)
PAPER_METHODS = ["mse", "spo", "dfl", "blackbox", "identity",
                 "nce", "pointLTR", "pairLTR", "listLTR", "lodl"]

METHOD_DISPLAY = {
    "mse": "MSE", "spo": "SPO+", "dfl": "DFL", "blackbox": "Blackbox",
    "identity": "Identity", "nce": "NCE",
    "pointLTR": "pt-LTR", "pairLTR": "pr-LTR", "listLTR": "L-LTR",
    "lodl": "LODL",
}


# ============================================================
# Load paper (original) benchmark numbers
# ============================================================

df = pd.read_csv(CSV_PATH)
df.columns = ["source", "problem", "variant", "metric", "method", "value"]

BM_PROB_MAP = {
    ("Knapsack", "Gen"):            "knapsack",
    ("Knapsack", "Energy"):         "knapsack-real",
    ("Scheduling", "Energy"):       "energy",
    ("Budget Allocation", None):    "budgetalloc",
    ("TopK", "Cubic"):              "cubic",
    ("Bipartite Matching", None):   "bipartitematching",
    ("Portfolio", None):            "portfolio",
}
BM_METHOD_MAP = {
    "two-stage": "mse",
    "spo": "spo", "dfl": "dfl", "blackbox": "blackbox", "identity": "identity",
    "nce": "nce", "pointLTR": "pointLTR", "pairLTR": "pairLTR",
    "listLTR": "listLTR", "lodl": "lodl",
}

def _bm_prob(row):
    p, v = row["problem"], row["variant"]
    v = v if isinstance(v, str) and v.strip() else None
    return BM_PROB_MAP.get((p, v))

def _parse(v):
    s = str(v).strip()
    if s in ("—", "N/A", "-", ""):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

bm = df[(df["source"] == "Benchmark matrix") &
        df["metric"].isin(["Regret", "Regret (%)"])].copy()

paper_regret = {m: {t: np.nan for t in ORIGINAL_7} for m in PAPER_METHODS}
for _, row in bm.iterrows():
    task   = _bm_prob(row)
    method = BM_METHOD_MAP.get(row["method"])
    if task is None or method is None:
        continue
    v = _parse(row["value"])
    if np.isnan(v):
        continue
    paper_regret[method][task] = v


# ============================================================
# Load rerun numbers
# ============================================================

with open(RERUN_JSON) as f:
    rerun = json.load(f)

DISP_TO_INT = {d: i for i, d in METHOD_DISPLAY.items()}

rerun_regret = {m: {t: np.nan for t in ALL_13} for m in PAPER_METHODS}
for task, mdict in rerun["regret"].items():
    if task not in ALL_13:
        continue
    for disp, val in mdict.items():
        m = DISP_TO_INT.get(disp)
        if m is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            rerun_regret[m][task] = v


# ============================================================
# Rank computation
# ============================================================

def avg_rank(regret_data, methods, tasks):
    n = len(methods)
    per_task = {m: [] for m in methods}
    for t in tasks:
        pairs = [(m, regret_data[m].get(t, np.nan)) for m in methods]
        finite = sorted([(m, v) for m, v in pairs if np.isfinite(v)],
                        key=lambda x: x[1])
        rank = {m: i + 1 for i, (m, _) in enumerate(finite)}
        for m, v in pairs:
            if not np.isfinite(v):
                rank[m] = n
        for m in methods:
            per_task[m].append(rank[m])
    return {m: float(np.mean(per_task[m])) for m in methods}


def to_ordinal(avg_ranks):
    sorted_methods = sorted(avg_ranks.keys(), key=lambda m: avg_ranks[m])
    return {m: i + 1 for i, m in enumerate(sorted_methods)}


# ============================================================
# Plot helpers
# ============================================================

BOX_W = 1.15
BOX_H = 0.78
COL_X = [0.0, 1.7]


def y_for_pos(pos, max_rows):
    return max_rows - pos


def rank_color(pos, N):
    frac = (pos - 1) / max(N - 1, 1)
    return plt.cm.RdYlGn_r(frac)


def draw_box(ax, cx, pos, N_denom, label, max_rows, alpha=1.0):
    y = y_for_pos(pos, max_rows)
    color = rank_color(pos, N_denom)
    rect = Rectangle((cx - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
                     facecolor=color, edgecolor="black",
                     linewidth=0.8, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, y, label, ha="center", va="center",
            fontsize=9, color="black", zorder=4)
    return y


def draw_arrow(ax, x0, y0, x1, y1, color="#444444", alpha=0.55):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=10,
        color=color, linewidth=0.8, alpha=alpha,
        shrinkA=0, shrinkB=0, zorder=2,
    )
    ax.add_patch(arrow)


# ============================================================
# Figure builder
# ============================================================

def make_figure(tasks, out_tag):
    n_methods = len(PAPER_METHODS)
    max_rows  = n_methods

    col1 = avg_rank(paper_regret, PAPER_METHODS, tasks)
    col2 = avg_rank(rerun_regret, PAPER_METHODS, tasks)
    col1_pos = to_ordinal(col1)
    col2_pos = to_ordinal(col2)

    fig, ax = plt.subplots(figsize=(6.0, 7.5))

    for m, p in col1_pos.items():
        draw_box(ax, COL_X[0], p, n_methods, METHOD_DISPLAY[m], max_rows)
    for m, p in col2_pos.items():
        draw_box(ax, COL_X[1], p, n_methods, METHOD_DISPLAY[m], max_rows)

    for m in PAPER_METHODS:
        y1 = y_for_pos(col1_pos[m], max_rows)
        y2 = y_for_pos(col2_pos[m], max_rows)
        draw_arrow(ax,
                   COL_X[0] + BOX_W / 2, y1,
                   COL_X[1] - BOX_W / 2, y2)

    headers = [
        ("Original", COL_X[0]),
        ("More Hyperparameter\nTuning", COL_X[1]),
    ]
    for title, cx in headers:
        ax.text(cx, max_rows + 1.0, title, ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_xlim(-0.95, COL_X[1] + 1.6)
    ax.set_ylim(-1.1, max_rows + 3.0)
    ax.set_yticks([y_for_pos(r, max_rows) for r in range(1, max_rows + 1)])
    ax.set_yticklabels([str(r) for r in range(1, max_rows + 1)], fontsize=9)
    ax.set_ylabel("Rank (1 = best)", fontsize=10)
    ax.set_xticks([])
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", left=False)

    plt.suptitle("Method Ranks", fontsize=13, y=0.985)
    plt.tight_layout(rect=[0, 0, 1.0, 0.96])

    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"fig_rank_transition_2col_{out_tag}.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

    return {
        "tasks": tasks,
        "methods": PAPER_METHODS,
        "col1_mean_rank": col1,
        "col1_ordinal":   col1_pos,
        "col2_mean_rank": col2,
        "col2_ordinal":   col2_pos,
    }


# ============================================================
# Run both figures
# ============================================================

summary = {
    "7tasks": make_figure(ORIGINAL_7, "7tasks"),
    "5tasks": make_figure(TASKS_5, "5tasks"),
}

with open(os.path.join(OUT_DIR, "fig_rank_transition_2col_data.json"), "w") as f:
    json.dump(summary, f, indent=2, default=float)
print("Saved: results/fig_rank_transition_2col_data.json")
