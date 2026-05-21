"""
fig_rank_transition.py
----------------------
Four-column rank bump chart showing how methods move as we progress from
the original benchmark paper to the full rerun:

  Col 1: Original paper numbers                  (11 methods, 7 tasks)
  Col 2: Rerun numbers, same methods/tasks       (11 methods, 7 tasks)
         [arrows from Col 1 -> Col 2: HP-tuning effect]
  Col 3: All 15 methods ranked on the same 7 tasks, rerun numbers.
         Paper methods keep their slots; the 4 new methods (perturb, qptl,
         pg, dad) slot in at their rank among 15.
  Col 4: All methods ranked across all rerun tasks (15 methods, 13 tasks)
         [arrows from Col 3 -> Col 4: task-expansion effect, all methods]

Data sources:
  results/comparable_methods_tables.csv  (Benchmark matrix rows = paper)
  results/bench_bump_rerun_data.json     (rerun regret per method x task)

Boxes colored by rank within their column's ranking (green = best, red = worst),
using RdYlGn_r.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize

# ---- Paths ----
CSV_PATH   = "results/comparable_methods_tables.csv"
RERUN_JSON = "results/bench_bump_rerun_data.json"
OUT_DIR    = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Task sets ----
ORIGINAL_7 = ["knapsack", "knapsack-real", "energy", "budgetalloc",
              "cubic", "bipartitematching", "portfolio"]
ALL_13 = ORIGINAL_7 + ["asurv", "cook_county", "speed_humps",
                       "sp_synth", "sp_planted", "shortestpath"]

# ---- Method sets ----
PAPER_METHODS = ["mse", "spo", "dfl", "blackbox", "identity",
                 "nce", "pointLTR", "pairLTR", "listLTR", "lodl", "cpLayer"]
NEW_METHODS   = ["perturb", "qptl", "pg", "dad", "mse_train", "mse_val"]
ALL_METHODS   = PAPER_METHODS + NEW_METHODS   # 17 total

METHOD_DISPLAY = {
    "mse": "MSE", "spo": "SPO+", "dfl": "DFL", "blackbox": "Blackbox",
    "identity": "Identity", "nce": "NCE",
    "pointLTR": "pt-LTR", "pairLTR": "pr-LTR", "listLTR": "L-LTR",
    "lodl": "LODL", "cpLayer": "cpLayer",
    "perturb": "Perturb", "qptl": "QPTL", "pg": "PG", "dad": "DAD",
    "mse_train": "MSE (train-sel)", "mse_val": "MSE (val-sel)",
}

USE_ABSOLUTE = {"portfolio"}   # portfolio regret is raw, others are %


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
    "listLTR": "listLTR", "lodl": "lodl", "cpLayer": "cpLayer",
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

rerun_regret = {m: {t: np.nan for t in ALL_13} for m in ALL_METHODS}
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
    """Per task, rank by regret ascending (1 = best). NaN => worst (rank=n).
    Returns method -> mean rank."""
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


# Column 1: paper methods ranked on ORIGINAL numbers, 7 tasks
col1_ranks_raw = avg_rank(paper_regret, PAPER_METHODS, ORIGINAL_7)

# Column 2: paper methods ranked on RERUN numbers, 7 tasks
col2_ranks_raw = avg_rank(rerun_regret, PAPER_METHODS, ORIGINAL_7)

# Column 3: all 15 methods ranked on 7 original tasks (rerun numbers)
col3_ranks_raw = avg_rank(rerun_regret, ALL_METHODS, ORIGINAL_7)

# Column 4: ALL 15 methods ranked across ALL 13 tasks
col4_ranks_raw = avg_rank(rerun_regret, ALL_METHODS, ALL_13)


def to_ordinal(avg_ranks):
    """Convert mean ranks -> ordinal positions 1..N (ties broken by mean rank)."""
    sorted_methods = sorted(avg_ranks.keys(), key=lambda m: avg_ranks[m])
    return {m: i + 1 for i, m in enumerate(sorted_methods)}


col1_pos = to_ordinal(col1_ranks_raw)            # 1..11
col2_pos = to_ordinal(col2_ranks_raw)            # 1..11
col3_pos = to_ordinal(col3_ranks_raw)            # 1..15
col4_pos = to_ordinal(col4_ranks_raw)            # 1..15


# ============================================================
# Plot
# ============================================================

N_COL1 = len(PAPER_METHODS)    # 11
N_COL2 = len(PAPER_METHODS)    # 11
N_COL3 = len(ALL_METHODS)      # 15 slots (4 drawn)
N_COL4 = len(ALL_METHODS)      # 15
MAX_ROWS = max(N_COL1, N_COL2, N_COL3, N_COL4)   # 15

COL_X = [0.0, 1.7, 3.4, 5.1]
BOX_W = 1.15
BOX_H = 0.78

def y_for_pos(pos):
    """Rank 1 at top; y decreases with rank."""
    return MAX_ROWS - pos

def rank_color(pos, N):
    frac = (pos - 1) / max(N - 1, 1)
    return plt.cm.RdYlGn_r(frac)

def draw_box(ax, cx, pos, N_denom, label, alpha=1.0):
    y = y_for_pos(pos)
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


fig, ax = plt.subplots(figsize=(10.5, 9.0))

# --- Col 1: paper methods, original numbers ---
for m, p in col1_pos.items():
    draw_box(ax, COL_X[0], p, N_COL1, METHOD_DISPLAY[m])

# --- Col 2: paper methods, rerun numbers ---
for m, p in col2_pos.items():
    draw_box(ax, COL_X[1], p, N_COL2, METHOD_DISPLAY[m])

# --- Col 3: all 15 methods ranked on 7 tasks (rerun) ---
#     Paper methods are drawn in a muted style to keep emphasis on
#     the 4 new methods that are appearing for the first time.
for m, p in col3_pos.items():
    alpha = 1.0 if m in NEW_METHODS else 0.55
    draw_box(ax, COL_X[2], p, N_COL3, METHOD_DISPLAY[m], alpha=alpha)

# --- Col 4: all 15 methods, rerun, all tasks ---
for m, p in col4_pos.items():
    draw_box(ax, COL_X[3], p, N_COL4, METHOD_DISPLAY[m])


# --- Arrows Col 1 -> Col 2 (same method) ---
for m in PAPER_METHODS:
    y1 = y_for_pos(col1_pos[m])
    y2 = y_for_pos(col2_pos[m])
    draw_arrow(ax,
               COL_X[0] + BOX_W / 2, y1,
               COL_X[1] - BOX_W / 2, y2)

# --- Arrows Col 3 -> Col 4 (all 15 methods) ---
for m in ALL_METHODS:
    y3 = y_for_pos(col3_pos[m])
    y4 = y_for_pos(col4_pos[m])
    emphasis = m in NEW_METHODS
    draw_arrow(ax,
               COL_X[2] + BOX_W / 2, y3,
               COL_X[3] - BOX_W / 2, y4,
               alpha=0.75 if emphasis else 0.4)


# --- Column headers ---
headers = [
    ("Original\npaper", "7 tasks × 11 methods", COL_X[0]),
    ("Rerun\n(HP tuned)", "7 tasks × 11 methods", COL_X[1]),
    ("+ 4 new\nmethods", "7 tasks × 15 methods", COL_X[2]),
    ("+ 6 new\ntasks", "Shortest Path ×3, Speed Humps,\nCook County, Cranes", COL_X[3]),
]
for title, sub, cx in headers:
    ax.text(cx, MAX_ROWS + 1.3, title, ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(cx, MAX_ROWS + 0.15, sub, ha="center", va="bottom",
            fontsize=8.5, color="#444444")

# --- y ticks: rank numbers ---
ax.set_xlim(-0.95, 6.15)
ax.set_ylim(-1.1, MAX_ROWS + 3.0)
ax.set_yticks([y_for_pos(r) for r in range(1, MAX_ROWS + 1)])
ax.set_yticklabels([str(r) for r in range(1, MAX_ROWS + 1)], fontsize=9)
ax.set_ylabel("Rank (1 = best)", fontsize=10)
ax.set_xticks([])
for spine in ["top", "right", "bottom"]:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="y", left=False)

# --- Colorbar ---
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=Normalize(vmin=0, vmax=1))
sm.set_array([])
cax = fig.add_axes([0.92, 0.32, 0.018, 0.36])
cb = fig.colorbar(sm, cax=cax)
cb.set_ticks([0, 0.5, 1])
cb.set_ticklabels(["best", "mid", "worst"])
cb.set_label("Rank within column", fontsize=9)
cb.ax.tick_params(labelsize=8)

plt.suptitle("Method ranks: paper → rerun → + new methods → all 13 tasks",
             fontsize=13, y=0.985)

plt.tight_layout(rect=[0, 0, 0.9, 0.96])
for ext in ("png", "pdf"):
    out = os.path.join(OUT_DIR, f"fig_rank_transition.{ext}")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

# --- Companion JSON: dump ranks used in the figure ---
summary = {
    "col1_original_paper":      {"methods": PAPER_METHODS, "tasks": ORIGINAL_7,
                                  "mean_rank": col1_ranks_raw,
                                  "ordinal":   col1_pos},
    "col2_rerun_same":          {"methods": PAPER_METHODS, "tasks": ORIGINAL_7,
                                  "mean_rank": col2_ranks_raw,
                                  "ordinal":   col2_pos},
    "col3_all15_7tasks":        {"methods": ALL_METHODS, "tasks": ORIGINAL_7,
                                  "new_methods_emphasized": NEW_METHODS,
                                  "mean_rank": col3_ranks_raw,
                                  "ordinal":   col3_pos},
    "col4_full_rerun":          {"methods": ALL_METHODS, "tasks": ALL_13,
                                  "mean_rank": col4_ranks_raw,
                                  "ordinal":   col4_pos},
}
with open(os.path.join(OUT_DIR, "fig_rank_transition_data.json"), "w") as f:
    json.dump(summary, f, indent=2, default=float)
print("Saved: results/fig_rank_transition_data.json")
