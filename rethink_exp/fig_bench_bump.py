"""
fig_bench_bump.py
-----------------
Bump chart of all methods × tasks from the Phase 1/2 benchmark re-run.

Selection criterion: best config (LR × batch for Phase 1; HP value for Phase 2)
is chosen by minimum val regret (min(eval) from val_logs.csv). Test regret from
results.npy is reported for the selected config.

Y-axis: relative regret × 100 (%) for all tasks except portfolio, which uses
absolute regret.  Log scale, inverted (lower = better → top of chart).

Outputs three figures:
  results/fig_bench_bump_rest.png      -- knapsack, kn-real, energy, cubic, bipartite
  results/fig_bench_bump_budgetalloc.png
  results/fig_bench_bump_portfolio.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ---- Paths ----
RESULTS_ROOT   = "saved_records"
BEST_JSON_PATH = "bench_p1_best.json"
OUT_DIR        = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Problem config (mirrors collect_bench_p1/p2) ----
PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio"]

PROB_ARG = {
    "knapsack":          "knapsack",
    "knapsack-real":     "knapsack",
    "energy":            "energy",
    "budgetalloc":       "budgetalloc",
    "cubic":             "cubic",
    "bipartitematching": "bipartitematching",
    "portfolio":         "portfolio",
}

PROB_VERSION = {
    "knapsack":          "gen",
    "knapsack-real":     "energy",
    "energy":            "energy",
    "budgetalloc":       "real",
    "cubic":             "gen",
    "bipartitematching": "cora",
    "portfolio":         "real",
}

PROB_DISPLAY = {
    "knapsack":          "Knapsack\n(Gen)",
    "knapsack-real":     "Knapsack\n(Real)",
    "energy":            "Scheduling\n(Energy)",
    "budgetalloc":       "Budget Allocation",
    "cubic":             "TopK\n(Cubic)",
    "bipartitematching": "Bipartite\nMatch.",
    "portfolio":         "Portfolio",
}

# ---- Method config ----
METHODS = ["mse", "dfl", "identity", "spo", "nce", "blackbox",
           "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg",
           "qptl", "cpLayer"]

# Extra one-off data points not part of the sweep (portfolio-only sigma=0 run)
# Best chosen by min val regret across sd1_w5_s0_lr* runs.
EXTRA_POINTS = {
    "dfl_direct": {
        "label":  "DFL (no surrogate)",
        "style":  ("h", "#27ae60", 55, 0.95),   # green hexagon
        "values": {"portfolio": 0.25484},         # test regret, abs
        "group":  "surrogate_gradient",
    },
}

METHOD_DISPLAY = {
    "mse":      "MSE",
    "dfl":      "DFL",
    "identity": "Identity",
    "spo":      "SPO+",
    "nce":      "NCE",
    "blackbox": "Blackbox",
    "pointLTR": "pt-LTR",
    "pairLTR":  "pr-LTR",
    "listLTR":  "L-LTR",
    "lodl":     "LODL",
    "perturb":  "Perturb",
    "pg":       "PG",
    "qptl":     "QPTL",
    "cpLayer":  "cpLayer",
}

METHOD_PROBLEMS = {
    "qptl":    {"knapsack", "bipartitematching", "portfolio"},
    "cpLayer": {"knapsack", "bipartitematching", "portfolio"},
    "pg":      {"knapsack", "knapsack-real", "energy", "cubic", "bipartitematching", "portfolio"},
}

LRS          = ["1e-2", "5e-3", "1e-3"]
BATCH_LABELS = ["default", "alt"]
USE_ABSOLUTE = {"portfolio"}

# ---- Phase 2 HP sweep definitions (mirrors collect_bench_p2.py) ----
HP_SWEEPS = {
    "dfl": {
        "hp": "dflalpha",
        "vals": ["0.001", "0.01", "0.1", "1.0", "10.0"],
        "tag_fn": lambda v: f"alpha{v}",
    },
    "blackbox": {
        "hp": "lambd",
        "vals": ["0.01", "0.05", "0.1", "0.5", "1.0"],
        "tag_fn": lambda v: f"lam{v}",
    },
    "qptl": {
        "hp": "tau",
        "vals": ["0.1", "0.5", "1.0", "5.0", "10.0"],
        "tag_fn": lambda v: f"tau{v}",
    },
    "listLTR": {
        "hp": "tau",
        "vals": ["0.1", "0.5", "1", "5", "10"],
        "tag_fn": lambda v: f"tau{v}",
    },
    "lodl": {
        "hp": "num_samples",
        "vals": ["100", "250", "500", "1000", "2000"],
        "tag_fn": lambda v: f"ns{v}",
    },
    "perturb": {
        "hp": "sigma",
        "vals": ["0.1", "0.5", "1.0", "2.0", "5.0"],
        "tag_fn": lambda v: f"s{v.replace('.','p')}_n10",
    },
    "pg": {
        "hp": "sigma",
        "vals": ["0.01", "0.05", "0.1", "0.5", "1.0"],
        "tag_fn": lambda v: f"s{v.replace('.','p')}",
    },
}


# ============================================================
# Data loading helpers
# ============================================================

def _out_dir(prob, method, prefix):
    parg = PROB_ARG[prob]
    pver = PROB_VERSION[prob]
    return os.path.join(RESULTS_ROOT, f"{parg}-{pver}", method, prefix)


def _read_val_score(d):
    """Return min(eval) from val_logs.csv, or None if unavailable."""
    p = os.path.join(d, "val_logs.csv")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p)
        if "eval" not in df.columns or df.empty:
            return None
        return float(df["eval"].min())
    except Exception:
        return None


def _read_test_regret(prob, method, prefix):
    """Return (abs_regret, rel_regret) from results.npy, or (None, None)."""
    d = _out_dir(prob, method, prefix)
    rpath = os.path.join(d, "results.npy")
    if not os.path.exists(rpath):
        return None, None
    try:
        r = np.load(rpath, allow_pickle=True)
        regret  = float(np.mean(np.array(r[1], dtype=float)))
        opt     = float(np.mean(np.abs(np.array(r[0], dtype=float))))
        rel_r   = regret / opt if opt > 0 else None
        return regret, rel_r
    except Exception:
        return None, None


def _metric(prob, abs_r, rel_r):
    """Return the scalar metric used for this problem (abs for portfolio, rel otherwise)."""
    return abs_r if prob in USE_ABSOLUTE else rel_r


def get_best_result(prob, method, best_json):
    """
    Return the best test regret (abs, rel) for (prob, method) by selecting
    the configuration with the lowest validation regret.

    Considers:
      - All 6 Phase 1 configs (3 LRs × 2 batches)
      - All Phase 2 HP configs (using Phase 1 best LR/batch)

    Val selection: min(eval) in val_logs.csv.
    Fallback: if val_logs unavailable, use test regret for selection.
    """
    allowed = METHOD_PROBLEMS.get(method, None)
    if allowed is not None and prob not in allowed:
        return None, None

    candidates = []  # list of (val_score, abs_r, rel_r)

    # ---- Phase 1: all LR × batch combos ----
    for batch in BATCH_LABELS:
        for lr in LRS:
            prefix = f"bench_p1_{method}_{batch}_lr{lr}"
            d = _out_dir(prob, method, prefix)
            abs_r, rel_r = _read_test_regret(prob, method, prefix)
            if abs_r is None:
                continue
            val_score = _read_val_score(d)
            # Fallback to test metric if no val_logs
            if val_score is None:
                val_score = _metric(prob, abs_r, rel_r) or abs_r
            candidates.append((val_score, abs_r, rel_r))

    # ---- Phase 2: method-specific HP sweep ----
    if method in HP_SWEEPS:
        sweep = HP_SWEEPS[method]
        cfg = best_json.get(method, {}).get(prob, {})
        lr   = cfg.get("lr", "1e-2")
        batch = cfg.get("batch", "default")
        for v in sweep["vals"]:
            tag    = sweep["tag_fn"](v)
            prefix = f"bench_p2_{method}_{tag}_{batch}_lr{lr}"
            d = _out_dir(prob, method, prefix)
            abs_r, rel_r = _read_test_regret(prob, method, prefix)
            if abs_r is None:
                continue
            val_score = _read_val_score(d)
            if val_score is None:
                val_score = _metric(prob, abs_r, rel_r) or abs_r
            candidates.append((val_score, abs_r, rel_r))

    if not candidates:
        return None, None

    best = min(candidates, key=lambda x: x[0])
    return best[1], best[2]  # abs_r, rel_r


# ============================================================
# Build regret matrix
# ============================================================

if not os.path.exists(BEST_JSON_PATH):
    raise FileNotFoundError(f"{BEST_JSON_PATH} not found. Run collect_bench_p1.py first.")

with open(BEST_JSON_PATH) as f:
    best_json = json.load(f)

# regret_vals[method][prob] = displayed regret value (rel*100 or abs)
regret_vals = {}
for method in METHODS:
    regret_vals[method] = {}
    for prob in PROBLEMS:
        abs_r, rel_r = get_best_result(prob, method, best_json)
        if abs_r is None:
            regret_vals[method][prob] = np.nan
        elif prob in USE_ABSOLUTE:
            regret_vals[method][prob] = abs_r
        else:
            regret_vals[method][prob] = rel_r * 100 if rel_r is not None else np.nan

# Inject extra one-off points
for key, ep in EXTRA_POINTS.items():
    regret_vals[key] = {p: ep["values"].get(p, np.nan) for p in PROBLEMS}


# ============================================================
# Marker / color scheme
# ============================================================

MSE_C  = "#1f77b4"   # blue
LTRL_C = "#e67e22"   # orange  (L-LTR)

# Gray shades
G1 = "#bbbbbb"   # light gray
G2 = "#888888"   # medium gray
G3 = "#444444"   # dark gray
G4 = "#222222"   # very dark gray  (Perturb diamond, LODL hexagon need distinct shades)

# Per-method style: (marker, color, size, alpha)
#   Surrogate Gradient → hexagon ("h"):   DFL, Blackbox, Identity, Perturb, SPO+
#   Continuous          → triangle-up ("^"): CPLayer, QPTL
#   LTR                 → square ("s"):   NCE, pt-LTR, pr-LTR; L-LTR is orange square
#   Trained Surrogate   → circle ("o"):   LODL
G5 = "#666666"   # mid-gray for SPO+
G6 = "#999999"   # light-mid gray for PG
METHOD_STYLE = {
    "mse":      ("*",  MSE_C,  80,  1.0),
    # Surrogate Gradient methods (gray hexagons, light → dark)
    "dfl":      ("h",  G1,     50,  0.90),
    "blackbox": ("h",  G2,     50,  0.90),
    "identity": ("h",  G3,     50,  0.90),
    "perturb":  ("h",  G4,     50,  0.90),
    "spo":      ("P",  G5,     50,  0.90),
    "pg":       ("P",  G6,     50,  0.90),
    # Trained Surrogate (gray circle)
    "lodl":     ("o",  G2,     44,  0.85),
    # Continuous methods (gray triangles)
    "cpLayer":  ("^",  G1,     50,  0.90),
    "qptl":     ("^",  G2,     50,  0.90),
    # LTR methods (gray squares, except L-LTR which is orange)
    "nce":      ("s",  G1,     44,  0.85),
    "pointLTR": ("s",  G2,     44,  0.85),
    "pairLTR":  ("s",  G3,     44,  0.85),
    "listLTR":  ("s",  LTRL_C, 50,  0.90),
}

def method_style(m):
    if m in METHOD_STYLE:
        return METHOD_STYLE[m]
    if m in EXTRA_POINTS:
        return EXTRA_POINTS[m]["style"]
    return ("o", G2, 38, 0.7)


# ============================================================
# Plotting helper
# ============================================================

plt.rcParams.update({
    "font.family":         "sans-serif",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.bottom":  False,
})

np.random.seed(42)
jitter_x = {m: np.random.uniform(-0.28, 0.28) for m in METHODS}
jitter_x["mse"]        = -0.15   # slightly left so star doesn't overlap
jitter_x["listLTR"]    =  0.0    # center the highlighted orange square
jitter_x["perturb"]    =  0.0    # center the perturb hexagon
jitter_x["dfl_direct"] =  0.10   # slight right offset for extra point



def _choose_5_ticks(v_min, v_max):
    """Return 5 tick values evenly spaced in log space between v_min and v_max."""
    lo = np.log10(max(v_min, 1e-9))
    hi = np.log10(max(v_max, lo + 1e-6))
    return 10 ** np.linspace(lo, hi, 5)




def _fmt_val(x, is_portfolio):
    """Format a tick value: 3 decimal places for small %, 3 sig figs for large."""
    if is_portfolio:
        return f"{x:.3f}"
    if x < 1.0:
        return f"{x:.3f}%"
    if x < 10.0:
        return f"{x:.2f}%"
    return f"{x:.1f}%"


def style_ax(ax, prob, show_ylabel=False):
    is_portfolio = prob in USE_ABSOLUTE

    # Determine y-limits: relative override or data-driven
    vals = [regret_vals[m][prob] for m in METHODS
            if np.isfinite(regret_vals[m].get(prob, np.nan))]
    if not vals:
        return

    # Pad a fraction of the log-range on each side so best lands near top
    # and worst near bottom. Use a tighter fraction for narrow-range columns.
    log_range = np.log10(max(max(vals), 1e-9)) - np.log10(max(min(vals), 1e-9))
    pad_frac = 0.15
    log_pad = log_range * pad_frac + 0.005
    v_min = max(10 ** (np.log10(max(min(vals), 1e-9)) - log_pad), 1e-4)
    v_max = 10 ** (np.log10(max(max(vals), 1e-9)) + log_pad)

    ticks = _choose_5_ticks(v_min, v_max)

    ax.set_yscale("log")
    ax.set_ylim(v_max, v_min)   # inverted: best at top
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([])

    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks(ticks)

    fmt_fn = lambda x, _: _fmt_val(x, is_portfolio)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_fn))
    ax.tick_params(axis="y", labelsize=6.5, length=0, pad=-3)

    if show_ylabel:
        unit = "Regret (log scale)" if is_portfolio else "Relative regret (log scale)"
        ax.set_ylabel(unit, fontsize=7, fontweight="bold", labelpad=24)

    # 5 horizontal reference lines at the tick positions
    for y in ticks:
        ax.axhline(y, color="#dddddd", linewidth=0.5, linestyle="-", zorder=0)

    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["left"].set_zorder(1)


def draw_bump(task_list, title, fname, figsize=None):
    n_tasks = len(task_list)
    if figsize is None:
        figsize = (max(3.0, 1.4 * n_tasks + 2.4), 3.0)

    def _le(marker, color, label, ms=7):
        return Line2D([0], [0], marker=marker, color="w",
                      markerfacecolor=color, markeredgecolor="white",
                      markersize=ms, markeredgewidth=0.5, label=label)

    def _hdr(text):
        """Invisible-marker header entry for a group label."""
        return Line2D([0], [0], color="none",
                      label=r"$\bf{" + text.replace(" ", r"\ ") + r"}$")

    def _extra_les(group):
        """Legend entries for EXTRA_POINTS belonging to `group` with data in task_list."""
        out = []
        for ep in EXTRA_POINTS.values():
            if ep.get("group") != group:
                continue
            if not any(np.isfinite(ep["values"].get(p, np.nan)) for p in task_list):
                continue
            mkr, col, sz, alp = ep["style"]
            out.append(_le(mkr, col, ep["label"], ms=7))
        return out

    legend_els = [
        # Decision-unaware baseline
        Line2D([0], [0], marker="*", color="w", markerfacecolor=MSE_C,
               markeredgecolor="white", markersize=10, markeredgewidth=0.8,
               label="MSE  (Decision-Unaware)"),
        # Surrogate Gradient group
        _hdr("Surrogate Gradient"),
        _le("h",  G1,     "DFL"),
        _le("h",  G2,     "Blackbox"),
        _le("h",  G3,     "Identity"),
        _le("h",  G4,     "Perturb"),
        _le("P",  G5,     "SPO+"),
        _le("P",  G6,     "PG"),
        *_extra_les("surrogate_gradient"),
        # Trained Surrogate group
        _hdr("Trained Surrogate"),
        _le("o",  G2,     "LODL"),
        # Continuous group
        _hdr("Continuous"),
        _le("^",  G1,     "cpLayer"),
        _le("^",  G2,     "QPTL"),
        # Statistical group
        _hdr("Statistical"),
        _le("s",  G1,     "NCE"),
        _le("s",  G2,     "pt-LTR"),
        _le("s",  G3,     "pr-LTR"),
        _le("s",  LTRL_C, "L-LTR"),
    ]

    single = (n_tasks == 1)
    add_avg_rank = (n_tasks == 5)   # only for the 5-task REST chart

    # GridSpec: task columns + optional avg-rank column
    n_cols = n_tasks + (1 if add_avg_rank else 0)
    width_ratios = [1] * n_tasks + ([1.1] if add_avg_rank else [])

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(1, n_cols, figure=fig,
                   width_ratios=width_ratios, wspace=0.10)

    axes = [fig.add_subplot(gs[0, j]) for j in range(n_tasks)]

    for j, (ax, prob) in enumerate(zip(axes, task_list)):
        style_ax(ax, prob, show_ylabel=(j == 0))

        # Standard methods
        for m in METHODS:
            yval = regret_vals[m].get(prob, np.nan)
            if not np.isfinite(yval):
                continue
            mkr, col, sz, alp = method_style(m)
            xpos = jitter_x[m]
            ax.scatter(xpos, yval, s=sz, color=col, edgecolors="white",
                       linewidths=0.5, zorder=3, marker=mkr, alpha=alp)

        # Extra one-off points
        for key, ep in EXTRA_POINTS.items():
            yval = ep["values"].get(prob, np.nan)
            if not np.isfinite(yval):
                continue
            mkr, col, sz, alp = ep["style"]
            ax.scatter(jitter_x.get(key, 0.0), yval, s=sz, color=col,
                       edgecolors="white", linewidths=0.5, zorder=4,
                       marker=mkr, alpha=alp)

        ax.set_xlabel(PROB_DISPLAY.get(prob, prob), fontsize=6.5,
                      fontweight="bold", labelpad=3)

    # ---- Average Rank column (5-task chart only) ----
    if add_avg_rank:
        ax_avg = fig.add_subplot(gs[0, n_tasks])

        # Compute per-task ranks (1=best, lower regret = better rank)
        # Methods with NaN for a task are excluded from that task's ranking.
        avg_rank = {}
        for m in METHODS:
            ranks = []
            for prob in task_list:
                col_vals = {mm: regret_vals[mm].get(prob, np.nan)
                            for mm in METHODS
                            if np.isfinite(regret_vals[mm].get(prob, np.nan))}
                if m not in col_vals:
                    continue
                sorted_methods = sorted(col_vals, key=col_vals.__getitem__)
                ranks.append(sorted_methods.index(m) + 1)
            avg_rank[m] = float(np.mean(ranks)) if ranks else np.nan

        valid_ranks = [v for v in avg_rank.values() if np.isfinite(v)]
        max_rank = max(valid_ranks) if valid_ranks else len(METHODS)

        ax_avg.set_ylim(max_rank + 0.5, 0.5)   # inverted: rank 1 at top
        ax_avg.set_xlim(-0.45, 0.45)
        ax_avg.set_xticks([])
        # Tick labels only at odd ranks; lines at every rank
        labeled = [r for r in range(1, int(max_rank) + 1) if r % 2 == 1]
        ax_avg.set_yticks(labeled)
        ax_avg.set_yticklabels([str(r) for r in labeled])
        ax_avg.tick_params(axis="y", labelsize=6.5, length=0, pad=2)
        ax_avg.spines["left"].set_linewidth(0.6)
        ax_avg.spines["left"].set_color("#cccccc")
        ax_avg.spines["top"].set_visible(False)
        ax_avg.spines["right"].set_visible(False)
        ax_avg.spines["bottom"].set_visible(False)

        # Horizontal guide lines at every integer rank
        for r in range(1, int(max_rank) + 1):
            ax_avg.axhline(r, color="#dddddd", linewidth=0.5, zorder=0)

        for m in METHODS:
            ar = avg_rank.get(m, np.nan)
            if not np.isfinite(ar):
                continue
            mkr, col, sz, alp = method_style(m)
            xpos = jitter_x[m]
            ax_avg.scatter(xpos, ar, s=sz, color=col, edgecolors="white",
                           linewidths=0.5, zorder=3, marker=mkr, alpha=alp)

        ax_avg.set_xlabel("Avg\nRank", fontsize=6.5, fontweight="bold", labelpad=3)

    if single:
        fig.legend(handles=legend_els, loc="upper center",
                   bbox_to_anchor=(0.5, 0.0), ncol=2,
                   frameon=True, framealpha=0.88, edgecolor="#cccccc",
                   fontsize=7, handletextpad=0.3, borderpad=0.4, labelspacing=0.15)
        plt.subplots_adjust(top=0.85, bottom=0.03, left=0.20, right=0.95)
    else:
        fig.legend(handles=legend_els, loc="upper left",
                   bbox_to_anchor=(0.82, 0.92), ncol=1,
                   frameon=True, framealpha=0.88, edgecolor="#cccccc",
                   fontsize=7, handletextpad=0.3, borderpad=0.4, labelspacing=0.15)
        plt.subplots_adjust(top=0.88, bottom=0.22, left=0.10, right=0.81)

    fig.suptitle(title, fontsize=8, fontweight="bold", y=1.01)

    # Replace default tick labels with manually placed text that has an
    # opaque white background, so the y-axis spine is fully occluded.
    all_axes = axes + ([ax_avg] if add_avg_rank else [])
    for ax in all_axes:
        ax.spines["left"].set_zorder(2)
        # Read tick positions and formatted labels, then hide originals
        fig.canvas.draw()
        tick_positions = ax.get_yticks()
        tick_labels    = [t.get_text() for t in ax.get_yticklabels()]
        ax.set_yticklabels([""] * len(tick_positions))
        # Place new text at each tick, right-aligned, with white bbox
        for yval, txt in zip(tick_positions, tick_labels):
            if not txt:
                continue
            ax.text(0.0, yval, txt, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6.5,
                    bbox=dict(facecolor="white", edgecolor="none",
                              boxstyle="square,pad=0.15"),
                    zorder=10, clip_on=False)

    out_png = os.path.join(OUT_DIR, fname)
    out_pdf = os.path.join(OUT_DIR, fname.replace(".png", ".pdf"))
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}  +  {out_pdf}")


# ============================================================
# Three plots
# ============================================================

REST = ["knapsack", "knapsack-real", "energy", "cubic", "bipartitematching"]

draw_bump(
    REST,
    title="Benchmark re-run — Relative regret % (best ↑)",
    fname="fig_bench_bump_rest.png",
    figsize=(8.5, 3.0),
)

draw_bump(
    ["budgetalloc"],
    title="Benchmark re-run — Budget Allocation\nRelative regret % (best ↑)",
    fname="fig_bench_bump_budgetalloc.png",
    figsize=(2.0, 2.8),
)

draw_bump(
    ["portfolio"],
    title="Benchmark re-run — Portfolio\nAbsolute regret (best ↑)",
    fname="fig_bench_bump_portfolio.png",
    figsize=(2.0, 2.8),
)

# ---- Print summary table ----
print("\nRegret summary (Phase 1/2 best, val-selected):")
print(f"{'method':>10}", end="")
for p in PROBLEMS:
    print(f"  {p[:12]:>12}", end="")
print()
print("-" * (10 + 14 * len(PROBLEMS)))
for m in METHODS:
    print(f"{m:>10}", end="")
    for p in PROBLEMS:
        v = regret_vals[m].get(p, np.nan)
        if np.isfinite(v):
            print(f"  {v:>12.4f}", end="")
        else:
            print(f"  {'—':>12}", end="")
    print()

# ---- Export JSON for JS visualization ----

def _nan_to_none(d):
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in d.items()}

# Compute avg ranks for each chart grouping
def _avg_ranks(task_list, all_methods):
    ranks = {}
    for m in all_methods:
        rs = []
        for prob in task_list:
            col = {mm: regret_vals[mm].get(prob, np.nan) for mm in all_methods
                   if np.isfinite(regret_vals[mm].get(prob, np.nan))}
            if m not in col:
                continue
            srt = sorted(col, key=col.__getitem__)
            rs.append(srt.index(m) + 1)
        ranks[m] = round(float(np.mean(rs)), 4) if rs else None
    return ranks

all_methods_inc_extra = list(METHODS) + list(EXTRA_POINTS.keys())

export = {
    "problems": PROBLEMS,
    "problem_display": PROB_DISPLAY,
    "methods": METHODS,
    "method_display": METHOD_DISPLAY,
    "absolute_tasks": list(USE_ABSOLUTE),
    "regret": {
        p: _nan_to_none(regret_vals.get(m, {}))
        if False else  # per-problem dict of method→value
        {METHOD_DISPLAY.get(m, m): (None if np.isnan(regret_vals[m].get(p, np.nan))
                                    else regret_vals[m][p])
         for m in METHODS}
        for p in PROBLEMS
    },
    "extra_points": {
        k: {
            "label": ep["label"],
            "style": {"marker": ep["style"][0], "color": ep["style"][1],
                      "size": ep["style"][2], "alpha": ep["style"][3]},
            "group": ep.get("group"),
            "values": _nan_to_none(ep["values"]),
        }
        for k, ep in EXTRA_POINTS.items()
    },
    "avg_rank": _avg_ranks(PROBLEMS, METHODS),
    "style": {
        METHOD_DISPLAY.get(m, m): {"marker": s[0], "color": s[1],
                                   "size": s[2], "alpha": s[3]}
        for m, s in METHOD_STYLE.items()
    },
    "method_groups": {
        "Decision-Unaware": ["MSE"],
        "Surrogate Gradient": ["DFL", "Blackbox", "Identity", "Perturb", "SPO+", "PG"],
        "Trained Surrogate": ["LODL"],
        "Continuous": ["cpLayer", "QPTL"],
        "Statistical": ["NCE", "pt-LTR", "pr-LTR", "L-LTR"],
    },
    "chart_groupings": {
        "rest": ["knapsack", "knapsack-real", "energy", "cubic", "bipartitematching"],
        "budgetalloc": ["budgetalloc"],
        "portfolio": ["portfolio"],
    },
}

json_path = os.path.join(OUT_DIR, "bench_bump_data.json")
with open(json_path, "w") as f:
    json.dump(export, f, indent=2)
print(f"\nExported {json_path}")
