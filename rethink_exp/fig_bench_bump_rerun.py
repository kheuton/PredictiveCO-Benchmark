"""
fig_bench_bump_rerun.py
-----------------------
Bump charts for the 2026 benchmark RE-RUN (14 problems × 15 methods).

Differences from fig_bench_bump.py:
  - Adds new problems: asurv, cook_county, speed_humps, sp_synth, sp_planted,
    pg_misspec, shortestpath (warcraft)
  - Adds new methods: pg, dad
  - Adds HP sweep for dad (stein_weight)
  - Groups tasks into multiple figures: classic, spatial/topk, shortest-path

Data loading: directly scans saved_records/ and reads val_logs.csv +
results.npy per run. Uses bench_p1_best.json to locate Phase 2 best
(LR, batch). Does NOT depend on sweep manifests. Partial results are
rendered as NaN (missing marker).
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
BEST_JSON_PATH = "bench_p1_best_val.json"
OUT_DIR        = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Problem config (mirrors collect_bench_p1.py) ----
PROBLEMS = [
    "knapsack", "knapsack-real", "energy", "budgetalloc",
    "cubic", "bipartitematching", "portfolio",
    "asurv", "cook_county", "speed_humps",
    "sp_synth", "sp_planted", "pg_misspec", "shortestpath",
]

PROB_ARG = {
    "knapsack":          "knapsack",
    "knapsack-real":     "knapsack",
    "energy":            "energy",
    "budgetalloc":       "budgetalloc",
    "cubic":             "cubic",
    "bipartitematching": "bipartitematching",
    "portfolio":         "portfolio",
    "asurv":             "asurv",
    "cook_county":       "cook_county",
    "speed_humps":       "speed_humps",
    "sp_synth":          "sp_synth",
    "sp_planted":        "sp_planted",
    "pg_misspec":        "pg_misspec",
    "shortestpath":      "shortestpath",
}

PROB_VERSION = {
    "knapsack":          "gen",
    "knapsack-real":     "energy",
    "energy":            "energy",
    "budgetalloc":       "real",
    "cubic":             "gen",
    "bipartitematching": "cora",
    "portfolio":         "real",
    "asurv":             "real",
    "cook_county":       "real",
    "speed_humps":       "real",
    "sp_synth":          "synth",
    "sp_planted":        "planted",
    "pg_misspec":        "v3",
    "shortestpath":      "warcraft",
}

PROB_DISPLAY = {
    "knapsack":          "Knapsack\n(Gen)",
    "knapsack-real":     "Knapsack\n(Real)",
    "energy":            "Scheduling\n(Energy)",
    "budgetalloc":       "Budget\nAllocation",
    "cubic":             "TopK\n(Cubic)",
    "bipartitematching": "Bipartite\nMatch.",
    "portfolio":         "Portfolio",
    "asurv":             "ASurv\n(TopK)",
    "cook_county":       "Cook Cty.\n(TopK)",
    "speed_humps":       "Speed Humps\n(TopK)",
    "sp_synth":          "SP Synth\n(5×5)",
    "sp_planted":        "SP Planted\n(5×5)",
    "pg_misspec":        "PG Misspec\n(v3)",
    "shortestpath":      "Warcraft\n(12×12)",
}

# ---- Method config ----
METHODS = ["mse", "mse_train", "mse_val",
           "dfl", "identity", "spo", "nce", "blackbox",
           "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg", "dad",
           "qptl", "cpLayer"]

METHOD_DISPLAY = {
    "mse":       "MSE",
    "mse_train": "MSE (train-sel)",
    "mse_val":   "MSE (val-sel)",
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
    "dad":      "DAD",
    "qptl":     "QPTL",
    "cpLayer":  "cpLayer",
}

# PG is valid on all problems except budgetalloc and shortestpath
# (see CLAUDE.md Solvers for comparability).
PG_PROBS = set(PROBLEMS) - {"budgetalloc", "shortestpath"}

METHOD_PROBLEMS = {
    "qptl":    {"knapsack", "bipartitematching", "portfolio"},
    "cpLayer": {"knapsack", "bipartitematching", "portfolio"},
    "pg":      PG_PROBS,
    # dad valid on all 13 problems
}

LRS          = ["1e-2", "5e-3", "1e-3", "5e-2", "1e-1"]
BATCH_LABELS = ["default", "alt"]
USE_ABSOLUTE = {"portfolio"}

# ---- Phase 2 HP sweeps (mirror collect_bench_p2.py / submit_bench_p2.sh) ----
HP_SWEEPS = {
    "dfl":      {"hp": "dflalpha",    "vals": ["0.001", "0.01", "0.1", "1.0", "10.0"],
                 "tag_fn": lambda v: f"alpha{v}"},
    "blackbox": {"hp": "lambd",       "vals": ["0.01", "0.05", "0.1", "0.5", "1.0"],
                 "tag_fn": lambda v: f"lam{v}"},
    "qptl":     {"hp": "tau",         "vals": ["0.1", "0.5", "1.0", "5.0", "10.0"],
                 "tag_fn": lambda v: f"tau{v}"},
    "listLTR":  {"hp": "tau",         "vals": ["0.1", "0.5", "1", "5", "10"],
                 "tag_fn": lambda v: f"tau{v}"},
    "lodl":     {"hp": "num_samples", "vals": ["100", "250", "500", "1000", "2000"],
                 "tag_fn": lambda v: f"ns{v}"},
    "perturb":  {"hp": "sigma",       "vals": ["0.1", "0.5", "1.0", "2.0", "5.0"],
                 "tag_fn": lambda v: f"s{v.replace('.','p')}_n10"},
    "pg":       {"hp": "sigma",       "vals": ["0.01", "0.05", "0.1", "0.5", "1.0"],
                 "tag_fn": lambda v: f"s{v.replace('.','p')}"},
    "dad":      {"hp": "stein_weight","vals": ["0.1", "0.5", "1.0", "2.0", "5.0"],
                 "tag_fn": lambda v: f"sw{v.replace('.','p')}"},
}


# ============================================================
# Data loading
# ============================================================

def _out_dir(prob, method, prefix):
    return os.path.join(RESULTS_ROOT,
                        f"{PROB_ARG[prob]}-{PROB_VERSION[prob]}",
                        method, prefix)


def _read_val_score(d):
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
    d = _out_dir(prob, method, prefix)
    rpath = os.path.join(d, "results.npy")
    if not os.path.exists(rpath):
        return None, None
    try:
        r = np.load(rpath, allow_pickle=True)
        regret = float(np.mean(np.array(r[1], dtype=float)))
        opt    = float(np.mean(np.abs(np.array(r[0], dtype=float))))
        rel_r  = regret / opt if opt > 0 else None
        return regret, rel_r
    except Exception:
        return None, None


def _metric(prob, abs_r, rel_r):
    return abs_r if prob in USE_ABSOLUTE else rel_r


def get_best_result(prob, method, best_json):
    allowed = METHOD_PROBLEMS.get(method, None)
    if allowed is not None and prob not in allowed:
        return None, None

    # mse_train / mse_val use a non-val-regret selection criterion (train MSE
    # and val MSE respectively). Their winners in best_json were picked by
    # collect_bench_p1.py using the correct criterion — do NOT re-select by
    # val regret here. Use only that single run's test regret.
    if method in ("mse_train", "mse_val"):
        cfg = best_json.get(method, {}).get(prob)
        if cfg is None:
            return None, None
        prefix = f"bench_p1_{method}_{cfg['batch']}_lr{cfg['lr']}"
        abs_r, rel_r = _read_test_regret(prob, method, prefix)
        return abs_r, rel_r

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
            if val_score is None:
                val_score = _metric(prob, abs_r, rel_r) or abs_r
            candidates.append((val_score, abs_r, rel_r))

    # ---- Phase 2: method-specific HP sweep ----
    if method in HP_SWEEPS:
        sweep = HP_SWEEPS[method]
        cfg = best_json.get(method, {}).get(prob, {})
        lr    = cfg.get("lr", "1e-2")
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
    return best[1], best[2]


# ============================================================
# Build regret matrix
# ============================================================

if not os.path.exists(BEST_JSON_PATH):
    raise FileNotFoundError(f"{BEST_JSON_PATH} not found. Run collect_bench_p1.py first.")

with open(BEST_JSON_PATH) as f:
    best_json = json.load(f)

regret_vals = {}
n_missing_by_method = {}
for method in METHODS:
    regret_vals[method] = {}
    miss = 0
    for prob in PROBLEMS:
        abs_r, rel_r = get_best_result(prob, method, best_json)
        if abs_r is None:
            regret_vals[method][prob] = np.nan
            miss += 1
        elif prob in USE_ABSOLUTE:
            regret_vals[method][prob] = abs_r
        else:
            regret_vals[method][prob] = rel_r * 100 if rel_r is not None else np.nan
    n_missing_by_method[method] = miss


# ============================================================
# Marker / color scheme
# ============================================================

MSE_C  = "#1f77b4"
MSE_TRAIN_C = "#7fb3d5"  # lighter blue: MSE selected on train MSE
MSE_VAL_C   = "#0e3a5e"  # darker blue:  MSE selected on val MSE
LTRL_C = "#e67e22"

G1 = "#bbbbbb"; G2 = "#888888"; G3 = "#444444"; G4 = "#222222"
G5 = "#666666"; G6 = "#999999"
DAD_C = "#8e44ad"   # purple for DAD (new method, visually distinct)

METHOD_STYLE = {
    "mse":       ("*",  MSE_C,       80,  1.0),
    "mse_train": ("*",  MSE_TRAIN_C, 80,  1.0),
    "mse_val":   ("*",  MSE_VAL_C,   80,  1.0),
    # Surrogate Gradient (hexagons)
    "dfl":      ("h",  G1,     50,  0.90),
    "blackbox": ("h",  G2,     50,  0.90),
    "identity": ("h",  G3,     50,  0.90),
    "perturb":  ("h",  G4,     50,  0.90),
    # Finite-difference / gradient estimator (plus marker)
    "spo":      ("P",  G5,     50,  0.90),
    "pg":       ("P",  G6,     50,  0.90),
    # Trained surrogate (circle)
    "lodl":     ("o",  G2,     44,  0.85),
    # Decision-aware denoising (diamond) — new method
    "dad":      ("D",  DAD_C,  42,  0.90),
    # Continuous (triangles)
    "cpLayer":  ("^",  G1,     50,  0.90),
    "qptl":     ("^",  G2,     50,  0.90),
    # Statistical / LTR (squares)
    "nce":      ("s",  G1,     44,  0.85),
    "pointLTR": ("s",  G2,     44,  0.85),
    "pairLTR":  ("s",  G3,     44,  0.85),
    "listLTR":  ("s",  LTRL_C, 50,  0.90),
}

def method_style(m):
    return METHOD_STYLE.get(m, ("o", G2, 38, 0.7))


# ============================================================
# Plotting
# ============================================================

plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.bottom": False,
})

np.random.seed(42)
jitter_x = {m: np.random.uniform(-0.28, 0.28) for m in METHODS}
jitter_x["mse"]       = -0.30
jitter_x["mse_train"] = -0.15
jitter_x["mse_val"]   = -0.22
jitter_x["listLTR"] =  0.0
jitter_x["perturb"] =  0.0
jitter_x["dad"]     =  0.18


def _choose_5_ticks(v_min, v_max):
    lo = np.log10(max(v_min, 1e-9))
    hi = np.log10(max(v_max, lo + 1e-6))
    return 10 ** np.linspace(lo, hi, 5)


def _fmt_val(x, is_portfolio):
    if is_portfolio:
        return f"{x:.3f}"
    if x < 1.0:
        return f"{x:.3f}%"
    if x < 10.0:
        return f"{x:.2f}%"
    return f"{x:.1f}%"


def style_ax(ax, prob, show_ylabel=False):
    is_portfolio = prob in USE_ABSOLUTE

    vals = [regret_vals[m][prob] for m in METHODS
            if np.isfinite(regret_vals[m].get(prob, np.nan))]
    if not vals:
        return

    log_range = np.log10(max(max(vals), 1e-9)) - np.log10(max(min(vals), 1e-9))
    pad_frac = 0.15
    log_pad = log_range * pad_frac + 0.005
    v_min = max(10 ** (np.log10(max(min(vals), 1e-9)) - log_pad), 1e-4)
    v_max = 10 ** (np.log10(max(max(vals), 1e-9)) + log_pad)

    ticks = _choose_5_ticks(v_min, v_max)
    ax.set_yscale("log")
    ax.set_ylim(v_max, v_min)
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

    for y in ticks:
        ax.axhline(y, color="#dddddd", linewidth=0.5, zorder=0)

    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["left"].set_zorder(1)


def _legend_elements(exclude_methods=()):
    def _le(mkr, col, lbl, ms=7):
        return Line2D([0], [0], marker=mkr, color="w",
                      markerfacecolor=col, markeredgecolor="white",
                      markersize=ms, markeredgewidth=0.5, label=lbl)
    def _hdr(t):
        return Line2D([0], [0], color="none",
                      label=r"$\bf{" + t.replace(" ", r"\ ") + r"}$")
    def _star(col, lbl):
        return Line2D([0], [0], marker="*", color="w", markerfacecolor=col,
                      markeredgecolor="white", markersize=10, markeredgewidth=0.8,
                      label=lbl)
    elements = [
        _hdr("Decision-Unaware (MSE)"),
        _star(MSE_C,       "MSE  (val regret sel.)"),
        _star(MSE_VAL_C,   "MSE  (val MSE sel.)"),
        _star(MSE_TRAIN_C, "MSE  (train MSE sel.)"),
        _hdr("Surrogate Gradient"),
        _le("h", G1, "DFL"), _le("h", G2, "Blackbox"),
        _le("h", G3, "Identity"), _le("h", G4, "Perturb"),
        _le("P", G5, "SPO+"), _le("P", G6, "PG"),
        _hdr("Trained Surrogate"),
        _le("o", G2, "LODL"),
    ]
    if "dad" not in exclude_methods:
        elements += [
            _hdr("Decision-Aware Denoising"),
            _le("D", DAD_C, "DAD"),
        ]
    cont_pairs = [("cpLayer", _le("^", G1, "cpLayer")),
                  ("qptl",    _le("^", G2, "QPTL"))]
    cont_kept = [le for (m, le) in cont_pairs if m not in exclude_methods]
    if cont_kept:
        elements.append(_hdr("Continuous"))
        elements.extend(cont_kept)
    elements += [
        _hdr("Statistical"),
        _le("s", G1, "NCE"),    _le("s", G2, "pt-LTR"),
        _le("s", G3, "pr-LTR"), _le("s", LTRL_C, "L-LTR"),
    ]
    return elements


def draw_bump(task_list, title, fname, figsize=None, add_avg_rank=False,
              exclude_methods=()):
    n_tasks = len(task_list)
    if figsize is None:
        figsize = (max(3.0, 1.4 * n_tasks + 2.6), 3.2)

    single = (n_tasks == 1)
    n_cols = n_tasks + (1 if add_avg_rank else 0)
    width_ratios = [1] * n_tasks + ([1.1] if add_avg_rank else [])

    methods = [m for m in METHODS if m not in exclude_methods]

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(1, n_cols, figure=fig,
                   width_ratios=width_ratios, wspace=0.10)
    axes = [fig.add_subplot(gs[0, j]) for j in range(n_tasks)]

    for j, (ax, prob) in enumerate(zip(axes, task_list)):
        style_ax(ax, prob, show_ylabel=(j == 0))
        for m in methods:
            y = regret_vals[m].get(prob, np.nan)
            if not np.isfinite(y):
                continue
            mkr, col, sz, alp = method_style(m)
            ax.scatter(jitter_x[m], y, s=sz, color=col,
                       edgecolors="white", linewidths=0.5,
                       zorder=3, marker=mkr, alpha=alp)
        ax.set_xlabel(PROB_DISPLAY.get(prob, prob), fontsize=6.5,
                      fontweight="bold", labelpad=3)

    if add_avg_rank:
        ax_avg = fig.add_subplot(gs[0, n_tasks])
        avg_rank = {}
        for m in methods:
            ranks = []
            for prob in task_list:
                col_vals = {mm: regret_vals[mm].get(prob, np.nan)
                            for mm in methods
                            if np.isfinite(regret_vals[mm].get(prob, np.nan))}
                if m not in col_vals:
                    continue
                srt = sorted(col_vals, key=col_vals.__getitem__)
                ranks.append(srt.index(m) + 1)
            avg_rank[m] = float(np.mean(ranks)) if ranks else np.nan
        valid = [v for v in avg_rank.values() if np.isfinite(v)]
        max_rank = max(valid) if valid else len(methods)

        ax_avg.set_ylim(max_rank + 0.5, 0.5)
        ax_avg.set_xlim(-0.45, 0.45)
        ax_avg.set_xticks([])
        labeled = [r for r in range(1, int(max_rank) + 1) if r % 2 == 1]
        ax_avg.set_yticks(labeled)
        ax_avg.set_yticklabels([str(r) for r in labeled])
        ax_avg.tick_params(axis="y", labelsize=6.5, length=0, pad=2)
        ax_avg.spines["left"].set_linewidth(0.6)
        ax_avg.spines["left"].set_color("#cccccc")
        for sp in ("top", "right", "bottom"):
            ax_avg.spines[sp].set_visible(False)
        for r in range(1, int(max_rank) + 1):
            ax_avg.axhline(r, color="#dddddd", linewidth=0.5, zorder=0)
        for m in methods:
            ar = avg_rank.get(m, np.nan)
            if not np.isfinite(ar):
                continue
            mkr, col, sz, alp = method_style(m)
            ax_avg.scatter(jitter_x[m], ar, s=sz, color=col,
                           edgecolors="white", linewidths=0.5,
                           zorder=3, marker=mkr, alpha=alp)
        ax_avg.set_xlabel("Avg\nRank", fontsize=6.5, fontweight="bold", labelpad=3)

    legend_els = _legend_elements(exclude_methods=exclude_methods)
    if single:
        fig.legend(handles=legend_els, loc="upper center",
                   bbox_to_anchor=(0.5, 0.0), ncol=2,
                   frameon=True, framealpha=0.88, edgecolor="#cccccc",
                   fontsize=7, handletextpad=0.3, borderpad=0.4, labelspacing=0.15)
        plt.subplots_adjust(top=0.85, bottom=0.03, left=0.20, right=0.95)
    else:
        fig.legend(handles=legend_els, loc="upper left",
                   bbox_to_anchor=(0.82, 0.94), ncol=1,
                   frameon=True, framealpha=0.88, edgecolor="#cccccc",
                   fontsize=7, handletextpad=0.3, borderpad=0.4, labelspacing=0.15)
        plt.subplots_adjust(top=0.88, bottom=0.22, left=0.08, right=0.81)

    fig.suptitle(title, fontsize=8, fontweight="bold", y=1.01)

    all_axes = axes + ([ax_avg] if add_avg_rank else [])
    for ax in all_axes:
        ax.spines["left"].set_zorder(2)
        fig.canvas.draw()
        tp = ax.get_yticks()
        tl = [t.get_text() for t in ax.get_yticklabels()]
        ax.set_yticklabels([""] * len(tp))
        for yv, txt in zip(tp, tl):
            if not txt:
                continue
            ax.text(0.0, yv, txt, transform=ax.get_yaxis_transform(),
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
# Chart groupings
# ============================================================

# Classic benchmark tasks (same as original fig, minus portfolio/budgetalloc)
CLASSIC = ["knapsack", "knapsack-real", "energy", "cubic", "bipartitematching"]

# New spatial / TopK real-world tasks (added Apr 2026)
SPATIAL = ["asurv", "cook_county", "speed_humps"]

# Shortest-path family
SP_FAMILY = ["sp_synth", "sp_planted", "shortestpath"]

draw_bump(
    CLASSIC,
    title="Benchmark re-run, Tasks with Limited Decision-Aware benefit, Relative regret % (best ↑)",
    fname="fig_bench_bump_rerun_classic.png",
    figsize=(8.5, 3.2),
    add_avg_rank=True,
    exclude_methods=["dad", "cpLayer", "qptl"],
)

draw_bump(
    SPATIAL,
    title="Benchmark re-run — Spatial TopK tasks — Relative regret % (best ↑)",
    fname="fig_bench_bump_rerun_spatial.png",
    figsize=(6.0, 3.2),
    add_avg_rank=True,
)

draw_bump(
    SP_FAMILY,
    title="Benchmark re-run — Shortest-path tasks — Relative regret % (best ↑)",
    fname="fig_bench_bump_rerun_sp.png",
    figsize=(6.0, 3.2),
    add_avg_rank=True,
)

draw_bump(
    PROBLEMS,
    title=f"Benchmark re-run — All {len(PROBLEMS)} tasks — Relative regret % (best ↑)",
    fname="fig_bench_bump_rerun_all.png",
    figsize=(15.5, 3.4),
    add_avg_rank=True,
)

draw_bump(
    ["budgetalloc"],
    title="Benchmark re-run — Budget Allocation\nRelative regret % (best ↑)",
    fname="fig_bench_bump_rerun_budgetalloc.png",
    figsize=(2.2, 2.8),
    exclude_methods=["dad", "cpLayer", "qptl"],
)

draw_bump(
    ["portfolio"],
    title="Benchmark re-run — Portfolio\nAbsolute regret (best ↑)",
    fname="fig_bench_bump_rerun_portfolio.png",
    figsize=(2.2, 2.8),
    exclude_methods=["dad", "cpLayer", "qptl"],
)

draw_bump(
    ["pg_misspec"],
    title="Benchmark re-run — PG Misspec (v3)\nRelative regret % (best ↑)",
    fname="fig_bench_bump_rerun_pg_misspec.png",
    figsize=(2.2, 2.8),
    exclude_methods=["cpLayer", "qptl"],
)


# ============================================================
# Console summary
# ============================================================

print("\nRegret summary (Phase 1/2 best, val-selected):")
hdr = f"{'method':>10}"
for p in PROBLEMS:
    hdr += f"  {p[:12]:>12}"
print(hdr)
print("-" * (10 + 14 * len(PROBLEMS)))
for m in METHODS:
    row = f"{m:>10}"
    for p in PROBLEMS:
        v = regret_vals[m].get(p, np.nan)
        row += f"  {v:>12.4f}" if np.isfinite(v) else f"  {'—':>12}"
    print(row)

print(f"\nMissing cells per method (out of {len(PROBLEMS)} problems):")
for m in METHODS:
    miss = n_missing_by_method[m]
    allowed = METHOD_PROBLEMS.get(m, set(PROBLEMS))
    n_valid = len(allowed) if m in METHOD_PROBLEMS else len(PROBLEMS)
    n_invalid = len(PROBLEMS) - n_valid
    actual_miss = miss - n_invalid
    print(f"  {m:>10}: missing={actual_miss}/{n_valid}  (N/A cells: {n_invalid})")


# ============================================================
# Export JSON (for any JS viz)
# ============================================================

def _nan_to_none(d):
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in d.items()}

export = {
    "problems": PROBLEMS,
    "problem_display": PROB_DISPLAY,
    "methods": METHODS,
    "method_display": METHOD_DISPLAY,
    "absolute_tasks": list(USE_ABSOLUTE),
    "regret": {
        p: {METHOD_DISPLAY.get(m, m): (None if np.isnan(regret_vals[m].get(p, np.nan))
                                       else regret_vals[m][p])
            for m in METHODS}
        for p in PROBLEMS
    },
    "chart_groupings": {
        "classic":      CLASSIC,
        "spatial":      SPATIAL,
        "sp_family":    SP_FAMILY,
        "budgetalloc":  ["budgetalloc"],
        "portfolio":    ["portfolio"],
    },
    "style": {
        METHOD_DISPLAY.get(m, m): {"marker": s[0], "color": s[1],
                                   "size": s[2], "alpha": s[3]}
        for m, s in METHOD_STYLE.items()
    },
}

json_path = os.path.join(OUT_DIR, "bench_bump_rerun_data.json")
with open(json_path, "w") as f:
    json.dump(export, f, indent=2)
print(f"\nExported {json_path}")
