"""
fig_specification_contrast.py
-----------------------------
Well-specified vs mis-specified spotlight (Committee Plan #5).

`sp_synth` and `sp_planted` are the designed-in well/mis contrast from the
SPO+ and PG papers: same 5x5 grid shortest-path problem with the same linear
prediction head (`--n_layers 1`), but the true c(x) is polynomial degree 6 in
`sp_synth` (mis-spec) vs. linear in `sp_planted` (well-spec).

Theoretical prediction (Elmachtoub et al. 2025, arxiv:2011.03030):
  - Well-spec  -> decision-blind training (MSE) should be competitive.
  - Mis-spec   -> decision-aware methods should win.

Outputs:
  - results/fig_specification_contrast.{png,pdf}  (scatter + table)
  - docs/tables/specification_contrast.md
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ----- config -----
LOSS_MATRIX = "loss_matrix.json"
OUT_FIG_DIR = "results"
OUT_TAB     = "docs/tables/specification_contrast.md"

METHODS = ["mse", "mse_train", "mse_val",
           "dfl", "identity", "spo", "nce", "blackbox",
           "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg",
           "qptl", "cpLayer", "dad"]

METHOD_DISPLAY = {
    "mse": "MSE", "mse_train": "MSE (train-sel)", "mse_val": "MSE (val-sel)",
    "dfl": "DFL", "identity": "Identity", "spo": "SPO+",
    "nce": "NCE", "blackbox": "Blackbox",
    "pointLTR": "pt-LTR", "pairLTR": "pr-LTR", "listLTR": "L-LTR",
    "lodl": "LODL", "perturb": "Perturb", "pg": "PG",
    "qptl": "QPTL", "cpLayer": "cpLayer", "dad": "DAD",
}

# Match family grouping from fig_bench_bump_rerun.py
FAMILY = {
    "mse":       ("Decision-blind",    "*", "#1f77b4", 180),
    "mse_train": ("Decision-blind",    "*", "#7fb3d5", 180),
    "mse_val":   ("Decision-blind",    "*", "#0e3a5e", 180),
    # Surrogate gradient
    "dfl":      ("Surrogate gradient", "h", "#bbbbbb", 90),
    "blackbox": ("Surrogate gradient", "h", "#888888", 90),
    "identity": ("Surrogate gradient", "h", "#444444", 90),
    "perturb":  ("Surrogate gradient", "h", "#222222", 90),
    # Finite-difference / estimator
    "spo":      ("FD / estimator",     "P", "#666666", 90),
    "pg":       ("FD / estimator",     "P", "#999999", 90),
    # Trained surrogate
    "lodl":     ("Trained surrogate",  "o", "#888888", 80),
    # Continuous
    "cpLayer":  ("Continuous",         "^", "#bbbbbb", 90),
    "qptl":     ("Continuous",         "^", "#888888", 90),
    # Statistical / LTR
    "nce":      ("Statistical / LTR",  "s", "#bbbbbb", 80),
    "pointLTR": ("Statistical / LTR",  "s", "#888888", 80),
    "pairLTR":  ("Statistical / LTR",  "s", "#444444", 80),
    "listLTR":  ("Statistical / LTR",  "s", "#e67e22", 95),
    # Denoising
    "dad":      ("Denoising",          "D", "#8e44ad", 80),
}


def to_float(v):
    if v is None:
        return np.nan
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return np.nan
    return float(v)


def load_pair(data):
    x, y = {}, {}
    for m in METHODS:
        e_s = data.get("sp_synth",   {}).get(m)
        e_p = data.get("sp_planted", {}).get(m)
        x[m] = to_float(e_s["metrics"].get("test_regret")) if e_s else np.nan
        y[m] = to_float(e_p["metrics"].get("test_regret")) if e_p else np.nan
    return x, y


# Manual label offsets (in points) to reduce overlap in the crowded lower-left
LABEL_OFFSET = {
    "mse":      (6, -10),
    "perturb":  (6, 6),
    "pg":       (6, -4),
    "pointLTR": (6, 4),
    "pairLTR":  (-8, -12),
    "listLTR":  (6, -8),
    "spo":      (6, 4),
    "nce":      (6, 4),
    "dfl":      (6, 4),
    "blackbox": (-8, -12),
    "identity": (6, -10),
    "lodl":     (6, 4),
    "dad":      (6, 6),
    "qptl":     (6, 4),
    "cpLayer":  (6, -10),
}


def _plot_points(ax, x, y, lo, hi, annotate=True, methods=None):
    methods = methods or METHODS
    # y=x reference
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=0.8, alpha=0.6, zorder=0)
    # MSE reference lines
    mse_x, mse_y = x["mse"], y["mse"]
    if np.isfinite(mse_x):
        ax.axvline(mse_x, color="#1f77b4", lw=0.6, alpha=0.35, zorder=0)
    if np.isfinite(mse_y):
        ax.axhline(mse_y, color="#1f77b4", lw=0.6, alpha=0.35, zorder=0)
    for m in methods:
        xm, ym = x[m], y[m]
        if not (np.isfinite(xm) and np.isfinite(ym)):
            continue
        fam, marker, color, size = FAMILY[m]
        ax.scatter(xm, ym, marker=marker, s=size, c=color,
                   edgecolors="black", linewidths=0.6, zorder=3)
        if annotate:
            dx, dy = LABEL_OFFSET.get(m, (5, 3))
            ax.annotate(METHOD_DISPLAY[m], (xm, ym),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=8.5, color="black")


def make_figure(x, y, out_png, out_pdf):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.25})
    ax_full, ax_zoom = axes

    # ---- Left: full range ----
    finite = [(x[m], y[m]) for m in METHODS
              if np.isfinite(x[m]) and np.isfinite(y[m])]
    if not finite:
        raise RuntimeError("No finite (sp_synth, sp_planted) pair found.")
    xs_all, ys_all = zip(*finite)
    lo = min(min(xs_all), min(ys_all)) * 0.9
    hi = max(max(xs_all), max(ys_all)) * 1.05

    _plot_points(ax_full, x, y, lo, hi, annotate=True)
    ax_full.set_xlim(lo, hi)
    ax_full.set_ylim(lo, hi)
    ax_full.set_aspect("equal")
    ax_full.grid(True, alpha=0.25, linestyle=":")
    ax_full.set_xlabel("sp_synth test regret  (mis-spec: deg-6 poly DGP)",
                       fontsize=10)
    ax_full.set_ylabel("sp_planted test regret  (well-spec: linear DGP)",
                       fontsize=10)
    ax_full.set_title("(a) All methods", fontsize=11, fontweight="bold")
    ax_full.text(0.02, 0.97, "lower-left = better on both",
                 transform=ax_full.transAxes, fontsize=8, color="gray",
                 ha="left", va="top")
    ax_full.text(0.98, 0.97, "y = x (no spec effect)",
                 transform=ax_full.transAxes, fontsize=8, color="gray",
                 ha="right", va="top")

    # Box highlighting the zoom region
    zoom_lo, zoom_hi = 0.04, 0.12
    ax_full.plot([zoom_lo, zoom_hi, zoom_hi, zoom_lo, zoom_lo],
                 [zoom_lo, zoom_lo, zoom_hi, zoom_hi, zoom_lo],
                 color="#c0392b", lw=1.0, alpha=0.8, zorder=4)

    # ---- Right: zoom on lower-left cluster ----
    cluster_methods = [m for m in METHODS
                       if np.isfinite(x[m]) and np.isfinite(y[m])
                       and x[m] <= zoom_hi and y[m] <= zoom_hi]
    _plot_points(ax_zoom, x, y, zoom_lo, zoom_hi,
                 annotate=True, methods=cluster_methods)
    ax_zoom.set_xlim(zoom_lo, zoom_hi)
    ax_zoom.set_ylim(zoom_lo, zoom_hi)
    ax_zoom.set_aspect("equal")
    ax_zoom.grid(True, alpha=0.25, linestyle=":")
    ax_zoom.set_xlabel("sp_synth test regret", fontsize=10)
    ax_zoom.set_ylabel("sp_planted test regret", fontsize=10)
    ax_zoom.set_title("(b) Lower-left cluster (competitive methods)",
                      fontsize=11, fontweight="bold")
    # Same red border to indicate zoom source
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor("#c0392b")
        spine.set_linewidth(1.2)

    # Family legend (once, on the zoom panel so it doesn't crowd the full plot)
    seen = []
    handles = []
    for m in METHODS:
        fam, marker, color, size = FAMILY[m]
        if fam in seen:
            continue
        seen.append(fam)
        handles.append(Line2D([0], [0], marker=marker, color="w",
                              markerfacecolor=color, markeredgecolor="black",
                              markersize=np.sqrt(size), label=fam))
    ax_zoom.legend(handles=handles, loc="lower right", fontsize=8,
                   framealpha=0.95, title="Family", title_fontsize=8)

    fig.suptitle("Specification contrast on 5×5 DAG shortest-path "
                 "(same 1-layer prediction head)",
                 fontsize=12, fontweight="bold", y=1.00)

    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_table(x, y, out_path):
    lines = ["# Specification contrast — `sp_synth` (mis-spec) vs `sp_planted` (well-spec)",
             "",
             "Same 5×5 DAG shortest-path, same 1-layer linear prediction head. "
             "`sp_synth` has polynomial-degree-6 DGP (mis-specified), "
             "`sp_planted` has linear DGP (well-specified). "
             "All values: relative test decision regret at best Phase-1/2 config "
             "(best-val-regret checkpoint).",
             "",
             "| Method | sp_synth (mis-spec) | sp_planted (well-spec) | Δ (synth − planted) |",
             "|---|---:|---:|---:|"]

    def fmt(v):
        if not np.isfinite(v):
            return "—"
        return f"{v:.4f}"

    # Sort by sp_synth regret ascending, with NaNs last
    order = sorted(METHODS,
                   key=lambda m: (not np.isfinite(x[m]), x[m] if np.isfinite(x[m]) else np.inf))
    for m in order:
        xm, ym = x[m], y[m]
        delta = (xm - ym) if (np.isfinite(xm) and np.isfinite(ym)) else np.nan
        lines.append(f"| {METHOD_DISPLAY[m]} | {fmt(xm)} | {fmt(ym)} | {fmt(delta)} |")

    lines += [
        "",
        "**Reading this table.**",
        "- Δ > 0 → method does *worse* on mis-spec than on well-spec.",
        "- Δ < 0 → method does *better* on mis-spec — surprising and diagnostic.",
        "- Expected theoretical pattern (Elmachtoub et al. 2025): MSE should have "
        "Δ > 0 (well-spec easier for prediction-only), decision-aware methods "
        "should have smaller or negative Δ.",
        "",
        "*Caveat:* Configs are test-selected (Phase-1 leakage, see "
        "`memory/phase1_hp_test_leakage.md`). Plan #7 fix pending.",
        "",
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    with open(LOSS_MATRIX) as f:
        data = json.load(f)
    x, y = load_pair(data)

    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    make_figure(x, y,
                os.path.join(OUT_FIG_DIR, "fig_specification_contrast.png"),
                os.path.join(OUT_FIG_DIR, "fig_specification_contrast.pdf"))
    make_table(x, y, OUT_TAB)

    print("Wrote:")
    print(f"  {OUT_FIG_DIR}/fig_specification_contrast.png")
    print(f"  {OUT_FIG_DIR}/fig_specification_contrast.pdf")
    print(f"  {OUT_TAB}")

    # Quick summary
    print("\nMSE delta:", f"{x['mse'] - y['mse']:+.4f}")
    print("Decision-aware best (sp_synth):",
          min((m for m in METHODS if np.isfinite(x[m]) and m != 'mse'),
              key=lambda m: x[m]))
    print("Decision-aware best (sp_planted):",
          min((m for m in METHODS if np.isfinite(y[m]) and m != 'mse'),
              key=lambda m: y[m]))


if __name__ == "__main__":
    main()
