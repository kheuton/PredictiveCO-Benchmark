"""
fig_loss_matrix.py
------------------
Per-problem heatmap of the (method × six-metric) loss matrix.

For each of the 13 benchmark problems we render one figure with:
  - Rows: the 15 methods.
  - Columns: six metrics — pred loss {train, val, test} and decision regret
    {train, val, test}.
  - Colour: log-scale value (per-problem colormap; pred and regret on separate
    colormaps so their scales don't wash each other out).

Inputs:
  - `loss_matrix.json` written by `rethink_exp/collect_loss_matrix.py`.

Outputs:
  - `results/fig_loss_matrix/<problem>.png` / `.pdf`
  - `results/fig_loss_matrix_all.png` — 13-panel grid (one per problem)

Usage:
    python rethink_exp/fig_loss_matrix.py
    python rethink_exp/fig_loss_matrix.py --problem budgetalloc
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio", "asurv", "cook_county",
            "speed_humps", "sp_synth", "sp_planted", "shortestpath"]

ALL_METHODS = ["mse", "mse_train", "mse_val", "dfl", "identity", "spo", "nce", "blackbox",
               "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg",
               "qptl", "cpLayer", "dad"]

PRED_COLS = [("train_pred_loss", "Train\npred"),
             ("val_pred_loss",   "Val\npred"),
             ("test_pred_loss",  "Test\npred")]
REG_COLS  = [("train_regret", "Train\nregret"),
             ("val_regret",   "Val\nregret"),
             ("test_regret",  "Test\nregret")]

OUTPUT_DIR = "results/fig_loss_matrix"


def to_float(v):
    if v is None:
        return np.nan
    if isinstance(v, str):
        if v == "inf":
            return np.inf
        try:
            return float(v)
        except ValueError:
            return np.nan
    return float(v)


def fmt_cell(v):
    if not np.isfinite(v):
        if np.isnan(v):
            return ""
        return "∞"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    if abs(v) >= 0.001:
        return f"{v:.3f}"
    return f"{v:.1e}"


def build_matrix(prob_data):
    """Return (pred_mat, reg_mat) shape (n_methods, 3) each."""
    pred = np.full((len(ALL_METHODS), 3), np.nan)
    reg  = np.full((len(ALL_METHODS), 3), np.nan)
    for i, method in enumerate(ALL_METHODS):
        entry = prob_data.get(method)
        if entry is None:
            continue
        m = entry["metrics"]
        for j, (k, _) in enumerate(PRED_COLS):
            pred[i, j] = to_float(m.get(k))
        for j, (k, _) in enumerate(REG_COLS):
            reg[i, j] = to_float(m.get(k))
    return pred, reg


def _safe_log_norm(mat):
    vals = mat[np.isfinite(mat) & (mat > 0)]
    if vals.size == 0:
        return LogNorm(vmin=1e-6, vmax=1.0)
    vmin = max(vals.min(), 1e-8)
    vmax = vals.max()
    if vmax <= vmin:
        vmax = vmin * 10
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_one_problem(prob, prob_data, ax=None, annotate=True):
    pred, reg = build_matrix(prob_data)
    full = np.concatenate([pred, reg], axis=1)   # (M, 6)
    col_labels = [lbl for _, lbl in PRED_COLS] + [lbl for _, lbl in REG_COLS]

    standalone = ax is None
    if standalone:
        # 15 methods x 6 cols, 0.32" per row, 0.7" per col, plus labels margin
        fig, ax = plt.subplots(figsize=(5.6, 6.4))
    else:
        fig = ax.figure

    # Use two separate colour scales — plot pred and regret as stacked imshows.
    pred_norm = _safe_log_norm(pred)
    reg_norm  = _safe_log_norm(reg)

    # Composite display: map pred and regret to colors separately, stitch
    cmap_pred = plt.get_cmap("Blues")
    cmap_reg  = plt.get_cmap("Reds")

    def _rgba(mat, norm, cmap):
        # replace inf with a finite sentinel > vmax so it saturates dark
        disp = np.where(np.isinf(mat), norm.vmax * 10, mat)
        disp = np.where(np.isnan(disp), norm.vmin / 10, disp)
        return cmap(norm(disp))

    img_left  = _rgba(pred, pred_norm, cmap_pred)
    img_right = _rgba(reg,  reg_norm,  cmap_reg)
    # grey out NaN cells
    nan_left  = np.isnan(pred)
    nan_right = np.isnan(reg)
    grey = np.array([0.92, 0.92, 0.92, 1.0])
    img_left[nan_left] = grey
    img_right[nan_right] = grey

    composite = np.concatenate([img_left, img_right], axis=1)
    ax.imshow(composite, aspect="auto", interpolation="nearest")
    # force cells wide enough to fit annotation text without overlap
    ax.set_ylim(len(ALL_METHODS) - 0.5, -0.5)

    # separator between pred and regret columns
    ax.axvline(2.5, color="k", lw=1.0)

    if annotate:
        for i in range(len(ALL_METHODS)):
            for j in range(6):
                val = full[i, j]
                if np.isnan(val):
                    continue
                txt = fmt_cell(val)
                # pick text color by background darkness
                rgb = composite[i, j, :3]
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                col = "white" if lum < 0.45 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=col)

    ax.set_xticks(range(6))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(ALL_METHODS)))
    ax.set_yticklabels(ALL_METHODS, fontsize=9)
    # column group labels
    ax.text(1, -0.14, "Prediction loss", ha="center", va="top",
            fontsize=10, fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.text(4, -0.14, "Decision regret", ha="center", va="top",
            fontsize=10, fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.set_title(prob, fontsize=11, fontweight="bold")

    if standalone:
        fig.subplots_adjust(left=0.18, right=0.98, top=0.93, bottom=0.18)
        return fig
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss_matrix_json", default="loss_matrix.json")
    ap.add_argument("--problem", default=None)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    with open(args.loss_matrix_json) as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    problems = [args.problem] if args.problem else PROBLEMS
    for prob in problems:
        fig = plot_one_problem(prob, data.get(prob, {}))
        for ext in ("png", "pdf"):
            p = os.path.join(args.output_dir, f"{prob}.{ext}")
            fig.savefig(p, dpi=160)
        plt.close(fig)
        print(f"  wrote {args.output_dir}/{prob}.{{png,pdf}}")

    if args.problem is None:
        # 13-panel grid, 4 cols x 4 rows (one slot empty)
        n = len(PROBLEMS)
        ncols = 4
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6.0 * ncols, 4.6 * nrows))
        for i, prob in enumerate(PROBLEMS):
            r, c = divmod(i, ncols)
            plot_one_problem(prob, data.get(prob, {}), ax=axes[r, c],
                             annotate=True)
        for i in range(len(PROBLEMS), nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r, c].axis("off")
        fig.suptitle("Benchmark loss matrix — prediction vs decision, "
                     "train / val / test (best-val-regret checkpoint)",
                     fontsize=13, fontweight="bold", y=1.00)
        fig.tight_layout()
        out = "results/fig_loss_matrix_all.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
