#!/usr/bin/env python
"""
Visualize PredictiveCO benchmark results, with focus on perturb analysis.

Usage:
    python rethink_exp/visualize_results.py
    python rethink_exp/visualize_results.py --prefix bench --sigmas 01,05,10
    python rethink_exp/visualize_results.py --no-show  # save only, don't display

Produces:
  1. Bar chart per problem: all methods side by side (eval_metric)
  2. Combined ranking heatmap across all problems
  3. Sigma sensitivity plot: x=sigma, y=eval_metric, one line per problem
  4. LaTeX table of results
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_with_perturb import (
    PROBLEM_CONFIG,
    collect_all_results,
    print_results_table,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---- Color scheme ----
BASELINE_COLOR = "#4C72B0"
PERTURB_COLORS = {
    "perturb_s01": "#E04040",
    "perturb_s05": "#E07020",
    "perturb_s10": "#E0A020",
}
METHOD_CATEGORIES = {
    "PtO": ["mse", "bce", "mae", "ce"],
    "PnO": [
        "dfl", "spo", "nce", "blackbox", "identity", "cpLayer",
        "pointLTR", "listLTR", "pairLTR", "lodl", "qptl",
    ],
    "Perturb": [],  # filled dynamically
}


def get_method_color(method):
    """Assign colors by category."""
    if method in PERTURB_COLORS:
        return PERTURB_COLORS[method]
    if method.startswith("perturb"):
        return "#E05050"
    if method in METHOD_CATEGORIES["PtO"]:
        return "#6BAED6"
    return "#4C72B0"


def plot_per_problem_bars(all_results, output_dir, show=True):
    """Bar chart for each problem showing eval_metric per method."""
    n_problems = len(all_results)
    fig, axes = plt.subplots(
        2, (n_problems + 1) // 2,
        figsize=(6 * ((n_problems + 1) // 2), 10),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, (prob_key, df) in enumerate(all_results.items()):
        ax = axes_flat[idx]
        df_valid = df.dropna(subset=["eval_metric"])
        if len(df_valid) == 0:
            ax.set_title(f"{prob_key}\n(no results)")
            continue

        methods = df_valid["method"].tolist()
        values = df_valid["eval_metric"].values
        colors = [get_method_color(m) for m in methods]

        bars = ax.bar(range(len(methods)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Eval Metric")
        ax.set_title(prob_key, fontsize=11, fontweight="bold")

        # Highlight best
        best_idx = np.nanargmin(values)
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(2)

    # Hide unused axes
    for idx in range(len(all_results), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("PredictiveCO Benchmark: Eval Metric by Problem", fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "per_problem_bars.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved: {path}")
    path_png = os.path.join(output_dir, "per_problem_bars.png")
    plt.savefig(path_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {path_png}")
    if show:
        plt.show()
    plt.close()


def plot_ranking_heatmap(all_results, output_dir, show=True):
    """Heatmap of method rankings across problems."""
    pivot, rank_df, _ = print_results_table(all_results)

    # Remove avg_rank column for heatmap
    rank_cols = [c for c in rank_df.columns if c != "avg_rank"]
    rank_display = rank_df[rank_cols].copy()
    rank_display.columns = [c.replace("_rank", "") for c in rank_display.columns]

    # Add average rank
    rank_display["AVG"] = rank_df["avg_rank"]

    fig, ax = plt.subplots(figsize=(max(10, len(rank_display.columns) * 1.2), max(6, len(rank_display) * 0.4)))

    # Color: lower rank (better) = green, higher = red
    cmap = cm.RdYlGn_r
    im = ax.imshow(rank_display.values, cmap=cmap, aspect="auto", vmin=1, vmax=len(rank_display))

    ax.set_xticks(range(len(rank_display.columns)))
    ax.set_xticklabels(rank_display.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rank_display)))
    ax.set_yticklabels(rank_display.index, fontsize=9)

    # Annotate cells
    for i in range(len(rank_display)):
        for j in range(len(rank_display.columns)):
            val = rank_display.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val > len(rank_display) * 0.6 else "black"
                ax.text(j, i, f"{val:.0f}" if j < len(rank_display.columns) - 1 else f"{val:.1f}",
                        ha="center", va="center", fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Rank (1=best)")
    ax.set_title("Method Rankings Across Problems", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "ranking_heatmap.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved: {path}")
    path_png = os.path.join(output_dir, "ranking_heatmap.png")
    plt.savefig(path_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {path_png}")
    if show:
        plt.show()
    plt.close()


def plot_sigma_sensitivity(all_results, sigma_tags, output_dir, show=True):
    """
    Sigma sensitivity plot: for each problem, plot eval_metric vs sigma.
    Also shows horizontal lines for baseline methods.
    """
    # Parse sigma values from tags
    sigma_map = {}
    for tag in sigma_tags:
        # tag like "01" -> 0.1, "05" -> 0.5, "10" -> 1.0
        try:
            # Reconstruct float: insert decimal point
            if len(tag) == 1:
                sigma_map[tag] = float(tag)
            elif len(tag) == 2:
                sigma_map[tag] = float(f"0.{tag[-1]}")
                # Better: try to figure out from common patterns
                if tag == "01":
                    sigma_map[tag] = 0.1
                elif tag == "05":
                    sigma_map[tag] = 0.5
                elif tag == "10":
                    sigma_map[tag] = 1.0
                elif tag == "25":
                    sigma_map[tag] = 0.25
                else:
                    sigma_map[tag] = float(tag) / 10.0
            elif len(tag) == 3:
                sigma_map[tag] = float(tag) / 100.0
            else:
                sigma_map[tag] = float(tag) / (10 ** (len(tag) - 1))
        except ValueError:
            sigma_map[tag] = float(tag)

    sigma_values = [sigma_map[t] for t in sigma_tags]

    n_problems = len(all_results)
    fig, axes = plt.subplots(
        2, (n_problems + 1) // 2,
        figsize=(5 * ((n_problems + 1) // 2), 8),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, (prob_key, df) in enumerate(all_results.items()):
        ax = axes_flat[idx]

        # Get perturb results
        perturb_vals = []
        for tag in sigma_tags:
            method_name = f"perturb_s{tag}"
            row = df[df["method"] == method_name]
            if len(row) > 0:
                perturb_vals.append(row.iloc[0]["eval_metric"])
            else:
                perturb_vals.append(np.nan)

        # Plot perturb line
        valid_mask = ~np.isnan(perturb_vals)
        if any(valid_mask):
            ax.plot(
                np.array(sigma_values)[valid_mask],
                np.array(perturb_vals)[valid_mask],
                "ro-", linewidth=2, markersize=8, label="perturb", zorder=10,
            )

        # Plot horizontal lines for top baselines
        baseline_df = df[~df["method"].str.startswith("perturb")].dropna(subset=["eval_metric"])
        if len(baseline_df) > 0:
            # Sort by eval_metric and show top 5
            baseline_df = baseline_df.sort_values("eval_metric").head(5)
            cmap = cm.tab10
            for bi, (_, brow) in enumerate(baseline_df.iterrows()):
                ax.axhline(
                    y=brow["eval_metric"],
                    color=cmap(bi),
                    linestyle="--",
                    alpha=0.7,
                    label=brow["method"],
                )

        ax.set_xlabel("σ (sigma)")
        ax.set_ylabel("Eval Metric")
        ax.set_title(prob_key, fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    for idx in range(len(all_results), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Perturb: Sigma Sensitivity Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "sigma_sensitivity.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved: {path}")
    path_png = os.path.join(output_dir, "sigma_sensitivity.png")
    plt.savefig(path_png, bbox_inches="tight", dpi=150)
    print(f"Saved: {path_png}")
    if show:
        plt.show()
    plt.close()


def generate_latex_table(all_results, output_dir):
    """Generate a LaTeX table of results."""
    pivot, rank_df, _ = print_results_table(all_results)

    # Bold the best value per column
    latex_rows = []
    for method in pivot.index:
        row_vals = []
        for col in pivot.columns:
            val = pivot.loc[method, col]
            if np.isnan(val):
                row_vals.append("--")
            else:
                # Check if best in column
                col_vals = pivot[col].dropna()
                is_best = val == col_vals.min()
                formatted = f"{val:.4f}"
                if is_best:
                    formatted = f"\\textbf{{{formatted}}}"
                row_vals.append(formatted)
        # Add average rank
        avg_rank = rank_df.loc[method, "avg_rank"]
        avg_str = f"{avg_rank:.1f}" if not np.isnan(avg_rank) else "--"
        row_vals.append(avg_str)
        latex_rows.append(f"  {method} & " + " & ".join(row_vals) + " \\\\")

    # Build table
    n_cols = len(pivot.columns) + 2  # method + problems + avg_rank
    col_spec = "l" + "c" * (n_cols - 1)
    header_cols = " & ".join(pivot.columns) + " & Avg Rank"

    latex = f"""\\begin{{table}}[ht]
\\centering
\\caption{{PredictiveCO Benchmark Results (Eval Metric, lower is better)}}
\\label{{tab:benchmark_results}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_spec}}}
\\toprule
Method & {header_cols} \\\\
\\midrule
"""
    # Separate baselines from perturb
    baseline_rows = [r for r in latex_rows if "perturb" not in r]
    perturb_rows = [r for r in latex_rows if "perturb" in r]

    latex += "\n".join(baseline_rows)
    if perturb_rows:
        latex += "\n\\midrule\n"
        latex += "\n".join(perturb_rows)

    latex += f"""
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""

    path = os.path.join(output_dir, "benchmark_table.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PredictiveCO results")
    parser.add_argument("--prefix", type=str, default="bench")
    parser.add_argument("--sigmas", type=str, default="01,05,10")
    parser.add_argument("--problem", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="resource/figs")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    args = parser.parse_args()

    sigma_tags = args.sigmas.split(",")
    problems = [args.problem] if args.problem else None
    show = not args.no_show

    os.makedirs(args.output_dir, exist_ok=True)

    print("Collecting results...")
    all_results = collect_all_results(
        problems=problems, prefix=args.prefix, sigmas=sigma_tags
    )

    # Check if any results exist
    any_data = False
    for df in all_results.values():
        if df.dropna(subset=["eval_metric"]).shape[0] > 0:
            any_data = True
            break

    if not any_data:
        print("\nNo results found yet. Run experiments first, then re-run this script.")
        print("Generating empty template plots for reference...")

    print("\nGenerating visualizations...")
    plot_per_problem_bars(all_results, args.output_dir, show=show)
    plot_ranking_heatmap(all_results, args.output_dir, show=show)
    plot_sigma_sensitivity(all_results, sigma_tags, args.output_dir, show=show)
    generate_latex_table(all_results, args.output_dir)

    print("\nDone! All figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
