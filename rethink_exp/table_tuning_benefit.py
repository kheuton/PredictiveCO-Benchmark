"""
table_tuning_benefit.py
-----------------------
Emit ``docs/tables/tuning_benefit.tex`` -- head-to-head comparison of every
method's *fixed-HP* run vs its *val-tuned-HP* run, addressing committee
feedback item #2 ("for any claim that tuning improves performance, add a
head-to-head comparison").

Source: ``bench_p2_best_val.json`` (built by
``rethink_exp/collect_bench_p2_val.py``). Each Phase-2-tunable entry carries

  * the val-best HP run  -- the "tuned" cell
  * the default-HP run at the same (LR, batch) -- the "fixed" cell

For non-Phase-2 methods (mse, identity, spo, nce, pointLTR, pairLTR, cpLayer)
there is no method-specific HP to tune; we still report their test regret in
the cell but mark Delta as "--".

Each cell shows three lines (top to bottom):

    fixed
    tuned
    Delta

where Delta = (fixed - tuned) / |fixed| * 100, in percent. Positive Delta
means tuning helped. Cells colored green when Delta >= 5%, red when
Delta <= -5%, neutral otherwise. Missing combinations show "--".

Below the main grid we append a one-row summary: per-method mean Delta
across all problems where it ran.
"""

import argparse
import json
import os
import sys

import numpy as np

P2_BEST_VAL_PATH = "bench_p2_best_val.json"
OUT_PATH = "docs/tables/tuning_benefit.tex"

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio", "asurv", "cook_county",
            "speed_humps", "sp_synth", "sp_planted", "pg_misspec", "shortestpath"]

# Short labels for column headers (defined in the caption).
PROB_LABELS = {
    "knapsack": "KS", "knapsack-real": "KS-E", "energy": "En",
    "budgetalloc": "BA", "cubic": "Cu", "bipartitematching": "BM",
    "portfolio": "Pf", "asurv": "AS", "cook_county": "CC",
    "speed_humps": "SH", "sp_synth": "sp\\textsubscript{s}",
    "sp_planted": "sp\\textsubscript{p}",
    "pg_misspec": "pg\\textsubscript{ms}",
    "shortestpath": "SP-W",
}

ALL_METHODS = ["mse", "mse_train", "mse_val", "dfl", "identity", "spo", "nce", "blackbox",
               "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg"]

METHOD_LABELS = {
    "mse": "MSE",
    "mse_train": "MSE\\textsubscript{train}",
    "mse_val":   "MSE\\textsubscript{val}",
    "dfl": "DFL", "identity": "Identity",
    "spo": "SPO\\textsuperscript{+}", "nce": "NCE", "blackbox": "Blackbox",
    "pointLTR": "ptLTR", "pairLTR": "prLTR", "listLTR": "lsLTR",
    "lodl": "LODL", "perturb": "DPO", "pg": "PG",
}

# Methods with a Phase-2 method-specific HP sweep.
P2_TUNABLE = {"dfl", "blackbox", "listLTR", "lodl", "perturb", "pg"}


def fmt_num(x, digits=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{digits}f}"


def fmt_delta(d):
    if d is None or not np.isfinite(d):
        return "--"
    sign = "+" if d > 0 else ("" if d == 0 else "")
    return rf"{sign}{d:.1f}\%"


def color_for(delta):
    """Return a LaTeX color command prefix or empty string."""
    if delta is None or not np.isfinite(delta):
        return ""
    if delta >= 5.0:
        return r"\cellcolor{green!18}"
    if delta <= -5.0:
        return r"\cellcolor{red!18}"
    return ""


def cell_content(entry):
    """Return (latex_string, delta_pct or None) for one (method, problem) cell."""
    if entry is None:
        return r"\multicolumn{1}{c}{--}", None

    tuned_test = entry.get("test")
    fixed_blk  = entry.get("fixed")
    if entry.get("phase", 1) == 1 or fixed_blk is None:
        # Non-P2 method: report only the test number, no Delta.
        return rf"\makecell[c]{{--\\{fmt_num(tuned_test)}\\--}}", None

    fixed_test = fixed_blk.get("test")
    if tuned_test is None or fixed_test is None or fixed_test == 0:
        delta = None
    else:
        delta = (fixed_test - tuned_test) / abs(fixed_test) * 100.0

    pre = color_for(delta)
    body = rf"\makecell[c]{{{fmt_num(fixed_test)}\\{fmt_num(tuned_test)}\\{fmt_delta(delta)}}}"
    return pre + body, delta


def render_table(p2_best_val):
    rows = []
    method_deltas = {}  # method -> list of deltas (for summary row)
    for m in ALL_METHODS:
        cells = [METHOD_LABELS[m]]
        ds = []
        for prob in PROBLEMS:
            entry = p2_best_val.get(m, {}).get(prob)
            cell, delta = cell_content(entry)
            cells.append(cell)
            if delta is not None and np.isfinite(delta):
                ds.append(delta)
        rows.append(" & ".join(cells) + r" \\")
        method_deltas[m] = ds

    # Summary row: mean Delta per method (only for methods with at least one
    # tunable problem).
    summary_cells = [r"\textit{mean $\Delta$}"]
    for prob in PROBLEMS:
        summary_cells.append("")  # column-by-column empty
    # Append a final mean-per-method block at the right is awkward; instead,
    # render the summary as a separate \multicolumn line below.

    n_cols = 1 + len(PROBLEMS)
    col_spec = "@{}l" + "c" * len(PROBLEMS) + "@{}"
    headers = [""] + [PROB_LABELS[p] for p in PROBLEMS]
    head_line = " & ".join(headers) + r" \\"

    # Per-method mean-Delta sentence (rendered in caption). Sort by descending mean.
    summary_lines = []
    for m, ds in method_deltas.items():
        if not ds:
            continue
        summary_lines.append((m, float(np.mean(ds)), len(ds)))
    summary_lines.sort(key=lambda x: -x[1])
    summary_caption = ", ".join(
        rf"{METHOD_LABELS[m]}\,{d:+.1f}\% ({n})" for m, d, n in summary_lines
    )

    head = (
        r"\begin{table*}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Benefit of tuning method-specific hyperparameters. Each "
        r"cell shows three numbers stacked top-to-bottom: \emph{fixed} "
        r"(method-specific HP at default values, with the Phase-1 val-best "
        r"$(\eta, b)$), \emph{tuned} (best HP value across the Phase-2 grid, "
        r"selected by minimum validation decision regret), and "
        r"$\Delta=({\rm fixed}-{\rm tuned})/|{\rm fixed}|$ (positive = tuning "
        r"helps). Numbers are test relative regret (absolute regret for "
        r"\textit{Pf}). Methods without a method-specific HP show only the "
        r"middle (tuned) row. Cells shaded green when $\Delta\geq 5\%$, red "
        r"when $\Delta\leq -5\%$. Per-method mean $\Delta$ across applicable "
        r"problems (descending): " + summary_caption + r". Columns: "
        r"KS=knapsack-synth, KS-E=knapsack-energy, En=energy, BA=budgetalloc, "
        r"Cu=cubic, BM=bipartite, Pf=portfolio, AS=asurv, CC=cook\_county, "
        r"SH=speed\_humps, sp\textsubscript{s}=sp\_synth, "
        r"sp\textsubscript{p}=sp\_planted, SP-W=warcraft.}" "\n"
        r"\label{tab:tuning-benefit}" "\n"
        r"\setlength{\tabcolsep}{2pt}" "\n"
        r"\renewcommand{\arraystretch}{0.95}" "\n"
        r"\resizebox{\textwidth}{!}{%" "\n"
        rf"\begin{{tabular}}{{{col_spec}}}" "\n"
        r"\toprule" "\n"
        + head_line + "\n" +
        r"\midrule"
    )
    body = "\n".join(rows)
    foot = (
        r"\bottomrule" "\n"
        r"\end{tabular}%" "\n"
        r"}" "\n"
        r"\end{table*}"
    )
    return f"{head}\n{body}\n{foot}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2_best_json", default=P2_BEST_VAL_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.p2_best_json):
        print(f"error: {args.p2_best_json} not found. "
              f"Run `python rethink_exp/collect_bench_p2_val.py` first.")
        sys.exit(1)
    with open(args.p2_best_json) as f:
        p2_best_val = json.load(f)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tex = render_table(p2_best_val)
    with open(args.out, "w") as f:
        f.write(tex)

    n_rows = len(ALL_METHODS)
    n_cols = len(PROBLEMS)
    print(f"wrote {args.out}  ({n_rows} method rows x {n_cols} problem cols)")


if __name__ == "__main__":
    main()
