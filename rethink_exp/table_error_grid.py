r"""
table_error_grid.py
-------------------
Emit a 15-method by 13-problem grid where every cell shows three numbers
(train / val / test) for one error metric. Two metrics are supported via
``--metric``:

  * ``pred``      -- prediction MSE / BCE / ... per problem
  * ``decision``  -- decision regret (absolute, mean over instances)

Source: ``loss_matrix.json`` produced by ``rethink_exp/collect_loss_matrix.py``.
Output: ``docs/tables/pred_error.tex`` (default for ``--metric pred``)
or    ``docs/tables/decision_error.tex`` (default for ``--metric decision``).

Addresses committee feedback item #3: the train and val numbers make
overfitting visible alongside the test number.

The page is wrapped in a ``landscape`` environment (requires
``\usepackage{pdflscape}`` and ``\usepackage{booktabs,makecell}`` in the
thesis preamble -- already standard).
"""

import argparse
import json
import os
import sys

import numpy as np

LOSS_MATRIX_PATH = "loss_matrix.json"

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio", "asurv", "cook_county",
            "speed_humps", "sp_synth", "sp_planted", "shortestpath"]

PROB_LABELS = {
    "knapsack": "KS", "knapsack-real": "KS-E", "energy": "En",
    "budgetalloc": "BA", "cubic": "Cu", "bipartitematching": "BM",
    "portfolio": "Pf", "asurv": "AS", "cook_county": "CC",
    "speed_humps": "SH", "sp_synth": r"sp\textsubscript{s}",
    "sp_planted": r"sp\textsubscript{p}", "shortestpath": "SP-W",
}

ALL_METHODS = ["mse", "dfl", "identity", "spo", "nce", "blackbox",
               "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg",
               "qptl", "cpLayer", "dad"]

METHOD_LABELS = {
    "mse": "MSE", "dfl": "DFL", "identity": "Identity",
    "spo": r"SPO\textsuperscript{+}", "nce": "NCE", "blackbox": "Blackbox",
    "pointLTR": "ptLTR", "pairLTR": "prLTR", "listLTR": "lsLTR",
    "lodl": "LODL", "perturb": "DPO", "pg": "PG",
    "qptl": "QPTL", "cpLayer": "cpLayer", "dad": "DAD",
}

# (key in the metrics dict) for each (metric, split). For ``decision`` we use
# absolute regret on every split so the three numbers in a cell live on the
# same scale.
METRIC_KEYS = {
    "pred": dict(train="train_pred_loss",
                 val="val_pred_loss",
                 test="test_pred_loss"),
    "decision": dict(train="train_regret",
                     val="val_regret",
                     test="test_regret_abs"),
}

DEFAULT_OUT = {
    "pred": "docs/tables/pred_error.tex",
    "decision": "docs/tables/decision_error.tex",
}


def fmt(x, sigdigs=3):
    """Compact number format. 3 sig digits with switch to scientific for very
    small / very large magnitudes. ``--`` when missing. Output is always
    valid in LaTeX text-mode (no ``\\!`` thin-spaces, no math-only macros)."""
    if x is None:
        return "--"
    if isinstance(x, str):
        if x == "inf":
            return r"$\infty$"
        return x
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 10000 or ax < 1e-3:
        # e.g. 1.3e-04 -> "1.3e\textminus 4"; keep ASCII for portability
        s = f"{x:.1e}"          # "1.3e-04"
        mant, exp = s.split("e")
        exp = int(exp)          # strips leading zeros / sign
        return f"{mant}e{exp}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.3f}"
    return f"{x:.{sigdigs}f}"


def cell(entry, keys):
    if entry is None:
        return r"\multicolumn{1}{c}{--}"
    m = entry.get("metrics", {})
    a = fmt(m.get(keys["train"]))
    b = fmt(m.get(keys["val"]))
    c = fmt(m.get(keys["test"]))
    return rf"\makecell[r]{{{a}\\{b}\\{c}}}"


def render_table(loss_matrix, metric):
    keys = METRIC_KEYS[metric]
    cap = {
        "pred": (
            "Prediction error per (method, task). Each cell shows three numbers "
            "stacked top-to-bottom: train / val / test prediction loss "
            "(MSE for regression problems, BCE for binary problems). "
            "Selected at the val-best (LR, batch, method-specific HP) per cell."
        ),
        "decision": (
            "Decision error per (method, task). Each cell shows three numbers "
            "stacked top-to-bottom: train / val / test mean absolute regret. "
            "Selected at the val-best (LR, batch, method-specific HP) per cell. "
            "Lower is better; values are not directly comparable across tasks "
            "(scale differs). "
            "Train and val regret read from \\texttt{train\\_logs.csv} / "
            "\\texttt{val\\_logs.csv} at the best-val-regret epoch; test regret "
            "from \\texttt{results.npy}."
        ),
    }[metric]
    cap += (
        " Cells reading \\texttt{--} indicate (a) the method does not apply to "
        "the task (e.g.\\ QPTL/cpLayer outside QP-form, PG on warcraft); "
        "(b) the test prediction loss has not yet been computed by "
        "\\texttt{eval\\_test\\_pred.py}; or (c) the run is still pending. "
        "Columns: KS=knapsack-synth, KS-E=knapsack-energy, En=energy, "
        "BA=budgetalloc, Cu=cubic, BM=bipartite, Pf=portfolio, AS=asurv, "
        "CC=cook\\_county, SH=speed\\_humps, "
        "sp\\textsubscript{s}=sp\\_synth, sp\\textsubscript{p}=sp\\_planted, "
        "SP-W=warcraft."
    )

    label = {"pred": "tab:pred-error", "decision": "tab:decision-error"}[metric]
    title = {"pred": "Prediction error",
             "decision": "Decision error (absolute regret)"}[metric]

    rows = []
    for m in ALL_METHODS:
        cells = [METHOD_LABELS[m]]
        for prob in PROBLEMS:
            cells.append(cell(loss_matrix.get(prob, {}).get(m), keys))
        rows.append(" & ".join(cells) + r" \\")

    col_spec = "@{}l" + "r" * len(PROBLEMS) + "@{}"
    head_cells = [""] + [PROB_LABELS[p] for p in PROBLEMS]
    head_line = " & ".join(head_cells) + r" \\"

    head = (
        r"\begin{landscape}" "\n"
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        rf"\caption{{{title}: {cap}}}" "\n"
        rf"\label{{{label}}}" "\n"
        r"\setlength{\tabcolsep}{3pt}" "\n"
        r"\renewcommand{\arraystretch}{0.95}" "\n"
        r"\resizebox{\linewidth}{!}{%" "\n"
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
        r"\end{table}" "\n"
        r"\end{landscape}"
    )
    return f"{head}\n{body}\n{foot}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["pred", "decision"], required=True)
    ap.add_argument("--loss_matrix", default=LOSS_MATRIX_PATH)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.loss_matrix):
        print(f"error: {args.loss_matrix} not found. Run "
              f"`python rethink_exp/collect_loss_matrix.py` first.")
        sys.exit(1)
    with open(args.loss_matrix) as f:
        loss_matrix = json.load(f)

    out = args.out or DEFAULT_OUT[args.metric]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tex = render_table(loss_matrix, args.metric)
    with open(out, "w") as f:
        f.write(tex)
    print(f"wrote {out}  (15 methods x 13 problems, 3 numbers per cell)")


if __name__ == "__main__":
    main()
