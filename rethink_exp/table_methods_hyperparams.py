"""
table_methods_hyperparams.py
----------------------------
Emit ``docs/tables/methods_hyperparams.tex`` -- for every method in the
rerun, list each tuned hyperparameter with a short description and the
val-search grid. Addresses committee feedback item #2.

Layout (one row per HP, methods separated by ``\\midrule``):

    Method        | Symbol | Description                   | Val search grid
    ------------- | ------ | ----------------------------- | ---------------
    MSE           | $\\eta$ | Adam learning rate           | $\\{10^{-2},...\\}$
                  | $b$    | batch (full / minibatch)     | $\\{\\text{full},\\text{mini}\\}$
    DFL           | ...    | ...                          | ...
                  | $\\alpha$ | MSE-blend weight added ...| $\\{10^{-3},...,10\\}$
    ...

Universal training settings (n_epochs=300, patience=40, etc.) live in the
caption -- they're never tuned and don't deserve table rows.

Two columns from the prior version are *hidden* (data still kept in the
``METHODS`` dict for future use):

* ``ref`` (\\citep{...} bibkey)
* ``fixed_extra`` (method-specific HPs that are NOT tuned)

The script cross-checks every "fixed" entry against ``default.yaml`` at
runtime and aborts on drift.
"""

import os
import sys

try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

OUT_PATH = "docs/tables/methods_hyperparams.tex"
DEFAULT_YAML = "openpto/config/models/default.yaml"

# Universal training settings shared by every method (caption text).
UNIVERSAL_FIXED = (
    r"All methods share fixed training settings: "
    r"$\texttt{n\_epochs}{=}300$, "
    r"$\texttt{patience}{=}40$, "
    r"$\texttt{seed}{=}2023$, "
    r"$\texttt{n\_ptr\_epochs}{=}0$, "
    r"prediction model = 2-layer dense MLP with 32 hidden units "
    r"(\texttt{Resnet18} for the Warcraft shortest-path task only), "
    r"Adam optimiser. Selection criterion for every cell of the search "
    r"grid is minimum validation decision regret, "
    r"except for MSE\textsubscript{train} (minimum training MSE) and "
    r"MSE\textsubscript{val} (minimum validation MSE) which are kept as "
    r"reference baselines."
)

# Two HPs swept for every method.
SHARED_TUNED = [
    dict(symbol=r"\eta", desc="Adam learning rate",
         grid=r"$\{10^{-2},\,5{\!\times\!}10^{-3},\,10^{-3}\}$"),
    dict(symbol=r"b",    desc="batch (full vs minibatch)",
         grid=r"$\{\mathrm{full},\,\mathrm{mini}\}$"),
]


# Per-method spec. ``ref`` and ``fixed_extra`` are kept for future use but
# are *not* rendered. ``tuned_extra`` lists the method-specific HPs swept
# in Phase 2 -- one dict per HP, each producing its own row.
METHODS = [
    dict(
        key="mse", display="MSE", ref=r"\citep{de2018end}",
        tuned_extra=[],
        fixed_extra=[("reduction", "mean"),
                     ("selection signal", "val regret")],
    ),
    dict(
        key="mse_train", display=r"MSE\textsubscript{train}",
        ref=r"\citep{de2018end}",
        tuned_extra=[],
        fixed_extra=[("reduction", "mean"),
                     ("selection signal", "train MSE")],
    ),
    dict(
        key="mse_val", display=r"MSE\textsubscript{val}",
        ref=r"\citep{de2018end}",
        tuned_extra=[],
        fixed_extra=[("reduction", "mean"),
                     ("selection signal", "val MSE")],
    ),
    dict(
        key="dfl", display="DFL", ref=r"\citep{wilder2019melding}",
        tuned_extra=[
            dict(symbol=r"\alpha", desc=r"MSE-blend weight added to decision loss",
                 grid=r"$\{10^{-3},\,10^{-2},\,10^{-1},\,1,\,10\}$"),
        ],
        fixed_extra=[("reduction", "mean")],
    ),
    dict(
        key="identity", display="Identity", ref=r"\citep{sahoo2023backprop}",
        tuned_extra=[],
        fixed_extra=[("reduction", "mean")],
    ),
    dict(
        key="spo", display=r"SPO\textsuperscript{+}",
        ref=r"\citep{elmachtoub2022smart}",
        tuned_extra=[],
        fixed_extra=[("reduction", "mean")],
    ),
    dict(
        key="nce", display="NCE", ref=r"\citep{mulamba2021contrastive}",
        tuned_extra=[],
        fixed_extra=[("reduction", "mean")],
    ),
    dict(
        key="blackbox", display="Blackbox", ref=r"\citep{vlastelica2019differentiation}",
        tuned_extra=[
            dict(symbol=r"\lambda", desc=r"interpolation step in the relaxed gradient",
                 grid=r"$\{0.01,\,0.05,\,0.1,\,0.5,\,1\}$"),
        ],
        fixed_extra=[("reduction", "mean")],
    ),
    dict(
        key="pointLTR", display="pointLTR", ref=r"\citep{mandi2022decision}",
        tuned_extra=[],
        fixed_extra=[("reduction", "mean")],
    ),
    dict(
        key="pairLTR", display="pairLTR", ref=r"\citep{mandi2022decision}",
        tuned_extra=[],
        fixed_extra=[("reduction", "mean")],
    ),
    dict(
        key="listLTR", display="listLTR", ref=r"\citep{mandi2022decision}",
        tuned_extra=[
            dict(symbol=r"\tau", desc=r"softmax temperature over candidate solutions",
                 grid=r"$\{0.1,\,0.5,\,1,\,5,\,10\}$"),
        ],
        fixed_extra=[("reduction", "mean")],
    ),
    dict(
        key="lodl", display="LODL", ref=r"\citep{shah2022decisionfocused}",
        tuned_extra=[
            dict(symbol=r"K", desc=r"\# Monte-Carlo samples for surrogate fitting",
                 grid=r"$\{100,\,250,\,500,\,10^{3},\,2{\!\times\!}10^{3}\}$"),
        ],
        fixed_extra=[
            ("sampling", "random_hessian"),
            ("model_type", "weightedmse"),
            ("num_iters", 100),
            ("losslr", "1e-3"),
            ("patience", 25),
        ],
    ),
    dict(
        key="perturb", display=r"DPO (perturbed)", ref=r"\citep{berthet2020learning}",
        tuned_extra=[
            dict(symbol=r"\sigma", desc=r"Gaussian noise scale (held $M{=}10$)",
                 grid=r"$\{0.1,\,0.5,\,1,\,2,\,5\}$"),
            dict(symbol=r"M", desc=r"\# perturbation samples per gradient (held $\sigma{=}1$)",
                 grid=r"$\{5,\,10,\,25,\,50,\,100\}$"),
        ],
        fixed_extra=[("reduction", "mean"), ("noise", "normal")],
    ),
    dict(
        key="pg", display="PG", ref=r"\citep{sahoo2023backprop}",
        tuned_extra=[
            dict(symbol=r"h", desc=r"finite-difference step size",
                 grid=r"$\{0.01,\,0.05,\,0.1,\,0.5,\,1\}$"),
        ],
        fixed_extra=[("reduction", "mean")],
    ),
    dict(
        key="cpLayer", display="cpLayer", ref=r"\citep{agrawal2019differentiable}",
        tuned_extra=[],
        fixed_extra=[("reduction", "mean")],
    ),
]


def _load_defaults():
    if not _HAVE_YAML:
        return None
    with open(DEFAULT_YAML) as f:
        return yaml.safe_load(f)


def _validate(defaults):
    if defaults is None:
        print("(skipped default-yaml cross-check; activate "
              "pco_bench_rhel7 to enable)", file=sys.stderr)
        return
    bad = []
    for m in METHODS:
        d = defaults.get(m["key"])
        if d is None:
            continue
        for name, expected in m["fixed_extra"]:
            if name not in d:
                continue
            actual = d[name]
            if actual != expected:
                bad.append((m["key"], name, expected, actual))
    if bad:
        for k, name, exp, act in bad:
            print(f"  default mismatch: {k}.{name} table={exp!r} yaml={act!r}",
                  file=sys.stderr)
        print(f"\n{len(bad)} fixed-HP defaults drifted from {DEFAULT_YAML}.",
              file=sys.stderr)
        sys.exit(1)


def _baseline_methods():
    return [m for m in METHODS if not m.get("tuned_extra")]


def _tunable_methods():
    return [m for m in METHODS if m.get("tuned_extra")]


def render_table():
    # ---- Section 1: shared HPs (one block, applies to every method). ----
    shared_lines = []
    n = len(SHARED_TUNED)
    for i, hp in enumerate(SHARED_TUNED):
        method_cell = (rf"\multirow{{{n}}}{{*}}{{\textit{{all methods}}}}"
                       if i == 0 else "")
        shared_lines.append(
            f"{method_cell} & ${hp['symbol']}$ & {hp['desc']} & {hp['grid']} \\\\"
        )
    shared_block = "\n".join(shared_lines)

    # ---- Section 2: method-specific HPs (one block per Phase-2-tunable
    # method, separated by \midrule). ----
    blocks = []
    for m in _tunable_methods():
        rows = m["tuned_extra"]
        n = len(rows)
        # Skip \multirow when there's only one row to avoid an unhelpful
        # \multirow{1}{*}{...} wrapper.
        method_cell = (m["display"] if n == 1
                       else rf"\multirow{{{n}}}{{*}}{{{m['display']}}}")
        body_lines = []
        for i, hp in enumerate(rows):
            mc = method_cell if i == 0 else ""
            body_lines.append(
                f"{mc} & ${hp['symbol']}$ & {hp['desc']} & {hp['grid']} \\\\"
            )
        blocks.append("\n".join(body_lines))
    method_specific_block = "\n\\midrule\n".join(blocks)

    # ---- Caption + names of baseline-only methods. ----
    baseline_names = ", ".join(m["display"] for m in _baseline_methods())
    caption = (
        r"Hyperparameters per method. " + UNIVERSAL_FIXED + r" "
        r"The top block ($\eta$, $b$) is shared by every method; the bottom "
        r"block lists each method-specific hyperparameter swept during "
        r"Phase 2 after the Phase-1 $(\eta, b)$ winner is fixed. Baseline "
        r"methods that have no method-specific hyperparameter --- " +
        baseline_names + r" --- differ only in their choice of $(\eta, b)$."
    )

    head = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        rf"\caption{{{caption}}}" "\n"
        r"\label{tab:methods-hyperparams}" "\n"
        r"\setlength{\tabcolsep}{6pt}" "\n"
        r"\renewcommand{\arraystretch}{1.1}" "\n"
        r"\begin{tabular}{@{}l l p{2.6in} l@{}}" "\n"
        r"\toprule" "\n"
        r"Method & Symbol & Description & Val search grid \\" "\n"
        r"\midrule"
    )
    foot = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )

    return (
        f"{head}\n"
        f"{shared_block}\n"
        r"\midrule" "\n"
        f"{method_specific_block}\n"
        f"{foot}\n"
    )


def main():
    defaults = _load_defaults()
    _validate(defaults)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tex = render_table()
    with open(OUT_PATH, "w") as f:
        f.write(tex)
    n_shared = len(SHARED_TUNED)
    n_specific = sum(len(m.get("tuned_extra", [])) for m in METHODS)
    n_baseline = len(_baseline_methods())
    n_tunable = len(_tunable_methods())
    print(f"wrote {OUT_PATH}  "
          f"({n_shared} shared rows + {n_specific} method-specific rows; "
          f"{n_tunable} tunable methods, {n_baseline} baseline-only listed in caption)")


if __name__ == "__main__":
    main()
