"""
table_decision_problems.py
--------------------------
Emit ``docs/tables/decision_problems.tex`` — one row per benchmark problem
with its predicted parameter, decision variable, objective, constraints, and
instance counts. Notation matches the DPO chapter
(``Thesis/content/chapters/dpo/sec_methods.tex``):

    z^*(x, theta) = argmin_{z in Z} E_{p_theta(Y|X=x)} [ c(z, Y) ]

with linear cost c(z, y) = <z, y> on most problems.

Sources: ``../dfl-wiki/raw/PtOPnO/PtOPnO.md`` for the eight original benchmark
problems (knapsack, knapsack-real, energy, budgetalloc, cubic,
bipartitematching, portfolio); the codebase
(``openpto/problems/<Class>.py`` and ``openpto/config/probs/<problem>.yaml``)
for the six new problems added in this rerun (asurv, cook_county, speed_humps,
sp_synth, sp_planted, shortestpath).
"""

import os

OUT_PATH = "docs/tables/decision_problems.tex"

# Display order matches the order used elsewhere in the rerun
# (collect_bench_p1.py, fig_bench_bump_rerun.py, etc.).
PROBLEMS_IN_ORDER = [
    "knapsack", "knapsack-real", "energy", "budgetalloc", "cubic",
    "bipartitematching", "portfolio", "asurv", "cook_county", "speed_humps",
    "sp_synth", "sp_planted", "shortestpath",
]

# All math is wrapped in $...$ at render time; cells use \makecell{...} so the
# arrays/braces flow vertically when needed.
PROBLEM_SPECS = {
    "knapsack": dict(
        display="Knapsack (synth.)",
        pred=r"\hat y\in\mathbb{R}^{20}",
        decision=r"z\in\{0,1\}^{20}",
        objective=r"\max_z\,\langle z,y\rangle",
        constraints=r"$\sum_i w_i z_i\le 30$",
        sizes=r"320 / 80 / 200",
        source=r"polynomial DGP (deg 4)",
    ),
    "knapsack-real": dict(
        display="Knapsack (energy)",
        pred=r"\hat y\in\mathbb{R}^{48}",
        decision=r"z\in\{0,1\}^{48}",
        objective=r"\max_z\,\langle z,y\rangle",
        constraints=r"$\sum_i w_i z_i\le C$",
        sizes=r"442 / 110 / 237",
        source=r"SEMO energy prices",
    ),
    "energy": dict(
        display="Energy scheduling",
        pred=r"\hat y\in\mathbb{R}^{48}",
        decision=r"z\in\{0,1\}^{M\times R\times 48}",
        objective=r"\min_z\,\sum_{j,m,t} z^{jmt}\!\sum_{t'} p^j_{t'}\,y_{t'}",
        constraints=r"job-sched.; $M{=}3$, $R{=}1$",
        sizes=r"520 / 130 / 139",
        source=r"SEMO energy prices",
    ),
    "budgetalloc": dict(
        display="Budget allocation",
        pred=r"\hat y\in[0,1]^{I\times T}",
        decision=r"z\in\{0,1\}^{I}",
        objective=r"\max_z\,\sum_t\!\Big[1-\prod_i (1-y_{it}\,p_i\,z_i)\Big]",
        constraints=r"$\sum_i z_i\le B$, $B{=}1$",
        sizes=r"320 / 80 / 200",
        source=r"Yahoo! Webscope (5 sites, 10 users)",
    ),
    "cubic": dict(
        display="Cubic Top-$K$",
        pred=r"\hat y\in\mathbb{R}^{50}",
        decision=r"z\in\{0,1\}^{50}",
        objective=r"\max_z\,\langle z,y\rangle",
        constraints=r"$\sum_i z_i = K$, $K{=}5$",
        sizes=r"200 / 50 / 400",
        source=r"$y_i=10x_i^{3}-6.5x_i$",
    ),
    "bipartitematching": dict(
        display="Bipartite matching",
        pred=r"\hat y\in\mathbb{R}^{50\times 50}",
        decision=r"z\in\{0,1\}^{50\times 50}",
        objective=r"\max_z\,\sum_{i,j} y_{ij}\,z_{ij}",
        constraints=r"$\sum_j z_{ij}\le 1$, $\sum_i z_{ij}\le 1$",
        sizes=r"16 / 4 / 6",
        source=r"Cora citation network",
    ),
    "portfolio": dict(
        display="Portfolio optim.",
        pred=r"\hat y\in\mathbb{R}^{50}",
        decision=r"z\in\Delta^{49}\subset[0,1]^{50}",
        objective=r"\max_z\,\langle z,y\rangle - \alpha\,z^{\!\top}\!\Sigma z",
        constraints=r"$\sum_i z_i = 1$, $z\succeq 0$; $\alpha{=}0.1$",
        sizes=r"320 / 80 / 200",
        source=r"S\&P\,500 daily returns (Quandl)",
    ),
    "asurv": dict(
        display="Aerial survey",
        pred=r"\hat y\in\mathbb{R}_{\ge 0}^{1338}",
        decision=r"z\in\{0,1\}^{1338}",
        objective=r"\max_z\,\langle z,y\rangle",
        constraints=r"$\sum_i z_i\le 50$",
        sizes=r"15 / 6 / 6 yrs",
        source=r"wildlife counts, 1338 sites",
    ),
    "cook_county": dict(
        display="Cook County opioid",
        pred=r"\hat y\in\mathbb{R}_{\ge 0}^{1328}",
        decision=r"z\in\{0,1\}^{1328}",
        objective=r"\max_z\,\langle z,y\rangle",
        constraints=r"$\sum_i z_i\le 100$",
        sizes=r"4 / 1 / 2 yrs",
        source=r"opioid deaths, 1328 tracts",
    ),
    "speed_humps": dict(
        display="NYC speed humps",
        pred=r"\hat y\in\mathbb{R}_{\ge 0}^{2107}",
        decision=r"z\in\{0,1\}^{2107}",
        objective=r"\max_z\,\langle z,y\rangle",
        constraints=r"$\sum_i z_i\le 107$",
        sizes=r"5 / 2 / 4 yrs",
        source=r"pedestrian injuries, 2107 tracts",
    ),
    "sp_synth": dict(
        display="Shortest path (synth.)",
        pred=r"\hat y\in\mathbb{R}^{E}",
        decision=r"z\in\{0,1\}^{E}",
        objective=r"\min_z\,\langle z,y\rangle",
        constraints=r"$s{\to}t$ path on $5{\times}5$ DAG",
        sizes=r"320 / 80 / 10000",
        source=r"polynomial DGP (deg 6); SPO\textsuperscript{+} setup",
    ),
    "sp_planted": dict(
        display="Shortest path (planted)",
        pred=r"\hat y\in\mathbb{R}^{E}",
        decision=r"z\in\{0,1\}^{E}",
        objective=r"\min_z\,\langle z,y\rangle",
        constraints=r"$s{\to}t$ path on $5{\times}5$ DAG w/ planted arcs",
        sizes=r"320 / 80 / 10000",
        source=r"polynomial DGP (deg 6); PG setup",
    ),
    "shortestpath": dict(
        display="Warcraft shortest path",
        pred=r"\hat y\in\mathbb{R}_{\ge 0}^{144}",
        decision=r"z\in\{0,1\}^{144}",
        objective=r"\min_z\,\sum_v y_v\,z_v",
        constraints=r"$s{\to}t$ path on $12{\times}12$ 8-grid",
        sizes=r"8000 / 1000 / 1000",
        source=r"Warcraft tile maps (RGB images)",
    ),
}


def render_row(prob):
    s = PROBLEM_SPECS[prob]
    cells = [
        s["display"],
        f"${s['pred']}$",
        f"${s['decision']}$",
        f"${s['objective']}$",
        s["constraints"],   # raw LaTeX (already-wrapped math)
        s["sizes"],
        s["source"],
    ]
    return " & ".join(cells) + r" \\"


def render_table():
    head = r"""\begin{table}[t]
\centering
\caption{Decision problem solved by each experiment. $\hat y$ is the
predicted parameter; $z$ is the decision variable with feasible set
$\mathcal Z$. The reported decision is
$z^{\star}(x,\theta)=\arg\min_{z\in\mathcal Z}\,
\mathbb{E}_{p_\theta(Y\mid X=x)}[c(z,Y)]$ (or the analogous $\arg\max$ when
the objective is to be maximized). Sizes are train/val/test instance counts
(real-world tasks measured in years).}
\label{tab:decision-problems}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lllllll@{}}
\toprule
Task & $\hat y$ & $z\in\mathcal Z$ & Objective $c(z,y)$ & Constraints & Sizes (train/val/test) & Source \\
\midrule"""
    body = "\n".join(render_row(p) for p in PROBLEMS_IN_ORDER)
    foot = r"""\bottomrule
\end{tabular}%
}
\end{table}"""
    return f"{head}\n{body}\n{foot}\n"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tex = render_table()
    with open(OUT_PATH, "w") as f:
        f.write(tex)
    n = len(PROBLEMS_IN_ORDER)
    print(f"wrote {OUT_PATH}  ({n} rows)")


if __name__ == "__main__":
    main()
