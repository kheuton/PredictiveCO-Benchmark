# Table: Benchmark Methods Categorised

This table categorises the 15 methods in the Phase-1/Phase-2 benchmark sweep
along axes relevant to the reviewer's feedback: which loss each method is
actually minimising, how it becomes *decision-aware*, what class of
combinatorial problem it applies to, its per-epoch solver-call cost, and the
number of hyper-parameters tuned.

Symbols are typeset with LaTeX inline math so the table renders identically
in Markdown viewers with math support (GitHub, VS Code, Obsidian, Pandoc) and
when included in a LaTeX manuscript.

---

| # | Method | Family | Loss / Surrogate mechanism | CO scope | Solves / batch | # tunable HPs | Phase-2 HP swept | Reference |
|---|---|---|---|---|---|---|---|---|
| 1 | **MSE** $^{\ast\dagger}$ | PtO | $\lVert \hat c - c \rVert_2^2$ — prediction loss only | Any | 0 | 2 (lr, bs) | — | — |
| 2 | **Identity** (NID) | PnO | Direct objective; $\partial z/\partial \hat c \equiv -I$ | Any | 1 | 2 | — | Sahoo et al., 2023 |
| 3 | **DFL** | PnO | Direct objective with $z$ detached, $+\ \alpha\cdot\mathrm{MSE}$ | Any | 1 | 3 | $\alpha \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}$ | Wilder et al., 2019 |
| 4 | **SPO+** | PnO | Convex upper bound; solve at $2\hat c - c$ | LP / MILP | 1 | 2 | — | Elmachtoub & Grigas, 2022 |
| 5 | **Blackbox** (DBB) | PnO | Finite-difference on $z$ with step $\lambda$ | LP / MILP | 2 | 3 | $\lambda \in \{0.01, 0.05, 0.1, 0.5, 1\}$ | Vlastelica et al., 2020 |
| 6 | **PG** | PnO | Finite-difference on objective, direction $c$, step $\sigma$ | 12 / 13 $^{\ddagger}$ | 2 | 3 | $\sigma \in \{0.01, 0.05, 0.1, 0.5, 1\}$ | Huang et al., 2024 |
| 7 | **Perturb** (DPO) | PnO | MC smoothing $\mathbb{E}_{\varepsilon}[z(\hat c + \sigma\varepsilon)]$ via reparam. | Any | $n$ | 4 | $\sigma \in \{0.1, 0.5, 1, 2, 5\}$; $n \in \{5, 10, 25, 50, 100\}$ | Berthet et al., 2020 |
| 8 | **DAD** | PnO | Stein / OS-VGC with per-item $\sigma_j = \lVert \hat y_j - y_j\rVert$ | Any | $1 + n$ | 4 | $w_{\mathrm{Stein}} \in \{0.1, 0.5, 1, 2, 5\}$; $n \in \{5, 10, 25, 50, 100\}$ | Gupta et al., 2024 |
| 9 | **pointLTR** | PnO | MSE over pooled solutions: $\sum_{z\in\mathcal{P}} (\hat c\cdot z - c\cdot z)^2$ | Any | 1 | 2 | — | Mandi et al., 2022 |
| 10 | **pairLTR** | PnO | Hinge: $z^{\star}$ vs. rest of pool $\mathcal{P}$ | Any | 1 | 2 | — | Mandi et al., 2022 |
| 11 | **listLTR** | PnO | Softmax cross-entropy over pool, temp. $\tau$ | Any | 1 | 3 | $\tau \in \{0.1, 0.5, 1, 5, 10\}$ | Mandi et al., 2022 |
| 12 | **NCE** | PnO | Contrastive $\sum_{z\in\mathcal{P}}[\hat c\cdot z^{\star} - \hat c\cdot z]$ | Any | 1 | 2 | — | Mulamba et al., 2021 |
| 13 | **LODL** | PnO | Learned local surrogate $\ell_\theta(\hat c, c)$ from $N_s$ offline samples | Any | $N_s$ offline | 3 | $N_s \in \{100, 250, 500, 1000, 2000\}$ | Shah et al., 2022 |
| 14 | **QPTL** | PnO | KKT implicit diff. of regularised QP, temp. $\tau$ | QP only (3 / 13) | 1 | 3 | $\tau \in \{0.1, 0.5, 1, 5, 10\}$ | Donti et al., 2017 |
| 15 | **cpLayer** | PnO | KKT implicit diff. via `cvxpylayer` | Convex only (3 / 13) | 1 | 2 | — | Agrawal et al., 2019 |

---

### Footnotes

$^{\ast}$  The `mse` method in the sweep dispatches to $\mathrm{BCE}$, $\mathrm{CE}$, or $\mathrm{MAE}$ when the problem's two-stage loss so indicates — all four are prediction-loss baselines and share this row.

$^{\dagger}$  **MSE, and in fact every row in this table, is *also* decision-aware via the finite grid search over `(lr, batch-size)` whose winner is picked by decision regret.** This matches the reviewer's observation: "hyper-parameter tuning relative to decision-loss is technically a decision-aware method. It differs from other decision-aware methods in the sense that the space that you search over is finite and smaller, which allows you to use grid search rather than gradient-based optimisation approaches" — a constrained form of model selection that gives MSE a decision-aware advantage often missed in the PtO-vs-PnO literature (see also [Elmachtoub et al., 2025, AISTATS](../misspec.pdf) for a well-specified-regime analysis).

**Known methodological issue (to be fixed — see `docs/committee_feedback_plan.md` #7).** Within each run, the best epoch's checkpoint is selected by *validation* decision regret (`ExpManager.py:599`). However, the *across-config* Phase-1 step (`collect_bench_p1.py`) currently picks the winning `(lr, batch)` by *test* decision regret, since `results.npy[1]` is the test eval (`ExpManager.py:659`). This is a form of test-set leakage through hyper-parameter selection. The reviewer's point about HP grid search being decision-aware still holds — in fact it holds more strongly — but the current sweep's winning configs are test-selected, not val-selected. Planned fix: read best-val-regret rows from `val_logs.csv` to rebuild `bench_p1_best.json`, then re-run Phase 2 on the corrected configs.

$^{\ddagger}$  **PG** is defined for every problem except `shortestpath` (warcraft images). The finite-difference direction $c$ and the detached-solver pattern work for both linear and non-linear objectives (e.g., the sub-modular objective of `budgetalloc`).

### Column notes

- **Family.** `PtO` (Predict-then-Optimise) — loss does not involve the CO solver during training. `PnO` (Predict-and-Optimise) — loss uses the solver or a differentiable surrogate thereof.
- **CO scope.** "Any" covers all 13 benchmark problems (10 synthetic/real + 3 shortest-path). "LP / MILP" means any linear-objective integer programme. "QP only" and "Convex only" are the restricted scopes where a differentiable convex-programming layer is defined — namely `knapsack`, `bipartitematching`, `portfolio`.
- **Solves / batch.** Number of CO-solver invocations per mini-batch element per forward–backward pass. `n` is the Phase-2 MC sample count. `offline` denotes solves paid once upfront (LODL).
- **# tunable HPs.** Counts the hyper-parameters this benchmark treats as tunable: always `lr` and `batch size` (Phase 1), plus any method-specific HPs swept in Phase 2. Methods with higher counts are more at-risk in the small-data regime (reviewer's point).
- **Phase-2 HP swept.** Exact values from `shells/slurm/submit_bench_p2.sh`. The default (Phase-1) value is starred in that script; all values in the sweep are listed here.
- **References.** Short author–year keys; see BibTeX in `refs.bib` (to be added).

### How to read this for the dissertation chapter

1. **Blind vs. gradient decision-aware axis.** MSE is the only row using the *finite-grid* mechanism; rows 2–15 use *gradient-based* decision-awareness.  The reviewer asks us to be explicit about this — the footnote frames it, and the #HPs column quantifies the extra tuning burden.
2. **Applicability axis.** Rows 14, 15 (QPTL, cpLayer) apply to only 3 / 13 problems; PG to 12 / 13.  When reporting average rank across problems, missing entries materially shift results — this table documents why some cells are blank in the benchmark grid.
3. **Cost axis.** For real-time or large-instance deployment, `Solves / batch` is the practical bottleneck.  DPO, DAD, LODL pay the most.
