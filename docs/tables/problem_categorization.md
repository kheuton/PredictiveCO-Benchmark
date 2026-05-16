# Table: Benchmark Problems Categorised

The 13 benchmark problems cover four specification bins relevant to the
reviewer's feedback: *well-specified* (hypothesis class contains the true
DGP), *mis-specified* (hypothesis class cannot represent it),
*under-determined / small-data* (limited supervision for a rich mapping),
and *real-world unknown* (DGP unknown; we report observed outcomes).

Symbols are in LaTeX math so the table renders identically in Markdown
with math support and when included in a LaTeX manuscript.

---

| # | Problem | Data | DGP (true $c = f(x)$) | Pred. model | Obj.\ class | $\partial_c$ linear? | Train / Val / Test | Specification bin | Expected DPO vs. MSE |
|---|---|---|---|---|---|---|---|---|---|
| 1  | `knapsack`           | Synthetic $^{a}$       | $c_i = 5 \cdot [(B x_i/\sqrt{p} + 3)/3.5]^{7}\cdot(1+\eta_i)$, $\eta\!\sim\!\mathcal{U}(-0.5,0.5)$, $B_{ij}\!\sim\!\mathrm{Bern}(0.5)$ fixed, $x\!\sim\!\mathcal{N}(0,I)$ | Dense MLP | 0/1 KP (LP-relax.\ OK via DP)            | Yes ($\langle c,z\rangle$)      | 320 / 80 / 200       | **Well-specified**$^{b}$           | MSE tied or wins (linear obj; capacity ample) |
| 2  | `knapsack-real`      | Real (UCI kp800) | Unknown                                                                                                                                                                   | Dense MLP | 0/1 KP                                    | Yes                             | real splits          | **Real / unknown**                 | Empirical                                      |
| 3  | `cubic`              | Synthetic              | $c = 10(x^{3} - 0.65 x)$, $x\!\sim\!\mathcal{U}(-1,1)$, **noise-free**                                                                                                    | Dense MLP | Top-$k$ on cube                           | Yes                             | 200 / 50 / 400       | **Well-specified** (noise-free)    | MSE wins (zero irreducible risk)              |
| 4  | `bipartitematching`  | Semi-synthetic         | Cora citation features $x$; edge labels via Bernoulli $\to$ BCE two-stage target                                                                                           | Dense MLP | LP (integer sol.\ via Gurobi)             | Yes                             | 16 / 4 / 6 graphs    | **Small-data, real features**      | High variance; either can win                 |
| 5  | `budgetalloc`        | Synthetic              | Linear click-probabilities $y_{ij}\!\in\![0,1]$                                                                                                                           | Dense MLP | Max $\sum_i w_i\bigl(1 - \prod_j (1 - z_j y_{ij})\bigr)$ (submodular) | **No** (nonlinear in $y$) | 320 / 80 / 200       | **Mis-specified by objective**$^{c}$ | DPO wins (observed: $\sim\!11\times$)         |
| 6  | `portfolio`          | Synth.\ returns        | Linear factor model over features                                                                                                                                         | Dense MLP | Markowitz QP: $\max\langle c,z\rangle - \alpha\,z^\top \Sigma z$ | Yes (linear in $c$, quad.\ in $z$) | 320 / 80 / 200       | **Well-specified**                 | MSE tied                                      |
| 7  | `sp_synth`           | Synthetic (SPO+ paper) | Polynomial degree 6: $c = \bigl((Bx/\sqrt{p} + 3)/3.5\bigr)^{6}\cdot(1+\eta)$ on $5\!\times\!5$ grid                                                                        | Dense, **1 layer** | Shortest path (DAG DP)                     | Yes                             | 320 / 80 / 10000     | **Mis-specified by design**$^{d}$  | Gradient methods win (DPO/SPO+/DFL)           |
| 8  | `sp_planted`         | Synthetic (PG paper)   | Two-path planted costs on $5\!\times\!5$ grid: linear in features                                                                                                        | Dense, **1 layer** | Shortest path (DAG DP)                     | Yes                             | 320 / 80 / 10000     | **Well-specified by design**$^{d}$ | MSE tied                                      |
| 9  | `shortestpath`       | Real images (Warcraft 12$\times$12) | Image $\to$ true vertex weights (dataset-provided labels)                                                                                                                 | ResNet18 | Shortest path (Dijkstra, 8-grid)           | Yes                             | 8000 / 2000 / 1000   | **Under-determined (rich mapping)**| Deep net + many samples $\to$ MSE competitive |
| 10 | `energy`             | Real (ICON prices)     | Real 48-period day-ahead electricity prices                                                                                                                              | Dense MLP | Unit-commitment MILP                       | Yes                             | 550 / 100 / $\sim$150| **Real / unknown**                 | Empirical                                      |
| 11 | `asurv` (aerial surveillance) | Real (NYC) | Real lat/long + temporal features; target = observed counts                                                                                                              | Dense MLP | Top-$k$ ($k=50$) over 1338 locations      | Yes                             | 15 / 6 / 6 periods   | **Real / unknown (temporal)**      | Empirical                                      |
| 12 | `cook_county`        | Real (Cook County)     | 13 features (SVI themes, lagged deaths, coords) $\to$ tract-level opioid deaths                                                                                          | Dense MLP | Top-$k$ ($k=100$) over 1328 tracts        | Yes                             | 4 / 1 / 2 years      | **Real / unknown (distribution shift)** | Empirical                                  |
| 13 | `speed_humps`        | Real (NYC)             | 9 features $\to$ tract-level pedestrian injury counts                                                                                                                    | Dense MLP | Top-$k$ ($k=107$) over 2107 tracts        | Yes                             | 5 / 2 / 4 years      | **Real / unknown (distribution shift: 2020–23)** | Empirical                          |

---

### Footnotes

$^{a}$ *Knapsack synthetic DGP.* `openpto/problems/Knapsack.py` uses $c_i = 5\cdot [(B x_i/\sqrt{p} + 3)/3.5]^{\mathrm{poly\_deg}}\cdot(1+\eta_i)$ with $\mathrm{poly\_deg}=7$, $p=5$ features, multiplicative uniform noise of width $0.5$; $B$ is a fixed $\mathrm{Bernoulli}(0.5)$ random projection. A dense 2-layer MLP is an approximate universal approximator for this polynomial on the observed support, modulo the irreducible noise.

$^{b}$ "Well-specified" here is in the sense of *hypothesis-class expressivity* — the pred. model can represent the conditional mean. There is still irreducible noise, so test regret is lower-bounded by the Bayes risk; we observe a $\sim$0.052 oracle ceiling (see `memory/oracle_ceiling.md`).

$^{c}$ `budgetalloc` is the clearest **mis-specification case by *objective*** rather than by hypothesis class: the true click-probabilities $y$ are predictable, but the CO objective $1 - \prod_j (1 - z_j y_{ij})$ is nonlinear in $y$, so errors in $\hat y$ compound non-linearly in the decision. Per Elmachtoub et al.\ (2025, AISTATS — `../misspec.pdf`), this is the regime where decision-aware surrogates most clearly beat MSE. Observed: DPO $\approx$ 0.032 vs.\ best-MSE 0.366 (Phase-2 sweep, $\sim$11$\times$ gap).

$^{d}$ `sp_synth` and `sp_planted` are the **designed-in well/mis contrast** from the SPO+ and PG papers respectively: same 5$\times$5 grid shortest-path problem with the same *linear* prediction head (`--n_layers 1`), but the true $c(x)$ is polynomial degree 6 in `sp_synth` (mis-spec) vs.\ linear in `sp_planted` (well-spec). This is our cleanest pair for isolating the *hypothesis-class* mis-spec effect.

### Column notes

- **DGP.** The true cost function $c = f(x)$ in data-generation code. "Unknown" for real-data tasks where only observations exist.
- **Pred.\ model.** Default architecture used in the sweep; see `shells/slurm/submit_bench_p1.sh` and `openpto/config/probs/*.yaml`.
- **Obj.\ class.** The combinatorial structure (LP, MILP, QP, shortest path, top-$k$, submodular max).
- **$\partial_c$ linear?** Whether the objective is linear in the predicted coefficients $c$ — the key distinction for the Elmachtoub et al.\ (2025) decision-loss analysis. Only `budgetalloc` is nonlinear.
- **Train / Val / Test.** After the 20% val split off the training instances. "$\sim$" flags datasets with fixed real splits where the exact count depends on the data file.
- **Specification bin.** Our categorisation: *well-specified* (hypothesis class expressive enough), *mis-specified* (by hypothesis class or by objective), *small-data* (few instances relative to mapping complexity), *real / unknown* (no controlled DGP).
- **Expected DPO vs.\ MSE.** *Hypothesis going in,* not the observed result. The dissertation chapter will compare expected vs.\ observed.

### How to read this for the dissertation chapter

1. **Controlled mis-spec contrast.** `sp_synth` vs.\ `sp_planted` is the cleanest in-benchmark comparison: same problem, same architecture, DGP flipped. If DPO beats MSE on `sp_synth` but not `sp_planted`, that's direct evidence for the Elmachtoub story.
2. **Nonlinear-objective mis-spec.** `budgetalloc` stands alone as the non-linear-objective case; the $\sim$11$\times$ DPO advantage here is the largest in the benchmark.
3. **Real / unknown.** Rows 10–13 are where distribution-shift questions live — `speed_humps` (2020–23 test fold), `cook_county` (year-based split), `asurv` (temporal). These are the chapters' real-world evidence, not the synthetic ones.
4. **Well-specified baseline.** `knapsack`, `cubic`, `portfolio`, `sp_planted` — methods should converge; large gaps flag optimisation failure, not representational mis-spec.
