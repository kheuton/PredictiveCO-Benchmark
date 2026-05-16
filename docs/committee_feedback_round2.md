# Committee Feedback Round 2 --- Tables

Companion to `committee_feedback_summary.md` (round 1, 2026-04-17). Round 2
feedback lives in `../Thesis/thesis_changelog.md` and asks for four things:

1. **Formally define the decision problem solved by each experiment.**
2. **Document how each surrogate is used per experiment** --- which HPs are
   *fixed* vs *tuned on a validation set via decision loss* --- and where any
   tuning-improvement claim is made, **add a head-to-head comparison** of the
   same surrogate with fixed vs val-tuned HPs.
3. **Report decision and prediction training error** alongside the existing
   test metrics, to make overfitting visible.
4. Interpolation-figure revision *(out of scope here --- experiments live
   outside this repo).*

Status as of 2026-04-23: items 1--3 complete (5 LaTeX tables + supporting
data pipeline). Item 4 still open.

The four-of-four work is also val-selected end-to-end: `bench_p1_best_val.json`
drives `submit_bench_p2.sh`, and `collect_bench_p2_val.py` now picks the
Phase-2 best HP per (method, problem) by val regret instead of test regret.
This closes the only remaining test-leakage path that round-1's plan #7
identified.

---

## How to regenerate

```bash
# Pipeline (val-selected end-to-end). Steps 0a-0d build the data; 1-5 emit tex.
python rethink_exp/collect_bench_p1.py --metric val          # bench_p1_best_val.json
python rethink_exp/collect_bench_p2_val.py                   # bench_p2_best_val.json
python rethink_exp/eval_test_pred.py                         # *.json under each run dir
python rethink_exp/collect_loss_matrix.py \
       --p2_best_json bench_p2_best_val.json                 # loss_matrix.json + docs/tables/loss_matrix/*.md

python rethink_exp/table_decision_problems.py                # docs/tables/decision_problems.tex
python rethink_exp/table_methods_hyperparams.py              # docs/tables/methods_hyperparams.tex
python rethink_exp/table_tuning_benefit.py                   # docs/tables/tuning_benefit.tex
python rethink_exp/table_error_grid.py --metric pred         # docs/tables/pred_error.tex      (wide 15x13 grid; backup)
python rethink_exp/table_error_grid.py --metric decision     # docs/tables/decision_error.tex  (wide 15x13 grid; backup)
python rethink_exp/table_error_per_problem.py                # docs/tables/error_per_problem.tex (one sub-table per problem; recommended)
```

Standalone preview: `cd docs/tables && latexmk -pdf main.tex` (compile off the
HPC --- the cluster has no `pdflatex`).

---

## 1. Decision problem per task &nbsp;&nbsp;&#9745;

**What it is.** One row per benchmark problem (13 total) listing the
predicted parameter $\hat y$, the decision variable $z\in\mathcal Z$, the
objective $c(z,y)$ in display math, the constraints, and the train/test
sizes. Notation matches `Thesis/content/chapters/dpo/sec_methods.tex`:

$$z^\star(x,\theta) = \arg\min_{z\in\mathcal Z}\,
  \mathbb{E}_{p_\theta(Y\mid X=x)}[c(z,Y)],\qquad
  c(z,y)=\langle z,y\rangle\ \text{(linear case)}.$$

**Sources.** PtOPnO paper (`../dfl-wiki/raw/PtOPnO/PtOPnO.md`) for the
eight original problems; `openpto/problems/<Class>.py` and
`openpto/config/probs/<problem>.yaml` for the six rerun additions
(`asurv`, `cook_county`, `speed_humps`, `sp_synth`, `sp_planted`,
`shortestpath`).

**Artifacts.**
- Script: `rethink_exp/table_decision_problems.py` (pure formatter; data
  hard-coded in `PROBLEM_SPECS`).
- Output: `docs/tables/decision_problems.tex` (13 rows + header).

---

## 2a. Methods $\times$ hyperparameters &nbsp;&nbsp;&#9745;

**What it is.** One row per method (15 total) splitting hyperparameters into
two columns: *tuned on val* (LR$\times$batch always; method-specific HP for
the eight Phase-2-tunable methods) vs *fixed* (universal training settings
in the caption; method-specific defaults from `default.yaml` in the
right-most column). Selection criterion is uniform across rows: minimum
validation decision regret.

**Load-bearing point.** Makes explicit which knobs the chapter tunes and
which it leaves at canonical defaults. Reading the table top-to-bottom
distinguishes the seven baseline-style methods (`mse, identity, spo, nce,
pointLTR, pairLTR, cpLayer`) that have no method-specific HP from the eight
methods (`dfl, blackbox, qptl, listLTR, lodl, perturb, pg, dad`) that get a
Phase-2 sweep.

**Artifacts.**
- Script: `rethink_exp/table_methods_hyperparams.py`. Cross-validates the
  hard-coded fixed-HP defaults against `openpto/config/models/default.yaml`
  at runtime; aborts on drift.
- Output: `docs/tables/methods_hyperparams.tex` (15 rows + header).

---

## 2b. Benefit of tuning &nbsp;&nbsp;&#9745;

**What it is.** A 15$\times$13 grid where each cell stacks three numbers:

```
 fixed         <- method-specific HP at default, Phase-1 best (LR, batch)
 tuned         <- best HP across the Phase-2 grid (val-selected)
 Delta (%)     <- (fixed - tuned) / |fixed| * 100, positive = tuning helps
```

Cells shaded green when $\Delta\geq 5\%$, red when $\Delta\leq -5\%$.
Per-method mean $\Delta$ across applicable problems is sorted descending in
the caption so the largest tuning benefits surface immediately.

**Load-bearing point.** Directly answers the reviewer's request that any
tuning claim be backed by a head-to-head comparison. For example, DFL on
Knapsack: fixed $\alpha{=}0.1$ gives test regret 0.242, tuned
$\alpha{=}10.0$ gives 0.067 ($+72.3\%$ improvement); LODL on knapsack-real:
fixed $K{=}500$ gives 0.082, tuned $K{=}2000$ gives 0.071 ($+13.4\%$).
Methods with no Phase-2 HP (MSE, Identity, SPO+, NCE, ptLTR, prLTR,
cpLayer) appear as the middle (tuned-only) row with $\Delta{=}{-}{-}$.

**Artifacts.**
- Script: `rethink_exp/table_tuning_benefit.py`. Reads `bench_p2_best_val.json`
  --- both the `winner` block and the `fixed` block written for every Phase-2
  entry by `collect_bench_p2_val.py`.
- Output: `docs/tables/tuning_benefit.tex` (15 method rows $\times$ 13 cols,
  landscape).

---

## 3. Train / val / test error tables &nbsp;&nbsp;&#9745;

Two layouts are produced from the same `loss_matrix.json` source. The
**per-problem layout is recommended for the chapter** -- it's much easier
to read; the wide grid is kept as a one-glance backup.

### 3a. Per-problem layout (recommended)

**What it is.** 13 small tables, one per problem, each 15 method rows x
6 columns: `Pred Train | Pred Val | Pred Test | Reg Train | Reg Val | Reg Test`,
with grouped header (`\\multicolumn` + `\\cmidrule`) separating the two
metric blocks. Per-column precision auto-scales to the column's median
magnitude so even very small values stay visible (e.g.\ MSE pred on
budgetalloc $\approx 0.0004$ rather than rounded to "0.000"). The lowest
test value in each metric block is bolded.

**Load-bearing point.** Surfaces the train-vs-test gap per (method,
problem) without forcing the reader to scan a wide page. Examples worth
flagging in chapter prose: cubic shows a near-zero gap for MSE
(train/val/test $\approx 0.003/0.002/0.004$) -- no overfitting -- whereas
Identity / Blackbox / DAD on the same problem have train/val regret
$\sim 1.5$ and test $\sim 1.6$, indicating high but consistent error
rather than overfit. Conversely on cook\_county the small sample size
visibly inflates val/test regret relative to train for several methods.

**Artifacts.**
- Script: `rethink_exp/table_error_per_problem.py`.
- Output: `docs/tables/error_per_problem.tex` (single file with 13
  `\\begin{table}` blocks, each labelled `tab:err-<problem>`; LaTeX floats
  them).

### 3b. Wide grid (backup)

**What it is.** 15$\times$13 landscape grid, three numbers per cell.
Useful for at-a-glance comparison across problems but dense.

Decision regret is in **absolute** units on all three splits (not directly
comparable across problems). Prediction loss is the problem's two-stage
loss (MSE / BCE), aligned across splits by construction.

**Sources.** `loss_matrix.json` -- canonical 13$\times$15$\times$8 tensor
from `collect_loss_matrix.py` reading per-run `train_logs.csv`,
`val_logs.csv`, `results.npy`, and `test_pred_loss.json`.

**Artifacts.**
- Script: `rethink_exp/table_error_grid.py` (one script, two metrics).
- Outputs: `docs/tables/pred_error.tex`, `docs/tables/decision_error.tex`.
- Standalone preview wrapper: `docs/tables/main.tex` includes both layouts.

---

## Pipeline support (round-2 plumbing)

These changes were necessary to make the round-2 tables val-selected
end-to-end and reproducible:

- **`rethink_exp/collect_bench_p2_val.py`** *(new).* Mirrors
  `collect_bench_p2.py` but selects best HP by min val regret read from
  `val_logs.csv`. For energy `mse/dfl/identity` (which run with
  `--skip_solver_eval` and have no `val_logs.csv`) it falls back to parsing
  `Iter N, val MSE (no solver): X` lines in `log.txt`, mirroring
  `collect_bench_p1.py`'s fallback. Also writes a `fixed` block per
  Phase-2-tunable entry --- the same prefix pattern at the default HP value
  --- so `table_tuning_benefit.py` can compute the head-to-head delta
  without redoing path math. Output: `bench_p2_best_val.json` (170/173
  cells full as of 2026-04-23; 2 partial, 1 missing).

- **`rethink_exp/collect_loss_matrix.py`** *(patched).* Added
  `--p2_best_json bench_p2_best_val.json` so `resolve_best_run` uses the
  pre-resolved val winner instead of doing its own test-regret HP search.
  Added `test_regret_abs` and `test_regret_rel` to the per-cell metrics
  dict so downstream tables can pick the unit they need. Updated the per-
  problem markdown footers to drop the obsolete leakage caveat.

- **`docs/tables/main.tex`** *(new).* Standalone preview that `\input`s all
  five tables. Uses `\providecommand\citep` so it compiles without a `.bib`;
  drop into the thesis tree to pick up natbib + bibliography for real.

---

## Coverage and gaps (2026-04-23)

- `bench_p2_best_val.json`: 170 full cells, 2 partial
  (`lodl/energy 4/5`, `lodl/shortestpath 2/5`), 1 missing
  (`listLTR/shortestpath` --- never submitted).
- `loss_matrix.json`: 172 cells with `test_pred_loss`; 1 diverged (inf).
- `eval_test_pred.py` filled 111 new `test_pred_loss.json` files in this
  round (the val-selected best run differs from the prior test-selected
  best for those cells).

When a Phase-2 winner config changes (e.g. after rerunning failed jobs)
re-spotcheck against source CSVs:

- `mse/knapsack` pred error: matches `tail -1 saved_records/knapsack-gen/mse/<best_prefix>/val_logs.csv` and `test_pred_loss.json`.
- `perturb/budgetalloc` decision: matches `mean(np.load(.../results.npy)[1])`.
- `dad/cook_county` decision: matches the same.
