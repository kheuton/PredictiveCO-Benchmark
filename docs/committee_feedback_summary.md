# Committee Feedback — Results Summary

Companion to `docs/committee_feedback_plan.md`. The reviewer asked for four things:
(1) experiments organized by data + hypothesis-class setting; (2) a method
categorization table; (3) train/val/test loss for both predictions and decisions;
(4) real-data emphasis surfacing distribution shift / overfitting. Six deliverables
were scoped; all six are complete. This document summarizes what each produced,
where the artifacts live, and the load-bearing finding.

Status as of 2026-04-17: **6 / 6 complete.** Plan #7 (test-leakage fix in
`collect_bench_p1.py`) remains open and gates final table reporting.

---

## 1. Method categorization table ☑ 2026-04-16

**What it is.** One-page reference classifying all 15 benchmark methods across
family (PtO / PnO), decision-awareness (blind / gradient / HP-grid-only),
surrogate mechanism, valid CO class, key Phase-2 HP, per-epoch cost proxy,
and paper citation.

**Load-bearing point.** MSE / BCE / CE / MAE are labelled *decision-aware via
HP grid* — Phase 1 selects (LR, batch) on a finite grid of validation decision
regret, so every "prediction-only" method in the chapter is technically a
coarse decision-aware method. This directly addresses the reviewer's reframing.

**Artifacts.**
- `docs/tables/method_categorization.md`
- `docs/tables/method_categorization.tex`

---

## 2. Problem / setting categorization table ☑ 2026-04-16

**What it is.** 13 benchmark problems binned by synth/real, hypothesis class,
objective linearity, size, specification regime, and expected MSE-vs-DPO
pattern.

**Load-bearing point.** Enables chapter prose of the form *"on
well-specified problems X, on mis-specified problems Y"*. Separates the
`sp_synth` / `sp_planted` pair as the designed-in well-vs-mis contrast, and
flags `cubic` as a special case: dense MLP can approximate the true DGP
(well-specified in principle), but DPO fails numerically, not from
mis-specification.

**Artifacts.**
- `docs/tables/problem_categorization.md`
- `docs/tables/problem_categorization.tex`

---

## 3. Train × Val × Test, prediction × decision loss matrix ☑ 2026-04-16

**What it is.** For each of 13 problems × 15 methods at Phase-2 best config,
report six numbers: {train, val, test} × {pred MSE, decision regret}. Test
pred MSE — which `ExpManager` does not save — is filled in by an offline
eval of `tr_pred_best.pt` against the test set.

**Load-bearing point.** Makes advertising-vs-implementation mismatches visible:
e.g. DFL beats MSE only on a minority of problems even though it is the
canonical "decision-aware" method; real-data problems show sharp val→test
regret gaps (distribution shift); and on `budgetalloc` DPO wins by ~11× on
test decision regret (0.056 vs 0.366), consistent with the nonlinear /
linear objective split.

**Artifacts.**
- `rethink_exp/collect_loss_matrix.py` (collector + writer)
- `rethink_exp/fig_loss_matrix.py` (per-problem heatmap)
- `loss_matrix.json` (raw data, 13 × 15 × 6)
- `docs/tables/loss_matrix/summary.md` (headline test-regret table)
- `docs/tables/loss_matrix/{asurv, bipartitematching, budgetalloc, cook_county, cubic, energy, knapsack, knapsack-real, portfolio, shortestpath, speed_humps, sp_planted, sp_synth}.md` (per-problem detail)
- `results/fig_loss_matrix/{problem}.{png,pdf}` (13 heatmaps)
- `results/fig_loss_matrix_all.png` (combined)

**Caveat.** Configs are Phase-1 test-selected; Plan #7 fix pending.

---

## 4. Small-data sweep ☑ 2026-04-17

**What it is.** 3 problems × 5 methods × 4 train-sizes = **60 SLURM jobs**,
each method at its Phase-1-best (LR, batch). Train-size grid: {50, 100, 200,
320} instances for knapsack and sp_synth; {1, 2, 3, 4} years for cook_county.
A single `--train_subsample_n` flag trims axis 0 of `get_train_data()` —
works uniformly for synthetic instances and CookCounty timesteps.

**Load-bearing point.** Reviewer hypothesis: methods needing more HP tuning
degrade faster at small N. Observed:
- **MSE is broadly robust** at small N — on knapsack its curve is nearly
  flat from N=50 to N=320.
- **Perturb on cook_county is counter-intuitively worse with more data**
  (regret 0.42 at 1 yr → 0.48 at 4 yrs). Not a bug; a real finding — flagged
  for follow-up.
- **LODL is the most small-N-sensitive** method across all three problems.

**Artifacts.**
- `shells/slurm/submit_small_data.sh` (60-job sweep, Phase-1 best HPs)
- `rethink_exp/fig_small_data.py` (3-panel plot + console summary table)
- `openpto/config/utils_conf.py` — added `--train_subsample_n` flag
- `rethink_exp/main_results.py` — subsample wrapper (axis-0 slice)
- `results/fig_small_data.{png,pdf}`
- `saved_records/{prob}-{version}/{method}/small_n{size}_*` (60 run dirs)

---

## 5. Well-specified vs mis-specified spotlight ☑ 2026-04-16

**What it is.** `sp_synth` (mis-spec: degree-6 polynomial DGP, 1-layer
linear head) vs `sp_planted` (well-spec on the decision-relevant edges:
linear DGP on 2 "risky" arcs, polynomial on irrelevants). 2-panel scatter
(x = sp_synth test regret, y = sp_planted test regret), plus table.

**Load-bearing point.** The naive theoretical prediction (Elmachtoub et
al. 2025) is that MSE Δ = `synth − planted` should be **positive** (well-spec
easier for prediction-only) and decision-aware methods should flip Δ
negative. What we see:

| Method | sp_synth | sp_planted | Δ |
|---|---:|---:|---:|
| SPO+    | 0.056 | 0.052 | +0.004 |
| Perturb | 0.072 | 0.106 | **−0.034** |
| MSE     | 0.096 | 0.102 | **−0.006** |
| DFL     | 0.488 | 0.228 | +0.260 |
| Identity| 0.715 | 0.402 | +0.313 |
| DAD     | 0.747 | 0.430 | +0.317 |

- **MSE Δ is mildly negative** (−0.006) — theoretical prediction fails here;
  MSE is slightly *worse* on well-spec than on mis-spec.
- **SPO+ wins both regimes** (0.056 / 0.052) — the cleanest decision-aware
  method.
- **Identity / Blackbox / DAD tank on sp_synth** (0.71–0.75): their
  gradients are dominated by an irrelevant signal when the DGP is nonlinear.

The real message: decision-aware methods don't universally win on
mis-spec; which method wins depends on whether its surrogate can route
gradient through the solver for this DAG structure.

**Artifacts.**
- `rethink_exp/fig_specification_contrast.py`
- `results/fig_specification_contrast.{png,pdf}`
- `docs/tables/specification_contrast.md`

---

## 6. Real-data interpolation ☑ 2026-04-17

**What it is.** Port of `fig_kn_multi_interpolation.py` to cook_county: the
MSE↔Perturb endpoints are connected by
θ(α) = (1−α)·θ_MSE + α·θ_Perturb for α ∈ [−0.5, 1.5], 30 grid points. At
each α evaluate on train/val/test: relative decision regret (TopK solver)
and prediction MSE. 6-panel figure.

**Load-bearing point.** Replicates the knapsack "no-barrier local-min"
structure on a real, noisy, time-series problem, but with a different
headline finding:
- **No barrier** between MSE and Perturb — landscape is smooth across
  α ∈ [−0.5, 1.5].
- **Train regret is essentially flat** (0.31–0.33): with only 4 timesteps ×
  1328 tracts, both endpoints memorize the training set equivalently.
- **Test regret minimum near α ≈ −0.16** (0.185) — slightly *beyond MSE
  away from Perturb*. Perturb endpoint is 0.210.
- **Pred MSE rises monotonically with α** on val (1.60 → 3.47) and test
  (1.95 → 3.59): Perturb trades prediction calibration for decision
  quality, but on this linear-objective TopK problem the trade buys nothing
  on held-out data.
- **Consistent with the emerging paper message** (`docs/paper_message_apr2026.md`):
  DPO only beats MSE when the CO objective is nonlinear in predicted costs.
  cook_county is linear → MSE wins, even though there is no barrier
  between the two solutions.

**Artifacts.**
- `rethink_exp/fig_realdata_interpolation.py`
- `results/fig_realdata_interpolation.{png,pdf,npz}`

**Endpoints used.**
- MSE: `saved_records/cook_county-real/mse/bench_p1_mse_default_lr1e-3/checkpoints/tr_pred_best.pt` (test 0.188)
- Perturb: `saved_records/cook_county-real/perturb/bench_p2_perturb_s1p0_n50_default_lr1e-3/checkpoints/tr_pred_best.pt` (test 0.211)

---

## Cross-cutting caveat: Plan #7 (open)

All reported numbers in Plans #3 / #5 / #6 depend on Phase-1 (LR, batch)
configs that were selected by **test** decision regret instead of val.
Within-run checkpoint selection is val-correct; only the across-config
Phase-1 step leaks. Full trace and fix plan in Plan #7 of
`docs/committee_feedback_plan.md` and `memory/phase1_hp_test_leakage.md`.

Fixing this does not require re-training — only re-reading `val_logs.csv`
and re-collecting. Scheduled before any paper-table-quality claim.

---

## Artifact index (flat list)

```
docs/
  committee_feedback_plan.md                         # master plan (this doc's parent)
  committee_feedback_summary.md                      # <-- this file
  paper_message_apr2026.md                           # linearity→δ narrative
  tables/
    method_categorization.{md,tex}                   # Plan #1
    problem_categorization.{md,tex}                  # Plan #2
    specification_contrast.md                        # Plan #5
    loss_matrix/
      summary.md                                     # Plan #3 headline
      {13 problems}.md                               # Plan #3 detail

rethink_exp/
  collect_loss_matrix.py                             # Plan #3
  fig_loss_matrix.py                                 # Plan #3
  fig_small_data.py                                  # Plan #4
  fig_specification_contrast.py                      # Plan #5
  fig_realdata_interpolation.py                      # Plan #6

shells/slurm/
  submit_small_data.sh                               # Plan #4 (60 jobs)

openpto/config/utils_conf.py                         # Plan #4 — added --train_subsample_n
rethink_exp/main_results.py                          # Plan #4 — subsample wrapper

results/
  fig_loss_matrix/{problem}.{png,pdf}                # Plan #3 (13 per-problem)
  fig_loss_matrix_all.png                            # Plan #3 combined
  fig_small_data.{png,pdf}                           # Plan #4
  fig_specification_contrast.{png,pdf}               # Plan #5
  fig_realdata_interpolation.{png,pdf,npz}           # Plan #6
loss_matrix.json                                     # Plan #3 raw
```
