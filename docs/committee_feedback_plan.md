# Committee Feedback — Response Plan

Reviewer asked for: (1) experiments organized by **data + hypothesis-class settings** (well-spec / mis-spec / small-data / real); (2) a **method categorization table** (note: HP tuning against decision loss is technically decision-aware); (3) reporting **train / val / test** loss for **both predictions and decisions**; (4) real-data emphasis to surface distribution shift / overfitting.

Six planned deliverables, ordered by effort-to-value. Status: ☐ todo · ◐ in progress · ☑ done.

---

## 1. ☑ Method Categorization Table ✅ 2026-04-16

**Goal.** One-page table summarising all 15 benchmark methods across axes that make the decision-blind / decision-aware story explicit, including the reviewer's point that MSE + LR grid search is *technically* decision-aware (finite grid over decision loss).

**Deliverable.**
- `docs/tables/method_categorization.md` (markdown)
- `docs/tables/method_categorization.tex` (LaTeX for chapter)

**Columns (draft).**
| Col | Values |
|---|---|
| Method | 15 methods registered in `wrapper_loss.py` |
| Family | PtO / PnO |
| Decision-aware? | No (blind) / Yes (gradient) / Yes (HP grid only) |
| Surrogate mechanism | none / direct-diff / smoothing / finite-diff / MC / convex-relax / learned |
| Valid CO class | Any / LP only / QP only / TopK / DAG |
| Key Phase 2 HP | dflalpha, lambd, σ, n_samples, τ, stein_weight, — |
| Per-epoch cost proxy | # solver calls per batch × (anything extra) |
| Paper | citation key |

**Plan.**
1. Enumerate methods from `openpto/method/wrapper_loss.py`; cross-reference each model in `openpto/method/Models/`.
2. Fill the table by inspecting each model's `forward` + Phase 2 submit script.
3. **Footnote:** MSE / BCE / CE / MAE are classified as *decision-aware via HP search* because our Phase 1 picks (LR, batch) minimise a finite grid over validation decision regret. This is the reviewer's key reframing.

**Effort.** ~2 hrs, zero compute.

**Open questions.** Exact wording of the "HP-grid decision-aware" footnote — do we want it as a column entry or a table note?

---

## 2. ☑ Problem / Setting Categorization Table ✅ 2026-04-16

**Goal.** Put each of the 13 benchmark problems into one of the reviewer's bins so the chapter can then say "on *well-specified* problems we see X, on *mis-specified* we see Y".

**Deliverable.**
- `docs/tables/problem_categorization.md` + `.tex`

**Columns (draft).**
| Col | Values |
|---|---|
| Problem | 13 names |
| Synth / Real | 2-level |
| Hypothesis class used | linear / dense-MLP / ResNet18 |
| Objective linearity in c | linear / nonlinear (submodular, etc.) |
| Size (train / val / test) | instance counts |
| Specification bin | well / mis / under-determined / real (unknown) |
| Noise in Y | none / additive / real-world |
| Expected pattern | MSE-favoured / DPO-favoured / ambiguous |

**Plan.**
1. For each problem: read `openpto/config/probs/<name>.yaml` + `openpto/problems/<Name>.py` for DGP.
2. For objective linearity: inspect `get_objective` — most are `<c, z>` (linear); `budgetalloc` is submodular (nonlinear); note `portfolio` is quadratic in z but linear in c.
3. Specification bin: set by (hypothesis class capacity) vs (true DGP). `sp_synth` and `sp_planted` are designed-in well/mis contrast — that's our clearest pair.
4. "Expected pattern" column = our hypothesis going in, not the result. Chapter will then compare to observed.

**Effort.** ~3 hrs, zero compute.

**Open questions.** Where does `cubic` fall? Memory says dense MLP can approximate `10(X³ − 0.65X)` → well-specified in principle, and MSE does reach 0.001. But DPO fails numerically, not from mis-spec. Need to decide how to label that in the table — "well-specified; DPO fails for non-mis-spec reasons" footnote?

---

## 3. ☑ Train × Val × Test, Prediction × Decision Loss Matrix ✅ 2026-04-16

**Goal.** For each (problem, method) at its Phase 2 best config, report six numbers: {train, val, test} × {pred MSE, decision regret}. Reveals overfitting (train ≪ val), advertising-vs-implementation mismatches (e.g., DFL not beating MSE on train decision loss), and distribution shift on real data (val ≠ test).

**Deliverable.**
- `rethink_exp/collect_loss_matrix.py` → writes `loss_matrix.json` + markdown tables.
- `rethink_exp/fig_loss_matrix.py` → one figure per problem: 15 methods × 6 numbers as a compact heatmap.

**Inputs available.**
- `train_logs.csv` / `val_logs.csv`: columns = `epoch, obj, loss, pred_loss, eval`. `pred_loss` = prediction MSE; `eval` = decision regret. ✅ have everything for train + val.
- `results.npy`: `[Objs_test_opt, eval_value]` — test decision regret ✅; **test prediction MSE is not saved** ✗.

**Gotcha.** To get test prediction MSE, we need one of:
  - (a) modify `ExpManager.py` to save test `pred_loss` alongside `eval_value` (small one-line change in the `print_metrics(..., "Final", ...)` block near line 650, then re-run Phase 2 — expensive);
  - (b) re-evaluate the saved `tr_pred_best.pt` checkpoint on the test set offline (cheap, but needs glue code to load problem + checkpoint + run one forward pass);
  - (c) skip test pred MSE initially; report train + val for prediction, train + val + test for decision. Reviewer's explicit ask is all six, so we should do (b).

**Plan.**
1. Load `bench_p1_best.json` and Phase 2 results to identify each (problem, method)'s best config directory.
2. For each: parse `train_logs.csv`/`val_logs.csv` final row; parse `results.npy` for test regret.
3. Offline test-MSE pass: write `rethink_exp/eval_test_mse.py` that loads the problem, loads `tr_pred_best.pt`, runs `pred_model(X_test)` → computes MSE vs Y_test. Dump to `test_mse.json` alongside each run.
4. Assemble 13×15×6 tensor → per-problem figure + summary markdown tables.
5. Highlight pathologies: train regret ≪ test regret (overfitting), train pred MSE low but train regret high (prediction-quality ≠ decision-quality), val vs test divergence on real data (dist shift).

**Effort.** ~1 day coding + ~1 hr CPU for the offline pass.

**Open questions.** For perturb/DPO methods, do we use the best-val-regret checkpoint or the final checkpoint? `ExpManager` saves `tr_pred_best.pt` by val regret. That is consistent with what the test number in `results.npy` is computed from, so use it.

---

## 4. ☑ Small-Data Sweep ✅ 2026-04-17

**Goal.** Reviewer says "small-data settings typically bad for methods needing more HP tuning". Test whether DPO / LODL / DAD degrade faster than MSE as training size shrinks.

**Deliverable.**
- `shells/slurm/submit_small_data.sh`
- `rethink_exp/fig_small_data.py` → x = train size, y = test regret, one line per method.

**Design.**
- **Problems.** Two synthetic (variable subsample is trivial) + one real if feasible:
  - `knapsack` (linear CO, dense MLP, standard benchmark).
  - `sp_synth` (well-spec, linear hypothesis, SPO+ DGP) *or* `budgetalloc` (nonlinear, DPO's only win — interesting to stress-test).
  - Real data: splits are hardcoded — subsampling means truncating the train set without touching val/test. Pick `cook_county` (smallest, 4 train years).
- **Methods.** 5: `mse`, `dfl`, `perturb`, `lodl`, `dad`. Covers blind + 4 decision-aware families.
- **Train sizes.** Synthetic: {50, 100, 200, 400}. Real (cook_county already tiny): use temporal truncation — {1 yr, 2 yr, 3 yr, 4 yr}.
- **HP.** Use each method's Phase-1 best (LR, batch) — don't re-tune per size; that's the point (reviewer: small-data punishes more-tuned methods).
- **Job count.** 3 problems × 5 methods × 4 sizes = **60 jobs**.
- **Walltime.** 1–2 hr each for synth, similar for cook_county → single preempt night.

**Code changes needed.**
- Add `--train_subsample_n` (synth) and `--train_subsample_years` (real) flags to `main_results.py`.
- For synth: trim the `(X_train, Y_train, Y_train_aux)` tuple after `problem.get_train_data()`. For `cook_county`: modify the loader to accept a subset of years.

**Plan.**
1. Add the subsample flag(s) — prefer a single `--train_frac` or `--train_n`.
2. Write submit script with the 60-job grid.
3. Dry-run + spot-check one config.
4. Submit to preempt; collect in the morning.
5. Plot with error bars if seed variance is meaningful (single seed=2023 per CLAUDE.md, so probably no bars — caveat in caption).

**Effort.** ~2 days coding + 1 night compute.

**Open questions.** Should we run multiple seeds to get error bars? Benchmark fidelity says `--seed 2023` is fixed. But for small-data, single-seed noise is likely large — a 3-seed sweep on the smallest size would be informative. Decide before submitting.

---

## 5. ☑ Well-Specified vs Mis-Specified Spotlight ✅ 2026-04-16

**Goal.** Single subplot / small table contrasting methods on `sp_synth` (well-specified: SPO+ DGP, linear model, linear objective) vs `sp_planted` (mis-specified: planted arcs DGP, same linear model). Directly addresses reviewer's theoretical point: decision-blind should win on well-spec, decision-aware on mis-spec.

**Deliverable.**
- `rethink_exp/fig_specification_contrast.py` → scatter: x = sp_synth test regret, y = sp_planted test regret, one point per method. MSE should sit in lower-left on sp_synth but upper-right on sp_planted; decision-aware methods should invert.

**Inputs.** Phase 2 results on `sp_synth` + `sp_planted` (via `bench_p2_best.json` equivalent).

**Plan.**
1. Verify Phase 2 is actually complete for these two problems — memory says sp_synth/sp_planted P2 was still pending as of 2026-04-12. **Run `python rethink_exp/sweep_status.py --phase 2` filtered to these problems first.**
2. If incomplete, this plan item is blocked until P2 finishes. Fall back to Phase 1 best configs as a near-substitute.
3. Make the scatter + a small table.
4. Prose in chapter: connect to `arxiv:2011.03030` (Elmachtoub) from reviewer.

**Effort.** Half-day, zero new compute (if P2 is done).

**Open questions.** If only 1 of the 2 problems is P2-complete, show Phase 1 best for the other and annotate.

---

## 6. ☑ Real-Data Interpolation ✅ 2026-04-17

**Goal.** Replicate the knapsack local-min interpolation story on a **real** problem. Reviewer wants more real-data insight; we have a unique mechanistic lens from the knapsack chapter.

**Deliverable.**
- `rethink_exp/fig_realdata_interpolation.py` → α ∈ [−0.5, 1.5], 30 grid points, 6-panel figure (3 regret + 3 pred MSE, across train/val/test). Port of `fig_kn_multi_interpolation.py` to cook_county.
- Output: `results/fig_realdata_interpolation.{png,pdf,npz}`.

**Endpoints.**
- MSE: `bench_p1_mse_default_lr1e-3/tr_pred_best.pt` (test regret 0.188)
- Perturb: `bench_p2_perturb_s1p0_n50_default_lr1e-3/tr_pred_best.pt` (test regret 0.211)

**Result (2026-04-17).**
- Train regret is essentially flat across α ∈ [−0.5, 1.5] (0.31–0.33): the training set is so small (4 timesteps × 1328 tracts) that both endpoints memorize it equivalently.
- **Val regret minimum near α ≈ 0.19** (0.154) — small basin slightly toward Perturb from MSE, but at val the two endpoints are close (0.157 vs 0.161).
- **Test regret minimum near α ≈ −0.16** (0.185): extrapolating *away from Perturb* (i.e., more-MSE direction) is marginally best on test.
- **Pred MSE rises monotonically with α** on val (1.60 → 3.47) and test (1.95 → 3.59): Perturb spends capacity solving for decision quality at the cost of prediction calibration, but on this linear-objective TopK problem that trade doesn't buy decision gains on held-out data.
- Narrative: unlike budgetalloc (nonlinear submodular → DPO wins), cook_county's TopK objective is linear → the MSE→Perturb barrier-less connection is real but the basin is essentially flat and the linear-objective regime favors MSE, consistent with the paper's emerging message.

---

## 7. ☐ Fix test-set leakage in Phase-1 config selection

**Issue (discovered 2026-04-16 while drafting Plan #1).** The across-config step
in Phase 1 — `rethink_exp/collect_bench_p1.py` — picks the winning
`(lr, batch)` per (method, problem) by the *test* decision regret. Trace:

- `results.npy[1]` stores `eval_value` from `ExpManager.py:659`, which is
  `results["test"]["eval"]["value"]` — the **test** eval.
- `collect_bench_p1.load_regret()` reads `results.npy[1]`; `best_config_for()`
  minimises over that.
- `bench_p1_best.json` even stores this under a misleadingly-named `"val"` key
  at `collect_bench_p1.py:174`.

Within-run checkpoint selection (`ExpManager.py:599` → `compare_result` on
`metrics["val"]["eval"]["value"]`) correctly uses val regret. The leakage is
only in the *across-config* Phase-1 step.

**Impact.** Reported Phase-1 best cells, and therefore the Phase-2 HP sweeps
launched from them, are advantaged by test-set HP tuning. The size of the
effect is unknown until we re-collect by val.

**Fix plan (no re-training required).**
1. Add a helper that reads `val_logs.csv` for each run and returns
   `min(eval)` (and `best_epoch`) — this is the best-val-regret achieved
   during training.
2. Modify `collect_bench_p1.py` to select `(lr, batch)` by that val-regret
   instead of the final-test-regret in `results.npy`.
3. Write a second `bench_p1_best_val.json` (don't overwrite the existing
   file immediately — useful to diff).
4. Produce a diff table: val-selected vs test-selected best configs,
   reporting (a) how many cells changed config, (b) how much the reported
   best regret moves when we switch metrics.
5. Once verified, point `submit_bench_p2.sh` at the val-selected JSON and
   re-run Phase 2 for any (method, problem) whose best config changed.

**Why now vs. later.** Later is fine — all data needed for the fix is on
disk; no compute burn to defer. Schedule before re-running Phase 2 for any
paper-table reporting.

**Effort.** ~2 hours for the script + diff table; Phase-2 re-runs depend on
the diff size.

---

## Global dependencies / order

1. (#1, #2) — pure-docs, independent, can be in parallel with everything else.
2. (#3) — depends on verifying Phase 2 best configs; needs offline test-MSE pass.
3. (#5) — depends on Phase 2 being complete for sp_synth + sp_planted. **Check first.**
4. (#4) — depends on a clean `--train_subsample_n` flag; independent of others.
5. (#6) — depends on saved checkpoints from Phase 2. Independent of #3/#4/#5.

**Suggested execution order.** #1 → #2 → **#7 (quick fix; blocks any table-quality claim in Phase 2)** → #3 → #5 → #4 → #6.

Refine each item before implementing.
