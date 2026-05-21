# Loss Matrix — budgetalloc (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-3, bs=alt | 0.0003 | 0.0003 | 0.0004 | 0.0392 | 0.0236 | 0.3657 |
| mse_train | lr=1e-3, bs=alt | 0.0003 | 0.0003 | 0.0004 | — | — | 0.3442 |
| mse_val | lr=1e-3, bs=alt | 0.0003 | 0.0002 | 0.0005 | — | — | 0.4202 |
| dfl | lr=1e-3, bs=default, dflalpha=0.001 | 0.2350 | 0.2307 | 0.2341 | 0.0305 | 0.0198 | 0.2282 |
| identity | lr=1e-2, bs=default | 1.2649 | 1.0798 | 1.0411 | 0.0193 | 0.0197 | 0.1465 |
| spo | lr=1e-2, bs=alt | 0.0597 | 0.0481 | 0.0509 | 0.0039 | 0.0025 | 0.0510 |
| nce | lr=5e-3, bs=alt | 0.0878 | 0.0736 | 0.0828 | 0.0154 | 0.0136 | 0.0863 |
| blackbox | lr=5e-3, bs=alt, lambd=0.05 | 0.2810 | 0.2402 | 0.2636 | 0.0193 | 0.0145 | 0.1063 |
| pointLTR | lr=5e-3, bs=default | 0.0012 | 0.0009 | 0.0012 | 0.0044 | 0.0022 | 0.0416 |
| pairLTR | lr=1e-2, bs=alt | 0.0471 | 0.0442 | 0.0459 | 0.0073 | 0.0074 | 0.0786 |
| listLTR | lr=5e-3, bs=default, tau=0.1 | 0.0016 | 0.0014 | 0.0017 | 0.0020 | 0.0007 | 0.0363 |
| lodl | lr=1e-3, bs=default, num_samples=2000 | 0.0266 | 0.0257 | 0.0271 | 0.0566 | 0.0478 | 0.4981 |
| perturb | lr=1e-2, bs=default, sigma=0.5 | 0.6407 | 0.5322 | 0.5697 | 0.0106 | 0.0031 | 0.0731 |
| pg | — | — | — | — | — | — | — |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-3, bs=default, stein_weight=2.0 | 0.2450 | 0.2371 | 0.2428 | 0.0386 | 0.0211 | 0.3221 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
