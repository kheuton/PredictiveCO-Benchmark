# Loss Matrix — speed_humps (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-2, bs=default | 6.1046 | 4.4247 | 3.0511 | 219.6000 | 169.0000 | 0.1825 |
| dfl | lr=1e-3, bs=default, dflalpha=0.001 | 29.4533 | 21.2432 | 9.7602 | 219.6000 | 190.5000 | 0.2322 |
| identity | lr=1e-3, bs=default | 29.4610 | 21.2498 | 9.7681 | 219.6000 | 190.5000 | 0.2322 |
| spo | lr=1e-2, bs=default | 28.3703 | 22.0816 | 22.3912 | 187.0000 | 150.5000 | 0.1730 |
| nce | lr=1e-3, bs=default | 32.7371 | 23.9847 | 11.0407 | 222.2000 | 184.0000 | 0.2720 |
| blackbox | lr=1e-3, bs=default, lambd=0.01 | 29.4610 | 21.2498 | 9.7681 | 219.6000 | 190.5000 | 0.2322 |
| pointLTR | lr=1e-3, bs=alt | 29.4605 | 21.2456 | 9.7633 | 219.6000 | 190.5000 | 0.2322 |
| pairLTR | lr=1e-2, bs=alt | 35.0805 | 25.9188 | 12.7456 | 220.2000 | 169.0000 | 0.2158 |
| listLTR | lr=1e-2, bs=alt, tau=0.1 | 33.1296 | 24.1378 | 10.9006 | 231.2000 | 188.0000 | 0.3291 |
| lodl | lr=1e-2, bs=default, num_samples=100 | 9.9054 | 7.2576 | 6.2701 | 215.2426 | 190.0336 | 0.2166 |
| perturb | lr=1e-2, bs=default, n_samples=5 | 25.9046 | 18.6799 | 9.3526 | 215.8000 | 184.5000 | 0.2010 |
| pg | lr=1e-3, bs=default, sigma=0.01 | 30.2726 | 21.9238 | 10.3130 | 209.4000 | 184.5000 | 0.2023 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-3, bs=default, stein_weight=0.5 | 30.0663 | 21.7265 | 10.0959 | 224.4000 | 191.0000 | 0.2135 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
