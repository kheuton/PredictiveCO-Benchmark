# Loss Matrix — cook_county (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=5e-2, bs=default | 1.1663 | 1.6822 | 1.9425 | 134.5000 | 91.0000 | 0.2012 |
| mse_train | lr=1e-1, bs=default | 0.8854 | 2.0816 | 2.6171 | — | — | 0.2794 |
| mse_val | lr=5e-3, bs=default | 1.1578 | 1.5345 | 2.1219 | — | — | 0.1946 |
| dfl | lr=5e-3, bs=default, dflalpha=0.001 | 1.8603 | 2.9091 | 4.8087 | 153.0000 | 102.0000 | 0.2012 |
| identity | lr=5e-3, bs=default | 1.8603 | 2.9091 | 4.8087 | 153.0000 | 102.0000 | 0.2012 |
| spo | lr=1e-3, bs=alt | 1.4992 | 2.5265 | 2.5207 | 131.0000 | 90.0000 | 0.2004 |
| nce | lr=1e-1, bs=alt | 1.6824 | 1.9399 | 2.2390 | 150.2500 | 108.0000 | 0.1836 |
| blackbox | lr=5e-3, bs=default, lambd=0.01 | 1.8603 | 2.9091 | 4.8087 | 153.0000 | 102.0000 | 0.2012 |
| pointLTR | lr=5e-3, bs=alt | 1.3136 | 1.8334 | 2.0528 | 145.5000 | 105.0000 | 0.2034 |
| pairLTR | lr=5e-2, bs=default | 2.4787 | 3.8549 | 3.9948 | 162.2500 | 106.0000 | 0.1975 |
| listLTR | lr=5e-2, bs=alt, tau=0.1 | 1.7942 | 3.1672 | 3.2871 | 184.2500 | 115.0000 | 0.2048 |
| lodl | lr=1e-3, bs=default, num_samples=250 | 1.1864 | 1.7740 | 1.9192 | 130.2639 | 86.0086 | 0.1734 |
| perturb | lr=1e-3, bs=default, n_samples=50 | 1.4504 | 2.6399 | 2.6973 | 134.2500 | 97.0000 | 0.2107 |
| pg | lr=1e-2, bs=default, sigma=0.5 | 1.3959 | 1.8120 | 2.0721 | 129.5000 | 96.0000 | 0.1990 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=5e-3, bs=default, stein_weight=0.5 | 2.4904 | 4.3515 | 7.4115 | 149.0000 | 102.0000 | 0.1953 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
