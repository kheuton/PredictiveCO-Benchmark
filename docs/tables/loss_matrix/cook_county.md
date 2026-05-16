# Loss Matrix — cook_county (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-2, bs=default | 1.1638 | 1.5422 | 2.1765 | 129.5000 | 93.0000 | 0.1953 |
| dfl | lr=5e-3, bs=default, dflalpha=0.001 | 1.8603 | 2.9091 | 4.8087 | 153.0000 | 102.0000 | 0.2012 |
| identity | lr=5e-3, bs=default | 1.8603 | 2.9091 | 4.8087 | 153.0000 | 102.0000 | 0.2012 |
| spo | lr=1e-3, bs=alt | 1.4992 | 2.5265 | 2.5207 | 131.0000 | 90.0000 | 0.2004 |
| nce | lr=1e-3, bs=alt | 2.0460 | 4.1571 | 4.4912 | 171.2500 | 112.0000 | 0.2180 |
| blackbox | lr=5e-3, bs=default, lambd=0.01 | 1.8603 | 2.9091 | 4.8087 | 153.0000 | 102.0000 | 0.2012 |
| pointLTR | lr=5e-3, bs=alt | 1.3136 | 1.8334 | 2.0528 | 145.5000 | 105.0000 | 0.2034 |
| pairLTR | lr=1e-2, bs=default | 2.0905 | 4.1757 | 4.5151 | 170.7500 | 113.0000 | 0.1895 |
| listLTR | lr=1e-2, bs=default, tau=0.1 | 2.0057 | 4.1304 | 4.5167 | 158.5000 | 128.0000 | 0.1953 |
| lodl | lr=1e-3, bs=default, num_samples=250 | 1.1864 | 1.7740 | 1.9192 | 130.2639 | 86.0086 | 0.1734 |
| perturb | lr=1e-3, bs=default, n_samples=50 | 1.4504 | 2.6399 | 2.6973 | 134.2500 | 97.0000 | 0.2107 |
| pg | lr=1e-2, bs=default, sigma=0.5 | 1.3959 | 1.8120 | 2.0721 | 129.5000 | 96.0000 | 0.1990 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=5e-3, bs=default, stein_weight=0.5 | 2.4904 | 4.3515 | 7.4115 | 149.0000 | 102.0000 | 0.1953 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
