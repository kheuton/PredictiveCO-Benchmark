# Loss Matrix — pg_misspec (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=5e-2, bs=default | 0.4272 | 0.3730 | 0.5613 | 0.0987 | 0.0738 | 0.3916 |
| mse_train | lr=5e-2, bs=default | 0.3200 | 0.2855 | 0.4209 | — | — | 0.6139 |
| mse_val | lr=5e-2, bs=alt | 0.3211 | 0.2846 | 0.4258 | — | — | 0.5962 |
| dfl | lr=5e-3, bs=alt, dflalpha=0.01 | 0.4594 | 0.4081 | 0.5966 | 0.1179 | 0.1003 | 0.5229 |
| identity | lr=1e-2, bs=default | 1.2661 | 1.3220 | 1.3923 | 0.2400 | 0.2441 | 1.0000 |
| spo | lr=5e-2, bs=default | 0.5808 | 0.5267 | 0.7343 | 0.0998 | 0.0646 | 0.3888 |
| nce | lr=1e-2, bs=default | 1.4399 | 1.3011 | 1.6341 | 0.3102 | 0.2609 | 1.4080 |
| blackbox | lr=1e-2, bs=default, lambd=0.01 | 1.2137 | 1.2653 | 1.3397 | 0.2400 | 0.2441 | 1.0000 |
| pointLTR | lr=5e-2, bs=default | 0.4612 | 0.4039 | 0.6007 | 0.0998 | 0.0646 | 0.3890 |
| pairLTR | lr=1e-2, bs=default | 0.4587 | 0.4083 | 0.5955 | 0.0998 | 0.0646 | 0.3889 |
| listLTR | lr=1e-2, bs=alt, tau=1 | 0.4502 | 0.3991 | 0.5861 | 0.0907 | 0.0646 | 0.3898 |
| lodl | lr=1e-1, bs=alt, num_samples=500 | 0.4824 | 0.4153 | 0.6100 | 0.0984 | 0.0711 | 0.3943 |
| perturb | lr=5e-2, bs=alt, sigma=0.5 | 6.3869 | 6.5992 | 7.1793 | 0.0998 | 0.0646 | 0.3893 |
| pg | lr=1e-2, bs=default, sigma=0.01 | 1.2137 | 1.2653 | 1.3397 | 0.2400 | 0.2441 | 1.0000 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-2, bs=default, stein_weight=0.1 | 1.2661 | 1.3220 | 1.3923 | 0.2400 | 0.2441 | 1.0000 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
