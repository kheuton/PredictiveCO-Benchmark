# Loss Matrix — knapsack (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-1, bs=default | 1.7255 | 2.6255 | 2.3587 | 1.6187 | 2.3375 | 0.0602 |
| mse_train | lr=5e-2, bs=alt | 1.3896 | 3.5375 | 2.8459 | — | — | 0.0633 |
| mse_val | lr=5e-3, bs=alt | 1.7332 | 2.6009 | 2.3790 | — | — | 0.0580 |
| dfl | lr=5e-2, bs=default, dflalpha=10.0 | 2.4057 | 3.1403 | 2.8560 | 2.0187 | 2.6375 | 0.0637 |
| identity | lr=5e-2, bs=alt | 48922214400.0000 | 50002145280.0000 | 37194768384.0000 | 9.5281 | 10.3250 | 0.2423 |
| spo | lr=5e-3, bs=default | 6.7790 | 8.1349 | 7.0631 | 2.2344 | 2.5875 | 0.0626 |
| nce | lr=1e-3, bs=alt | 26.1129 | 29.6709 | 25.9005 | 5.3937 | 5.4250 | 0.1483 |
| blackbox | lr=1e-2, bs=default, lambd=0.01 | 270.1588 | 281.2812 | 273.3376 | 9.4906 | 10.4125 | 0.2432 |
| pointLTR | lr=1e-2, bs=default | 1.9150 | 2.7931 | 2.4771 | 1.7000 | 2.2875 | 0.0580 |
| pairLTR | lr=1e-2, bs=alt | 26.0200 | 29.5525 | 25.8461 | 2.3719 | 2.7000 | 0.0789 |
| listLTR | lr=1e-2, bs=alt, tau=0.1 | 5.6813 | 7.3346 | 5.9990 | 1.5094 | 2.3375 | 0.0585 |
| lodl | lr=1e-2, bs=default, num_samples=500 | 11.9856 | 14.3657 | 12.4583 | 9.2647 | 10.0849 | 0.2403 |
| perturb | lr=1e-3, bs=alt, n_samples=100 | 16.8900 | 19.6905 | 16.7894 | 2.6094 | 3.5875 | 0.0800 |
| pg | lr=1e-2, bs=alt, sigma=0.1 | 19.9927 | 23.0340 | 20.7002 | 3.7219 | 5.1500 | 0.1114 |
| qptl | lr=5e-2, bs=alt, tau=0.1 | 10093.8076 | 10133.6943 | 10069.9482 | 7.2938 | 7.8750 | 0.1984 |
| cpLayer | lr=1e-2, bs=default | 270.1588 | 281.2812 | 273.3376 | 9.4906 | 10.4125 | 0.2432 |
| dad | lr=1e-3, bs=default, stein_weight=1.0 | 25.4054 | 29.0148 | 25.0377 | 9.7063 | 10.3625 | 0.2389 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
