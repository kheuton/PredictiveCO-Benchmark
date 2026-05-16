# Loss Matrix — knapsack (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-2, bs=alt | 1.8132 | 2.6829 | 2.3843 | 1.6906 | 2.3750 | 0.0544 |
| dfl | lr=1e-2, bs=default, dflalpha=10.0 | 2.6247 | 3.4682 | 3.0091 | 2.0969 | 2.8125 | 0.0671 |
| identity | lr=1e-2, bs=alt | 119.7241 | 129.2204 | 120.4049 | 9.3875 | 10.3500 | 0.2385 |
| spo | lr=5e-3, bs=default | 6.7790 | 8.1349 | 7.0631 | 2.2344 | 2.5875 | 0.0626 |
| nce | lr=1e-3, bs=alt | 26.1129 | 29.6709 | 25.9005 | 5.3937 | 5.4250 | 0.1483 |
| blackbox | lr=1e-2, bs=default, lambd=0.01 | 270.1588 | 281.2812 | 273.3376 | 9.4906 | 10.4125 | 0.2432 |
| pointLTR | lr=1e-2, bs=default | 1.9150 | 2.7931 | 2.4771 | 1.7000 | 2.2875 | 0.0580 |
| pairLTR | lr=1e-2, bs=alt | 26.0200 | 29.5525 | 25.8461 | 2.3719 | 2.7000 | 0.0789 |
| listLTR | lr=1e-2, bs=alt, tau=0.1 | 5.6813 | 7.3346 | 5.9990 | 1.5094 | 2.3375 | 0.0585 |
| lodl | lr=1e-2, bs=default, num_samples=500 | 11.9856 | 14.3657 | 12.4583 | 9.2647 | 10.0849 | 0.2403 |
| perturb | lr=1e-3, bs=alt, n_samples=100 | 16.8900 | 19.6905 | 16.7894 | 2.6094 | 3.5875 | 0.0800 |
| pg | lr=1e-2, bs=alt, sigma=0.1 | 19.9927 | 23.0340 | 20.7002 | 3.7219 | 5.1500 | 0.1114 |
| qptl | lr=1e-2, bs=alt, tau=0.1 | 339.0719 | 346.1346 | 345.0517 | 7.2844 | 8.1875 | 0.2007 |
| cpLayer | lr=1e-2, bs=default | 270.1588 | 281.2812 | 273.3376 | 9.4906 | 10.4125 | 0.2432 |
| dad | lr=1e-3, bs=default, stein_weight=1.0 | 25.4054 | 29.0148 | 25.0377 | 9.7063 | 10.3625 | 0.2389 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
