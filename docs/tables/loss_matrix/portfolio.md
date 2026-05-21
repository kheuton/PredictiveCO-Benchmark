# Loss Matrix — portfolio (absolute regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=5e-3, bs=default | 0.0005 | 0.0005 | 0.0005 | 0.2367 | 0.1894 | 0.2206 |
| mse_train | lr=1e-2, bs=alt | 0.0004 | 0.0004 | 0.0004 | — | — | 0.2399 |
| mse_val | lr=5e-2, bs=alt | 0.0004 | 0.0004 | 0.0004 | — | — | 0.2774 |
| dfl | lr=5e-2, bs=alt, dflalpha=0.1 | 0.1292 | 0.1217 | 0.1219 | 0.2788 | 0.2520 | 0.2556 |
| identity | lr=5e-3, bs=alt | 0.8291 | 0.8182 | 0.8200 | 0.2826 | 0.2714 | 0.2804 |
| spo | lr=1e-2, bs=alt | 0.1863 | 0.1840 | 0.1865 | 0.2348 | 0.2108 | 0.2298 |
| nce | lr=1e-2, bs=default | 0.9827 | 0.9851 | 0.9816 | 0.3095 | 0.2583 | 0.3099 |
| blackbox | lr=5e-3, bs=default, lambd=0.1 | 0.7530 | 0.7519 | 0.7425 | 0.2811 | 0.2588 | 0.2682 |
| pointLTR | lr=5e-3, bs=alt | 0.0006 | 0.0005 | 0.0005 | 0.2415 | 0.1925 | 0.2514 |
| pairLTR | lr=1e-3, bs=default | 0.1293 | 0.1234 | 0.1258 | 0.2657 | 0.2107 | 0.2554 |
| listLTR | lr=1e-3, bs=default, tau=0.5 | 0.0538 | 0.0568 | 0.0538 | 0.2405 | 0.1923 | 0.2323 |
| lodl | lr=5e-2, bs=default, num_samples=500 | 0.0021 | 0.0012 | 0.0013 | 0.2574 | 0.2068 | 0.2542 |
| perturb | lr=5e-2, bs=alt, n_samples=25 | 0.9817 | 0.9808 | 0.9787 | 0.2965 | 0.2684 | 0.3082 |
| pg | lr=1e-1, bs=alt, sigma=0.05 | 0.9958 | 0.9961 | 0.9968 | 0.3141 | 0.2590 | 0.3256 |
| qptl | lr=1e-3, bs=alt, tau=0.1 | 0.9005 | 0.8962 | 0.8967 | 0.2788 | 0.2693 | 0.2861 |
| cpLayer | lr=5e-3, bs=alt | 0.7318 | 0.7392 | 0.7274 | 0.2835 | 0.2637 | 0.2791 |
| dad | lr=1e-1, bs=alt, stein_weight=0.1 | 0.9989 | 0.9972 | 0.9991 | 0.2733 | 0.2473 | 0.2821 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
