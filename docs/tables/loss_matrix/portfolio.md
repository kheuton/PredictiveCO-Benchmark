# Loss Matrix — portfolio (absolute regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=5e-3, bs=default | 0.0005 | 0.0005 | 0.0005 | 0.2367 | 0.1894 | 0.2206 |
| dfl | lr=5e-3, bs=alt, dflalpha=0.1 | 0.4279 | 0.4220 | 0.3894 | 0.2842 | 0.2587 | 0.2761 |
| identity | lr=5e-3, bs=alt | 0.8291 | 0.8182 | 0.8200 | 0.2826 | 0.2714 | 0.2804 |
| spo | lr=1e-2, bs=alt | 0.1863 | 0.1840 | 0.1865 | 0.2348 | 0.2108 | 0.2298 |
| nce | lr=1e-2, bs=default | 0.9827 | 0.9851 | 0.9816 | 0.3095 | 0.2583 | 0.3099 |
| blackbox | lr=5e-3, bs=default, lambd=0.1 | 0.7530 | 0.7519 | 0.7425 | 0.2811 | 0.2588 | 0.2682 |
| pointLTR | lr=5e-3, bs=alt | 0.0006 | 0.0005 | 0.0005 | 0.2415 | 0.1925 | 0.2514 |
| pairLTR | lr=1e-3, bs=default | 0.1293 | 0.1234 | 0.1258 | 0.2657 | 0.2107 | 0.2554 |
| listLTR | lr=1e-3, bs=default, tau=0.5 | 0.0538 | 0.0568 | 0.0538 | 0.2405 | 0.1923 | 0.2323 |
| lodl | lr=5e-3, bs=alt, num_samples=500 | 0.0016 | 0.0012 | 0.0010 | 0.2395 | 0.2088 | 0.2421 |
| perturb | lr=1e-2, bs=default, sigma=1.0 | 0.2977 | 0.3307 | 0.2257 | 0.4195 | 0.3246 | 0.3822 |
| pg | lr=1e-3, bs=default, sigma=0.5 | 0.0253 | 0.0191 | 0.0192 | 0.3833 | 0.3215 | 0.3533 |
| qptl | lr=1e-3, bs=alt, tau=0.1 | 0.9005 | 0.8962 | 0.8967 | 0.2788 | 0.2693 | 0.2861 |
| cpLayer | lr=5e-3, bs=alt | 0.7318 | 0.7392 | 0.7274 | 0.2835 | 0.2637 | 0.2791 |
| dad | lr=1e-3, bs=default, stein_weight=5.0 | 0.0239 | 0.0177 | 0.0182 | 0.3891 | 0.3125 | 0.3492 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
