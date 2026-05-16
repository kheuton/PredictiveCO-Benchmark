# Loss Matrix — asurv (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-3, bs=default | 1.1206 | 0.8904 | 0.6760 | 120.3333 | 116.5000 | 0.6968 |
| dfl | lr=1e-3, bs=default, dflalpha=1.0 | 1.3430 | 1.3950 | 1.6069 | 126.8667 | 115.5000 | 0.7147 |
| identity | lr=1e-3, bs=default | 1.2828 | 1.2091 | 1.2990 | 129.2000 | 115.8333 | 0.7296 |
| spo | lr=1e-3, bs=alt | 1.4956 | 0.9950 | 0.7372 | 130.8000 | 115.8333 | 0.7107 |
| nce | lr=1e-2, bs=default | 32.3737 | 17.2839 | 12.9114 | 117.8000 | 118.6667 | 0.6451 |
| blackbox | lr=1e-3, bs=default, lambd=0.01 | 1.2828 | 1.2091 | 1.2990 | 129.2000 | 115.8333 | 0.7296 |
| pointLTR | lr=1e-3, bs=default | 1.2896 | 1.0709 | 0.9544 | 121.7333 | 116.3333 | 0.6998 |
| pairLTR | lr=1e-2, bs=default | 1.6701 | 1.0646 | 0.6774 | 123.8000 | 114.6667 | 0.7177 |
| listLTR | lr=5e-3, bs=default, tau=0.1 | 1.7127 | 1.0420 | 0.6938 | 161.6667 | 127.5000 | 0.7038 |
| lodl | lr=1e-3, bs=default, num_samples=100 | 1.2709 | 0.9593 | 0.7253 | 130.5278 | 116.3979 | 0.7227 |
| perturb | lr=1e-3, bs=default, sigma=1.0 | 1.2889 | 1.0132 | 0.8397 | 129.5333 | 114.5000 | 0.7137 |
| pg | lr=1e-2, bs=default, sigma=0.1 | 4.2984 | 2.7888 | 2.0712 | 118.8000 | 114.1667 | 0.6849 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=5e-3, bs=default, stein_weight=0.1 | 1.2779 | 1.1752 | 1.2127 | 128.5333 | 116.0000 | 0.7326 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
