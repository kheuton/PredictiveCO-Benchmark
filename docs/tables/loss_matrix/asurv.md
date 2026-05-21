# Loss Matrix — asurv (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-1, bs=default | 1.0765 | 0.8930 | 0.8313 | 116.7333 | 116.1667 | 0.6690 |
| mse_train | lr=5e-2, bs=default | 0.9311 | 1.2082 | 2.3832 | — | — | 0.7714 |
| mse_val | lr=1e-1, bs=default | 1.0510 | 0.8224 | 0.6465 | — | — | 0.6899 |
| dfl | lr=1e-3, bs=default, dflalpha=1.0 | 1.3430 | 1.3950 | 1.6069 | 126.8667 | 115.5000 | 0.7147 |
| identity | lr=1e-3, bs=default | 1.2828 | 1.2091 | 1.2990 | 129.2000 | 115.8333 | 0.7296 |
| spo | lr=5e-2, bs=default | 1.5174 | 1.2159 | 0.9577 | 115.4000 | 115.0000 | 0.7843 |
| nce | lr=1e-1, bs=default | 38492.0664 | 18265.6562 | 13292.4004 | 120.0667 | 117.8333 | 0.6431 |
| blackbox | lr=1e-3, bs=default, lambd=0.01 | 1.2828 | 1.2091 | 1.2990 | 129.2000 | 115.8333 | 0.7296 |
| pointLTR | lr=5e-2, bs=default | 2.3943 | 14.3645 | 38.8301 | 142.8000 | 114.0000 | 0.7326 |
| pairLTR | lr=1e-2, bs=default | 1.6701 | 1.0646 | 0.6774 | 123.8000 | 114.6667 | 0.7177 |
| listLTR | lr=5e-2, bs=alt, tau=0.1 | 1.6490 | 1.3552 | 1.3993 | 134.8000 | 117.3333 | 0.6988 |
| lodl | lr=1e-1, bs=default, num_samples=250 | 1.4991 | 1.0755 | 0.8759 | 118.9560 | 114.6827 | 0.6829 |
| perturb | lr=1e-3, bs=default, sigma=1.0 | 1.2889 | 1.0132 | 0.8397 | 129.5333 | 114.5000 | 0.7137 |
| pg | lr=5e-2, bs=default, sigma=0.01 | 28.1501 | 20.8553 | 20.3455 | 124.1333 | 111.8333 | 0.6998 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=5e-3, bs=default, stein_weight=0.1 | 1.2779 | 1.1752 | 1.2127 | 128.5333 | 116.0000 | 0.7326 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
