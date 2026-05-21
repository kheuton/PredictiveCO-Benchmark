# Loss Matrix — sp_synth (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-1, bs=alt | 0.4578 | 0.6317 | 1.0110 | 0.2906 | 0.2424 | 0.0899 |
| mse_train | lr=5e-3, bs=alt | 0.4492 | 0.6363 | 1.0035 | — | — | 0.0997 |
| mse_val | lr=1e-1, bs=alt | 0.4560 | 0.6236 | 1.0137 | — | — | 0.1022 |
| dfl | lr=1e-3, bs=alt, dflalpha=10.0 | 1.1445 | 1.3212 | 1.7521 | 1.0352 | 0.9911 | 0.2680 |
| identity | lr=1e-2, bs=default | 3.2452 | 3.7330 | 4.2763 | 2.5174 | 2.8545 | 0.7147 |
| spo | lr=5e-3, bs=default | 2.4391 | 2.8170 | 3.3823 | 0.1426 | 0.1339 | 0.0616 |
| nce | lr=5e-3, bs=default | 49.4541 | 50.7972 | 51.1349 | 0.3706 | 0.2933 | 0.0968 |
| blackbox | lr=1e-1, bs=alt, lambd=0.01 | 95.6391 | 96.5169 | 98.0549 | 2.4345 | 3.0959 | 0.7544 |
| pointLTR | lr=1e-1, bs=default | 1.2082 | 1.5156 | 1.9280 | 0.4176 | 0.2160 | 0.1152 |
| pairLTR | lr=1e-1, bs=default | 71.5766 | 75.2536 | 76.2338 | 0.1936 | 0.1770 | 0.0656 |
| listLTR | lr=5e-2, bs=alt, tau=1 | 2.0957 | 2.5388 | 3.0156 | 0.2226 | 0.1686 | 0.0681 |
| lodl | lr=1e-1, bs=alt, num_samples=500 | 1.7224 | 2.0441 | 2.4782 | 1.7981 | 1.7414 | 0.4908 |
| perturb | lr=5e-2, bs=default, n_samples=100 | 4.2195 | 4.6103 | 5.3240 | 0.1978 | 0.2728 | 0.0755 |
| pg | lr=1e-2, bs=alt, sigma=0.05 | 2.2257 | 2.6215 | 3.1479 | 0.3274 | 0.2880 | 0.1029 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=5e-2, bs=alt, stein_weight=5.0 | 1843.5302 | 1760.9943 | 1737.0659 | 2.3234 | 2.3412 | 0.7093 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
