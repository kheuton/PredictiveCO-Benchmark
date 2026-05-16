# Loss Matrix — sp_synth (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-2, bs=alt | 0.4494 | 0.6362 | 1.0044 | 0.3037 | 0.2666 | 0.0981 |
| dfl | lr=1e-3, bs=alt, dflalpha=10.0 | 1.1445 | 1.3212 | 1.7521 | 1.0352 | 0.9911 | 0.2680 |
| identity | lr=1e-2, bs=default | 3.2452 | 3.7330 | 4.2763 | 2.5174 | 2.8545 | 0.7147 |
| spo | lr=5e-3, bs=default | 2.4391 | 2.8170 | 3.3823 | 0.1426 | 0.1339 | 0.0616 |
| nce | lr=5e-3, bs=default | 49.4541 | 50.7972 | 51.1349 | 0.3706 | 0.2933 | 0.0968 |
| blackbox | lr=1e-3, bs=default, lambd=0.01 | 2.2797 | 2.7447 | 3.2515 | 2.6424 | 3.1142 | 0.7472 |
| pointLTR | lr=1e-2, bs=default | 1.0870 | 1.3365 | 1.7541 | 0.3206 | 0.2586 | 0.0953 |
| pairLTR | lr=1e-3, bs=default | 1.9757 | 2.4026 | 2.9125 | 0.2187 | 0.1925 | 0.0811 |
| listLTR | lr=5e-3, bs=default, tau=0.5 | 2.0081 | 2.4405 | 2.9228 | 0.3364 | 0.1832 | 0.0806 |
| lodl | lr=1e-2, bs=alt, num_samples=250 | 1.5254 | 1.8617 | 2.2620 | 1.9736 | 2.0483 | 0.6055 |
| perturb | lr=1e-2, bs=alt, n_samples=100 | 5.7041 | 6.2613 | 6.9021 | 0.1837 | 0.2016 | 0.0719 |
| pg | lr=1e-2, bs=alt, sigma=0.05 | 2.2257 | 2.6215 | 3.1479 | 0.3274 | 0.2880 | 0.1029 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-3, bs=default, stein_weight=2.0 | 2.2880 | 2.7530 | 3.2608 | 2.6386 | 3.1046 | 0.7501 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
