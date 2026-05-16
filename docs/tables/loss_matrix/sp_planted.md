# Loss Matrix — sp_planted (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-2, bs=alt | 0.3746 | 0.5086 | 0.6680 | 0.9582 | 0.9819 | 0.1022 |
| dfl | lr=1e-3, bs=default, dflalpha=10.0 | 2.1672 | 2.3081 | 2.4831 | 2.9419 | 2.8020 | 0.3035 |
| identity | lr=1e-3, bs=default | 3.7634 | 4.0507 | 4.1350 | 4.0434 | 4.0079 | 0.4265 |
| spo | lr=5e-3, bs=default | 3.6551 | 3.9443 | 3.9765 | 0.5118 | 0.4865 | 0.0537 |
| nce | lr=5e-3, bs=default | 231.5916 | 263.3741 | 241.4945 | 1.6343 | 1.6722 | 0.1566 |
| blackbox | lr=1e-2, bs=default, lambd=0.01 | 3.6730 | 3.9437 | 4.0341 | 4.1187 | 4.0412 | 0.4320 |
| pointLTR | lr=1e-2, bs=default | 0.9964 | 1.2676 | 1.3866 | 0.9439 | 0.7627 | 0.1006 |
| pairLTR | lr=5e-3, bs=alt | 3.3129 | 3.5759 | 3.6411 | 0.7598 | 0.7526 | 0.0826 |
| listLTR | lr=5e-3, bs=default, tau=0.5 | 3.2083 | 3.4807 | 3.5348 | 0.5379 | 0.4850 | 0.0560 |
| lodl | lr=5e-3, bs=default, num_samples=500 | 2.5087 | 2.7300 | 2.8633 | 3.8238 | 3.5574 | 0.3868 |
| perturb | lr=5e-3, bs=alt, n_samples=100 | 4.7379 | 5.1430 | 5.1132 | 0.9241 | 1.2230 | 0.1063 |
| pg | lr=5e-3, bs=alt, sigma=0.1 | 3.2814 | 3.5397 | 3.6056 | 0.8704 | 1.2083 | 0.1031 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-3, bs=default, stein_weight=5.0 | 3.6240 | 3.8933 | 3.9845 | 4.1256 | 4.0057 | 0.4309 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
