# Loss Matrix — sp_planted (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-1, bs=alt | 0.3787 | 0.5111 | 0.6715 | 0.9606 | 0.8645 | 0.1020 |
| mse_train | lr=5e-3, bs=alt | 0.3739 | 0.5083 | 0.6673 | — | — | 0.1019 |
| mse_val | lr=1e-1, bs=alt | 0.3804 | 0.5026 | 0.6679 | — | — | 0.1061 |
| dfl | lr=1e-3, bs=default, dflalpha=10.0 | 2.1672 | 2.3081 | 2.4831 | 2.9419 | 2.8020 | 0.3035 |
| identity | lr=1e-3, bs=default | 3.7634 | 4.0507 | 4.1350 | 4.0434 | 4.0079 | 0.4265 |
| spo | lr=5e-3, bs=default | 3.6551 | 3.9443 | 3.9765 | 0.5118 | 0.4865 | 0.0537 |
| nce | lr=5e-2, bs=alt | 12.6885 | 14.7692 | 13.5541 | 1.7414 | 1.6703 | 0.1681 |
| blackbox | lr=1e-2, bs=default, lambd=0.01 | 3.6730 | 3.9437 | 4.0341 | 4.1187 | 4.0412 | 0.4320 |
| pointLTR | lr=1e-2, bs=default | 0.9964 | 1.2676 | 1.3866 | 0.9439 | 0.7627 | 0.1006 |
| pairLTR | lr=5e-2, bs=default | 18.5622 | 21.0142 | 20.0812 | 0.9410 | 0.7198 | 0.0921 |
| listLTR | lr=5e-2, bs=alt, tau=0.5 | 3.5564 | 3.8618 | 3.9054 | 0.5568 | 0.4523 | 0.0581 |
| lodl | lr=5e-2, bs=alt, num_samples=100 | 0.9821 | 1.2184 | 1.3681 | 3.0496 | 3.0958 | 0.3106 |
| perturb | lr=5e-3, bs=alt, n_samples=100 | 4.7379 | 5.1430 | 5.1132 | 0.9241 | 1.2230 | 0.1063 |
| pg | lr=5e-3, bs=alt, sigma=0.1 | 3.2814 | 3.5397 | 3.6056 | 0.8704 | 1.2083 | 0.1031 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-3, bs=default, stein_weight=5.0 | 3.6240 | 3.8933 | 3.9845 | 4.1256 | 4.0057 | 0.4309 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
