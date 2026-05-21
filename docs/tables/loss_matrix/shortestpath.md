# Loss Matrix — shortestpath (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=5e-3, bs=alt | 0.0027 | 0.0039 | 0.0038 | 0.0016 | 0.0044 | 0.0002 |
| mse_train | lr=5e-3, bs=alt | 0.0046 | 0.0154 | 0.0153 | — | — | 0.0017 |
| mse_val | lr=1e-2, bs=alt | 0.0046 | 0.0087 | 0.0085 | — | — | 0.0008 |
| dfl | lr=1e-1, bs=default, dflalpha=0.001 | — | — | — | 14.4404 | 15.0139 | 0.4869 |
| identity | lr=5e-2, bs=default | — | — | — | 14.4404 | 15.0139 | 0.4869 |
| spo | lr=1e-3, bs=alt | 28.4902 | 27.6344 | 27.5705 | 0.0301 | 0.6242 | 0.0204 |
| nce | lr=1e-2, bs=default | 677758697472.0000 | 677759614976.0000 | 677763022848.0000 | 14.4404 | 15.0139 | 0.4869 |
| blackbox | lr=1e-2, bs=alt, lambd=0.01 | 90753296.0000 | 90752312.0000 | 90753728.0000 | 48.5208 | 47.7827 | 1.6554 |
| pointLTR | lr=5e-3, bs=alt | 0.2172 | 0.2231 | 0.2238 | 0.0189 | 0.0399 | 0.0015 |
| pairLTR | lr=1e-3, bs=alt | 24.2425 | 24.4186 | 23.7722 | 5.8748 | 6.6076 | — |
| listLTR | lr=1e-3, bs=default, tau=0.5 | 16.0247 | 16.1552 | 15.6225 | 0.0563 | 0.2397 | — |
| lodl | lr=1e-1, bs=alt, num_samples=250 | 11.0476 | 11.0657 | 10.7754 | 14.4544 | 14.9798 | 0.4869 |
| perturb | lr=1e-2, bs=alt, n_samples=5 | 4158.0171 | 4159.1118 | 4156.4692 | 25.9430 | 25.6095 | 0.8887 |
| pg | — | — | — | — | — | — | — |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | — | — | — | — | — | — | — |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
