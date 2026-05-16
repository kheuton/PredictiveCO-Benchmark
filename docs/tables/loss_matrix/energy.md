# Loss Matrix — energy (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=5e-3, bs=alt | — | — | 627.1152 | — | — | 0.0204 |
| dfl | lr=1e-3, bs=default, dflalpha=0.001 | — | — | 5280.3735 | — | — | 0.1715 |
| identity | lr=1e-3, bs=default | — | — | 5280.3413 | — | — | 0.1708 |
| spo | lr=5e-3, bs=default | 4466.1011 | 4373.2485 | 4796.3657 | 25146.8312 | 30827.9693 | 0.0181 |
| nce | lr=5e-3, bs=alt | 5337.8149 | 5247.8359 | 5266.6362 | 28207.1945 | 31784.1140 | 0.0189 |
| blackbox | lr=1e-2, bs=alt, lambd=0.01 | 7686.2896 | 7552.7271 | 7376.5151 | 52421.9735 | 56046.6854 | 0.0347 |
| pointLTR | lr=1e-2, bs=alt | 795.0256 | 705.0991 | 739.0831 | 33426.3835 | 39609.4984 | 0.0199 |
| pairLTR | lr=1e-2, bs=alt | 5373.0762 | 5283.3262 | 5299.4683 | 31482.3957 | 33365.4339 | 0.0204 |
| listLTR | lr=1e-2, bs=alt, tau=0.5 | 5358.2354 | 5268.1821 | 5285.9453 | 28797.7651 | 30974.0162 | 0.0200 |
| lodl | lr=1e-2, bs=alt, num_samples=250 | 866.2393 | 719.1458 | 838.3982 | 37155.2324 | 44247.6432 | 0.0240 |
| perturb | lr=5e-3, bs=alt, sigma=1.0 | 5507.3359 | 5408.4043 | 5418.2358 | 28943.8513 | 29455.5038 | 0.0197 |
| pg | lr=1e-2, bs=default, sigma=0.1 | 4783.4634 | 4675.8354 | 4758.2642 | 28816.2453 | 30131.2927 | 0.0193 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-2, bs=default, stein_weight=5.0 | 5401.0977 | 5310.9678 | 5325.1509 | 208123.5741 | 192400.7782 | 0.1171 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
