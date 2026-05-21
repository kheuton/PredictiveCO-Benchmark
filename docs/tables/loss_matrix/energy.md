# Loss Matrix — energy (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-1, bs=alt | 672.3024 | 565.3019 | 614.9795 | — | — | 0.0214 |
| mse_train | lr=1e-1, bs=alt | 671.8184 | 567.8830 | 611.0276 | — | — | 0.0214 |
| mse_val | lr=1e-1, bs=alt | 683.0355 | 569.8611 | 626.4077 | — | — | 0.0204 |
| dfl | lr=1e-3, bs=default, dflalpha=0.001 | 5353.3369 | 5264.2476 | 5280.3735 | — | — | 0.1715 |
| identity | lr=1e-3, bs=default | 5353.3188 | 5264.2344 | 5280.3413 | — | — | 0.1708 |
| spo | lr=1e-1, bs=default | 3769.3484 | 3609.1514 | 3851.3906 | 27866.3945 | 29116.4684 | 0.0200 |
| nce | lr=5e-3, bs=alt | 5337.8149 | 5247.8359 | 5266.6362 | 28207.1945 | 31784.1140 | 0.0189 |
| blackbox | lr=1e-1, bs=default, lambd=0.05 | 9507.9287 | 9317.4062 | 9059.8896 | 36107.4522 | 35115.1208 | 0.0239 |
| pointLTR | lr=1e-2, bs=alt | 795.0256 | 705.0991 | 739.0831 | 33426.3835 | 39609.4984 | 0.0199 |
| pairLTR | lr=1e-2, bs=alt | 5373.0762 | 5283.3262 | 5299.4683 | 31482.3957 | 33365.4339 | 0.0204 |
| listLTR | lr=1e-2, bs=alt, tau=0.5 | 5358.2354 | 5268.1821 | 5285.9453 | 28797.7651 | 30974.0162 | 0.0200 |
| lodl | lr=1e-1, bs=alt, num_samples=250 | 916.6742 | 765.9144 | 880.4314 | 35292.6245 | 38134.0296 | 0.0232 |
| perturb | lr=5e-3, bs=alt, sigma=1.0 | 5507.3359 | 5408.4043 | 5418.2358 | 28943.8513 | 29455.5038 | 0.0197 |
| pg | lr=5e-2, bs=alt, sigma=0.1 | 5557.6592 | 5372.4932 | 5443.8545 | 26599.9811 | 29595.5715 | 0.0194 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-1, bs=default, stein_weight=5.0 | 5513.8438 | 5421.7480 | 5431.6655 | 101446.6905 | 101483.0817 | 0.0655 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
