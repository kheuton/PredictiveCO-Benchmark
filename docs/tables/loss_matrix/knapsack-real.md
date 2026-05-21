# Loss Matrix — knapsack-real (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-1, bs=alt | 700.0698 | 870.1780 | 617.7440 | 62.3838 | 71.2527 | 0.0817 |
| mse_train | lr=1e-1, bs=alt | 633.4877 | 848.8085 | 581.1328 | — | — | 0.0777 |
| mse_val | lr=1e-1, bs=alt | 646.9794 | 824.5803 | 571.1873 | — | — | 0.0780 |
| dfl | lr=1e-1, bs=alt, dflalpha=1.0 | 727.5406 | 884.0659 | 623.3392 | 63.2435 | 71.2236 | 0.0818 |
| identity | lr=1e-1, bs=alt | 347938018557952.0000 | 383889646288896.0000 | 549216594165760.0000 | 136.2078 | 142.3287 | 0.1641 |
| spo | lr=5e-3, bs=default | 931.2012 | 1102.7317 | 692.3826 | 57.7352 | 69.1852 | 0.0701 |
| nce | lr=1e-3, bs=alt | 5417.2690 | 5693.7671 | 4967.2065 | 89.7958 | 89.4887 | 0.1205 |
| blackbox | lr=1e-1, bs=default, lambd=0.01 | 4980.9629 | 5253.0698 | 4508.8721 | 302.9746 | 303.6241 | 0.3729 |
| pointLTR | lr=5e-3, bs=default | 724.5469 | 869.1758 | 655.9338 | 62.8180 | 73.4898 | 0.0842 |
| pairLTR | lr=5e-2, bs=alt | 5442.1743 | 5720.0752 | 4991.7920 | 59.7348 | 69.9405 | 0.0812 |
| listLTR | lr=5e-2, bs=alt, tau=0.1 | 5081.0347 | 5372.9780 | 4663.4048 | 59.2174 | 74.1045 | 0.0784 |
| lodl | lr=1e-1, bs=default, num_samples=500 | 801.2936 | 973.6739 | 703.5408 | 67.1980 | 73.0174 | 0.0849 |
| perturb | lr=1e-3, bs=default, sigma=0.1 | 5390.0151 | 5668.4302 | 4948.1602 | 71.4112 | 71.6876 | 0.0954 |
| pg | lr=5e-2, bs=alt, sigma=0.05 | 48415.0234 | 44872.8906 | 45804.7812 | 57.0410 | 70.5813 | 0.0771 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-3, bs=default, stein_weight=0.1 | 5377.3145 | 5653.8813 | 4920.6538 | 303.7118 | 314.7110 | 0.3527 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
