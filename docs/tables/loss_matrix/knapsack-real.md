# Loss Matrix — knapsack-real (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=1e-2, bs=alt | 669.7254 | 848.1179 | 589.8736 | 61.3709 | 72.1325 | 0.0763 |
| dfl | lr=1e-2, bs=alt, dflalpha=1.0 | 712.2666 | 866.6654 | 621.0205 | 61.6613 | 71.6651 | 0.0711 |
| identity | lr=5e-3, bs=default | 53711.7930 | 57338.7227 | 74613.1797 | 256.1831 | 263.3486 | 0.3514 |
| spo | lr=5e-3, bs=default | 931.2012 | 1102.7317 | 692.3826 | 57.7352 | 69.1852 | 0.0701 |
| nce | lr=1e-3, bs=alt | 5417.2690 | 5693.7671 | 4967.2065 | 89.7958 | 89.4887 | 0.1205 |
| blackbox | lr=1e-2, bs=default, lambd=0.01 | 5273.4072 | 5549.2109 | 4813.4341 | 304.4567 | 310.9121 | 0.3657 |
| pointLTR | lr=5e-3, bs=default | 724.5469 | 869.1758 | 655.9338 | 62.8180 | 73.4898 | 0.0842 |
| pairLTR | lr=5e-3, bs=alt | 5448.0928 | 5725.7773 | 4996.3975 | 73.2887 | 71.2834 | 0.1098 |
| listLTR | lr=1e-2, bs=default, tau=0.1 | 5160.3940 | 5445.4668 | 4748.6426 | 61.0519 | 71.6754 | 0.0718 |
| lodl | lr=5e-3, bs=default, num_samples=500 | 890.1172 | 1051.2089 | 803.0926 | 76.8159 | 81.7505 | 0.0992 |
| perturb | lr=1e-3, bs=default, sigma=0.1 | 5390.0151 | 5668.4302 | 4948.1602 | 71.4112 | 71.6876 | 0.0954 |
| pg | lr=5e-3, bs=alt, sigma=0.01 | 4089.8008 | 4377.1899 | 3888.1423 | 61.0480 | 73.5847 | 0.0812 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-3, bs=default, stein_weight=0.1 | 5377.3145 | 5653.8813 | 4920.6538 | 303.7118 | 314.7110 | 0.3527 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
