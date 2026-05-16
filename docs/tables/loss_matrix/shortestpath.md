# Loss Matrix — shortestpath (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=5e-3, bs=alt | 0.0027 | 0.0039 | 0.0038 | 0.0016 | 0.0044 | 0.0002 |
| dfl | lr=1e-2, bs=alt, dflalpha=10.0 | 8.3120 | 7.2318 | 7.2258 | 12.5548 | 14.9070 | 0.5095 |
| identity | lr=1e-2, bs=default | 101641124263613153280.0000 | 104210207948702482432.0000 | 101002017736805580800.0000 | 193.5529 | 193.1707 | 6.4778 |
| spo | lr=1e-3, bs=alt | 28.4902 | 27.6344 | 27.5705 | 0.0301 | 0.6242 | 0.0204 |
| nce | lr=1e-2, bs=default | 677758697472.0000 | 677759614976.0000 | 677763022848.0000 | 14.4404 | 15.0139 | 0.4869 |
| blackbox | lr=1e-2, bs=alt, lambd=0.01 | 90753296.0000 | 90752312.0000 | 90753728.0000 | 48.5208 | 47.7827 | 1.6554 |
| pointLTR | lr=1e-3, bs=alt | 0.3610 | 0.3711 | 0.3799 | 0.1156 | 0.1606 | — |
| pairLTR | lr=1e-3, bs=alt | 24.2373 | 24.4136 | 23.7722 | 7.1311 | 8.1611 | — |
| listLTR | — | — | — | — | — | — | — |
| lodl | lr=1e-2, bs=default, num_samples=2000 | 34739950347878400.0000 | 35244196688297984.0000 | 34580443852439552.0000 | 14.4428 | 15.0268 | 0.4869 |
| perturb | lr=1e-2, bs=alt, n_samples=5 | 4158.0171 | 4159.1118 | 4156.4692 | 25.9430 | 25.6095 | 0.8887 |
| pg | — | — | — | — | — | — | — |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-2, bs=alt, stein_weight=1.0 | ∞ | ∞ | ∞ | 14.4404 | 15.0139 | 0.4869 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
