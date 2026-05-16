# Loss Matrix — bipartitematching (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=5e-3, bs=alt | 0.2718 | 0.2563 | 0.2903 | 36.9720 | 34.8565 | 0.9185 |
| dfl | lr=1e-2, bs=alt, dflalpha=1.0 | 3.2682 | 2.9699 | 3.1521 | 37.1960 | 33.1078 | 0.9431 |
| identity | lr=1e-3, bs=default | 5.3745 | 5.1709 | 5.2245 | 37.6976 | 35.0590 | 0.9223 |
| spo | lr=1e-2, bs=default | 0.9834 | 0.9461 | 0.9363 | 25.1576 | 33.1164 | 0.9220 |
| nce | lr=1e-2, bs=default | 0.5301 | 0.5452 | 0.5618 | 2.9168 | 32.3825 | 0.9000 |
| blackbox | lr=1e-2, bs=alt, lambd=0.01 | 4.6332 | 4.3206 | 4.4233 | 37.7052 | 34.8120 | 0.9192 |
| pointLTR | lr=5e-3, bs=alt | 0.2443 | 0.2230 | 0.2500 | 37.0910 | 34.6144 | 0.9215 |
| pairLTR | lr=1e-3, bs=default | 0.6042 | 0.6048 | 0.6046 | 0.4833 | 33.6166 | 0.9094 |
| listLTR | lr=1e-3, bs=default, tau=0.1 | 0.6531 | 0.6576 | 0.6564 | 1.2302 | 33.2265 | 0.8980 |
| lodl | lr=5e-3, bs=alt, num_samples=500 | 0.2148 | 0.2128 | 0.2281 | 37.1845 | 34.6727 | 0.9392 |
| perturb | lr=1e-2, bs=alt, n_samples=5 | 1.9944 | 1.7871 | 1.8973 | 36.8078 | 32.8032 | 0.9359 |
| pg | lr=5e-3, bs=alt, sigma=0.1 | 0.5690 | 0.5738 | 0.5826 | 33.4592 | 33.0229 | 0.9073 |
| qptl | lr=1e-3, bs=default, tau=0.1 | 5.2037 | 4.9935 | 5.0492 | 37.4907 | 35.0240 | 0.9205 |
| cpLayer | lr=1e-3, bs=default | 5.1200 | 4.9356 | 4.9805 | 37.6433 | 34.9329 | 0.9200 |
| dad | lr=5e-3, bs=default, stein_weight=1.0 | 0.3501 | 0.3590 | 0.3840 | 37.4792 | 33.4590 | 0.9174 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
