# Loss Matrix — cubic (relative regret)

Each row: one method at its best (Phase 2) HP config. Values at the **best-val-regret epoch** (val_logs.csv argmin). Test regret from results.npy. Test pred_loss from `test_pred_loss.json` if present (produced by `eval_test_pred.py`).

| Method | Config | Train pred | Val pred | Test pred | Train regret | Val regret | Test regret |
|---|---|---|---|---|---|---|---|
| mse | lr=5e-3, bs=alt | 0.0069 | 0.0067 | 0.0069 | 0.0033 | 0.0017 | 0.0003 |
| dfl | lr=5e-3, bs=alt, dflalpha=10.0 | 0.4745 | 0.4584 | 0.4848 | 0.2414 | 0.2169 | 0.0220 |
| identity | lr=5e-3, bs=alt | 2612.8682 | 2610.5671 | 2613.5698 | 1.4670 | 1.3873 | 0.1389 |
| spo | lr=1e-2, bs=default | 11897.3652 | 11813.5986 | 11982.1104 | 17.7905 | 17.5864 | 1.6041 |
| nce | lr=1e-2, bs=default | 35.4807 | 35.6857 | 35.7612 | 15.9021 | 15.5792 | 1.4260 |
| blackbox | lr=5e-3, bs=alt, lambd=0.01 | 2612.8682 | 2610.5671 | 2613.5698 | 1.4670 | 1.3873 | 0.1389 |
| pointLTR | lr=5e-3, bs=default | 0.0063 | 0.0065 | 0.0064 | 0.0059 | 0.0021 | 0.0005 |
| pairLTR | lr=1e-2, bs=default | 2.3726 | 2.3212 | 2.3947 | 0.0441 | 0.0389 | 0.0042 |
| listLTR | lr=1e-2, bs=default, tau=10 | 567.6691 | 567.6890 | 567.6980 | 0.0007 | 0.0006 | 0.0001 |
| lodl | lr=5e-3, bs=alt, num_samples=500 | 2.3523 | 2.3171 | 2.3694 | 0.5587 | 0.5713 | 0.0422 |
| perturb | lr=1e-2, bs=alt, sigma=0.5 | 33.4720 | 33.6029 | 33.5420 | 1.4229 | 1.3391 | 0.1352 |
| pg | lr=1e-3, bs=default, sigma=1.0 | 2.3764 | 2.3306 | 2.3970 | 0.6185 | 0.6374 | 0.0488 |
| qptl | — | — | — | — | — | — | — |
| cpLayer | — | — | — | — | — | — | — |
| dad | lr=1e-3, bs=default, stein_weight=0.5 | 2.4638 | 2.4093 | 2.4881 | 1.4794 | 1.3795 | 0.1407 |

Phase-1 (lr, batch) and Phase-2 method-specific HP are both selected by **val** decision regret (no test leakage). Source: `bench_p2_best_val.json` via `collect_bench_p2_val.py`.
