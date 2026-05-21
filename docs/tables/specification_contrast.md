# Specification contrast — `sp_synth` (mis-spec) vs `sp_planted` (well-spec)

Same 5×5 DAG shortest-path, same 1-layer linear prediction head. `sp_synth` has polynomial-degree-6 DGP (mis-specified), `sp_planted` has linear DGP (well-specified). All values: relative test decision regret at best Phase-1/2 config (best-val-regret checkpoint).

| Method | sp_synth (mis-spec) | sp_planted (well-spec) | Δ (synth − planted) |
|---|---:|---:|---:|
| SPO+ | 0.0616 | 0.0537 | 0.0079 |
| pr-LTR | 0.0656 | 0.0921 | -0.0265 |
| L-LTR | 0.0681 | 0.0581 | 0.0100 |
| Perturb | 0.0755 | 0.1063 | -0.0308 |
| MSE | 0.0899 | 0.1020 | -0.0121 |
| NCE | 0.0968 | 0.1681 | -0.0714 |
| MSE (train-sel) | 0.0997 | 0.1019 | -0.0022 |
| MSE (val-sel) | 0.1022 | 0.1061 | -0.0038 |
| PG | 0.1029 | 0.1031 | -0.0002 |
| pt-LTR | 0.1152 | 0.1006 | 0.0146 |
| DFL | 0.2680 | 0.3035 | -0.0354 |
| LODL | 0.4908 | 0.3106 | 0.1801 |
| DAD | 0.7093 | 0.4309 | 0.2784 |
| Identity | 0.7147 | 0.4265 | 0.2882 |
| Blackbox | 0.7544 | 0.4320 | 0.3224 |
| QPTL | — | — | — |
| cpLayer | — | — | — |

**Reading this table.**
- Δ > 0 → method does *worse* on mis-spec than on well-spec.
- Δ < 0 → method does *better* on mis-spec — surprising and diagnostic.
- Expected theoretical pattern (Elmachtoub et al. 2025): MSE should have Δ > 0 (well-spec easier for prediction-only), decision-aware methods should have smaller or negative Δ.

*Caveat:* Configs are test-selected (Phase-1 leakage, see `memory/phase1_hp_test_leakage.md`). Plan #7 fix pending.
