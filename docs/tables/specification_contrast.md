# Specification contrast — `sp_synth` (mis-spec) vs `sp_planted` (well-spec)

Same 5×5 DAG shortest-path, same 1-layer linear prediction head. `sp_synth` has polynomial-degree-6 DGP (mis-specified), `sp_planted` has linear DGP (well-specified). All values: relative test decision regret at best Phase-1/2 config (best-val-regret checkpoint).

| Method | sp_synth (mis-spec) | sp_planted (well-spec) | Δ (synth − planted) |
|---|---:|---:|---:|
| SPO+ | 0.0560 | 0.0520 | 0.0040 |
| L-LTR | 0.0620 | 0.0575 | 0.0045 |
| pr-LTR | 0.0667 | 0.0798 | -0.0130 |
| Perturb | 0.0719 | 0.1063 | -0.0343 |
| pt-LTR | 0.0932 | 0.0984 | -0.0052 |
| MSE | 0.0959 | 0.1022 | -0.0062 |
| NCE | 0.0968 | 0.1559 | -0.0591 |
| PG | 0.0987 | 0.1031 | -0.0044 |
| DFL | 0.4875 | 0.2275 | 0.2600 |
| LODL | 0.6025 | — | — |
| Identity | 0.7147 | 0.4021 | 0.3126 |
| DAD | 0.7471 | 0.4303 | 0.3167 |
| Blackbox | 0.7472 | 0.4307 | 0.3165 |
| QPTL | — | — | — |
| cpLayer | — | — | — |

**Reading this table.**
- Δ > 0 → method does *worse* on mis-spec than on well-spec.
- Δ < 0 → method does *better* on mis-spec — surprising and diagnostic.
- Expected theoretical pattern (Elmachtoub et al. 2025): MSE should have Δ > 0 (well-spec easier for prediction-only), decision-aware methods should have smaller or negative Δ.

*Caveat:* Configs are test-selected (Phase-1 leakage, see `memory/phase1_hp_test_leakage.md`). Plan #7 fix pending.
