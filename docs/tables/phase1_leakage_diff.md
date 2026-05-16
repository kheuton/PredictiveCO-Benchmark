# Phase 1 HP-selection diff: test-leakage fix

Old selection: `bench_p1_best.json` (picks by test regret — leaky)

New selection: `bench_p1_best_val.json` (picks by val regret — correct)

Reported test regret uses the same relative/absolute convention as `collect_bench_p1.py` (abs for `portfolio`, rel otherwise).

| method | problem | old (batch/lr) | new (batch/lr) | old_test | new_test | Δ | val_src | status |
|---|---|---|---|---|---|---|---|---|
| mse | knapsack | alt/1e-2 | alt/1e-2 | 0.0544 | 0.0544 | 0.0000 | val_eval |  |
| mse | knapsack-real | alt/1e-2 | alt/1e-2 | 0.0763 | 0.0763 | 0.0000 | val_eval |  |
| mse | energy | alt/1e-2 | alt/5e-3 | 0.0200 | 0.0204 | 0.0004 | val_pred_mse | CHANGED |
| mse | budgetalloc | alt/1e-3 | alt/1e-3 | 0.3657 | 0.3657 | 0.0000 | val_eval |  |
| mse | cubic | alt/1e-2 | alt/5e-3 | 0.0003 | 0.0003 | 0.0001 | val_eval | CHANGED |
| mse | bipartitematching | default/5e-3 | alt/5e-3 | 0.9172 | 0.9185 | 0.0013 | val_eval | CHANGED |
| mse | portfolio | default/5e-3 | default/5e-3 | 0.2206 | 0.2206 | 0.0000 | val_eval |  |
| mse | asurv | default/1e-2 | default/1e-3 | 0.6889 | 0.6968 | 0.0080 | val_eval | CHANGED |
| mse | cook_county | default/1e-3 | default/1e-2 | 0.1880 | 0.1953 | 0.0073 | val_eval | CHANGED |
| mse | speed_humps | default/1e-2 | default/1e-2 | 0.1825 | 0.1825 | 0.0000 | val_eval |  |
| mse | sp_synth | alt/1e-3 | alt/1e-2 | 0.0959 | 0.0981 | 0.0022 | val_eval | CHANGED |
| mse | sp_planted | alt/1e-2 | alt/1e-2 | 0.1022 | 0.1022 | 0.0000 | val_eval |  |
| mse | shortestpath | alt/1e-2 | alt/5e-3 | 0.0002 | 0.0002 | 0.0000 | val_eval | CHANGED |
| dfl | knapsack | default/1e-2 | default/1e-2 | 0.2418 | 0.2418 | 0.0000 | val_eval |  |
| dfl | knapsack-real | alt/1e-2 | alt/1e-2 | 0.0962 | 0.0962 | 0.0000 | val_eval |  |
| dfl | energy | alt/1e-2 | default/1e-3 | 0.1398 | 0.1715 | 0.0317 | val_pred_mse | CHANGED |
| dfl | budgetalloc | alt/1e-3 | default/1e-3 | 0.2453 | 0.3322 | 0.0869 | val_eval | CHANGED |
| dfl | cubic | alt/5e-3 | alt/5e-3 | 0.1353 | 0.1353 | 0.0000 | val_eval |  |
| dfl | bipartitematching | default/5e-3 | alt/1e-2 | 0.9167 | 0.9369 | 0.0202 | val_eval | CHANGED |
| dfl | portfolio | alt/1e-3 | alt/5e-3 | 0.2708 | 0.2761 | 0.0052 | val_eval | CHANGED |
| dfl | asurv | default/1e-2 | default/1e-3 | 0.7266 | 0.7296 | 0.0030 | val_eval | CHANGED |
| dfl | cook_county | default/1e-3 | default/5e-3 | 0.1960 | 0.2012 | 0.0051 | val_eval | CHANGED |
| dfl | speed_humps | default/5e-3 | default/1e-3 | 0.2317 | 0.2328 | 0.0010 | val_eval | CHANGED |
| dfl | sp_synth | default/1e-3 | alt/1e-3 | 0.7468 | 0.7581 | 0.0113 | val_eval | CHANGED |
| dfl | sp_planted | default/1e-2 | default/1e-3 | 0.4299 | 0.4305 | 0.0006 | val_eval | CHANGED |
| dfl | shortestpath | alt/1e-2 | alt/1e-2 | 0.7480 | 0.7480 | 0.0000 | val_eval |  |
| identity | knapsack | alt/1e-2 | alt/1e-2 | 0.2385 | 0.2385 | 0.0000 | val_eval |  |
| identity | knapsack-real | default/1e-2 | default/5e-3 | 0.3511 | 0.3514 | 0.0003 | val_eval | CHANGED |
| identity | energy | alt/1e-2 | default/1e-3 | 0.1397 | 0.1708 | 0.0311 | val_pred_mse | CHANGED |
| identity | budgetalloc | default/1e-2 | default/1e-2 | 0.1465 | 0.1465 | 0.0000 | val_eval |  |
| identity | cubic | default/1e-2 | alt/5e-3 | 0.1388 | 0.1389 | 0.0001 | val_eval | CHANGED |
| identity | bipartitematching | default/5e-3 | default/1e-3 | 0.9181 | 0.9223 | 0.0042 | val_eval | CHANGED |
| identity | portfolio | default/5e-3 | alt/5e-3 | 0.2685 | 0.2804 | 0.0118 | val_eval | CHANGED |
| identity | asurv | default/1e-2 | default/1e-3 | 0.7266 | 0.7296 | 0.0030 | val_eval | CHANGED |
| identity | cook_county | default/1e-3 | default/5e-3 | 0.1960 | 0.2012 | 0.0051 | val_eval | CHANGED |
| identity | speed_humps | default/5e-3 | default/1e-3 | 0.2287 | 0.2322 | 0.0036 | val_eval | CHANGED |
| identity | sp_synth | default/1e-2 | default/1e-2 | 0.7147 | 0.7147 | 0.0000 | val_eval |  |
| identity | sp_planted | alt/1e-2 | default/1e-3 | 0.4021 | 0.4265 | 0.0244 | val_eval | CHANGED |
| identity | shortestpath | default/1e-2 | default/1e-2 | 6.4778 | 6.4778 | 0.0000 | val_eval |  |
| spo | knapsack | alt/5e-3 | default/5e-3 | 0.0559 | 0.0626 | 0.0067 | val_eval | CHANGED |
| spo | knapsack-real | default/5e-3 | default/5e-3 | 0.0701 | 0.0701 | 0.0000 | val_eval |  |
| spo | energy | default/5e-3 | default/5e-3 | 0.0181 | 0.0181 | 0.0000 | val_eval |  |
| spo | budgetalloc | default/5e-3 | alt/1e-2 | 0.0311 | 0.0510 | 0.0199 | val_eval | CHANGED |
| spo | cubic | default/1e-2 | default/1e-2 | 1.6041 | 1.6041 | 0.0000 | val_eval |  |
| spo | bipartitematching | alt/1e-2 | default/1e-2 | 0.9184 | 0.9220 | 0.0036 | val_eval | CHANGED |
| spo | portfolio | alt/1e-3 | alt/1e-2 | 0.2245 | 0.2298 | 0.0052 | val_eval | CHANGED |
| spo | asurv | default/1e-2 | alt/1e-3 | 0.6759 | 0.7107 | 0.0348 | val_eval | CHANGED |
| spo | cook_county | default/5e-3 | alt/1e-3 | 0.1931 | 0.2004 | 0.0073 | val_eval | CHANGED |
| spo | speed_humps | default/1e-2 | default/1e-2 | 0.1730 | 0.1730 | 0.0000 | val_eval |  |
| spo | sp_synth | default/1e-2 | default/5e-3 | 0.0560 | 0.0616 | 0.0057 | val_eval | CHANGED |
| spo | sp_planted | alt/1e-2 | default/5e-3 | 0.0520 | 0.0537 | 0.0018 | val_eval | CHANGED |
| spo | shortestpath | alt/1e-3 | alt/1e-3 | 0.0204 | 0.0204 | 0.0000 | val_eval |  |
| nce | knapsack | alt/1e-2 | alt/1e-3 | 0.1391 | 0.1483 | 0.0092 | val_eval | CHANGED |
| nce | knapsack-real | default/1e-2 | alt/1e-3 | 0.1175 | 0.1205 | 0.0030 | val_eval | CHANGED |
| nce | energy | alt/5e-3 | alt/5e-3 | 0.0189 | 0.0189 | 0.0000 | val_eval |  |
| nce | budgetalloc | alt/5e-3 | alt/5e-3 | 0.0863 | 0.0863 | 0.0000 | val_eval |  |
| nce | cubic | default/1e-2 | default/1e-2 | 1.4260 | 1.4260 | 0.0000 | val_eval |  |
| nce | bipartitematching | alt/5e-3 | default/1e-2 | 0.8980 | 0.9000 | 0.0020 | val_eval | CHANGED |
| nce | portfolio | default/5e-3 | default/1e-2 | 0.2844 | 0.3099 | 0.0256 | val_eval | CHANGED |
| nce | asurv | default/1e-2 | default/1e-2 | 0.6451 | 0.6451 | 0.0000 | val_eval |  |
| nce | cook_county | alt/5e-3 | alt/1e-3 | 0.2158 | 0.2180 | 0.0022 | val_eval | CHANGED |
| nce | speed_humps | default/1e-2 | default/1e-3 | 0.1958 | 0.2720 | 0.0761 | val_eval | CHANGED |
| nce | sp_synth | default/5e-3 | default/5e-3 | 0.0968 | 0.0968 | 0.0000 | val_eval |  |
| nce | sp_planted | default/1e-2 | default/5e-3 | 0.1559 | 0.1566 | 0.0008 | val_eval | CHANGED |
| nce | shortestpath | alt/1e-3 | default/1e-2 | 0.4869 | — | — | val_eval | CHANGED |
| blackbox | knapsack | alt/1e-2 | default/1e-2 | 0.2420 | 0.2432 | 0.0011 | val_eval | CHANGED |
| blackbox | knapsack-real | default/1e-3 | default/1e-2 | 0.3515 | 0.3657 | 0.0142 | val_eval | CHANGED |
| blackbox | energy | alt/1e-2 | alt/1e-2 | 0.0662 | 0.0662 | 0.0000 | val_eval |  |
| blackbox | budgetalloc | alt/5e-3 | alt/5e-3 | 0.1702 | 0.1702 | 0.0000 | val_eval |  |
| blackbox | cubic | default/1e-2 | alt/5e-3 | 0.1388 | 0.1389 | 0.0001 | val_eval | CHANGED |
| blackbox | bipartitematching | default/1e-3 | alt/1e-2 | 0.9167 | 0.9264 | 0.0096 | val_eval | CHANGED |
| blackbox | portfolio | default/5e-3 | default/5e-3 | 0.2682 | 0.2682 | 0.0000 | val_eval |  |
| blackbox | asurv | default/1e-2 | default/1e-3 | 0.7266 | 0.7296 | 0.0030 | val_eval | CHANGED |
| blackbox | cook_county | default/1e-3 | default/5e-3 | 0.1960 | 0.2012 | 0.0051 | val_eval | CHANGED |
| blackbox | speed_humps | default/5e-3 | default/1e-3 | 0.2287 | 0.2322 | 0.0036 | val_eval | CHANGED |
| blackbox | sp_synth | default/1e-3 | default/1e-3 | 0.7472 | 0.7472 | 0.0000 | val_eval |  |
| blackbox | sp_planted | default/1e-3 | default/1e-2 | 0.4307 | 0.4320 | 0.0013 | val_eval | CHANGED |
| blackbox | shortestpath | alt/1e-2 | alt/1e-2 | 1.6554 | 1.6554 | 0.0000 | val_eval |  |
| pointLTR | knapsack | alt/1e-2 | default/1e-2 | 0.0562 | 0.0580 | 0.0018 | val_eval | CHANGED |
| pointLTR | knapsack-real | default/1e-3 | default/5e-3 | 0.0830 | 0.0842 | 0.0012 | val_eval | CHANGED |
| pointLTR | energy | default/1e-3 | alt/1e-2 | 0.0186 | 0.0199 | 0.0013 | val_eval | CHANGED |
| pointLTR | budgetalloc | default/5e-3 | default/5e-3 | 0.0416 | 0.0416 | 0.0000 | val_eval |  |
| pointLTR | cubic | default/5e-3 | default/5e-3 | 0.0005 | 0.0005 | 0.0000 | val_eval |  |
| pointLTR | bipartitematching | default/5e-3 | alt/5e-3 | 0.9015 | 0.9215 | 0.0199 | val_eval | CHANGED |
| pointLTR | portfolio | alt/1e-3 | alt/5e-3 | 0.2232 | 0.2514 | 0.0283 | val_eval | CHANGED |
| pointLTR | asurv | default/1e-2 | default/1e-3 | 0.6918 | 0.6998 | 0.0080 | val_eval | CHANGED |
| pointLTR | cook_county | default/1e-2 | alt/5e-3 | 0.1982 | 0.2034 | 0.0051 | val_eval | CHANGED |
| pointLTR | speed_humps | default/1e-3 | alt/1e-3 | 0.2248 | 0.2322 | 0.0074 | val_eval | CHANGED |
| pointLTR | sp_synth | alt/5e-3 | default/1e-2 | 0.0932 | 0.0953 | 0.0020 | val_eval | CHANGED |
| pointLTR | sp_planted | default/5e-3 | default/1e-2 | 0.0984 | 0.1006 | 0.0022 | val_eval | CHANGED |
| pointLTR | shortestpath | — | alt/1e-3 | — | — | — | val_eval | N/A |
| pairLTR | knapsack | alt/5e-3 | alt/1e-2 | 0.0771 | 0.0789 | 0.0018 | val_eval | CHANGED |
| pairLTR | knapsack-real | default/5e-3 | alt/5e-3 | 0.0833 | 0.1098 | 0.0266 | val_eval | CHANGED |
| pairLTR | energy | default/1e-3 | alt/1e-2 | 0.0194 | 0.0204 | 0.0009 | val_eval | CHANGED |
| pairLTR | budgetalloc | alt/1e-2 | alt/1e-2 | 0.0786 | 0.0786 | 0.0000 | val_eval |  |
| pairLTR | cubic | default/1e-2 | default/1e-2 | 0.0042 | 0.0042 | 0.0000 | val_eval |  |
| pairLTR | bipartitematching | alt/1e-2 | default/1e-3 | 0.9055 | 0.9094 | 0.0039 | val_eval | CHANGED |
| pairLTR | portfolio | alt/1e-2 | default/1e-3 | 0.2424 | 0.2554 | 0.0131 | val_eval | CHANGED |
| pairLTR | asurv | default/1e-2 | default/1e-2 | 0.7177 | 0.7177 | 0.0000 | val_eval |  |
| pairLTR | cook_county | default/1e-2 | default/1e-2 | 0.1895 | 0.1895 | 0.0000 | val_eval |  |
| pairLTR | speed_humps | default/1e-2 | alt/1e-2 | 0.2158 | 0.2158 | 0.0000 | val_eval | CHANGED |
| pairLTR | sp_synth | default/5e-3 | default/1e-3 | 0.0667 | 0.0811 | 0.0144 | val_eval | CHANGED |
| pairLTR | sp_planted | default/1e-3 | alt/5e-3 | 0.0798 | 0.0826 | 0.0029 | val_eval | CHANGED |
| pairLTR | shortestpath | — | alt/1e-3 | — | — | — | val_eval | N/A |
| listLTR | knapsack | default/1e-3 | alt/1e-2 | 0.0559 | 0.0585 | 0.0026 | val_eval | CHANGED |
| listLTR | knapsack-real | default/1e-2 | default/1e-2 | 0.0718 | 0.0718 | 0.0000 | val_eval |  |
| listLTR | energy | alt/5e-3 | alt/1e-2 | 0.0195 | 0.0201 | 0.0006 | val_eval | CHANGED |
| listLTR | budgetalloc | default/1e-2 | default/5e-3 | 0.0339 | 0.0363 | 0.0024 | val_eval | CHANGED |
| listLTR | cubic | default/1e-2 | default/1e-2 | 0.0055 | 0.0055 | 0.0000 | val_eval |  |
| listLTR | bipartitematching | default/1e-3 | default/1e-3 | 0.8980 | 0.8980 | 0.0000 | val_eval |  |
| listLTR | portfolio | default/1e-3 | default/1e-3 | 0.2384 | 0.2384 | 0.0000 | val_eval |  |
| listLTR | asurv | default/5e-3 | default/5e-3 | 0.7038 | 0.7038 | 0.0000 | val_eval |  |
| listLTR | cook_county | default/1e-2 | default/1e-2 | 0.1953 | 0.1953 | 0.0000 | val_eval |  |
| listLTR | speed_humps | alt/1e-2 | alt/1e-2 | 0.3291 | 0.3291 | 0.0000 | val_eval |  |
| listLTR | sp_synth | alt/1e-2 | default/5e-3 | 0.0688 | 0.0725 | 0.0036 | val_eval | CHANGED |
| listLTR | sp_planted | default/1e-3 | default/5e-3 | 0.0613 | 0.0621 | 0.0009 | val_eval | CHANGED |
| listLTR | shortestpath | alt/1e-2 | alt/5e-3 | 0.0158 | — | — | val_eval | CHANGED |
| lodl | knapsack | default/5e-3 | default/1e-2 | 0.2374 | 0.2403 | 0.0029 | val_eval | CHANGED |
| lodl | knapsack-real | alt/1e-2 | default/5e-3 | 0.0935 | 0.0992 | 0.0058 | val_eval | CHANGED |
| lodl | energy | alt/1e-2 | alt/1e-2 | 0.0244 | 0.0244 | 0.0000 | val_eval |  |
| lodl | budgetalloc | default/1e-3 | default/1e-3 | 0.5611 | 0.5611 | 0.0000 | val_eval |  |
| lodl | cubic | alt/5e-3 | alt/5e-3 | 0.0422 | 0.0422 | 0.0000 | val_eval |  |
| lodl | bipartitematching | default/5e-3 | alt/5e-3 | 0.9116 | 0.9392 | 0.0276 | val_eval | CHANGED |
| lodl | portfolio | default/1e-3 | alt/5e-3 | 0.2406 | 0.2421 | 0.0015 | val_eval | CHANGED |
| lodl | asurv | default/1e-2 | default/1e-3 | 0.7197 | 0.7227 | 0.0030 | val_eval | CHANGED |
| lodl | cook_county | default/1e-3 | default/1e-3 | 0.1814 | 0.1814 | 0.0000 | val_eval |  |
| lodl | speed_humps | default/1e-2 | default/1e-2 | 0.2207 | 0.2207 | 0.0000 | val_eval |  |
| lodl | sp_synth | alt/1e-2 | alt/1e-2 | 0.6025 | 0.6025 | 0.0000 | val_eval |  |
| lodl | sp_planted | default/1e-2 | default/5e-3 | 0.3825 | 0.3868 | 0.0042 | val_eval | CHANGED |
| perturb | knapsack | alt/5e-3 | alt/1e-3 | 0.1364 | 0.1382 | 0.0018 | val_eval | CHANGED |
| perturb | knapsack-real | alt/5e-3 | default/1e-3 | 0.0877 | 0.1052 | 0.0175 | val_eval | CHANGED |
| perturb | energy | default/5e-3 | alt/5e-3 | 0.0191 | 0.0197 | 0.0006 | val_eval | CHANGED |
| perturb | budgetalloc | alt/5e-3 | default/1e-2 | 0.0665 | 0.0778 | 0.0113 | val_eval | CHANGED |
| perturb | cubic | alt/1e-2 | alt/1e-2 | 0.1354 | 0.1354 | 0.0000 | val_eval |  |
| perturb | bipartitematching | alt/1e-3 | alt/1e-2 | 0.9254 | 0.9385 | 0.0132 | val_eval | CHANGED |
| perturb | portfolio | alt/1e-2 | default/1e-2 | 0.3234 | 0.3822 | 0.0587 | val_eval | CHANGED |
| perturb | asurv | default/1e-2 | default/1e-3 | 0.6978 | 0.7137 | 0.0159 | val_eval | CHANGED |
| perturb | cook_county | default/1e-3 | default/1e-3 | 0.4813 | 0.4813 | 0.0000 | val_eval |  |
| perturb | speed_humps | default/1e-2 | default/1e-2 | 0.2069 | 0.2069 | 0.0000 | val_eval |  |
| perturb | sp_synth | alt/1e-2 | alt/1e-2 | 0.1156 | 0.1156 | 0.0000 | val_eval |  |
| perturb | sp_planted | alt/5e-3 | alt/5e-3 | 0.1334 | 0.1334 | 0.0000 | val_eval |  |
| perturb | shortestpath | alt/1e-2 | alt/1e-2 | 0.7369 | 0.7369 | 0.0000 | val_eval |  |
| pg | knapsack | alt/1e-2 | alt/1e-2 | 0.1114 | 0.1114 | 0.0000 | val_eval |  |
| pg | knapsack-real | alt/5e-3 | alt/5e-3 | 0.0868 | 0.0868 | 0.0000 | val_eval |  |
| pg | energy | default/1e-2 | default/1e-2 | 0.0193 | 0.0193 | 0.0000 | val_eval |  |
| pg | cubic | default/1e-3 | default/1e-3 | 0.0716 | 0.0716 | 0.0000 | val_eval |  |
| pg | bipartitematching | alt/1e-3 | alt/5e-3 | 0.9017 | 0.9245 | 0.0228 | val_eval | CHANGED |
| pg | portfolio | alt/1e-3 | default/1e-3 | 0.3389 | 0.3563 | 0.0174 | val_eval | CHANGED |
| pg | asurv | default/1e-2 | default/1e-2 | 0.6849 | 0.6849 | 0.0000 | val_eval |  |
| pg | cook_county | default/1e-2 | default/1e-2 | 0.1975 | 0.1975 | 0.0000 | val_eval |  |
| pg | speed_humps | default/1e-3 | default/1e-3 | 0.1982 | 0.1982 | 0.0000 | val_eval |  |
| pg | sp_synth | alt/1e-2 | alt/1e-2 | 0.0987 | 0.0987 | 0.0000 | val_eval |  |
| pg | sp_planted | alt/5e-3 | alt/5e-3 | 0.1031 | 0.1031 | 0.0000 | val_eval |  |
| qptl | knapsack | alt/1e-2 | alt/1e-2 | 0.2007 | 0.2007 | 0.0000 | val_eval |  |
| qptl | bipartitematching | default/5e-3 | default/1e-3 | 0.9159 | 0.9205 | 0.0045 | val_eval | CHANGED |
| qptl | portfolio | alt/1e-3 | alt/1e-3 | 0.2861 | 0.2861 | 0.0000 | val_eval |  |
| cpLayer | knapsack | alt/1e-2 | default/1e-2 | 0.2420 | 0.2432 | 0.0011 | val_eval | CHANGED |
| cpLayer | bipartitematching | default/1e-2 | default/1e-3 | 0.9138 | 0.9200 | 0.0062 | val_eval | CHANGED |
| cpLayer | portfolio | alt/1e-2 | alt/5e-3 | 0.2655 | 0.2791 | 0.0136 | val_eval | CHANGED |
| dad | knapsack | default/1e-3 | default/1e-3 | 0.2389 | 0.2389 | 0.0000 | val_eval |  |
| dad | knapsack-real | default/1e-3 | default/1e-3 | 0.3392 | 0.3392 | 0.0000 | val_eval |  |
| dad | energy | default/1e-2 | default/1e-2 | 0.1245 | 0.1245 | 0.0000 | val_eval |  |
| dad | budgetalloc | default/1e-3 | default/1e-3 | 0.2642 | 0.2642 | 0.0000 | val_eval |  |
| dad | cubic | default/1e-3 | default/1e-3 | 1.3296 | 1.3296 | 0.0000 | val_eval |  |
| dad | bipartitematching | default/1e-2 | default/5e-3 | 0.9085 | 0.9174 | 0.0089 | val_eval | CHANGED |
| dad | portfolio | default/1e-3 | default/1e-3 | 0.3575 | 0.3575 | 0.0000 | val_eval |  |
| dad | asurv | default/1e-3 | default/5e-3 | 0.7197 | 0.7286 | 0.0089 | val_eval | CHANGED |
| dad | cook_county | default/1e-3 | default/5e-3 | 0.1931 | 0.1946 | 0.0015 | val_eval | CHANGED |
| dad | speed_humps | default/5e-3 | default/1e-3 | 0.2025 | 0.2130 | 0.0105 | val_eval | CHANGED |
| dad | sp_synth | default/1e-3 | default/1e-3 | 0.7473 | 0.7473 | 0.0000 | val_eval |  |
| dad | sp_planted | default/1e-3 | default/1e-3 | 0.4309 | 0.4309 | 0.0000 | val_eval |  |
| dad | shortestpath | alt/1e-2 | alt/1e-2 | 0.4869 | 0.4869 | 0.0000 | val_eval |  |

**Summary.** 74 unchanged, 96 changed out of 170 shared cells.
Test-regret movement on changed cells (new − old): mean +0.0099, median +0.0045, max +0.0869, min +0.0000. 93 cells worsen, 0 improve.
