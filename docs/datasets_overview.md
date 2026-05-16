# Datasets and Default Prediction Models

For each benchmark task in `openpto`, this document records:

1. The **source / citation** for the data (paper, dataset, generation procedure).
2. The **default prediction model** invoked when running the task with the
   sweep defaults (`shells/slurm/submit_bench_p1.sh` + `openpto/config/utils_conf.py`).

Defaults from `openpto/config/utils_conf.py` unless overridden in
`shells/slurm/submit_bench_p1.sh` (`PRED_MODEL_ARGS`):
`--pred_model dense`, `--n_layers 2`, `--n_hidden 32`, `--activation relu`.
The `dense` predictor is `MLP` in `openpto/method/Predicts/dense.py`:
two `Linear` layers separated by ReLU (one hidden layer of 32 units),
with the per-problem output activation set by `get_output_activation()`.

The 8 problems listed in the original benchmark paper (Tang & Yan, *PtOPnO*,
NeurIPS 2024 — `../dfl-wiki/raw/PtOPnO/PtOPnO.md`) are marked **[paper]**.
The remaining 7 were added to this fork — they are marked **[fork]**.

---

## 1. `knapsack` — Synthetic 0/1 knapsack [paper]

- **Source.** Synthetic data, polynomial generation procedure of
  Elmachtoub & Grigas (2022, "Smart Predict-Then-Optimize," *Management
  Science*) as adopted in PyEPO ([20] in the paper, ref [64] for code).
  Defining equation (PtOPnO Eq. 15): `y_i = (Bx_i / sqrt(p) + 3)^deg / 3.5^deg + 1`,
  multiplied by uniform multiplicative noise; `B ~ Bernoulli(0.5)` fixed,
  `x ~ N(0, I_p)`. Item weights drawn `U[3,8]`. `p=5`, `deg=4`, `num_items=20`,
  `capacity=30`.  See `openpto/problems/Knapsack.py` and
  `openpto/config/probs/knapsack.yaml`.
- **Train / val / test.** `--instances 400 --testinstances 200` (defaults)
  with `--val_frac 0.2` → **320 / 80 / 200** instances. Each instance is a
  single 20-item knapsack with 5-D feature vector per item.
- **Default predictor.** `dense` MLP (2 layers, 32 hidden, ReLU) with
  `identity` output activation.

## 2. `knapsack-real` (a.k.a. KE — knapsack-energy) — Real energy prices [paper]

- **Source.** Irish Single Electricity Market Operator (SEMO) day-ahead
  prices, Ifrim, O'Sullivan & Simonis (2012) [33]. The energy price of each
  48-slot day is reused as item value; resource usage on machines fills in as
  weights. Same dataset that drives `energy`. See
  `openpto/config/probs/knapsack-real.yaml`.
- **Train / val / test.** Hardcoded splits — do *not* pass `--instances` /
  `--testinstances`. Loaded from `prices2013.dat`; 552 train+val days are
  shuffled and split by `val_frac=0.2` → **442 / 110 / 237** days, each a
  48-slot 0/1 knapsack with 8-D features per slot. (Counts from
  `openpto/problems/Knapsack.py::get_energy_data` line 212 comment.)
- **Default predictor.** `dense` MLP (defaults).

## 3. `energy` — Energy-cost-aware scheduling [paper]

- **Source.** SEMO day-ahead electricity prices (Ifrim et al. 2012 [33]),
  midnight 1 Nov 2011 – 31 Dec 2013. 9-D feature vector per 30-min slot
  (calendar attributes, day-ahead weather, SEMO load forecast, wind and
  prices, actual wind speed, temperature, CO2 intensity, price). Loaded
  from `prices2013.dat`. See `openpto/problems/Energy.py`.
- **Train / val / test.** Hardcoded splits — do *not* pass `--instances` /
  `--testinstances`. With `val_frac=0.2`: 650 pretrain rows split into
  130 val + 520 train; test is the fixed tail starting at index 650 →
  **520 / 130 / 139** days (139 from PtOPnO Table 2). Legacy mode
  (`val_frac=None`) uses 550 / 100 / 139.
- **Default predictor.** `dense` MLP (defaults).

## 4. `budgetalloc` — Submodular budget allocation [paper]

- **Source.** Yahoo! Webscope dataset (G4 search-marketing tag, [74]); we
  follow the multi-linear relaxation of Wilder, Dilkina & Tambe (2019, [72])
  and LODL (Shah et al. 2022, [60]). 10 users × 5 websites with budget 1.
  See `openpto/problems/BudgetAllocation.py`.
- **Train / val / test.** `--instances 400 --testinstances 200` (defaults)
  with `--val_frac 0.2` → **320 / 80 / 200** instances. Each instance is a
  5-website × 10-user click-probability matrix.
- **Default predictor.** `dense` MLP (defaults), with `sigmoid` output
  activation.

## 5. `cubic` (Cubic Top-K) — Synthetic top-k [paper]

- **Source.** Synthetic data from LODL (Shah et al. 2022, [60]).
  `x ~ U(0,1)`, `y = 10 x^3 − 6.5 x` (noise-free). 50 items, k = 5.
  See `openpto/problems/CubicTopK.py`.
- **Train / val / test.** Benchmark settings require explicit
  `--instances 250 --testinstances 400` with `--val_frac 0.2` →
  **200 / 50 / 400** instances. Each instance is a vector of 50 items.
- **Default predictor.** `dense` MLP (defaults).

## 6. `bipartitematching` — Cora-based bipartite matching [paper]

- **Source.** Cora citation network ([48], [59], [76]); the bipartite
  matching split into 27 instances of 100 nodes was first used by Wilder,
  Dilkina & Tambe (2019, [72]) and reused by Ferber et al. (2020, [22]) and
  by Mandi et al. (2024, [44]). Each instance uses 1433-D bag-of-words
  features. See `openpto/problems/BipartiteMatching.py`.
- **Train / val / test.** Benchmark settings require explicit
  `--instances 20 --testinstances 6` with `--val_frac 0.2` →
  **16 / 4 / 6** graph instances of 100 nodes (matching cardinality 50).
- **Default predictor.** `dense` MLP (defaults).

## 7. `portfolio` — Markowitz portfolio optimisation [paper]

- **Source.** S&P 500 stock returns 2004–2017 (`Quandl WIKI` dataset, [56]).
  Features: return windows of 10 days/weeks/months/years plus rolling
  averages. 50 stocks, risk-aversion `α = 0.1`, full-investment constraint.
  See `openpto/problems/PortfolioOpt.py`.
- **Train / val / test.** `--instances 400 --testinstances 200` (defaults)
  with `--val_frac 0.2` → **320 / 80 / 200** instances. Each instance is a
  50-stock day with 50-D features per stock.
- **Default predictor.** `dense` MLP (defaults), `identity` output.

## 8. `advertising` (CA — Combinatorial Advertising) — Industrial dataset [paper]

- **Source.** New industrial dataset released with the PtOPnO paper:
  Finvolution Group fintech advertising records, 269 329 users over 203
  days in 2023. 7-day intervals (23 train + 6 test). Features: 41-D
  (encrypted user vector + recent marketing records). License covered in
  PtOPnO Appendix C.7. See `openpto/problems/Advertising.py`.
- **Train / val / test.** Hardcoded splits — **23 / 0 / 6** weekly
  instances (no val split is carved out; PtOPnO Table 2 reports 23 train +
  6 test, average 2 933 users per instance). The pretrain pickle exposes a
  separate set of "real" labelled days used for the BCE pretrain
  step; the train/test pickles are mock combinations.
- **Default predictor.** `cvr` (`CVRModel` in
  `openpto/method/Predicts/cvr_model.py`): a four-layer fully-connected
  network with hidden widths `4·H, 2·H, H/2, num_targets` and ReLU
  activations, `sigmoid` output, processing each user separately. The
  benchmark shell explicitly passes `--pred_model cvr`.

---

# New problems added in this fork

## 9. `shortestpath` (SP-W — Warcraft 12×12) [fork]

- **Source.** Warcraft II terrain images, introduced for Differentiation of
  Blackbox Combinatorial Solvers by Vlastelica et al. (ICLR 2020) and used
  in the DPO paper (Berthet et al., NeurIPS 2020). 96×96 RGB images on a
  12×12 grid, vertex-cost shortest path with Dijkstra (8-grid). Bundled
  under `./openpto/data/shortestpath/12x12/`. See
  `openpto/problems/Shortestpath.py` and
  `openpto/config/probs/shortestpath.yaml`.
- **Train / val / test.** Benchmark settings require explicit
  `--instances 10000 --testinstances 1000`. The train file holds 10 000
  images, of which `val_frac=0.2` reserves 2 000 — but the codepath then
  loads the *separate* val.npy file (1 000 images) and caps the val
  count at `min(2000, 1000)=1000`. Net: **8 000 / 1 000 / 1 000** images
  (96×96×3 each).
- **Default predictor.** `Resnet18`
  (`openpto/method/Predicts/cv_model.py::Resnet18`). PyTorch
  `torchvision.models.resnet18(pretrained=False)` with the first conv
  replaced by `Conv2d(3, 64, kernel=7, stride=2)` and the classifier head
  set to `num_targets = 144` (one weight per grid vertex). Set in
  `submit_bench_p1.sh` via `PRED_MODEL_ARGS[shortestpath]="--pred_model Resnet18"`.
  Requires GPU.

## 10. `sp_synth` — 5×5 grid shortest path, polynomial DGP [fork]

- **Source.** Synthetic shortest-path benchmark from Elmachtoub & Grigas
  (2022), the SPO+ paper, faithfully reproduced in
  `openpto/problems/ShortestpathSynth.py::_gen_synth_data`. 5 features,
  polynomial degree 6, multiplicative uniform noise of half-width 0.3,
  on a directed 5×5 grid (right + down edges; 40 edges).
- **Train / val / test.** Benchmark settings require explicit
  `--instances 400 --testinstances 10000` with `--val_frac 0.2` →
  **320 / 80 / 10 000** instances.
- **Default predictor.** `dense` linear model — single
  `Linear(num_features, num_edges)` because `submit_bench_p1.sh` sets
  `PRED_MODEL_ARGS[sp_synth]="--pred_model dense --n_layers 1"`. This is
  intentionally mis-specified vs. the polynomial DGP.

## 11. `sp_planted` — 5×5 grid shortest path, planted-arc DGP [fork]

- **Source.** Synthetic shortest-path benchmark from Gupta & Huang (the
  Perturbation-Gradient / PG paper, arXiv 2402.03256), reproduced in
  `openpto/problems/ShortestpathSynth.py::_gen_planted_data`. Same 5×5
  grid as `sp_synth` but with two planted paths (safe constant cost = 2,
  risky context-switched cost on `x_6`) and the remaining edges driven by a
  polynomial of the first 5 features plus additive Gaussian noise. 6
  features.
- **Train / val / test.** Benchmark settings require explicit
  `--instances 400 --testinstances 10000` with `--val_frac 0.2` →
  **320 / 80 / 10 000** instances.
- **Default predictor.** `dense` linear model (`--n_layers 1`); same as
  `sp_synth`. Well-specified relative to the planted-cost portion of the
  DGP.

## 12. `TSP` — Synthetic travelling salesman [fork]

- **Source.** Synthetic data generated in `openpto/problems/TSP.py` using
  the PyEPO-style polynomial generator (`gendata` and
  `gen_from_global_feats`); `num_nodes=20`, `num_features=20`, `poly_deg=4`,
  `noise_width=0.5`. Defined in the codebase only — no external dataset.
- **Train / val / test.** Inherits the global defaults
  `--instances 400 --testinstances 200` with `--val_frac 0.2` →
  **320 / 80 / 200** instances. Each instance is a 20-node graph with
  190-edge cost vector and 20-D node features.
- **Default predictor.** `dense` MLP (defaults). The PtOPnO paper does not
  benchmark this task; it is included as an additional CO problem in the
  fork.

## 13. `asurv` — Aerial surveillance top-K [fork]

- **Source.** Real spatio-temporal location data bundled under
  `openpto/data/asurv/`. 9 features per location: lat, long,
  month indicator, timestep feature, and five lagged target counts
  (1- to 5-step back). Top-K = 50. See `openpto/problems/AerialSurv.py`.
- **Train / val / test.** Hardcoded splits — do *not* pass `--instances`
  / `--testinstances`. **15 / 6 / 6** timesteps; each timestep covers
  1 338 locations (so the model sees 1 338 × {15, 6, 6} location-step
  pairs). Each timestep is one optimisation instance.
- **Default predictor.** `dense` MLP (defaults), `identity`/`none` output
  activation. Applied per location.

## 14. `cook_county` — Cook County opioid-mortality top-K [fork]

- **Source.** Real Cook County (Illinois) census-tract mortality data,
  loaded from `openpto/data/cook_county/`. 13 features per tract: SVI
  themes 1–4, total SVI, lat (`INTPTLAT`), long (`INTPTLON`),
  spatial-lagged deaths (`deaths_sp_lag`), and 5 temporal lags. Top-K =
  100. Originating from the Hughes Lab predictive-policy work (no external
  citation in the PtOPnO paper). See `openpto/problems/CookCounty.py`.
- **Train / val / test.** Hardcoded splits — do *not* pass `--instances`
  / `--testinstances`. **4 / 1 / 2** years; each year covers 1 328
  tracts. Each year is one optimisation instance.
- **Default predictor.** `dense` MLP (defaults), `identity`/`none` output.

## 15. `speed_humps` — NYC pedestrian-injury speed-hump siting [fork]

- **Source.** Real NYC Department of Transportation crash records bundled
  under `openpto/data/speed_humps/raw/`, pre-processed by
  `scripts/prep_speed_humps_data.py`. Targets:
  `injury_count = crashes × prob_ped_injured`. 9 features per tract:
  borough code, centroid lat/lon, crashes, lagged injury count, lagged
  crashes, lagged ped-injury probability, 3-yr and 5-yr rolling injury
  averages. Top-K = 107. Originating from this fork — no external
  citation. See `openpto/problems/SpeedHumps.py`.
- **Train / val / test.** Hardcoded splits — do *not* pass `--instances`
  / `--testinstances`. **5 / 2 / 4** years (Train: 2013–2017,
  Val: 2018–2019, Test: 2020–2023); each year covers 2 107 tracts.
- **Default predictor.** `dense` MLP (defaults), `identity`/`none` output.

---

## Summary table

Counts below are the **default benchmark sweep settings** (i.e. matching
`shells/slurm/submit_bench_p1.sh` and the CLAUDE.md "Benchmark Fidelity"
table; `val_frac=0.2` carved off training instances unless the dataset has
hardcoded splits).

| # | Task | Data source | Train / Val / Test (instances) | Default predictor |
|---|---|---|---|---|
| 1 | `knapsack` | Synthetic poly-DGP (Elmachtoub & Grigas 2022 / Demirovic et al. 2019) | 320 / 80 / 200 | dense MLP (2 layers × 32) |
| 2 | `knapsack-real` | SEMO energy prices (Ifrim et al. 2012) | 442 / 110 / 237 (days) | dense MLP |
| 3 | `energy` | SEMO energy prices (Ifrim et al. 2012) | 520 / 130 / 139 (days) | dense MLP |
| 4 | `budgetalloc` | Yahoo! Webscope G4 (Wilder et al. 2019) | 320 / 80 / 200 | dense MLP, sigmoid head |
| 5 | `cubic` | Synthetic (Shah et al. 2022, LODL) | 200 / 50 / 400 | dense MLP |
| 6 | `bipartitematching` | Cora citation graph (Wilder et al. 2019 split) | 16 / 4 / 6 (graphs) | dense MLP |
| 7 | `portfolio` | Quandl WIKI / S&P 500 2004–17 | 320 / 80 / 200 | dense MLP |
| 8 | `advertising` | Finvolution Group industrial dataset (released with PtOPnO) | 23 / 0 / 6 (weeks; ~2933 users each) | `cvr` 4-layer NN |
| 9 | `shortestpath` | Warcraft II 12×12 images (Vlastelica et al. 2020 / Berthet et al. 2020) | 8000 / 1000 / 1000 (images) | ResNet18 |
| 10 | `sp_synth` | Synthetic SPO+ poly-DGP on 5×5 grid (Elmachtoub & Grigas 2022) | 320 / 80 / 10000 | dense linear (1 layer) |
| 11 | `sp_planted` | Synthetic planted-arc DGP on 5×5 grid (Gupta & Huang 2024) | 320 / 80 / 10000 | dense linear (1 layer) |
| 12 | `TSP` | Synthetic PyEPO-style poly-DGP, 20 nodes | 320 / 80 / 200 | dense MLP |
| 13 | `asurv` | Real aerial-surveillance counts, 1 338 locations | 15 / 6 / 6 (timesteps) | dense MLP |
| 14 | `cook_county` | Real Cook County opioid mortality, 1 328 tracts | 4 / 1 / 2 (years) | dense MLP |
| 15 | `speed_humps` | Real NYC DOT crash records, 2 107 tracts | 5 / 2 / 4 (years) | dense MLP |

---

# Methods (`--opt_model` choices)

Each method is selected via `--opt_model` and dispatched in
`openpto/method/Models/wrapper_loss.py`. PtO baselines marked **[PtO]**
optimise a prediction loss with the solver only at evaluation time. PnO
methods marked **[PnO]** flow gradients from the decision objective back
into the predictor. The 11 methods benchmarked in the original PtOPnO paper
are marked **[paper]**; the two added in this fork are marked **[fork]**.
Reference numbers in brackets `[NN]` are the PtOPnO bibliography entry
numbers (`../dfl-wiki/raw/PtOPnO/PtOPnO.md`, References §).

## Two-stage / prediction baselines [PtO, paper]

- **`mse` — Mean-squared error.** Standard regression baseline; minimises
  `(ŷ − y)²` then runs the solver at test time. The original PtOPnO paper
  refers to this branch as the "two-stage" approach (Bertsimas & Kallus,
  *From Predictive to Prescriptive Analytics*, *Management Science* 2020,
  [7]). Source: `openpto/method/Models/MSE.py::MSE`.
- **`bce`, `ce`, `mae`, `msesum` — Variants** of the same two-stage idea
  with binary-cross-entropy / cross-entropy / mean-absolute-error / summed
  squared error losses; same citation [7].
- **`dfl` — Decision-Focused Learning ("straight-through" gradient).**
  Differentiates by passing the gradient of the decision objective directly
  through the predictor; the spirit is the straight-through estimator of
  Bengio, Léonard & Courville (*arXiv* 1308.3432, 2013, [5]) adapted to
  PtO as in Shah et al. (LODL, NeurIPS 2022, [60]). Source:
  `openpto/method/Models/MSE.py::DFL`.

## Continuous / KKT-based PnO [PnO, paper]

- **`cpLayer` — Differentiable convex-optimisation layers.**
  Agrawal, Amos, Barratt, Boyd, Diamond & Kolter, *Differentiable Convex
  Optimization Layers*, NeurIPS 2019 [1]; built on cone-program
  differentiation (Agrawal et al., *arXiv* 1904.09043 [2]) and OptNet
  (Amos & Kolter, ICML 2017 [3]). Source:
  `openpto/method/Models/cpLayer.py`.
- **`qptl` — Quadratic Programming Task Loss.**
  Wilder, Dilkina & Tambe, *Melding the Data-Decisions Pipeline*, AAAI
  2019 [72] — adds a small squared-norm regulariser so that linear /
  combinatorial programs can be differentiated through their KKT
  conditions. Source: `openpto/method/Models/QPTL.py`.

## Discrete / gradient-interpolation PnO [PnO, paper]

- **`spo` — SPO+ (Smart Predict-then-Optimize).**
  Elmachtoub & Grigas, *Smart "Predict, then Optimize"*, *Management
  Science* 2022 [20], with the SPO-relax discrete extension by Mandi,
  Stuckey & Guns, *Smart Predict-and-Optimize for Hard Combinatorial
  Optimization Problems*, AAAI 2020 [46]. Source:
  `openpto/method/Models/SPO.py`.
- **`blackbox` — Differentiation of Blackbox CO solvers.**
  Pogančić, Paulus, Musil, Martius & Rolínek, *Differentiation of Blackbox
  Combinatorial Solvers*, ICLR 2020 [54]. Source:
  `openpto/method/Models/Blackbox.py`.
- **`identity` — Identity-with-projection.**
  Sahoo, Paulus, Vlastelica, Musil, Kuleshov & Martius, *Backpropagation
  through Combinatorial Algorithms: Identity with Projection Works*,
  ICLR 2023 [58]. Source: `openpto/method/Models/Identity.py`.
- **`perturb` — Differentiable Perturbed Optimisers (DPO).**
  Berthet, Blondel, Teboul, Cuturi, Vert & Bach, *Learning with
  Differentiable Perturbed Optimizers*, NeurIPS 2020 [6]. The default
  `perturbed` class in this fork implements the soft-decision /
  Jacobian-vector-product formulation; the legacy `perturbed_reinforce`
  variant keeps the original REINFORCE-on-scalar-objective form for
  reproducibility. Source: `openpto/method/Models/perturbed.py`.

## Statistical PnO [PnO, paper]

- **`nce` — Noise-Contrastive Estimation for PnO.**
  Mulamba, Mandi, Diligenti, Lombardi, Bucarey & Guns, *Contrastive
  Losses and Solution Caching for Predict-and-Optimize*, IJCAI 2021 [49];
  builds on the original NCE of Gutmann & Hyvärinen, AISTATS 2010 [29].
  Source: `openpto/method/Models/NCE.py::NCE`.
- **`pointLTR`, `pairLTR`, `listLTR` — Decision-Focused Learning to Rank.**
  Mandi, Bucarey, Tchomba & Guns, *Decision-Focused Learning: Through the
  Lens of Learning to Rank*, ICML 2022 [44]; the underlying pointwise /
  pairwise / listwise ranking losses are due to Caruana, Baluja & Mitchell
  (NIPS 1995 [11]), Joachims (KDD 2002 [35]), and Cao, Qin, Liu, Tsai &
  Li (ICML 2007 [10]) respectively. Source:
  `openpto/method/Models/LTR.py`.

## Surrogate-objective PnO [PnO, paper]

- **`lodl` — Locally Optimised Decision Losses.**
  Shah, Wang, Wilder, Perrault & Tambe, *Decision-Focused Learning Without
  Decision-Making: Learning Locally Optimized Decision Losses*, NeurIPS
  2022 [60]. Source: `openpto/method/Models/LODLs.py`.

## Methods added in this fork [PnO, fork]

- **`pg` — Perturbation Gradient (PG).**
  Gupta & Huang, *Perturbation-Gradient Losses for Decision-Focused
  Learning*, *arXiv* 2402.03256 (2024). Deterministic finite-difference
  surrogate that uses the ground-truth cost vector as the perturbation
  direction. The header docstring of `openpto/method/Models/PG.py` cites
  this reference directly. Not in the original PtOPnO paper.
- **`dad` — Decision-Aware Denoising (DAD).**
  Gupta, Huang & Rusmevichientong, *Decision-Aware Denoising*, 2024
  (cited verbatim in the header of `openpto/method/Models/DAD.py`).
  Implements the Stein-corrected objective `J_stein = obj − stein_weight ·
  stein_bias`; the per-item variance-adaptive perturbation lifts the
  paper's exact-Stein result for top-K to a Monte-Carlo / reparameterised
  estimator that applies to any CO problem in the benchmark. Not in the
  original PtOPnO paper.

## Method-citation summary table

| `--opt_model` | Category | Citation (PtOPnO ref #) |
|---|---|---|
| `mse` / `bce` / `ce` / `mae` / `msesum` | PtO two-stage | Bertsimas & Kallus, *Manag. Sci.* 2020 [7] |
| `dfl` | PnO – discrete (straight-through) | Bengio et al. 2013 [5] / Shah et al. 2022 [60] |
| `cpLayer` | PnO – continuous (KKT) | Agrawal et al., NeurIPS 2019 [1] |
| `qptl` | PnO – continuous (KKT) | Wilder, Dilkina & Tambe, AAAI 2019 [72] |
| `spo` | PnO – discrete (subgradient) | Elmachtoub & Grigas 2022 [20]; Mandi et al. AAAI 2020 [46] |
| `blackbox` | PnO – discrete (gradient interp.) | Pogančić et al., ICLR 2020 [54] |
| `identity` | PnO – discrete (identity proj.) | Sahoo et al., ICLR 2023 [58] |
| `perturb` | PnO – discrete (perturbed solver) | Berthet et al., NeurIPS 2020 [6] |
| `nce` | PnO – statistical | Mulamba et al., IJCAI 2021 [49] |
| `pointLTR` / `pairLTR` / `listLTR` | PnO – statistical (LTR) | Mandi et al., ICML 2022 [44] |
| `lodl` | PnO – surrogate objective | Shah et al., NeurIPS 2022 [60] |
| `pg` (fork) | PnO – discrete (FD surrogate) | Gupta & Huang, *arXiv* 2402.03256 (2024) |
| `dad` (fork) | PnO – surrogate (Stein-corrected) | Gupta, Huang & Rusmevichientong, 2024 |
