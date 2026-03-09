# Perturbed Method: Sigma Sweep Diagnostic Experiment

## Motivation

The `perturbed` method performs middle-of-the-pack despite extensive hyperparameter search. The core issue is sigma: too small → zero gradient; too large → noisy gradient. The right sigma depends on cost magnitude, which varies by task and changes during training.

This experiment produces diagnostics to guide the design of an adaptive sigma algorithm.

## Setup

| Parameter | Value |
|---|---|
| Primary task | cubic (fast, known DGP, hard regret gap) |
| Future task | knapsack |
| Sigma values | 10, log-spaced 0.001–10 |
| Prediction models | MLP (standard), PolyPredModel (correctly specified) |
| n_samples | 100 |
| Sigma schedule | constant |
| FD gradient | every epoch |
| Logging | every epoch |

### Why cubic?
- Very fast (TopK solver = just sorting)
- Known DGP: `Y = 10(X³ - 0.65X)`, so we can build a correctly-specified model
- Has an "easy" solution (~1% regret via ranking) and a "hard" solution (~0%) — easy to tell if a method is learning anything useful

### MLP vs. PolyPredModel
- **MLP**: standard benchmark model, may be misspecified
- **PolyPredModel**: matches the true functional form, isolates gradient quality from model misspecification

## Diagnostic Variables

| Variable | How computed |
|---|---|
| `sigma_ratio` | `sigma / mean(|coeff_hat|)` — effective dimensionless ratio |
| `decision_regret` | Regret using hard TopK at eval |
| `cosine_sim` | Cosine similarity between perturbed backprop gradient and finite-difference gradient of hard regret |
| `rank_change_rate` | Fraction of perturbed solutions that differ from the unperturbed hard solution, normalized by 2K |
| `softness` | `mean(z̄_i · (1 - z̄_i))` |
| `dist_binary` | `‖z̄ - round(z̄)‖₂` |
| `entropy` | Binary entropy of z̄ |
| `grad_norm` | `‖∇ loss‖` |
| `fd_grad_norm` | `‖fd gradient‖` |

### Rank change rate
Measures whether sigma is causing meaningful exploration:
```
rank_change_rate = mean_n( ‖z_n - z_0‖₀ ) / (2K)
```
where z_0 = TopK(coeff_hat), z_n = TopK(coeff_hat + σεₙ). Range [0,1].
- ~0: sigma too small, all perturbations identical, zero gradient
- ~1: sigma too large, perturbations random, gradient is noise

### Finite difference gradient
Gradient of the *hard* decision regret w.r.t. coeff_hat via forward differences. For cubic (20 items, trivial sort solver) this is cheap. Provides a reference to measure whether the perturbed backprop gradient points in a useful direction.

## Key Visualization
Scatter of `cosine_sim` vs `rank_change_rate`, colored by sigma, across all epochs and both model types. Hypothesis: there is a sweet spot rank_change_rate that maximizes gradient quality — this would motivate an adaptive sigma that targets that rate.

## Files

| File | Purpose |
|---|---|
| `openpto/method/Predicts/poly_model.py` | PolyPredModel — works for any polynomial DGP problem |
| `openpto/method/Models/perturb_diag.py` | PerturbDiag — subclass of perturbed, stores z_bar and per-sample solutions |
| `openpto/diagnostics/perturb_metrics.py` | fd_gradient(), interiority_metrics(), rank_change_rate() |
| `rethink_exp/perturb_sigma_sweep.py` | Main experiment script (configurable to task) |
| `notebooks/perturb_sigma_analysis.ipynb` | Visualization |

## DGP Notes

**Cubic**: `Y = 10(X³ - 0.65X)`, X ~ Uniform[-1,1], 1 feature per item, no noise.
PolyPredModel: linear(1→1) per item + cubic polynomial. Essentially no free parameters for a correctly specified model.

**Knapsack (gen)**: `value ∝ (B @ feat / √d + 3)^poly_deg * noise`, B ~ Bernoulli(0.5) fixed random matrix.
PolyPredModel: linear(num_features→num_items) + degree-4 polynomial. Must learn B — still a real learning problem.
Both problems use `get_model_shape()` to return the correct dims, so PolyPredModel needs no problem-specific logic.
