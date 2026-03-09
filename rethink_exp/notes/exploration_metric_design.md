# Exploration Metric Design: Beyond RCR

## Motivation

RCR (Rank Change Rate) was the original metric for controlling sigma in the adaptive controller.
Two problems motivated replacing it:

### Problem 1: Non-binary decisions
RCR measures Hamming distance between perturbed solutions and z0, normalised by 2K
(number of items selected). This has no natural generalisation to non-binary CO problems
(flows, assignments, continuous portfolios, etc.).

### Problem 2: Adversarial cardinality changes
Consider a knapsack where z0 selects one heavy item, but a perturbed solution selects 100
light items of similar total value. RCR = 101/2 >> 1, despite the two solutions being nearly
equivalent in objective value. The metric is sensitive to the *topology* of the change, not
its *significance*.

---

## Proposed Metric: OCV_Y (Objective Coefficient of Variation under true costs)

```
OCV_Y = std_n(Y · z_n) / |Y · z_0|
```

where:
- Y is the true cost vector
- z_0 = solver(c_hat) — unperturbed hard decision
- z_n = solver(c_hat + σε_n) — perturbed solutions

### Why Y, not c_hat?

Using predicted costs c_hat is a mathematical convenience (z_0 = argmax c_hat·z, so
c_hat·z_n ≤ c_hat·z_0 always). But it only reflects true objective diversity when the
model is well-calibrated. Early in training, c_hat·z_n can be large even when Y·z_n ≈ Y·z_0
(genuinely equivalent solutions). Y is what matters for decision quality.

### Properties

| Property | OCV_Y |
|---|---|
| Generalises beyond binary | Yes — Y·z_n is a scalar for any CO problem |
| Adversarial case (heavy→100 light, similar Y value) | std ≈ 0 → OCV_Y ≈ 0. Correct: these solutions are equivalent |
| sigma → 0 | All z_n → z_0, std → 0 |
| sigma → ∞ | z_n random, std saturates at problem-determined ceiling |
| Monotone in sigma | Yes (approximately) — critical for use as a control target |

---

## FracImproving: Motivation and Why It's Not a Control Target

A natural candidate was:
```
FracImproving = mean_n[ 1{ Y·z_n > Y·z_0 } ]
```

The appeal: we'd prefer perturbed solutions that are *better* than z_0 under true costs,
as they provide direct improvement signal. Worse solutions also provide useful gradient
(contrastive signal), but improving solutions provide a more direct learning signal.

### Why FracImproving fails as a control target

FracImproving is NOT monotone in sigma:

- sigma ≈ 0: z_n ≈ z_0, FracImproving ≈ 0
- sigma medium: perturbations explore the neighbourhood, some find better solutions → FracImproving > 0
- sigma very large: z_n ≈ random draws from Z; if z_0 is already decent, random solutions
  mostly lose → FracImproving small again

Three distinct regimes all give FracImproving ≈ 0:
1. Sigma too small (no exploration)
2. Sigma too large (random, mostly worse)
3. Model already optimal (nothing beats z_0)

A controller cannot determine which direction to push sigma from FracImproving alone.

### OCV_Y is the continuous analogue of FracImproving

FracImproving counts binary events (Y·z_n > Y·z_0). OCV_Y measures the *spread* of Y·z_n
values — a soft, continuous version of the same idea. It inherits the conceptual motivation
(care about true objective diversity) while being monotone in sigma and therefore usable as
a control target.

---

## Two-Metric System

| Metric | Role |
|---|---|
| OCV_Y | Control target — monotone in sigma, tells controller which direction to push |
| FracImproving | Convergence diagnostic — if OCV_Y is at target and FracImproving ≈ 0, model is near-optimal |

Controller logic:
1. Drive sigma so OCV_Y hits a target value
2. Monitor FracImproving as a diagnostic
3. If OCV_Y is at target and FracImproving = 0 regardless of sigma: model has converged,
   perturbations cannot find improving solutions → training may be near-optimal

This is cleaner than the original RCR-based controller:
- Works for any CO problem
- Robust to adversarial solution topologies
- FracImproving = 0 is a meaningful signal (convergence), not just "sigma too small"

---

## Open Questions

1. What is the right OCV_Y target? (Analogous to finding RCR sweet spot at 0.3–0.4)
2. How does OCV_Y behave on knapsack vs cubic — is the target transferable across problems?
3. Normalisation: if |Y·z_0| ≈ 0, use mean_n(Y·z_n) instead, or add epsilon
4. Should OCV_Y use the coefficient of variation (std/mean) rather than std/|z_0 value|
   to be robust to the sign of the objective?
