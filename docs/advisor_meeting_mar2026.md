# Advisor Meeting Notes — March 2026

Two main topics:
1. Adaptive sigma control via Hamming-rate targeting (per-instance and per-item)
2. MSE + minibatch noise as a surprisingly strong baseline across all 7 benchmark problems

---

## 1. Adaptive Sigma via Hamming-Rate Controller

### Motivation

The perturbed optimizer (Berthet et al. DPO) draws noisy perturbations ε ~ N(0, σ²I) and
forms a soft decision z̃ = E_ε[z*(ĉ + σε)]. The signal quality depends heavily on σ:

- **σ too small**: all perturbations give the same optimal solution → gradient is zero
- **σ too large**: perturbations dominate the cost signal → solutions become random → gradient is noise

The question is how to set σ adaptively so it stays in the useful regime throughout training.

### The Hamming-Rate Controller

**Idea**: adjust σ so that, on average, exactly *k* items in the solution change when you
replace ĉ with ĉ + σε. This directly measures whether σ is "doing something useful."

**Target parameterization**: For a D-item knapsack, a Hamming-rate target of

    t = k / D

means we want k items to flip on each perturbation draw. For example, with D=20 items,
t = 0.05 corresponds to exactly **1 flip** — the minimum non-trivial perturbation size.
t = 0.10 = 2 flips, t = 0.20 = 4 flips, etc.

The controller runs per-instance: each training example n has its own σ_n, updated after
each epoch by comparing the measured per-instance Hamming rate to the target t.

### Per-Instance Results (20-item knapsack, benchmark scale)

![Per-instance Hamming sweep](../saved_records/knapsack-gen/perturb_adaptive_sweep/fig_kn_bench_hamming.png)

*Sweep over 5 targets × 3 LRs. Left: val regret vs target (solid=constant, dashed=decay).
Right: training curves for best configs.*

**Key results (constant target, best per LR):**

| LR    | Best test regret | Target |
|-------|-----------------|--------|
| 1e-2  | **0.087**       | 0.10   |
| 5e-3  | 0.095           | 0.10   |
| 5e-2  | 0.110           | 0.05   |
| MSE baseline | **0.064** | — |

**Target decay (linearly to 0 over epochs 100–300) makes things worse** — as σ → 0 the
gradient vanishes and the model freezes at a suboptimal solution.

The per-instance controller is competitive with the OCV-based controller (best ~0.084)
but neither beats MSE (0.064). The gap appears to be a local-minimum problem in weight space,
not a sigma-calibration problem.

---

### Extending to Per-Item Sigma

The per-instance controller gives every example the same target t. A natural refinement:
give each *item* its own σ, targeting the per-item Hamming rate separately. Items the model
gets wrong could be given a higher sigma (more exploration around their boundary), while
items the model already predicts correctly could be given lower sigma.

![Per-item summary](../saved_records/knapsack-gen/oracle_ceiling/fig_kn_peritem_summary.png)

*3-panel summary. Left: test regret by controller family (all worse than MSE).
Center: sigma trajectories over training (log scale) — sigma balloons to 30–100× in failing variants.
Right: val regret training curves showing divergence from MSE.*

**All per-item variants fail** (best test regret: 0.142 vs MSE 0.064).

---

### Why Does Sigma Blow Up for Some Items?

![Per-item sigma heatmap](../saved_records/knapsack-gen/oracle_ceiling/fig_kn_per_item_heatmap_t0.100_lr1e-2.png)

*Epoch × item heatmap of log₁₀(σ_item) for per-item Hamming controller (target=0.10, lr=1e-2).
Rows = epochs 1–150; columns = items 0–19. Bright = large σ, dark = small σ.*

The heatmap reveals a sharp split: a handful of items have σ → ∞ (bright columns) while
others stay low or collapse. The divergence happens within the first ~30 epochs.

![Per-item diagnostics](../saved_records/knapsack-gen/item_analysis/fig_kn_item_analysis.png)

*4-panel per-item breakdown. Top-left: item weights (heavier = harder to include).
Top-right: z* inclusion rate (fraction of training instances where item is in optimal solution).
Bottom-left: prediction error rate. Bottom-right: weight vs error scatter.*

**The split is explained by item weight:**

| Class | Items | Weight | z* rate | Error rate | Sigma behavior |
|-------|-------|--------|---------|------------|----------------|
| Light (boundary) | 2, 9, 13, 19 | 3.3–5.8 | 0.44–0.68 | 0.22–0.24 | **Balloon** (σ → 30–100) |
| Heavy (usually out) | 1, 3, 5, 8, 12, 14, 17 | 6.4–7.8 | 0.13–0.22 | 0.07–0.15 | Low / stable |

**Why light items balloon**: These items sit near the capacity boundary — whether they
are included depends sensitively on the other items chosen. Even with a large perturbation
σ, the combinatorial structure means the solution rarely flips these items (other heavier
items soak up the perturbation budget first). The controller measures Hamming rate < target
→ increases σ → still no flip → increases σ again → runaway.

**Why heavy items stay low**: Heavy items are almost always excluded from z* (weight too
large relative to capacity). The model learns this quickly, prediction error is low, and
the Hamming target is easily met with small σ.

**Root cause**: The Hamming rate is **not monotone in σ** for combinatorially rigid items.
Increasing σ indefinitely cannot make a structurally fixed item flip more often. The same
4 items (2, 9, 13, 19) balloon in *every* per-item variant tested, regardless of target
value (0.05, 0.10, 0.20) or sigma ceiling.

![Per-item training dynamics](../saved_records/knapsack-gen/oracle_ceiling/fig_kn_per_item_dynamics.png)

*Training dynamics for all 4 per-item variants (A–D) at lr=1e-2.
Top-left: val regret over training. Top-right: mean σ trajectory.
Bottom-left: per-item Hamming rate. Bottom-right: σ spread ratio (max/mean) — higher = more item disparity.*

---

### Summary: Hamming Controller

| Controller | Best test | Notes |
|---|---|---|
| Per-instance, constant target (t=0.10) | 0.087 | Best adaptive result |
| Per-instance, target decay | 0.093 | Decay hurts — gradient vanishes |
| Per-item Hamming (all 4 variants) | 0.142 | Sigma ballooning |
| MSE baseline | **0.064** | |
| Oracle ceiling (noiseless DGP) | 0.052 | |

The per-instance controller is the best we have found. The per-item extension fails because
the Hamming-rate control signal is unreliable for combinatorially rigid items.

---

## 2. MSE + Minibatch Is a Surprisingly Strong Baseline

### Finding

A simple sweep over batch size and learning rate reveals that **SGD minibatch noise
dramatically improves MSE** across almost all 7 benchmark problems — without any
algorithmic change.

**Setup**: 7 problems × 5 LRs × 4 batch configs (full-batch GD, bs=32, 64, 128) = 140 jobs.
All runs use the same network architecture (dense MLP) and 300 epochs.

### Results

| Problem | Old MSE (full-batch) | Best MSE (tuned) | Best config | Improvement |
|---------|---------------------|-----------------|-------------|-------------|
| knapsack | 0.0640 | **0.0537** | bs=32, lr=5e-2 | +16% |
| knapsack-real | 0.0840 | **0.0722** | bs=32, lr=5e-2 | +14% |
| energy | 0.0210 | **0.0197** | bs=32, lr=5e-2 | +6% |
| budgetalloc | 0.706 | **0.366** | bs=32, lr=1e-3 | +48% |
| cubic | 0.0010 | **0.0002** | bs=32, lr=5e-2 | +80% |
| bipartitematching | 0.924 | **0.917** | gd, lr=5e-3 | +1% |
| portfolio | 0.253 | **0.221** | gd, lr=5e-3 (abs) | +13% |

**bs=32 wins for 5/7 problems.** bipartitematching is batch-invariant (only 20 training
instances — all batch sizes give identical results). Portfolio also prefers full-batch
(small dataset, batch=32 would be just 4 samples).

### What Explains the Improvement?

Minibatch SGD adds gradient noise proportional to 1/√bs. For MSE, this acts as a
regularizer — it prevents the model from overfitting to the particular noise realization
in the training set, making the predictions generalize better to the test instances.

This is consistent with the knapsack minibatch sweep (Wave 8), where:
- MSE: 0.064 → **0.054** with bs=32
- Plain DPO: 0.074 → **0.084** with bs=32 (worse — DPO gradient variance compounds with batch noise)
- MSE + λ=10 reg: 0.061 → **0.056** with bs=128

The DPO method is hurt by small batches because the per-instance perturbed gradient already
has high variance. Adding batch variance on top makes the signal worse.

### Updated Multi-Problem Comparison

With the new MSE baselines, the picture is clear:

| Problem | Objective | Best MSE | Best DPO | Winner |
|---------|-----------|----------|----------|--------|
| knapsack | linear | **0.054** | 0.093 | MSE by 73% |
| knapsack-real | linear | **0.072** | 0.092 | MSE by 28% |
| energy | linear | **0.0197** | 0.0195 | ~tie (1%) |
| **budgetalloc** | **nonlinear** | 0.366 | **0.032** | **DPO by 91%** |
| cubic | nonlinear† | **0.0002** | 1.52 | MSE by >1000× |
| bipartitematching | linear | **0.917** | 0.944 | MSE by 3% |
| portfolio | convex-QP | **0.221** | 0.250 | MSE by 13% |

†Cubic: the *prediction* problem is nonlinear (Y = 10(X³ − 0.65X)) but the *optimization*
objective is linear in predicted costs. DPO fails due to coefficient-norm blowup, not misspecification.

**The takeaway**: DPO beats MSE only on budget allocation, the one problem where the
CO objective is genuinely nonlinear (submodular) in the predicted values — and even there,
MSE with minibatch tuning reduces the gap from 22× to 11×.

---

## Open Questions / Next Steps

1. **Budget allocation DPO surgery**: the surg_all perturb jobs stalled at epoch 20 and
   need resubmission. We want to know if surgery helps on the one problem where DPO actually works.

2. **Why does minibatch help MSE so much on cubic?** The 5× improvement (0.001 → 0.0002)
   on a noiseless DGP is surprising — minibatch noise shouldn't regularize a noiseless problem.
   This may be an optimization effect (escaping local minima) rather than generalization.

3. **Hamming rigidity**: Items 2, 9, 13, 19 are structurally rigid — what would a controller
   that accounts for this look like? One idea: detect ballooning items early and cap their
   sigma at the per-instance mean rather than continuing to drive it up.
