# Paper Message — April 2026

## Core Finding

**DPO methods (perturbed, SPO+, etc.) only beat MSE when the objective function is nonlinear with respect to predicted costs.**

- **Linear objectives** (knapsack, energy, cubic): MSE predictions directly translate to good decisions. Prediction error and decision error are aligned. δ ≈ 0 → ETO (MSE) wins.
- **Nonlinear/submodular objectives** (budget allocation): prediction errors interact nonlinearly with the optimization, so minimizing MSE is not aligned with minimizing regret. δ >> 0 → IEO (DPO) can win.

This is a practical, oracle-free diagnostic: **inspect the objective structure** before choosing a method.

---

## Benchmark Evidence

| Problem | Objective | MSE rel. regret | Best Perturb | Winner |
|---|---|---|---|---|
| knapsack-gen | linear (∑cᵢzᵢ) | ~6.4% | ~7.4% | **MSE** |
| knapsack-real | linear | ~8.4% | ~9.2% | **MSE** |
| energy | linear | ~2.1% | — | **MSE** |
| cubic | linear | ~0.1% | >>1.0 | **MSE** |
| bipartite matching | linear | ~92% | ~92% | tie |
| portfolio | nonlinear (exp utility) | ~6.90 | ~6.82 | ~tie / DPO slight edge |
| **budget allocation** | **submodular** (P(≥1 click)) | **~70.6%** | **~3.2%** | **DPO 22× better** |

Budget allocation is the single problem where MSE catastrophically fails and DPO wins decisively.

---

## Why Budget Allocation Is Different

The objective is P(at least one click) = 1 − ∏(1 − pᵢxᵢ).

- Optimal allocation depends on **marginal gains** in a multiplicative, diminishing-returns structure
- A small error in predicted pᵢ can flip which channel is at the margin, because marginal value depends on the **joint product** of all other probabilities
- MSE optimizes each pᵢ prediction independently, which is misaligned with what the submodular optimizer needs
- DPO gradient accounts for how the full allocation changes with predictions → correct signal

For knapsack (linear objective), predicting cᵢ well *is* predicting the decision well. For budget allocation, it isn't.

---

## Theoretical Backing

Elmachtoub et al. (2025, AISTATS) — "Dissecting the Impact of Model Misspecification":
- Define δ = v₀(w_{θ_KL}) − v₀(w_{θ*}) = regret gap between best ETO and best IEO solutions at population level
- **δ >> 0** (misspecified): IEO has "universal double benefit" in the two leading regret terms → IEO wins
- **δ ≈ 0** (well-specified): ETO has smaller estimation variance → ETO wins in the second-order term

Our finding: linearity of the objective is the practical determinant of δ.
- Linear objectives → best MSE solution ≈ best decision-optimal solution → δ ≈ 0
- Nonlinear objectives → MSE solution can be systematically suboptimal for decisions → δ >> 0

---

## Why DPO Still Fails Even When δ > 0 (Knapsack Mechanistic Story)

Even for knapsack (δ ≈ 0), it's worth noting *why* DPO fails mechanistically:

1. **Incoherent gradients**: per-instance DPO gradients point in different directions (cosine ∈ −0.2 to +0.4). Each instance makes different pairwise item swaps, so the average gradient has almost no alignment with the MSE-improving direction (0.9% of ‖G‖ at the local min).
2. **Gradient direction problem**: the DPO local minimum is far from the MSE solution. The perturbed loss pushes weights *away* from the MSE solution even from a warm start.
3. **MSE gradient is coherent**: every instance agrees on the item-level bias direction → clean, consistent signal.

For budget allocation, DPO likely has more coherent gradients because the submodular structure creates a consistent "prefer channels with higher marginal gain" signal across instances.

---

## Practical Recommendation

1. **Check objective linearity**: if your CO problem has a linear (or near-linear) objective in predicted costs, use MSE with minibatch SGD. Tune batch size and LR.
2. **If the objective is nonlinear** (submodular, exponential utility, products of probabilities): DPO methods are worth trying; plain perturb with moderate σ and lr=5e-3 is a reasonable starting point.
3. **Minibatch SGD matters even for MSE**: bs=32, lr=5e-2 gets knapsack from 0.064 → 0.054, approaching oracle ceiling.

---

## Open Questions

- Portfolio has an exponential utility (nonlinear) yet DPO only marginally edges out MSE (~1%). Why? The sigma=0 result (direct regret minimization) in the LODL paper suggests the problem has smooth, informative gradients — a different mechanism than DPO smoothing.
- Can gradient coherence be measured cheaply as a diagnostic? (cosine similarity across per-instance gradients at initialization)
- Does the linear/nonlinear split generalize beyond this benchmark? Are there other common nonlinear objective types (beyond submodular) where DPO helps?
