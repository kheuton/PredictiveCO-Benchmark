# Plan: σ=0 Fast Path — Native Solver Gradients in Perturbed Method

**Status**: Planned  
**Date**: 2026-02-26  

## Motivation

For problems with differentiable solvers (e.g. portfolio via `CvxpyLayer`), the
perturbation method's REINFORCE gradient is strictly worse than the solver's
native KKT-based gradient. The current code always uses REINFORCE, even when the
solver can provide exact gradients—because `perturbedOptFunc.apply()` detaches
from the solver's computation graph.

cpLayer already uses native gradients and gets **test_regret = 0.269** on
portfolio vs perturb's best of **0.278**. But cpLayer optimises the raw
objective, not the regret loss. The σ=0 fast path would let us test whether the
**regret loss formulation + native QP gradient** beats cpLayer.

## Design

Add a fast path in `perturbed.forward()` (around line 295 of
`openpto/method/Models/perturbed.py`). When `self.sigma == 0`:

1. **Skip `perturbedOptFunc.apply()`** entirely.
2. Call the solver **with gradients attached** (no `.detach()`, no `.cpu()`):
   ```python
   z_hat, _ = problem.get_decision(
       coeff_hat, params, self.ptoSolver, isTrain=True, **problem.init_API()
   )
   e_obj = problem.get_objective(coeff_true, z_hat, params)
   ```
3. Compute the same regret loss: `loss = |e_obj - obj_star|`.
4. Gradients flow through `get_objective → z_hat → get_decision → coeff_hat`
   via cvxpylayers' implicit differentiation.

### Caveats

- `get_decision` for portfolio doesn't accept `isTrain`; it always returns
  differentiable solutions from `CvxpyLayer`. Verify this is true and that
  calling it without `.detach().cpu()` works (device issues?).
- For Gurobi-based solvers (knapsack, bipartite, etc.), `sigma=0` is
  meaningless — the solver is non-differentiable and the fast path would
  produce zero gradients. Guard with a warning or restrict to cvxpy solvers.
- The portfolio `get_decision` moves inputs to CPU (`Y = Y.cpu()`). This may
  break autograd if the GPU tensor's grad_fn is lost. May need a small fix in
  `PortfolioOpt.get_decision` to preserve the computation graph.

## Files to Change

1. **`openpto/method/Models/perturbed.py`** — ~20 lines in `forward()`:
   - Before `perturbedOptFunc.apply(...)`, add:
     ```python
     if self.sigma == 0:
         z_hat, _ = problem.get_decision(
             coeff_hat, params, self.ptoSolver, **problem.init_API()
         )
         z_hat = torch.as_tensor(z_hat, device=coeff_hat.device, dtype=coeff_hat.dtype)
         e_obj = problem.get_objective(coeff_true, z_hat, params)
         # ... then fall through to regret computation
     ```
   - Wrap the existing `perturbedOptFunc.apply()` call in `else:`.

2. **`openpto/problems/PortfolioOpt.py`** — May need to remove `.cpu()` calls
   when gradients are required, or add an `isTrain` flag to conditionally
   preserve the computation graph.

3. **New YAML**: `openpto/config/models/perturb_s0_n1.yaml` with `sigma: 0`.

4. **Experiment**: Submit portfolio-only job with `sigma=0`, compare against
   cpLayer (0.269) and best perturb (0.278).

## Expected Outcome

If the regret loss is better than cpLayer's direct objective loss, we should
see test_regret < 0.269. If it matches ~0.269, it confirms the gradient
estimation (not the loss) was the bottleneck. If it's worse, it means the
regret formulation doesn't help for portfolio.
