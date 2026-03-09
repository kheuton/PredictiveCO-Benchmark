"""
Problem-agnostic diagnostic metrics for the perturbed optimizer.

All functions operate on detached tensors / numpy arrays and make no assumptions
about problem structure beyond the interfaces already defined on PTOProblem.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Interiority metrics
# ---------------------------------------------------------------------------

def interiority_metrics(z_bar: torch.Tensor) -> dict:
    """
    Compute interiority metrics for soft decision z_bar ∈ [0, 1]^(B, D).

    Returns a dict with:
      softness    — mean z̄_i(1 - z̄_i), peaks at 0.25 when maximally uncertain
      dist_binary — ‖z̄ - round(z̄)‖₂ / B, average distance to nearest binary solution
      entropy     — mean binary entropy H(z̄_i), peaks when z̄_i = 0.5
    """
    z = z_bar.float().detach()
    eps = 1e-7
    B = z.shape[0]

    softness = (z * (1.0 - z)).mean().item()

    dist_binary = (z - z.round()).norm().item() / B

    z_c = z.clamp(eps, 1.0 - eps)
    entropy = -(z_c * z_c.log() + (1.0 - z_c) * (1.0 - z_c).log()).mean().item()

    return {"softness": softness, "dist_binary": dist_binary, "entropy": entropy}


# ---------------------------------------------------------------------------
# Rank change rate
# ---------------------------------------------------------------------------

def rank_change_rate(
    perturbed_solutions: torch.Tensor,
    unperturbed_sol: torch.Tensor,
    K: int,
) -> float:
    """
    Fraction of perturbed solutions that differ from the unperturbed hard solution,
    normalised by 2K (so the result is in [0, 1]).

    perturbed_solutions : (N, B, n_items) — binary integer tensors
    unperturbed_sol     : (B, n_items)    — binary integer tensor
    K                   : budget (number of items selected)

    0 → every perturbation yields the same solution (sigma too small, zero gradient)
    1 → every perturbation yields a completely different solution (sigma too large)
    """
    sols = perturbed_solutions.float().detach()
    ref = unperturbed_sol.float().detach()

    # Hamming distance per sample per instance: (N, B)
    hamming = (sols - ref.unsqueeze(0)).abs().sum(dim=-1)
    return (hamming / (2.0 * K)).mean().item()


# ---------------------------------------------------------------------------
# Finite-difference gradient
# ---------------------------------------------------------------------------

def fd_gradient(
    coeff_hat: torch.Tensor,
    coeff_true: torch.Tensor,
    problem,
    ptoSolver,
    params,
    epsilon: float = 1e-3,
    opt_obj: torch.Tensor | None = None,
    max_instances: int | None = None,
) -> torch.Tensor:
    """
    Finite-difference gradient of the hard-decision regret w.r.t. coeff_hat.

        regret(c) = |f(coeff_true, z(c)) − f(coeff_true, z(coeff_true))|
        fd_grad[b, i] ≈ [regret(coeff_hat + ε·eᵢ) − regret(coeff_hat)] / ε

    All solver calls are made on CPU with no gradient tracking.

    Args:
        coeff_hat      : (B, ...) predicted costs (will be detached + moved to CPU)
        coeff_true     : (B, ...) ground-truth costs
        problem        : PTOProblem instance
        ptoSolver      : solver instance
        params         : auxiliary params (e.g. knapsack weights), may be None
        epsilon        : finite-difference step size
        opt_obj        : (B,) precomputed f(coeff_true, z*) to avoid recomputing.
                         If None it is computed here.
        max_instances  : if set, use only the first N instances (for speed)

    Returns:
        Gradient tensor of the same shape as coeff_hat[:max_instances].
    """
    coeff_hat = coeff_hat.detach().cpu()
    coeff_true = coeff_true.detach().cpu()

    if max_instances is not None:
        coeff_hat = coeff_hat[:max_instances]
        coeff_true = coeff_true[:max_instances]
        if params is not None and torch.is_tensor(params):
            params = params[:max_instances]

    B = coeff_hat.shape[0]
    D = coeff_hat[0].numel()
    init_api = problem.init_API()

    # --- Base regret ---
    z0, _ = problem.get_decision(coeff_hat, params, ptoSolver, **init_api)
    z0 = _to_tensor(z0, coeff_hat.dtype)
    obj0 = problem.get_objective(coeff_true, z0, params)
    obj0 = _to_tensor(obj0, coeff_hat.dtype)  # (B,)

    if opt_obj is None:
        z_star, _ = problem.get_decision(coeff_true, params, ptoSolver, **init_api)
        z_star = _to_tensor(z_star, coeff_hat.dtype)
        opt_obj_b = problem.get_objective(coeff_true, z_star, params)
        opt_obj_b = _to_tensor(opt_obj_b, coeff_hat.dtype)
    else:
        opt_obj_b = _to_tensor(opt_obj, coeff_hat.dtype)[:B]

    regret0 = (obj0 - opt_obj_b).abs()  # (B,)

    # --- FD loop over each dimension ---
    coeff_flat = coeff_hat.reshape(B, D)
    grad_flat = torch.zeros(B, D, dtype=coeff_hat.dtype)

    for i in range(D):
        perturbed = coeff_flat.clone()
        perturbed[:, i] += epsilon
        perturbed = perturbed.reshape(coeff_hat.shape)

        z_i, _ = problem.get_decision(perturbed, params, ptoSolver, **init_api)
        z_i = _to_tensor(z_i, coeff_hat.dtype)
        obj_i = problem.get_objective(coeff_true, z_i, params)
        obj_i = _to_tensor(obj_i, coeff_hat.dtype)
        regret_i = (obj_i - opt_obj_b).abs()

        grad_flat[:, i] = (regret_i - regret0) / epsilon

    return grad_flat.reshape(coeff_hat.shape)


def _to_tensor(x, dtype):
    if torch.is_tensor(x):
        return x.detach().cpu().to(dtype)
    return torch.as_tensor(x, dtype=dtype)
