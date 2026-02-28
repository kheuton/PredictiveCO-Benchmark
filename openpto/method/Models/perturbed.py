#!/usr/bin/env python
# coding: utf-8
"""
Perturbed optimization function

Based on the differentiable perturbation approach from:
  Berthet et al., "Learning with Differentiable Perturbed Optimizers", NeurIPS 2020.

The autograd backward pass follows the reference PyTorch implementation:
  https://github.com/tuero/perturbations-differential-pytorch
"""

import logging
import math

import numpy as np
import torch
from torch.distributions.gumbel import Gumbel
from torch.distributions.normal import Normal

from gurobipy import GRB  # pylint: disable=no-name-in-module

from openpto.method.Models.abcOptModel import optModel
from openpto.method.utils_method import do_reduction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise helpers (ported from reference perturbations.py)
# ---------------------------------------------------------------------------

_GUMBEL = "gumbel"
_NORMAL = "normal"
SUPPORTED_NOISES = (_GUMBEL, _NORMAL)


def sample_noise_with_gradients(noise, shape):
    """Sample noise and the gradient of log p(noise).

    For Normal noise the gradient equals the noise itself.
    For Gumbel noise the gradient is ``1 - exp(-noise)``.
    """
    if noise not in SUPPORTED_NOISES:
        raise ValueError(
            f"{noise} noise is not supported. Use one of {SUPPORTED_NOISES}"
        )
    if noise == _GUMBEL:
        sampler = Gumbel(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = 1 - torch.exp(-samples)
    elif noise == _NORMAL:
        sampler = Normal(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = samples
    return samples, gradients


# ---------------------------------------------------------------------------
# Sigma Scheduler
# ---------------------------------------------------------------------------

class SigmaScheduler:
    """
    Schedule the perturbation amplitude (sigma) over training epochs.

    Supported schedules:
        constant     – sigma is fixed (default behaviour).
        linear_decay – linear interpolation from sigma_start to sigma_end.
        cosine_decay – cosine annealing from sigma_start to sigma_end.
        step_decay   – multiply by gamma every step_size epochs.

    Usage:
        sched = SigmaScheduler("cosine_decay", sigma_start=1.0, sigma_end=0.01,
                               n_epochs=300)
        for epoch in range(1, 301):
            sigma = sched.step(epoch)
    """

    def __init__(
        self,
        schedule="constant",
        sigma_start=1.0,
        sigma_end=0.01,
        n_epochs=300,
        sigma_gamma=0.5,
        sigma_step_size=100,
        warmup_epochs=0,
    ):
        self.schedule = schedule
        self.sigma_start = float(sigma_start)
        self.sigma_end = float(sigma_end)
        self.n_epochs = max(n_epochs, 1)
        self.gamma = float(sigma_gamma)
        self.step_size = int(sigma_step_size)
        self.warmup_epochs = int(warmup_epochs)
        self.current_sigma = self.sigma_start

    def step(self, epoch):
        """Return the sigma value for the given *1-based* epoch.

        If ``warmup_epochs > 0``, sigma is held at ``sigma_start`` for the
        first ``warmup_epochs`` epochs; decay begins at epoch
        ``warmup_epochs + 1`` and runs over the remaining
        ``n_epochs - warmup_epochs`` epochs.
        """
        # During warmup, hold sigma constant at sigma_start
        if epoch <= self.warmup_epochs:
            self.current_sigma = self.sigma_start
            return self.current_sigma

        # Effective epoch and total for the decay phase
        decay_epoch = epoch - self.warmup_epochs
        decay_total = max(self.n_epochs - self.warmup_epochs, 1)

        if self.schedule == "constant":
            self.current_sigma = self.sigma_start
        elif self.schedule == "linear_decay":
            frac = min(decay_epoch / decay_total, 1.0)
            self.current_sigma = self.sigma_start + frac * (self.sigma_end - self.sigma_start)
        elif self.schedule == "cosine_decay":
            frac = min(decay_epoch / decay_total, 1.0)
            self.current_sigma = (
                self.sigma_end
                + 0.5 * (self.sigma_start - self.sigma_end) * (1 + math.cos(math.pi * frac))
            )
        elif self.schedule == "step_decay":
            n_steps = decay_epoch // self.step_size
            self.current_sigma = self.sigma_start * (self.gamma ** n_steps)
        else:
            raise ValueError(f"Unknown sigma schedule: {self.schedule}")
        return self.current_sigma

    def __repr__(self):
        warmup_str = f", warmup={self.warmup_epochs}" if self.warmup_epochs > 0 else ""
        return (
            f"SigmaScheduler(schedule={self.schedule}, start={self.sigma_start}, "
            f"end={self.sigma_end}, n_epochs={self.n_epochs}{warmup_str})"
        )


# ---------------------------------------------------------------------------
# Perturbed Optimization Model
# ---------------------------------------------------------------------------

class perturbed(optModel):
    """
    Reference:
    """

    def __init__(
        self,
        ptoSolver,
        n_samples=10,
        sigma=1.0,
        noise="normal",
        seed=135,
        output_activation="none",
        # Sigma scheduler params (optional)
        sigma_schedule="constant",
        sigma_start=None,
        sigma_end=0.01,
        sigma_n_epochs=300,
        sigma_gamma=0.5,
        sigma_step_size=100,
        sigma_warmup_epochs=0,
        **hyperparams,
    ):
        """
        Args:
            ptoSolver (optModel): an  optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            noise (str): noise distribution, 'normal' or 'gumbel'
            seed (int): random state seed
            output_activation (str): activation applied to predictions before
                perturbation. One of 'none', 'sigmoid', 'tanh', 'softplus'.
            sigma_schedule (str): 'constant', 'linear_decay', 'cosine_decay', 'step_decay'
            sigma_start (float): starting sigma (defaults to sigma if None)
            sigma_end (float): ending sigma for decay schedules
            sigma_n_epochs (int): total epochs for schedule computation
            sigma_gamma (float): multiplicative factor for step_decay
            sigma_step_size (int): epoch interval for step_decay
            sigma_warmup_epochs (int): hold sigma_start for this many epochs
                before starting the decay schedule (default: 0)

        """
        super().__init__(ptoSolver)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # noise distribution
        if noise not in SUPPORTED_NOISES:
            raise ValueError(
                f"{noise} noise not supported. Use one of {SUPPORTED_NOISES}"
            )
        self.noise = noise
        # random state (kept for solver reproducibility)
        self.rnd = np.random.RandomState(seed)
        # solution pool
        n_vars = ptoSolver.num_vars
        self.solpool = np.empty((0, n_vars), dtype=np.float64)

        # Output activation applied to predictions before perturbation
        _act_map = {
            "none": lambda x: x,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "softplus": torch.nn.functional.softplus,
        }
        if output_activation not in _act_map:
            raise ValueError(
                f"Unknown output_activation '{output_activation}'. "
                f"Choose from {list(_act_map.keys())}"
            )
        self.output_activation = _act_map[output_activation]

        # Per-epoch scale tracking
        self._batch_scales = []

        # Sigma scheduler
        if sigma_start is None:
            sigma_start = sigma
        self.sigma_scheduler = SigmaScheduler(
            schedule=sigma_schedule,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            n_epochs=sigma_n_epochs,
            sigma_gamma=sigma_gamma,
            sigma_step_size=sigma_step_size,
            warmup_epochs=sigma_warmup_epochs,
        )

    def step(self, epoch):
        """Update sigma according to the schedule. Called by ExpManager each epoch."""
        # Log scale diagnostics from the previous epoch
        if self._batch_scales:
            avg_scale = sum(self._batch_scales) / len(self._batch_scales)
            ratio = self.sigma / avg_scale if avg_scale > 0 else float('inf')
            logger.info(
                f"  [perturb] epoch {epoch}: "
                f"|coeff_hat| = {avg_scale:.4f}, "
                f"sigma = {self.sigma:.4f}, "
                f"sigma/|coeff_hat| = {ratio:.4f}"
            )
            self._batch_scales = []
        self.sigma = self.sigma_scheduler.step(epoch)
        return self.sigma

    def forward(
        self,
        problem,
        coeff_hat,
        params,
        coeff_true=None,
        **hyperparams,
    ):
        """
        Forward pass — expected regret formulation.

        loss = | f(θ*, z*) - E_ε[ f(θ*, z(θ̂ + σε)) ] |

        This is always non-negative and equals zero when every perturbed
        solution is optimal.  Gradients come from E[f] via REINFORCE;
        f(θ*, z*) is a detached constant.

        We use abs() rather than relying on modelSense for the sign,
        because modelSense describes the solver's internal direction
        (e.g. MINIMIZE a cost) which may differ from get_objective's
        direction (e.g. higher = better).
        """
        # Handle list inputs (e.g., advertising has variable-size instances
        # that cannot be batched into a single tensor)
        if isinstance(coeff_hat, list):
            losses = []
            for i in range(len(coeff_hat)):
                ch_i = coeff_hat[i].unsqueeze(0)   # (1, D1, ..., Dk)
                ct_i = coeff_true[i].unsqueeze(0)
                loss_i = self.forward(
                    problem, ch_i, params, ct_i, **hyperparams
                )
                losses.append(loss_i)
            return do_reduction(torch.stack(losses), hyperparams["reduction"])

        # Apply output activation (e.g. sigmoid) to bound predictions
        coeff_hat = self.output_activation(coeff_hat)

        # Track characteristic scale for diagnostics
        with torch.no_grad():
            self._batch_scales.append(coeff_hat.abs().mean().item())

        # E[obj(z_n)] via perturbedOptFunc — (B,)
        e_obj = perturbedOptFunc.apply(
            coeff_hat,
            self.ptoSolver,
            problem,
            coeff_true,
            params,
            self.n_samples,
            self.sigma,
            self.noise,
        )

        # Optimal objective f(θ*, z*) — constant, no gradient
        coeff_true_cpu = coeff_true.detach().cpu()
        z_star_np, _ = problem.get_decision(
            coeff_true_cpu, params, self.ptoSolver, **problem.init_API()
        )
        z_star = torch.as_tensor(
            z_star_np, device=coeff_hat.device, dtype=coeff_hat.dtype
        )
        obj_star = problem.get_objective(coeff_true, z_star, params)
        if not isinstance(obj_star, torch.Tensor):
            obj_star = torch.as_tensor(
                obj_star, device=coeff_hat.device, dtype=coeff_hat.dtype
            )
        obj_star = obj_star.to(device=coeff_hat.device, dtype=coeff_hat.dtype).detach()  # (B,)

        # Expected regret: abs makes this direction-agnostic
        regret = torch.abs(e_obj - obj_star)

        loss = do_reduction(regret, hyperparams["reduction"])
        return loss


class perturbedOptFunc(torch.autograd.Function):
    """
    Autograd function for perturbed optimization  (Berthet et al.).

    Follows the reference implementation from:
    https://github.com/tuero/perturbations-differential-pytorch

    Forward:  sample N perturbations of the cost vector, solve each perturbed
              problem, evaluate the objective on each *feasible discrete*
              solution using coeff_true, and return E[obj].
    Backward: REINFORCE-style gradient  ∂E[obj]/∂θ  using per-sample
              objectives as scalar scores.
    """

    @staticmethod
    def forward(
        ctx,
        coeff_hat,
        ptoSolver,
        problem,
        coeff_true,
        params,
        n_samples,
        sigma,
        noise_type,
    ):
        """
        Args:
            coeff_hat:   (B, D1, ..., Dk)  predicted cost coefficients
            ptoSolver:   combinatorial solver wrapper
            problem:     problem instance (get_decision, get_objective, init_API)
            coeff_true:  (B, D1, ..., Dk)  ground-truth cost coefficients
            params:      auxiliary params for solver / objective
            n_samples:   number of Monte-Carlo perturbation samples
            sigma:       perturbation amplitude
            noise_type:  'normal' or 'gumbel'
        Returns:
            e_obj: (B,)  E_eps[ obj(coeff_true, z(coeff_hat + sigma*eps)) ]
        """
        device = coeff_hat.device
        dtype = coeff_hat.dtype
        original_input_shape = coeff_hat.shape          # (B, D1, ..., Dk)
        B = original_input_shape[0]

        # ----- Sample noise on the correct device / dtype -----
        perturbed_input_shape = [n_samples] + list(original_input_shape)
        additive_noise, noise_gradient = sample_noise_with_gradients(
            noise_type, perturbed_input_shape
        )
        additive_noise = additive_noise.to(device=device, dtype=dtype)
        noise_gradient = noise_gradient.to(device=device, dtype=dtype)

        # ----- Perturbed inputs: (N, B, D1, ..., Dk) -----
        perturbed_input = coeff_hat.unsqueeze(0) + sigma * additive_noise

        # ----- Solve each sample separately -----
        # We loop over N rather than flattening to N*B, because some problems
        # (e.g. portfolio) have auxiliary parameters (covariance matrix) that
        # are batched at size B and cannot be expanded to N*B.
        init_api = problem.init_API()
        perturbed_solutions = []
        for n in range(n_samples):
            perturbed_n_cpu = perturbed_input[n].detach().cpu()  # (B, D1, ..., Dk)
            sols_n, _ = problem.get_decision(
                perturbed_n_cpu, params, ptoSolver, **init_api
            )
            if not isinstance(sols_n, torch.Tensor):
                sols_n = torch.as_tensor(sols_n, device=device, dtype=dtype)
            else:
                sols_n = sols_n.to(device=device, dtype=dtype)
            perturbed_solutions.append(sols_n)
        perturbed_solutions = torch.stack(perturbed_solutions, dim=0)  # (N, B, sol_D1, ...)

        # ----- Per-sample objectives scored under coeff_true -----
        per_sample_objs = []
        for n in range(n_samples):
            z_n = perturbed_solutions[n]                              # (B, sol_D1, ...)
            obj_n = problem.get_objective(coeff_true, z_n, params)    # (B,)
            if not isinstance(obj_n, torch.Tensor):
                obj_n = torch.as_tensor(obj_n, device=device, dtype=dtype)
            elif obj_n.device != device:
                obj_n = obj_n.to(device=device, dtype=dtype)
            per_sample_objs.append(obj_n.detach())
        per_sample_objs = torch.stack(per_sample_objs, dim=0)  # (N, B)

        # ----- Expected objective -----
        e_obj = per_sample_objs.mean(dim=0)                    # (B,)

        # ----- Save for backward -----
        ctx.save_for_backward(per_sample_objs, noise_gradient)
        ctx.n_samples = n_samples
        ctx.sigma = sigma
        ctx.original_input_shape = original_input_shape
        return e_obj

    @staticmethod
    def backward(ctx, dy):
        """
        REINFORCE-style gradient of E[obj] w.r.t. θ.

        per_sample_objs:  (N, B)               scalar objectives per sample
        noise_gradient:   (N, B, D1, ..., Dk)  ∇log p(ε)
        dy:               (B,)                 upstream gradient

        g_d = (1 / Nσ) Σ_n  obj_n · ∇log p(ε_n)_d · dy
        """
        per_sample_objs, noise_gradient = ctx.saved_tensors
        n_samples = ctx.n_samples
        sigma = ctx.sigma
        original_input_shape = ctx.original_input_shape

        # per_sample_objs is (N, B) — scalar output, so always full-reduce.
        output = per_sample_objs.unsqueeze(-1)       # (N, B, 1)
        dy_exp = dy.unsqueeze(-1)                    # (B, 1)

        flatten = lambda t: t.reshape(t.shape[0], t.shape[1], -1)
        noise_grad_flat = flatten(noise_gradient)    # (N, B, D)

        g = torch.einsum(
            "nbd,nb->bd",
            noise_grad_flat,
            torch.einsum("nbd,bd->nb", output, dy_exp),
        )
        g /= sigma * n_samples

        g = g.reshape(original_input_shape)
        # 8 inputs to forward: coeff_hat + 7 non-tensor args
        return g, None, None, None, None, None, None, None


# class perturbedFenchelYoung(optModel):
#     """
#     An autograd module for Fenchel-Young loss using perturbation techniques. The
#     use of the loss improves the algorithmic by the specific expression of the
#     gradients of the loss.

#     For the perturbed optimizer, the cost vector need to be predicted from
#     contextual data and are perturbed with Gaussian noise.

#     The Fenchel-Young loss allows to directly optimize a loss between the features
#     and solutions with less computation. Thus, allows us to design an algorithm
#     based on stochastic gradient descent.

#     Reference:
#     """

#     def __init__(
#         self,
#         ptoSolver,
#         n_samples=10,
#         sigma=1.0,
#         seed=135,
#         dataset=None,
#     ):
#         """
#         Args:
#             ptoSolver (optModel): an  optimization model
#             n_samples (int): number of Monte-Carlo samples
#             sigma (float): the amplitude of the perturbation
#             seed (int): random state seed
#
#             dataset (None/optDataset): the training data
#         """
#         super().__init__(ptoSolver)
#         # number of samples
#         self.n_samples = n_samples
#         # perturbation amplitude
#         self.sigma = sigma
#         # random state
#         self.rnd = np.random.RandomState(seed)
#         # build optimizer
#         self.pfy = perturbedFenchelYoungFunc()

#     def forward(self, coeff_hat, true_sol, reduction="mean"):
#         """
#         Forward pass
#         """
#         loss = self.pfy.apply(
#             coeff_hat,
#             true_sol,
#             self.ptoSolver,
#             self.n_samples,
#             self.sigma,
#             self.pool,
#             self.rnd,
#             self,
#         )
#         # reduction
#         loss = do_reduction(loss, hyperparams["reduction"])
#         return loss


# class perturbedFenchelYoungFunc(torch.autograd.Function):
#     """
#     A autograd function for Fenchel-Young loss using perturbation techniques.
#     """

#     @staticmethod
#     def forward(
#         ctx,
#         coeff_hat,
#         true_sol,
#         ptoSolver,
#         n_samples,
#         sigma,
#         pool,
#         rnd,
#         module,
#     ):
#         """
#         Forward pass for perturbed Fenchel-Young loss

#         Args:
#             coeff_hat (torch.tensor): a batch of predicted values of the cost
#             true_sol (torch.tensor): a batch of true optimal solutions
#             ptoSolver (optModel): an  optimization model
#             n_samples (int): number of Monte-Carlo samples
#             sigma (float): the amplitude of the perturbation
#             pool (ProcessPool): process pool object
#             rnd (RondomState): numpy random state
#
#             module (optModel): perturbedFenchelYoung module

#         Returns:
#             torch.tensor: solution expectations with perturbation
#         """
#         # get device
#         device = coeff_hat.device
#         # convert tenstor
#         cp = coeff_hat.detach().cpu().numpy()
#         w = true_sol.detach().cpu().numpy()
#         # sample perturbations
#         noises = rnd.normal(0, 1, size=(n_samples, *cp.shape))
#         ptb_c = cp + sigma * noises
#         # solve with perturbation
#         rand_sigma = np.random.uniform()
#         ptb_sols = _solve_in_pass(ptb_c, ptoSolver, pool)
#         sols = ptb_sols.reshape(-1, cp.shape[1])
#         # add into solpool
#         module.solpool = np.concatenate((module.solpool, sols))
#         # remove duplicate
#         module.solpool = np.unique(module.solpool, axis=0)
#         # solution expectation
#         e_sol = ptb_sols.mean(axis=1)
#         # difference
#         if ptoSolver.modelSense == GRB.MINIMIZE:
#             diff = w - e_sol
#         if ptoSolver.modelSense == GRB.MAXIMIZE:
#             diff = e_sol - w
#         # loss
#         loss = np.sum(diff**2, axis=1)
#         # convert to tensor
#         diff = torch.FloatTensor(diff).to(device)
#         loss = torch.FloatTensor(loss).to(device)
#         # save solutions
#         ctx.save_for_backward(diff)
#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass for perturbed Fenchel-Young loss
#         """
#         (grad,) = ctx.saved_tensors
#         grad_output = torch.unsqueeze(grad_output, dim=-1)
#         return grad * grad_output, None, None, None, None, None, None, None, None, None


# def _solve_in_pass(ptb_c, ptoSolver, pool):
#     """
#     A function to solve optimization in the forward pass
#     """
#     # number of instance
#     n_samples, ins_num = ptb_c.shape[0], ptb_c.shape[1]
#     # single-core
#     if processes == 1:
#         ptb_sols = []
#         for i in range(ins_num):
#             sols = []
#             # per sample
#             for j in range(n_samples):
#                 # solve
#                 ptoSolver.setObj(ptb_c[j, i])
#                 sol, _ = ptoSolver.solve()
#                 sols.append(sol)
#             ptb_sols.append(sols)
#     # multi-core
#     else:
#         # get class
#         model_type = type(ptoSolver)
#         # get args
#         args = getArgs(ptoSolver)
#         # parallel computing
#         ptb_sols = pool.amap(
#             _solveWithObj4Par,
#             ptb_c.transpose(1, 0, 2),
#             [args] * ins_num,
#             [model_type] * ins_num,
#         ).get()
#     return np.array(ptb_sols)


# def _cache_in_pass(ptb_c, ptoSolver, solpool):
#     """
#     A function to use solution pool in the forward/backward pass
#     """
#     # number of samples & instance
#     n_samples, ins_num, _ = ptb_c.shape
#     # init sols
#     ptb_sols = []
#     for j in range(n_samples):
#         # best solution in pool
#         solpool_obj = ptb_c[j] @ solpool.T
#         if ptoSolver.modelSense == GRB.MINIMIZE:
#             ind = np.argmin(solpool_obj, axis=1)
#         if ptoSolver.modelSense == GRB.MAXIMIZE:
#             ind = np.argmax(solpool_obj, axis=1)
#         ptb_sols.append(solpool[ind])
#     return np.array(ptb_sols).transpose(1, 0, 2)


# def _solveWithObj4Par(perturbed_costs, args, model_type):
#     """
#     A global function to solve function in parallel processors

#     Args:
#         perturbed_costs (np.ndarray): costsof objective function with perturbation
#         args (dict): optModel args
#         model_type (ABCMeta): optModel class type

#     Returns:
#         list: optimal solution
#     """
#     # rebuild model
#     ptoSolver = model_type(**args)
#     # per sample
#     sols = []
#     for cost in perturbed_costs:
#         # set obj
#         ptoSolver.setObj(cost)
#         # solve
#         sol, _ = ptoSolver.solve()
#         sols.append(sol)
#     return sols
