"""
PerturbDiag — diagnostic subclass of perturbed.

Identical training behaviour to perturbed, but also computes and stores
per-forward-pass diagnostics:
  last_z_bar               — (B, n_items)  soft decision z̄ = E[z(θ̂ + σε)]
  last_perturbed_solutions — (N, B, n_items) individual perturbed hard solutions

Uses perturbedSoftDecisionDiag, a drop-in replacement for perturbedSoftDecision
that returns both z_bar AND the stacked per-sample solutions.  The solutions
are detached from the autograd graph; gradients flow only through z_bar,
using the same Berthet JVP as the base class.
"""

import logging
import math

import torch

from openpto.method.Models.perturbed import (
    SigmaScheduler,
    optModel,
    do_reduction,
    sample_noise_with_gradients,
)
from openpto.diagnostics.perturb_metrics import rank_change_rate

from gurobipy import GRB  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modified autograd function — returns (z_bar, perturbed_solutions)
# ---------------------------------------------------------------------------

class perturbedSoftDecisionDiag(torch.autograd.Function):
    """
    Like perturbedSoftDecision but returns a second output:
        perturbed_solutions.detach()  — (N, B, sol_D...)

    Gradients flow only through z_bar (first output).  The second output is
    a leaf tensor; its incoming gradient in backward() will be None, which
    we simply ignore.
    """

    @staticmethod
    def forward(ctx, coeff_hat, ptoSolver, problem, params, n_samples, sigma, noise_type):
        device = coeff_hat.device
        dtype = coeff_hat.dtype
        original_input_shape = coeff_hat.shape

        # Sample noise
        perturbed_input_shape = [n_samples] + list(original_input_shape)
        additive_noise, noise_gradient = sample_noise_with_gradients(
            noise_type, perturbed_input_shape
        )
        additive_noise = additive_noise.to(device=device, dtype=dtype)
        noise_gradient = noise_gradient.to(device=device, dtype=dtype)

        perturbed_input = coeff_hat.unsqueeze(0) + sigma * additive_noise

        # Solve each sample
        init_api = problem.init_API()
        perturbed_solutions = []
        for n in range(n_samples):
            perturbed_n_cpu = perturbed_input[n].detach().cpu()
            sols_n, _ = problem.get_decision(
                perturbed_n_cpu, params, ptoSolver, **init_api
            )
            if not torch.is_tensor(sols_n):
                sols_n = torch.as_tensor(sols_n, device=device, dtype=dtype)
            else:
                sols_n = sols_n.to(device=device, dtype=dtype)
            perturbed_solutions.append(sols_n)
        perturbed_solutions = torch.stack(perturbed_solutions, dim=0)  # (N, B, sol_D...)

        z_bar = perturbed_solutions.mean(dim=0)  # (B, sol_D...)

        ctx.save_for_backward(perturbed_solutions, noise_gradient)
        ctx.n_samples = n_samples
        ctx.sigma = sigma
        ctx.original_input_shape = original_input_shape

        # Return z_bar (differentiable) and solutions (detached leaf for diagnostics)
        return z_bar, perturbed_solutions.detach()

    @staticmethod
    def backward(ctx, dz_bar, _d_solutions):
        # _d_solutions is None (second output is not used in any differentiable computation)
        perturbed_solutions, noise_gradient = ctx.saved_tensors
        n_samples = ctx.n_samples
        sigma = ctx.sigma
        original_input_shape = ctx.original_input_shape

        flatten = lambda t: t.reshape(t.shape[0], t.shape[1], -1)
        sol_flat = flatten(perturbed_solutions)      # (N, B, D_sol)
        noise_grad_flat = flatten(noise_gradient)    # (N, B, D_in)
        dy_flat = dz_bar.reshape(dz_bar.shape[0], -1)  # (B, D_sol)

        scores = torch.einsum("nbd,bd->nb", sol_flat, dy_flat)
        g = torch.einsum("nbd,nb->bd", noise_grad_flat, scores)
        g /= sigma * n_samples
        g = g.reshape(original_input_shape)

        return g, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# PerturbDiag
# ---------------------------------------------------------------------------

class PerturbDiag(optModel):
    """
    Diagnostic wrapper around the Berthet DPO (perturbed) loss.

    Training behaviour is identical to perturbed (same loss, same gradients).
    After each forward() call the following attributes are populated:
        last_z_bar               — (B, n_items) soft decision (detached)
        last_perturbed_solutions — (N, B, n_items) per-sample hard solutions (detached)

    These are used by the experiment script to compute rank_change_rate,
    interiority metrics, etc., without any additional solver calls.
    """

    def __init__(
        self,
        ptoSolver,
        n_samples: int = 10,
        sigma: float = 1.0,
        noise: str = "normal",
        seed: int = 135,
        output_activation: str = "none",
        loss_type: str = "regret",
        sigma_schedule: str = "constant",
        sigma_start=None,
        sigma_end: float = 0.01,
        sigma_n_epochs: int = 300,
        sigma_gamma: float = 0.5,
        sigma_step_size: int = 100,
        sigma_warmup_epochs: int = 0,
        **hyperparams,
    ):
        super().__init__(ptoSolver)

        if loss_type not in ("regret", "objective"):
            raise ValueError(f"Unknown loss_type '{loss_type}'.")
        self.loss_type = loss_type
        self.model_sense = ptoSolver.modelSense
        self.n_samples = n_samples
        self.sigma = sigma
        self.noise = noise

        _act_map = {
            "none": lambda x: x,
            "identity": lambda x: x,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "softplus": torch.nn.functional.softplus,
        }
        if output_activation not in _act_map:
            raise ValueError(f"Unknown output_activation '{output_activation}'.")
        self.output_activation = _act_map[output_activation]

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

        # Diagnostic outputs — populated after each forward()
        self.last_z_bar: torch.Tensor | None = None
        self.last_perturbed_solutions: torch.Tensor | None = None

        self._batch_scales: list[float] = []

    def step(self, epoch: int) -> float:
        """Update sigma per schedule.  Called by ExpManager each epoch."""
        self.sigma = self.sigma_scheduler.step(epoch)
        return self.sigma

    def forward(self, problem, coeff_hat, params, coeff_true=None, **hyperparams):
        """
        Forward pass — Berthet soft-decision, identical loss to perturbed.
        Also stores last_z_bar and last_perturbed_solutions for diagnostics.
        """
        if isinstance(coeff_hat, list):
            # Variable-size instances (e.g. Advertising) — no diagnostics in this path
            losses = []
            for i in range(len(coeff_hat)):
                loss_i = self.forward(
                    problem,
                    coeff_hat[i].unsqueeze(0),
                    params,
                    coeff_true[i].unsqueeze(0),
                    **hyperparams,
                )
                losses.append(loss_i)
            return do_reduction(torch.stack(losses), hyperparams["reduction"])

        coeff_hat = self.output_activation(coeff_hat)

        with torch.no_grad():
            self._batch_scales.append(coeff_hat.abs().mean().item())

        if self.sigma == 0:
            z_bar, _ = problem.get_decision(
                coeff_hat, params, self.ptoSolver, **problem.init_API()
            )
            if not torch.is_tensor(z_bar):
                z_bar = torch.as_tensor(z_bar, device=coeff_hat.device, dtype=coeff_hat.dtype)
            else:
                z_bar = z_bar.to(device=coeff_hat.device, dtype=coeff_hat.dtype)
            self.last_z_bar = z_bar.detach()
            self.last_perturbed_solutions = None
        else:
            z_bar, perturbed_solutions = perturbedSoftDecisionDiag.apply(
                coeff_hat,
                self.ptoSolver,
                problem,
                params,
                self.n_samples,
                self.sigma,
                self.noise,
            )
            self.last_z_bar = z_bar.detach()
            self.last_perturbed_solutions = perturbed_solutions  # already detached

        # Objective under true coefficients
        obj = problem.get_objective(coeff_true, z_bar, params)
        if not torch.is_tensor(obj):
            obj = torch.as_tensor(obj, dtype=coeff_hat.dtype)
        obj = obj.to(device=coeff_hat.device, dtype=coeff_hat.dtype)

        if self.loss_type == "objective":
            if self.model_sense == GRB.MAXIMIZE:
                loss = do_reduction(-obj, hyperparams["reduction"])
            else:
                loss = do_reduction(obj, hyperparams["reduction"])
        else:
            # Regret: |f(θ*, z̄) − f(θ*, z*)|
            coeff_true_cpu = coeff_true.detach().cpu()
            z_star_np, _ = problem.get_decision(
                coeff_true_cpu, params, self.ptoSolver, **problem.init_API()
            )
            z_star = torch.as_tensor(
                z_star_np, device=coeff_hat.device, dtype=coeff_hat.dtype
            )
            obj_star = problem.get_objective(coeff_true, z_star, params)
            if not torch.is_tensor(obj_star):
                obj_star = torch.as_tensor(obj_star, dtype=coeff_hat.dtype)
            obj_star = obj_star.to(device=coeff_hat.device, dtype=coeff_hat.dtype).detach()

            loss = do_reduction(torch.abs(obj - obj_star), hyperparams["reduction"])

        pred_loss_weight = float(hyperparams.get("pred_loss_weight", 0.0))
        if pred_loss_weight > 0.0 and coeff_true is not None:
            pred_mse = torch.nn.functional.mse_loss(coeff_hat, coeff_true)
            loss = loss + pred_loss_weight * pred_mse

        return loss


# ---------------------------------------------------------------------------
# AdaptiveSigmaPerturb — closed-loop RCR-targeting sigma controllers
# ---------------------------------------------------------------------------

class AdaptiveSigmaPerturb(PerturbDiag):
    """
    PerturbDiag with a closed-loop RCR-targeting sigma controller.

    Motivation: constant sigma fails because predicted-coefficient norms drift
    during training, causing sigma/||coeff|| to collapse and rank-change-rate
    (RCR) to fall below the exploration threshold (~0.15).

    Two update rules are available via ``mode``:

    ``"proportional"`` (default, no gain to tune):
        sigma_next = sigma_current * clip(rcr_target / rcr_current, 1/max_step, max_step)
        Assumes RCR ∝ sigma, so one step targets the desired RCR directly.
        Requires 1 extra solver call per forward pass (for unperturbed z0).

    ``"feedback"`` (classic integral controller):
        sigma_next = sigma_current * exp(alpha * (rcr_current - rcr_target))
        Requires tuning ``alpha`` (typical range 0.05–0.2).
        Requires 1 extra solver call per forward pass (for unperturbed z0).

    K (items selected per instance) is inferred from z0, so no problem-specific
    configuration is needed.

    Sigma is hard-clipped to [sigma_min, sigma_max] after every update.

    Parameters
    ----------
    rcr_target : float
        Desired rank-change-rate (default 0.3 — sweet spot from sweep experiments).
    mode : str
        ``"proportional"`` or ``"feedback"``.
    max_step : float
        Max multiplicative change in sigma per epoch for proportional mode (default 10).
    alpha : float
        Controller gain for feedback mode (default 0.05, ignored by proportional mode).
    sigma_min, sigma_max : float
        Hard bounds preventing runaway sigma.
    All other kwargs are forwarded to PerturbDiag unchanged.
    """

    def __init__(
        self,
        ptoSolver,
        rcr_target: float = 0.3,
        mode: str = "proportional",
        max_step: float = 10.0,
        alpha: float = 0.05,
        sigma_min: float = 1e-4,
        sigma_max: float = 1e3,
        **kwargs,
    ):
        super().__init__(ptoSolver, **kwargs)
        if mode not in ("proportional", "feedback"):
            raise ValueError(f"Unknown mode '{mode}'. Use 'proportional' or 'feedback'.")
        self.rcr_target = rcr_target
        self.mode = mode
        self.max_step = float(max_step)
        self.alpha = float(alpha)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        # Accumulate per-batch state across each epoch
        self._epoch_perturbed_sols: list = []
        self._epoch_z0: list = []

    def forward(self, problem, coeff_hat, params, coeff_true=None, **hyperparams):
        loss = super().forward(problem, coeff_hat, params, coeff_true, **hyperparams)

        if self.last_perturbed_solutions is not None:
            # Accumulate perturbed solutions (already detached)
            self._epoch_perturbed_sols.append(self.last_perturbed_solutions.cpu())

            # Unperturbed reference solution — one extra solver call per batch.
            # We apply output_activation to coeff_hat to match what the solver sees.
            with torch.no_grad():
                coeff_act = self.output_activation(coeff_hat.detach()).cpu()
                z0, _ = problem.get_decision(
                    coeff_act, params, self.ptoSolver, **problem.init_API()
                )
                if not torch.is_tensor(z0):
                    z0 = torch.as_tensor(z0, dtype=torch.float32)
                self._epoch_z0.append(z0.cpu().float())

        return loss

    def step(self, epoch: int) -> float:
        """Compute epoch-level RCR, update sigma, clear buffers."""
        # Discard batch-scale log (parent uses it only for SigmaScheduler path)
        self._batch_scales.clear()

        rcr = float("nan")
        if self._epoch_perturbed_sols and self._epoch_z0:
            all_sols = torch.cat(self._epoch_perturbed_sols, dim=1).float()  # (N, B_total, D)
            all_z0 = torch.cat(self._epoch_z0, dim=0).float()                # (B_total, D)

            # Infer K from unperturbed solutions (handles any TopK budget automatically)
            K = max(1, round(all_z0.sum(dim=-1).mean().item()))
            rcr = rank_change_rate(all_sols, all_z0, K)

            if self.mode == "proportional":
                # One-step update assuming RCR ∝ sigma.
                # Clamped to [1/max_step, max_step] to prevent overshooting.
                ratio = self.rcr_target / max(rcr, 1e-6)
                ratio = max(1.0 / self.max_step, min(self.max_step, ratio))
                self.sigma *= ratio
            else:  # "feedback"
                self.sigma *= math.exp(self.alpha * (rcr - self.rcr_target))

            self.sigma = float(max(self.sigma_min, min(self.sigma_max, self.sigma)))

        self._epoch_perturbed_sols.clear()
        self._epoch_z0.clear()

        logger.info(
            f"  [adaptive_perturb/{self.mode}] epoch {epoch}: "
            f"sigma={self.sigma:.5f}, rcr={rcr:.3f}, target={self.rcr_target:.2f}"
        )
        return self.sigma
