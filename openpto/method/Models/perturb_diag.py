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
from openpto.diagnostics.perturb_metrics import rank_change_rate, true_objective_metrics

from gurobipy import GRB  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)


def hamming_rate_per_instance(perturbed_sols, z0):
    """Per-instance mean fraction of items that flip between z0 and each z_n.

    Args:
        perturbed_sols: (N, B, D) float tensor
        z0:             (B, D) float tensor
    Returns:
        (B,) tensor — mean hamming rate per instance, in [0, 1]
    """
    flips = (perturbed_sols != z0.unsqueeze(0)).float()  # (N, B, D)
    return flips.mean(dim=(0, 2))                         # (B,)


def hamming_rate_per_item(perturbed_sols, z0):
    """Per-item mean fraction of instances/samples that flip item i.

    Args:
        perturbed_sols: (N, B, D) float tensor
        z0:             (B, D) float tensor
    Returns:
        (D,) tensor — mean hamming rate per item, averaged over N and B dims
    """
    flips = (perturbed_sols != z0.unsqueeze(0)).float()  # (N, B, D)
    return flips.mean(dim=(0, 1))                         # (D,)


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

        if torch.is_tensor(sigma) and sigma.ndim >= 1:
            n_spatial = additive_noise.ndim - 2   # dims after (N, B)
            B = original_input_shape[0]
            D = original_input_shape[-1] if len(original_input_shape) > 1 else 1
            if sigma.ndim == 2:
                # (B, D) per-inst×per-item → (1, B, D)
                sigma_noise = sigma.unsqueeze(0).to(device=device, dtype=dtype)
            elif sigma.ndim == 1 and len(original_input_shape) > 1 and sigma.shape[0] == D and D != B:
                # (D,) per-item → (1, 1, D)
                sigma_noise = sigma.view(1, 1, D).to(device=device, dtype=dtype)
            else:
                # (B,) per-instance → (1, B, 1, ...)
                sigma_noise = sigma.view(1, -1, *([1] * n_spatial)).to(device=device, dtype=dtype)
        else:
            sigma_noise = sigma
        perturbed_input = coeff_hat.unsqueeze(0) + sigma_noise * additive_noise

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
        if torch.is_tensor(sigma) and sigma.ndim >= 1:
            sigma_dev = sigma.to(g.device)
            if sigma_dev.ndim == 2:
                # (B, D) per-inst×per-item
                g /= sigma_dev * n_samples
            elif sigma_dev.ndim == 1:
                D = original_input_shape[-1] if len(original_input_shape) > 1 else 1
                B = original_input_shape[0]
                if sigma_dev.shape[0] == D and D != B:
                    # (D,) per-item → (1, D) broadcast over (B, D)
                    g /= sigma_dev.unsqueeze(0) * n_samples
                else:
                    # (B,) per-instance → (B, 1) broadcast over (B, D)
                    g /= sigma_dev.unsqueeze(-1) * n_samples
        else:
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
        self.last_z_star: torch.Tensor | None = None

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

        _sigma_is_zero = (not torch.is_tensor(self.sigma)) and self.sigma == 0
        if _sigma_is_zero:
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
            self.last_z_star = z_star.detach().cpu()
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
        ocv_target: float = 0.3,
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
        self.ocv_target = ocv_target
        self.mode = mode
        self.max_step = float(max_step)
        self.alpha = float(alpha)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        # Accumulate per-batch state across each epoch
        self._epoch_perturbed_sols: list = []
        self._epoch_z0: list = []
        self._epoch_coeff_true: list = []

        # Last computed metrics (for external inspection)
        self.last_ocv_y: float = float("nan")
        self.last_frac_improving: float = float("nan")

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

            # Accumulate true costs for OCV_Y computation
            if coeff_true is not None:
                self._epoch_coeff_true.append(coeff_true.detach().cpu())

        return loss

    def step(self, epoch: int) -> float:
        """Compute epoch-level OCV_Y, update sigma, clear buffers."""
        # Discard batch-scale log (parent uses it only for SigmaScheduler path)
        self._batch_scales.clear()

        ocv_y = float("nan")
        frac_improving = float("nan")
        rcr = float("nan")

        if self._epoch_perturbed_sols and self._epoch_z0 and self._epoch_coeff_true:
            all_sols = torch.cat(self._epoch_perturbed_sols, dim=1).float()  # (N, B_total, D)
            all_z0   = torch.cat(self._epoch_z0, dim=0).float()              # (B_total, D)
            all_y    = torch.cat(self._epoch_coeff_true, dim=0)              # (B_total, ...)

            # --- OCV_Y and FracImproving (primary control signal) ---
            metrics = true_objective_metrics(all_sols, all_z0, all_y)
            ocv_y = metrics["ocv_y"]
            frac_improving = metrics["frac_improving"]

            # --- RCR for backward-compat logging (binary problems only) ---
            try:
                K = max(1, round(all_z0.sum(dim=-1).mean().item()))
                rcr = rank_change_rate(all_sols, all_z0, K)
            except Exception:
                rcr = float("nan")

            # --- Update sigma using OCV_Y as control target ---
            if self.mode == "proportional":
                ratio = self.ocv_target / max(ocv_y, 1e-6)
                ratio = max(1.0 / self.max_step, min(self.max_step, ratio))
                self.sigma *= ratio
            else:  # "feedback"
                self.sigma *= math.exp(self.alpha * (ocv_y - self.ocv_target))

            self.sigma = float(max(self.sigma_min, min(self.sigma_max, self.sigma)))

        self.last_ocv_y = ocv_y
        self.last_frac_improving = frac_improving

        self._epoch_perturbed_sols.clear()
        self._epoch_z0.clear()
        self._epoch_coeff_true.clear()

        logger.info(
            f"  [adaptive_perturb/{self.mode}] epoch {epoch}: "
            f"sigma={self.sigma:.5f}, ocv_y={ocv_y:.4f}, frac_improving={frac_improving:.4f}, "
            f"rcr={rcr:.3f}, target={self.ocv_target:.4f}"
        )
        return self.sigma


# ---------------------------------------------------------------------------
# PerInstanceAdaptiveSigma — per-instance sigma controller
# ---------------------------------------------------------------------------

class PerInstanceAdaptiveSigma(AdaptiveSigmaPerturb):
    """
    Per-instance sigma controller.

    Each training instance gets its own sigma value, updated each epoch via the
    proportional OCV_Y controller. Works with:
      - ExpManager (mini-batch, shuffle=True): requires 'inst_idx' passed from DataLoader
        (see ExpDataset.__getitem__ and ExpManager changes).
      - Sweep script (full-batch): inst_idx defaults to torch.arange(B) since B = N_train.

    Additional params:
        sigma_ema : float — EMA smoothing for per-instance sigma (default 0.7).
    """

    def __init__(self, ptoSolver, sigma_ema: float = 0.7, **kwargs):
        super().__init__(ptoSolver, **kwargs)
        self.sigma_ema = float(sigma_ema)
        self.sigma_vec: torch.Tensor | None = None   # (N_train,), lazy init
        self._epoch_indices: list = []               # accumulated instance indices

    def forward(self, problem, coeff_hat, params, coeff_true=None, inst_idx=None, **hyperparams):
        B = coeff_hat.shape[0]
        if self.sigma_vec is not None:
            if inst_idx is None:
                # Full-batch path: batch position = dataset index
                idx = torch.arange(B)
            else:
                idx = inst_idx.long()
            self.sigma = self.sigma_vec[idx].detach().to(coeff_hat.device)  # (B,) tensor, no grad

        loss = super().forward(problem, coeff_hat, params, coeff_true, **hyperparams)

        # Accumulate indices for step()
        if self.last_perturbed_solutions is not None:
            if inst_idx is None:
                self._epoch_indices.append(torch.arange(B))
            else:
                self._epoch_indices.append(inst_idx.long().cpu())

        return loss

    def step(self, epoch: int) -> float:
        if not (self._epoch_perturbed_sols and self._epoch_z0 and self._epoch_coeff_true):
            mean_sigma = float(self.sigma_vec.mean()) if self.sigma_vec is not None else (
                self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean()))
            return mean_sigma

        all_sols    = torch.cat(self._epoch_perturbed_sols, dim=1).float()  # (N, B_total, D)
        all_z0      = torch.cat(self._epoch_z0, dim=0).float()              # (B_total, D)
        all_y       = torch.cat(self._epoch_coeff_true, dim=0)              # (B_total, ...)
        all_indices = torch.cat(self._epoch_indices, dim=0)                 # (B_total,)
        B_total     = all_z0.shape[0]

        # Lazy init sigma_vec using the max index seen to infer N_train
        if self.sigma_vec is None:
            N_train = int(all_indices.max().item()) + 1
            init_sigma = self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean())
            self.sigma_vec = torch.full((N_train,), float(init_sigma))

        # Per-instance OCV_Y (all tensors are detached, no grad needed)
        with torch.no_grad():
            z_n   = all_sols.reshape(all_sols.shape[0], B_total, -1)
            z_0   = all_z0.reshape(B_total, -1)
            y     = all_y.reshape(B_total, -1)
            obj_n = torch.einsum("nbd,bd->nb", z_n, y)   # (N, B_total)
            obj_0 = torch.einsum("bd,bd->b",   z_0, y)   # (B_total,)

            ocv_y_per = obj_n.std(dim=0) / (obj_0.abs() + 1e-6)          # (B_total,)
            frac_per  = (obj_n > obj_0.unsqueeze(0)).float().mean(dim=0)  # (B_total,)

            # Proportional update per instance
            ratio = (self.ocv_target / ocv_y_per.clamp(min=1e-6)).clamp(
                1.0 / self.max_step, self.max_step
            )
            sigma_proposed = (self.sigma_vec[all_indices] * ratio).clamp(
                self.sigma_min, self.sigma_max
            )

            # EMA update (only for instances seen this epoch)
            self.sigma_vec[all_indices] = (
                self.sigma_ema * self.sigma_vec[all_indices]
                + (1 - self.sigma_ema) * sigma_proposed
            )

        # Global stats
        ocv_y_global          = ocv_y_per.mean().item()
        self.last_ocv_y       = ocv_y_global
        self.last_frac_improving = frac_per.mean().item()
        self.sigma            = self.sigma_vec.mean().item()  # scalar for compat/logging

        self._epoch_perturbed_sols.clear()
        self._epoch_z0.clear()
        self._epoch_coeff_true.clear()
        self._epoch_indices.clear()

        logger.info(
            f"  [per_instance/prop] epoch {epoch}: "
            f"sigma mean={self.sigma:.5f} min={self.sigma_vec.min():.5f} "
            f"max={self.sigma_vec.max():.5f} ocv_y={ocv_y_global:.4f} "
            f"frac={self.last_frac_improving:.4f}"
        )
        return self.sigma


# ---------------------------------------------------------------------------
# PerInstanceHammingPerturb — per-instance Hamming-rate controller
# ---------------------------------------------------------------------------

class PerInstanceHammingPerturb(PerInstanceAdaptiveSigma):
    """
    Per-instance sigma controller using Hamming rate as control target.

    Hamming rate = mean fraction of items that flip selection status between
    z_0 and z_n, averaged over perturbation samples.

    Targets "enough noise to learn something new, but not enough to change
    all the predictions." For a 20-item knapsack:
        hamming_target=0.05 → ~1 item flips per perturbation
        hamming_target=0.10 → ~2 items flip
        hamming_target=0.20 → ~4 items flip

    Unlike OCV_Y (which is objective-scale dependent), Hamming rate is
    purely structural — it measures decision change directly, with a
    natural [0, 1] range that's problem-scale invariant.

    OCV_Y and FracImproving are still computed and logged for diagnostics.
    """

    def __init__(
        self,
        ptoSolver,
        hamming_target: float = 0.10,
        hamming_warmup_epochs: int = 0,
        hamming_decay_epochs: int = 0,
        **kwargs,
    ):
        super().__init__(ptoSolver, **kwargs)
        self.hamming_target        = float(hamming_target)
        self.hamming_warmup_epochs = int(hamming_warmup_epochs)
        self.hamming_decay_epochs  = int(hamming_decay_epochs)
        self.last_hamming: float   = float("nan")
        self._current_hamming_target: float = float(hamming_target)

    def _schedule_target(self, epoch: int) -> float:
        """Return the Hamming target for this epoch.

        Constant for epochs <= warmup_epochs, then linearly decays to 0
        over the next decay_epochs epochs.
        """
        if self.hamming_decay_epochs <= 0 or epoch <= self.hamming_warmup_epochs:
            return self.hamming_target
        elapsed = epoch - self.hamming_warmup_epochs
        frac    = min(elapsed / self.hamming_decay_epochs, 1.0)
        return self.hamming_target * (1.0 - frac)

    def step(self, epoch: int) -> float:
        if not (self._epoch_perturbed_sols and self._epoch_z0):
            mean_sigma = (
                float(self.sigma_vec.mean()) if self.sigma_vec is not None
                else (self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean()))
            )
            return mean_sigma

        all_sols    = torch.cat(self._epoch_perturbed_sols, dim=1).float()  # (N, B_total, D)
        all_z0      = torch.cat(self._epoch_z0, dim=0).float()              # (B_total, D)
        all_indices = torch.cat(self._epoch_indices, dim=0)                 # (B_total,)
        B_total     = all_z0.shape[0]

        # Lazy init sigma_vec
        if self.sigma_vec is None:
            N_train = int(all_indices.max().item()) + 1
            init_sigma = self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean())
            self.sigma_vec = torch.full((N_train,), float(init_sigma))

        with torch.no_grad():
            # Per-instance Hamming rate (control signal)
            hamming_per = hamming_rate_per_instance(all_sols, all_z0)  # (B_total,)

            # OCV_Y and FracImproving for diagnostic logging
            if self._epoch_coeff_true:
                all_y  = torch.cat(self._epoch_coeff_true, dim=0)
                y      = all_y.reshape(B_total, -1)
                z_n    = all_sols.reshape(all_sols.shape[0], B_total, -1)
                z_0    = all_z0.reshape(B_total, -1)
                obj_n  = torch.einsum("nbd,bd->nb", z_n, y)
                obj_0  = torch.einsum("bd,bd->b",   z_0, y)
                ocv_y_per  = obj_n.std(dim=0) / (obj_0.abs() + 1e-6)
                frac_per   = (obj_n > obj_0.unsqueeze(0)).float().mean(dim=0)
            else:
                ocv_y_per = torch.zeros(B_total)
                frac_per  = torch.zeros(B_total)

            # Proportional update using scheduled Hamming target
            self._current_hamming_target = self._schedule_target(epoch)
            ratio = (self._current_hamming_target / hamming_per.clamp(min=1e-6)).clamp(
                1.0 / self.max_step, self.max_step
            )
            sigma_proposed = (self.sigma_vec[all_indices] * ratio).clamp(
                self.sigma_min, self.sigma_max
            )

            # EMA update
            self.sigma_vec[all_indices] = (
                self.sigma_ema * self.sigma_vec[all_indices]
                + (1 - self.sigma_ema) * sigma_proposed
            )

        self.last_hamming        = hamming_per.mean().item()
        self.last_ocv_y          = ocv_y_per.mean().item()
        self.last_frac_improving = frac_per.mean().item()
        self.sigma               = self.sigma_vec.mean().item()

        self._epoch_perturbed_sols.clear()
        self._epoch_z0.clear()
        self._epoch_coeff_true.clear()
        self._epoch_indices.clear()

        logger.info(
            f"  [per_instance_hamming/prop] epoch {epoch}: "
            f"sigma mean={self.sigma:.5f} min={self.sigma_vec.min():.5f} "
            f"max={self.sigma_vec.max():.5f} hamming={self.last_hamming:.4f} "
            f"ocv_y={self.last_ocv_y:.4f} frac={self.last_frac_improving:.4f} "
            f"target={self._current_hamming_target:.4f}"
        )
        return self.sigma


# ---------------------------------------------------------------------------
# PerInstanceHammingStarPerturb — dynamic hamming(z*, z_0) target per instance
# ---------------------------------------------------------------------------

class PerInstanceHammingStarPerturb(PerInstanceHammingPerturb):
    """
    Per-instance sigma controller where the Hamming target is dynamic:
        target_i = min(hamming(z*_i, z0_i) / D, hamming_target)

    z*_i is the optimal decision under true costs y_i, and D is the number of
    items (decision dimension). The cap (``hamming_target``, inherited) prevents
    runaway sigma when z0 is far from z* early in training. As training converges,
    hamming(z*_i, z0_i) → 0, so each instance's target → 0 and sigma
    self-terminates per-instance.

    z* is cached permanently after epoch 1 (true costs are fixed data, so z*
    is deterministic). From epoch 2 onward there is zero extra overhead.

    Before the cache is warm (epoch 0), falls back to the inherited
    ``hamming_target`` fixed value.

    Requires ``loss_type="regret"`` (the default) so that PerturbDiag.forward()
    computes and stores ``last_z_star``.
    """

    def __init__(self, ptoSolver, **kwargs):
        super().__init__(ptoSolver, **kwargs)
        self.z_star_cache: torch.Tensor | None = None  # (N_train, D), filled after epoch 1
        self._cache_warm: bool = False
        self._epoch_z_star: list = []  # accumulates (inst_idx, z_star) pairs until warm

    def forward(self, problem, coeff_hat, params, coeff_true=None, inst_idx=None, **hyperparams):
        loss = super().forward(problem, coeff_hat, params, coeff_true, inst_idx=inst_idx, **hyperparams)
        # Accumulate z* for cache building (only until cache is warm)
        if not self._cache_warm and self.last_z_star is not None:
            B = coeff_hat.shape[0]
            idx = inst_idx.long().cpu() if inst_idx is not None else torch.arange(B)
            self._epoch_z_star.append((idx, self.last_z_star.cpu()))
        return loss

    def step(self, epoch: int) -> float:
        # Build z_star_cache from this epoch's accumulation
        if not self._cache_warm and self._epoch_z_star:
            all_idx = torch.cat([pair[0] for pair in self._epoch_z_star])
            all_zs  = torch.cat([pair[1] for pair in self._epoch_z_star], dim=0)
            if self.z_star_cache is None:
                N_train = int(all_idx.max().item()) + 1
                D = all_zs.shape[-1]
                self.z_star_cache = torch.zeros(N_train, D)
            self.z_star_cache[all_idx] = all_zs.float()
            self._cache_warm = True
        self._epoch_z_star.clear()

        # Same early-exit as parent if buffers are empty
        if not (self._epoch_perturbed_sols and self._epoch_z0):
            mean_sigma = (
                float(self.sigma_vec.mean()) if self.sigma_vec is not None
                else (self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean()))
            )
            return mean_sigma

        all_sols    = torch.cat(self._epoch_perturbed_sols, dim=1).float()  # (N, B_total, D)
        all_z0      = torch.cat(self._epoch_z0, dim=0).float()              # (B_total, D)
        all_indices = torch.cat(self._epoch_indices, dim=0)                 # (B_total,)
        B_total     = all_z0.shape[0]

        # Lazy init sigma_vec
        if self.sigma_vec is None:
            N_train = int(all_indices.max().item()) + 1
            init_sigma = self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean())
            self.sigma_vec = torch.full((N_train,), float(init_sigma))

        with torch.no_grad():
            # Per-instance Hamming rate (control signal, normalised by D)
            hamming_per = hamming_rate_per_instance(all_sols, all_z0)  # (B_total,)

            # OCV_Y and FracImproving for diagnostic logging
            if self._epoch_coeff_true:
                all_y  = torch.cat(self._epoch_coeff_true, dim=0)
                y      = all_y.reshape(B_total, -1)
                z_n    = all_sols.reshape(all_sols.shape[0], B_total, -1)
                z_0    = all_z0.reshape(B_total, -1)
                obj_n  = torch.einsum("nbd,bd->nb", z_n, y)
                obj_0  = torch.einsum("bd,bd->b",   z_0, y)
                ocv_y_per  = obj_n.std(dim=0) / (obj_0.abs() + 1e-6)
                frac_per   = (obj_n > obj_0.unsqueeze(0)).float().mean(dim=0)
            else:
                ocv_y_per = torch.zeros(B_total)
                frac_per  = torch.zeros(B_total)

            # Dynamic per-instance Hamming target: min(hamming(z*_i, z0_i) / D, cap)
            # Normalised by D to match the scale of hamming_rate_per_instance.
            # Capped at hamming_target to prevent runaway sigma when z0 is far from z*.
            if self.z_star_cache is not None:
                z_star_batch = self.z_star_cache[all_indices].float()  # (B_total, D)
                D = float(all_z0.shape[-1])
                hamming_to_star = (z_star_batch != all_z0).float().sum(dim=-1)  # (B_total,)
                dynamic_target = (hamming_to_star / D).clamp(min=1e-6, max=self.hamming_target)
            else:
                # Fallback before cache is warm: use fixed hamming_target
                dynamic_target = torch.full((B_total,), self.hamming_target)

            ratio = (dynamic_target / hamming_per.clamp(min=1e-6)).clamp(
                1.0 / self.max_step, self.max_step
            )
            sigma_proposed = (self.sigma_vec[all_indices] * ratio).clamp(
                self.sigma_min, self.sigma_max
            )

            # EMA update
            self.sigma_vec[all_indices] = (
                self.sigma_ema * self.sigma_vec[all_indices]
                + (1 - self.sigma_ema) * sigma_proposed
            )

        self.last_hamming        = hamming_per.mean().item()
        self.last_ocv_y          = ocv_y_per.mean().item()
        self.last_frac_improving = frac_per.mean().item()
        self.sigma               = self.sigma_vec.mean().item()
        mean_dyn_target = dynamic_target.mean().item()

        self._epoch_perturbed_sols.clear()
        self._epoch_z0.clear()
        self._epoch_coeff_true.clear()
        self._epoch_indices.clear()

        logger.info(
            f"  [per_instance_hamming_star] epoch {epoch}: "
            f"sigma mean={self.sigma:.5f} min={self.sigma_vec.min():.5f} "
            f"max={self.sigma_vec.max():.5f} hamming={self.last_hamming:.4f} "
            f"dyn_target={mean_dyn_target:.4f} cache_warm={self._cache_warm} "
            f"ocv_y={self.last_ocv_y:.4f} frac={self.last_frac_improving:.4f}"
        )
        return self.sigma


# ---------------------------------------------------------------------------
# CoeffRelativeSigmaPerturb — sigma proportional to predicted cost magnitude
# ---------------------------------------------------------------------------

class CoeffRelativeSigmaPerturb(PerturbDiag):
    """
    Coefficient-relative sigma: sigma_{b,i} = alpha * |ĉ_{b,i}|

    No controller loop. Sigma is recomputed from the current predictions every
    forward pass, so it automatically tracks predicted-cost magnitude per item
    and per instance.  This keeps sigma naturally bounded — it can only grow as
    large as the predicted costs themselves.

    The single hyperparameter alpha scales the relative noise level:
        alpha = 1.0  → noise std equals the absolute predicted cost per item
        alpha < 1.0  → smaller perturbation (tighter noise)
        alpha > 1.0  → larger perturbation (looser noise)

    Parameters
    ----------
    alpha : float
        Noise-to-cost ratio (default 1.0).
    All other kwargs forwarded to PerturbDiag.
    """

    def __init__(self, ptoSolver, alpha: float = 1.0, **kwargs):
        super().__init__(ptoSolver, **kwargs)
        self.alpha = float(alpha)
        self._last_sigma_mean: float = float("nan")
        self.last_ocv_y: float = float("nan")
        self.last_frac_improving: float = float("nan")
        self.last_hamming: float = float("nan")

    def forward(self, problem, coeff_hat, params, coeff_true=None, **hyperparams):
        # Detach so sigma has no gradient; shape (B, D) handled by generalised backward
        with torch.no_grad():
            self.sigma = self.alpha * coeff_hat.abs()
        return super().forward(problem, coeff_hat, params, coeff_true, **hyperparams)

    def step(self, epoch: int) -> float:
        # No update — sigma is recomputed from coeff_hat each forward pass.
        # Just return the last mean sigma for external logging.
        if torch.is_tensor(self.sigma):
            self._last_sigma_mean = float(self.sigma.mean().item())
        return self._last_sigma_mean


# ---------------------------------------------------------------------------
# PerItemHammingPerturb — per-item (D,) sigma controller
# ---------------------------------------------------------------------------

class PerItemHammingPerturb(AdaptiveSigmaPerturb):
    """
    Per-item sigma controller using Hamming rate as control target.

    sigma_vec is a (D,) tensor — one sigma per item/decision-dimension.
    Control signal: per-item Hamming rate = mean fraction of N×B perturbations
    that flip item i (averaged over all instances and samples).

    This allows items that are easy to determine (rarely flip) to use small
    sigma, while ambiguous items (frequently flip) get larger sigma.

    Parameters
    ----------
    hamming_target : float
        Desired per-item Hamming rate (default 0.10).
    sigma_ema : float
        EMA smoothing factor for sigma_vec updates (default 0.7).
    sigma_min : float or None
        Minimum clamp for sigma_vec. None disables clamping.
    All other kwargs forwarded to AdaptiveSigmaPerturb.
    """

    def __init__(
        self,
        ptoSolver,
        hamming_target: float = 0.10,
        sigma_ema: float = 0.7,
        sigma_min=1e-4,
        **kwargs,
    ):
        # Pass sigma_min=0.0 to parent (parent's step() is never called for this class)
        super().__init__(ptoSolver, sigma_min=sigma_min if sigma_min is not None else 0.0, **kwargs)
        self.hamming_target = float(hamming_target)
        self.sigma_ema = float(sigma_ema)
        self.sigma_min_item = sigma_min  # None or float
        self.sigma_vec: torch.Tensor | None = None  # (D,), lazy init
        self.last_hamming: float = float("nan")
        self.last_hamming_vec: torch.Tensor | None = None   # (D,) for external inspection
        self.last_sigma_vec_item: torch.Tensor | None = None  # (D,) snapshot after each step

    def forward(self, problem, coeff_hat, params, coeff_true=None, **hyperparams):
        if self.sigma_vec is not None:
            # Set sigma to (D,) tensor so perturbedSoftDecisionDiag uses per-item broadcast
            self.sigma = self.sigma_vec.to(coeff_hat.device)
        return super().forward(problem, coeff_hat, params, coeff_true, **hyperparams)

    def step(self, epoch: int) -> float:
        self._batch_scales.clear()

        if not (self._epoch_perturbed_sols and self._epoch_z0):
            mean_sigma = (
                float(self.sigma_vec.mean()) if self.sigma_vec is not None
                else (self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean()))
            )
            self._epoch_perturbed_sols.clear()
            self._epoch_z0.clear()
            self._epoch_coeff_true.clear()
            return mean_sigma

        all_sols = torch.cat(self._epoch_perturbed_sols, dim=1).float()  # (N, B_total, D)
        all_z0   = torch.cat(self._epoch_z0, dim=0).float()              # (B_total, D)
        B_total  = all_z0.shape[0]
        D        = all_z0.shape[-1]

        # Lazy init sigma_vec
        if self.sigma_vec is None:
            init_sigma = self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean())
            self.sigma_vec = torch.full((D,), float(init_sigma))

        with torch.no_grad():
            # Per-item Hamming rate (mean over N and B dims)
            hamming_item = hamming_rate_per_item(all_sols, all_z0)  # (D,)

            # Proportional update
            ratio = (self.hamming_target / hamming_item.clamp(min=1e-6)).clamp(
                1.0 / self.max_step, self.max_step
            )
            sigma_proposed = self.sigma_vec * ratio  # (D,)
            if self.sigma_min_item is not None:
                sigma_proposed = sigma_proposed.clamp(min=self.sigma_min_item)

            # EMA update
            self.sigma_vec = (
                self.sigma_ema * self.sigma_vec
                + (1 - self.sigma_ema) * sigma_proposed
            )

        self.last_hamming = hamming_item.mean().item()
        self.last_hamming_vec = hamming_item.clone()
        self.last_sigma_vec_item = self.sigma_vec.clone()  # (D,) snapshot for logging
        self.sigma = float(self.sigma_vec.mean().item())

        # Diagnostic OCV_Y
        if self._epoch_coeff_true:
            all_y = torch.cat(self._epoch_coeff_true, dim=0)
            with torch.no_grad():
                z_n = all_sols.reshape(all_sols.shape[0], B_total, -1)
                z_0 = all_z0.reshape(B_total, -1)
                y   = all_y.reshape(B_total, -1)
                obj_n = torch.einsum("nbd,bd->nb", z_n, y)
                obj_0 = torch.einsum("bd,bd->b",   z_0, y)
                ocv_y_per = obj_n.std(dim=0) / (obj_0.abs() + 1e-6)
                frac_per  = (obj_n > obj_0.unsqueeze(0)).float().mean(dim=0)
            self.last_ocv_y          = ocv_y_per.mean().item()
            self.last_frac_improving = frac_per.mean().item()

        self._epoch_perturbed_sols.clear()
        self._epoch_z0.clear()
        self._epoch_coeff_true.clear()

        logger.info(
            f"  [per_item_hamming] epoch {epoch}: "
            f"sigma mean={self.sigma:.5f} min={self.sigma_vec.min():.5f} "
            f"max={self.sigma_vec.max():.5f} hamming_mean={self.last_hamming:.4f} "
            f"hamming_std={self.last_hamming_vec.std().item():.4f} "
            f"ocv_y={self.last_ocv_y:.4f} frac={self.last_frac_improving:.4f} "
            f"target={self.hamming_target:.4f}"
        )
        return self.sigma


# ---------------------------------------------------------------------------
# PerInstItemHammingPerturb — per-instance × per-item (B, D) sigma controller
# ---------------------------------------------------------------------------

class PerInstItemHammingPerturb(AdaptiveSigmaPerturb):
    """
    Per-instance × per-item sigma controller.

    sigma_mat is an (N_train, D) tensor — one sigma per (instance, item) pair.
    Control signal: per-instance × per-item Hamming rate =
        flips.mean(dim=0) → (B_total, D)  (mean over N perturbation samples).

    Parameters
    ----------
    hamming_target : float
        Desired per-instance×per-item Hamming rate (default 0.10).
    sigma_ema : float
        EMA smoothing factor for sigma_mat updates (default 0.7).
    sigma_min : float or None
        Minimum clamp for sigma_mat. None disables clamping.
    All other kwargs forwarded to AdaptiveSigmaPerturb.
    """

    def __init__(
        self,
        ptoSolver,
        hamming_target: float = 0.10,
        sigma_ema: float = 0.7,
        sigma_min=1e-4,
        **kwargs,
    ):
        super().__init__(ptoSolver, sigma_min=sigma_min if sigma_min is not None else 0.0, **kwargs)
        self.hamming_target = float(hamming_target)
        self.sigma_ema = float(sigma_ema)
        self.sigma_min_item = sigma_min  # None or float
        self.sigma_mat: torch.Tensor | None = None  # (N_train, D), lazy init
        self._epoch_indices: list = []
        self.last_hamming: float = float("nan")
        self.last_hamming_vec: torch.Tensor | None = None   # (D,) mean over instances
        self.last_sigma_vec_item: torch.Tensor | None = None  # (D,) mean over instances, snapshot

    def forward(self, problem, coeff_hat, params, coeff_true=None, inst_idx=None, **hyperparams):
        B = coeff_hat.shape[0]
        if self.sigma_mat is not None:
            if inst_idx is None:
                idx = torch.arange(B)
            else:
                idx = inst_idx.long()
            self.sigma = self.sigma_mat[idx].detach().to(coeff_hat.device)  # (B, D)

        loss = super().forward(problem, coeff_hat, params, coeff_true, **hyperparams)

        # Accumulate indices for step()
        if self.last_perturbed_solutions is not None:
            if inst_idx is None:
                self._epoch_indices.append(torch.arange(B))
            else:
                self._epoch_indices.append(inst_idx.long().cpu())

        return loss

    def step(self, epoch: int) -> float:
        self._batch_scales.clear()

        if not (self._epoch_perturbed_sols and self._epoch_z0):
            mean_sigma = (
                float(self.sigma_mat.mean()) if self.sigma_mat is not None
                else (self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean()))
            )
            self._epoch_perturbed_sols.clear()
            self._epoch_z0.clear()
            self._epoch_coeff_true.clear()
            self._epoch_indices.clear()
            return mean_sigma

        all_sols    = torch.cat(self._epoch_perturbed_sols, dim=1).float()  # (N, B_total, D)
        all_z0      = torch.cat(self._epoch_z0, dim=0).float()              # (B_total, D)
        all_indices = torch.cat(self._epoch_indices, dim=0)                 # (B_total,)
        B_total     = all_z0.shape[0]
        D           = all_z0.shape[-1]

        # Lazy init sigma_mat
        if self.sigma_mat is None:
            N_train = int(all_indices.max().item()) + 1
            init_sigma = self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean())
            self.sigma_mat = torch.full((N_train, D), float(init_sigma))

        with torch.no_grad():
            # Per-instance × per-item Hamming rate: (N, B_total, D) → (B_total, D)
            flips = (all_sols != all_z0.unsqueeze(0)).float()  # (N, B_total, D)
            hamming_per = flips.mean(dim=0)                     # (B_total, D)

            # Proportional update
            ratio = (self.hamming_target / hamming_per.clamp(min=1e-6)).clamp(
                1.0 / self.max_step, self.max_step
            )
            sigma_proposed = self.sigma_mat[all_indices] * ratio  # (B_total, D)
            if self.sigma_min_item is not None:
                sigma_proposed = sigma_proposed.clamp(min=self.sigma_min_item)

            # EMA update (only for instances seen this epoch)
            self.sigma_mat[all_indices] = (
                self.sigma_ema * self.sigma_mat[all_indices]
                + (1 - self.sigma_ema) * sigma_proposed
            )

        self.last_hamming = hamming_per.mean().item()
        self.last_hamming_vec = hamming_per.mean(dim=0).clone()  # (D,) mean over instances
        self.last_sigma_vec_item = self.sigma_mat.mean(dim=0).clone()  # (D,) snapshot for logging
        self.sigma = float(self.sigma_mat.mean().item())

        # Diagnostic OCV_Y
        if self._epoch_coeff_true:
            all_y = torch.cat(self._epoch_coeff_true, dim=0)
            with torch.no_grad():
                z_n = all_sols.reshape(all_sols.shape[0], B_total, -1)
                z_0 = all_z0.reshape(B_total, -1)
                y   = all_y.reshape(B_total, -1)
                obj_n = torch.einsum("nbd,bd->nb", z_n, y)
                obj_0 = torch.einsum("bd,bd->b",   z_0, y)
                ocv_y_per = obj_n.std(dim=0) / (obj_0.abs() + 1e-6)
                frac_per  = (obj_n > obj_0.unsqueeze(0)).float().mean(dim=0)
            self.last_ocv_y          = ocv_y_per.mean().item()
            self.last_frac_improving = frac_per.mean().item()

        self._epoch_perturbed_sols.clear()
        self._epoch_z0.clear()
        self._epoch_coeff_true.clear()
        self._epoch_indices.clear()

        logger.info(
            f"  [per_inst_item_hamming] epoch {epoch}: "
            f"sigma mean={self.sigma:.5f} min={self.sigma_mat.min():.5f} "
            f"max={self.sigma_mat.max():.5f} hamming_mean={self.last_hamming:.4f} "
            f"hamming_item_std={self.last_hamming_vec.std().item():.4f} "
            f"ocv_y={self.last_ocv_y:.4f} frac={self.last_frac_improving:.4f} "
            f"target={self.hamming_target:.4f}"
        )
        return self.sigma


# ---------------------------------------------------------------------------
# FlipMarginSigmaPerturb — per-(instance, item) sigma guided by flip margin
# ---------------------------------------------------------------------------

class FlipMarginSigmaPerturb(PerInstItemHammingPerturb):
    """
    Per-instance × per-item sigma controller guided by whether the model's
    current decision z0 agrees with the oracle optimal decision z* item-by-item.

    Target assignment for each (instance b, item i):
        target_{b,i} = error_target   if  z*_{b,i} ≠ z0_{b,i}  (model wrong → more flips)
                       correct_target if  z*_{b,i} = z0_{b,i}  (model right → fewer flips)

    Instances with regret < regret_threshold are blended toward the inherited
    hamming_target so sigma doesn't move much for already-solved instances.

    A hard per-item ceiling  σ_{b,i} ≤ sigma_coeff_max × |ĉ_{b,i}|  prevents
    the coeff-relative runaway seen in CoeffRelativeSigmaPerturb.

    z* is cached permanently after epoch 1 (true costs are fixed data).
    Before the cache is warm, falls back to the parent proportional controller.

    Requires ``loss_type="regret"`` (default in PerturbDiag) so that
    ``self.last_z_star`` is populated by each forward pass.

    Parameters
    ----------
    error_target : float
        Hamming rate target for items where the model is wrong (default 0.20).
    correct_target : float
        Hamming rate target for items where the model is correct (default 0.01).
    regret_threshold : float
        Regret level above which the full error/correct split applies (default 0.01).
    sigma_coeff_max : float or None
        Hard ceiling: σ_{b,i} ≤ sigma_coeff_max × |ĉ_{b,i}|. None disables (default 5.0).
    All other kwargs forwarded to PerInstItemHammingPerturb.
    """

    def __init__(
        self,
        ptoSolver,
        error_target: float = 0.20,
        correct_target: float = 0.01,
        regret_threshold: float = 0.01,
        sigma_coeff_max: float | None = 5.0,
        **kwargs,
    ):
        super().__init__(ptoSolver, **kwargs)
        self.error_target     = float(error_target)
        self.correct_target   = float(correct_target)
        self.regret_threshold = float(regret_threshold)
        self.sigma_coeff_max  = sigma_coeff_max
        # z* cache — filled after epoch 1 (same mechanism as PerInstanceHammingStarPerturb)
        self.z_star_cache: torch.Tensor | None = None  # (N_train, D), lazy init
        self._cache_warm: bool = False
        self._epoch_z_star: list = []     # (idx, z_star) pairs until cache warm
        self._epoch_coeff_hat: list = []  # coeff_hat (B, D) for sigma ceiling

    def forward(self, problem, coeff_hat, params, coeff_true=None, inst_idx=None, **hyperparams):
        loss = super().forward(
            problem, coeff_hat, params, coeff_true, inst_idx=inst_idx, **hyperparams
        )
        # Accumulate z* until cache is warm
        if not self._cache_warm and self.last_z_star is not None:
            B   = coeff_hat.shape[0]
            idx = inst_idx.long().cpu() if inst_idx is not None else torch.arange(B)
            self._epoch_z_star.append((idx, self.last_z_star.cpu()))
        # Accumulate coeff_hat for the per-item sigma ceiling
        if self.sigma_coeff_max is not None and self.last_perturbed_solutions is not None:
            self._epoch_coeff_hat.append(coeff_hat.detach().cpu())
        return loss

    def step(self, epoch: int) -> float:
        # Build z_star_cache (same logic as PerInstanceHammingStarPerturb)
        if not self._cache_warm and self._epoch_z_star:
            all_idx = torch.cat([pair[0] for pair in self._epoch_z_star])
            all_zs  = torch.cat([pair[1] for pair in self._epoch_z_star], dim=0)
            if self.z_star_cache is None:
                N_train = int(all_idx.max().item()) + 1
                D       = all_zs.shape[-1]
                self.z_star_cache = torch.zeros(N_train, D)
            self.z_star_cache[all_idx] = all_zs.float()
            self._cache_warm = True
        self._epoch_z_star.clear()

        # Collect coeff_hat before any clearing
        coeff_hat_epoch = (
            torch.cat(self._epoch_coeff_hat, dim=0).float()
            if self._epoch_coeff_hat else None
        )
        self._epoch_coeff_hat.clear()

        # Fall back to parent if z* not yet cached or epoch buffers empty
        if self.z_star_cache is None or not (self._epoch_perturbed_sols and self._epoch_z0):
            return super().step(epoch)

        # ---- Gather epoch tensors ----
        all_sols    = torch.cat(self._epoch_perturbed_sols, dim=1).float()  # (N, B_total, D)
        all_z0      = torch.cat(self._epoch_z0, dim=0).float()              # (B_total, D)
        all_indices = torch.cat(self._epoch_indices, dim=0)                 # (B_total,)
        B_total     = all_z0.shape[0]
        D           = all_z0.shape[-1]

        # Lazy init sigma_mat
        if self.sigma_mat is None:
            N_train    = int(all_indices.max().item()) + 1
            init_sigma = self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean())
            self.sigma_mat = torch.full((N_train, D), float(init_sigma))

        with torch.no_grad():
            # Per-(instance, item) Hamming rate: (N, B_total, D) → (B_total, D)
            flips       = (all_sols != all_z0.unsqueeze(0)).float()
            hamming_per = flips.mean(dim=0)  # (B_total, D)

            # Oracle decisions for this batch
            z_star_batch = self.z_star_cache[all_indices].float()  # (B_total, D)

            # Per-instance normalised regret using true costs
            if self._epoch_coeff_true:
                all_y   = torch.cat(self._epoch_coeff_true, dim=0).float()  # (B_total, D)
                opt_obj = (all_y * z_star_batch).sum(dim=-1)                 # (B_total,)
                cur_obj = (all_y * all_z0).sum(dim=-1)                       # (B_total,)
                regret_i = ((opt_obj - cur_obj) / (opt_obj.abs() + 1e-8)).clamp(0)
            else:
                regret_i = torch.ones(B_total)

            # Regret-based blending weight: 0 = low regret (ignore), 1 = high regret (full split)
            w_i = (regret_i / (self.regret_threshold + 1e-8)).clamp(0.0, 1.0)  # (B_total,)

            # Per-(instance, item) differentiated Hamming target
            wrong_bij    = (z_star_batch != all_z0).float()       # (B_total, D)
            correct_bij  = 1.0 - wrong_bij
            split_target = (
                self.error_target * wrong_bij + self.correct_target * correct_bij
            )  # (B_total, D)
            target_bij = (
                w_i.unsqueeze(-1) * split_target
                + (1.0 - w_i.unsqueeze(-1)) * self.hamming_target
            )  # (B_total, D)

            # Proportional update
            ratio = (target_bij / hamming_per.clamp(min=1e-6)).clamp(
                1.0 / self.max_step, self.max_step
            )
            sigma_proposed = self.sigma_mat[all_indices] * ratio  # (B_total, D)

            # Per-item coeff-relative hard ceiling
            if self.sigma_coeff_max is not None and coeff_hat_epoch is not None:
                sigma_hard_max = (self.sigma_coeff_max * coeff_hat_epoch.abs()).clamp(min=1e-6)
                sigma_proposed = sigma_proposed.clamp(max=sigma_hard_max)

            if self.sigma_min_item is not None:
                sigma_proposed = sigma_proposed.clamp(min=self.sigma_min_item)

            # EMA update for seen instances
            self.sigma_mat[all_indices] = (
                self.sigma_ema * self.sigma_mat[all_indices]
                + (1.0 - self.sigma_ema) * sigma_proposed
            )

        self.last_hamming        = hamming_per.mean().item()
        self.last_hamming_vec    = hamming_per.mean(dim=0).clone()  # (D,)
        self.last_sigma_vec_item = self.sigma_mat.mean(dim=0).clone()  # (D,)
        self.sigma               = float(self.sigma_mat.mean().item())

        mean_regret    = regret_i.mean().item()
        mean_wrong_frac = wrong_bij.mean().item()

        # OCV_Y diagnostic
        if self._epoch_coeff_true:
            all_y = torch.cat(self._epoch_coeff_true, dim=0).float()
            z_n   = all_sols.reshape(all_sols.shape[0], B_total, -1)
            y     = all_y.reshape(B_total, -1)
            z_0   = all_z0.reshape(B_total, -1)
            obj_n = torch.einsum("nbd,bd->nb", z_n, y)
            obj_0 = torch.einsum("bd,bd->b",   z_0, y)
            self.last_ocv_y          = (obj_n.std(dim=0) / (obj_0.abs() + 1e-6)).mean().item()
            self.last_frac_improving = (obj_n > obj_0.unsqueeze(0)).float().mean().item()

        self._epoch_perturbed_sols.clear()
        self._epoch_z0.clear()
        self._epoch_coeff_true.clear()
        self._epoch_indices.clear()

        logger.info(
            f"  [flip_margin] epoch {epoch}: "
            f"sigma mean={self.sigma:.5f} min={self.sigma_mat.min():.5f} "
            f"max={self.sigma_mat.max():.5f} hamming_mean={self.last_hamming:.4f} "
            f"wrong_frac={mean_wrong_frac:.4f} regret_mean={mean_regret:.4f} "
            f"ocv_y={self.last_ocv_y:.4f} frac={self.last_frac_improving:.4f} "
            f"cache_warm={self._cache_warm}"
        )
        return self.sigma


# ---------------------------------------------------------------------------
# HammingProportionalSigmaPerturb — no-setpoint per-item sigma controller
# ---------------------------------------------------------------------------

class HammingProportionalSigmaPerturb(PerItemHammingPerturb):
    """
    Per-item sigma controller with NO feedback setpoint.

    sigma_i is assigned proportionally to the empirical flip rate:
        sigma_i = sigma_scale × (hamming_ema_i ** alpha)   (normalized to mean = sigma_scale)

    Key insight: all prior Hamming controllers fail because they have a setpoint.
    When rigid items can't reach the target flip rate, sigma escalates without
    bound. Removing the setpoint entirely prevents ballooning:
      - rigid items (hamming → 0): sigma → 0 (no wasted perturbation)
      - boundary items (hamming ≈ 0.05–0.15): sigma ∝ their natural flip rate
      - alpha controls the shape of the mapping

    Parameters
    ----------
    alpha : float
        Exponent in sigma ∝ hamming^alpha.
        alpha=1.0 → linear (emphasizes boundary items)
        alpha=0.5 → sqrt (compresses dynamic range, gentler)
        alpha=0.0 → uniform sigma = plain perturb (sanity check)
    sigma_scale : float
        Overall magnitude multiplier; mean(sigma_vec) ≈ sigma_scale after update.
    warmup_epochs : int
        Number of epochs to collect hamming stats before first assignment.
    update_freq : int
        Re-estimate sigma every N epochs.
    sigma_ema : float
        EMA for hamming accumulation AND sigma update. High = slow change.
    sigma_min : float or None
        Floor for sigma_vec entries.
    sigma_max : float
        Ceiling for sigma_vec entries.
    All other kwargs forwarded to PerItemHammingPerturb / AdaptiveSigmaPerturb.
    """

    def __init__(
        self,
        ptoSolver,
        hp_alpha: float = 0.5,
        sigma_scale: float = 1.0,
        warmup_epochs: int = 10,
        update_freq: int = 20,
        sigma_ema: float = 0.9,
        sigma_min: float = 1e-4,
        **kwargs,
    ):
        # sigma_max from kwargs sets the per-item ceiling; also forwarded to parent.
        sigma_max_hp = float(kwargs.get("sigma_max", 10.0))
        # hamming_target=0 passed to parent — not used (no setpoint)
        super().__init__(
            ptoSolver,
            hamming_target=0.0,
            sigma_ema=sigma_ema,
            sigma_min=sigma_min,
            **kwargs,
        )
        self.alpha         = float(hp_alpha)   # stored as self.alpha (exponent)
        self.sigma_scale   = float(sigma_scale)
        self.sigma_max_hp  = sigma_max_hp      # per-item ceiling
        self.warmup_epochs = int(warmup_epochs)
        self.update_freq   = int(update_freq)
        self.hamming_ema: torch.Tensor | None = None  # (D,), lazy init

    def step(self, epoch: int) -> float:
        self._batch_scales.clear()

        if not (self._epoch_perturbed_sols and self._epoch_z0):
            mean_sigma = (
                float(self.sigma_vec.mean()) if self.sigma_vec is not None
                else (self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean()))
            )
            self._epoch_perturbed_sols.clear()
            self._epoch_z0.clear()
            self._epoch_coeff_true.clear()
            return mean_sigma

        all_sols = torch.cat(self._epoch_perturbed_sols, dim=1).float()  # (N, B_total, D)
        all_z0   = torch.cat(self._epoch_z0, dim=0).float()              # (B_total, D)
        B_total  = all_z0.shape[0]
        D        = all_z0.shape[-1]

        # Lazy init sigma_vec and hamming_ema
        if self.sigma_vec is None:
            init_sigma = self.sigma if not torch.is_tensor(self.sigma) else float(self.sigma.mean())
            self.sigma_vec = torch.full((D,), float(init_sigma))
        if self.hamming_ema is None:
            # Init to uniform (neutral: no item favored over another)
            self.hamming_ema = torch.full((D,), float(self.sigma_scale))

        with torch.no_grad():
            # Per-item Hamming rate (mean over N and B dims)
            hamming_item = hamming_rate_per_item(all_sols, all_z0)  # (D,)

            # Update hamming EMA (high sigma_ema = slow change)
            ema = self.sigma_ema
            self.hamming_ema = ema * self.hamming_ema + (1.0 - ema) * hamming_item

            # Diagnostic OCV_Y
            if self._epoch_coeff_true:
                all_y = torch.cat(self._epoch_coeff_true, dim=0)
                z_n = all_sols.reshape(all_sols.shape[0], B_total, -1)
                z_0 = all_z0.reshape(B_total, -1)
                y   = all_y.reshape(B_total, -1)
                obj_n = torch.einsum("nbd,bd->nb", z_n, y)
                obj_0 = torch.einsum("bd,bd->b",   z_0, y)
                ocv_y_per = obj_n.std(dim=0) / (obj_0.abs() + 1e-6)
                frac_per  = (obj_n > obj_0.unsqueeze(0)).float().mean(dim=0)
                self.last_ocv_y          = ocv_y_per.mean().item()
                self.last_frac_improving = frac_per.mean().item()

            # Proportional sigma assignment (no setpoint)
            # Only update after warmup and at update_freq intervals
            if epoch >= self.warmup_epochs and epoch % self.update_freq == 0:
                raw = self.hamming_ema.clamp(min=1e-6) ** self.alpha  # (D,)
                raw_mean = raw.mean()
                if raw_mean > 1e-12:
                    sigma_target = raw / raw_mean * self.sigma_scale
                else:
                    sigma_target = torch.full((D,), self.sigma_scale)

                smin = self.sigma_min_item if self.sigma_min_item is not None else 0.0
                sigma_target = sigma_target.clamp(min=smin, max=self.sigma_max_hp)

                # EMA update (high ema = slow change → stable sigma)
                self.sigma_vec = ema * self.sigma_vec + (1.0 - ema) * sigma_target

        self.last_hamming        = hamming_item.mean().item()
        self.last_hamming_vec    = hamming_item.clone()
        self.last_sigma_vec_item = self.sigma_vec.clone()
        self.sigma = float(self.sigma_vec.mean().item())

        self._epoch_perturbed_sols.clear()
        self._epoch_z0.clear()
        self._epoch_coeff_true.clear()

        logger.info(
            f"  [hamming_proportional] epoch {epoch}: "
            f"sigma mean={self.sigma:.5f} min={self.sigma_vec.min():.5f} "
            f"max={self.sigma_vec.max():.5f} hamming_mean={self.last_hamming:.4f} "
            f"alpha={self.alpha:.2f} sigma_scale={self.sigma_scale:.3f} "
            f"ocv_y={self.last_ocv_y:.4f}"
        )
        return self.sigma
