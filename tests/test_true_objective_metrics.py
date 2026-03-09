"""
Unit tests for true_objective_metrics in perturb_metrics.py.
"""

import torch
import pytest
from openpto.diagnostics.perturb_metrics import true_objective_metrics


def make_binary(N, B, D, fill=None):
    if fill is not None:
        return torch.full((N, B, D), fill, dtype=torch.float32)
    return (torch.rand(N, B, D) > 0.5).float()


# ---------------------------------------------------------------------------
# OCV_Y = 0 when all perturbed solutions equal z0
# ---------------------------------------------------------------------------

def test_ocv_y_zero_when_all_equal():
    B, D, N = 8, 10, 20
    z0 = (torch.rand(B, D) > 0.5).float()
    z_n = z0.unsqueeze(0).expand(N, -1, -1)        # (N, B, D) all identical to z0
    y  = torch.randn(B, D)

    metrics = true_objective_metrics(z_n, z0, y)
    assert metrics["ocv_y"] == pytest.approx(0.0, abs=1e-5), \
        f"OCV_Y should be 0 when all z_n = z0, got {metrics['ocv_y']}"


# ---------------------------------------------------------------------------
# OCV_Y > 0 when perturbed solutions are diverse
# ---------------------------------------------------------------------------

def test_ocv_y_positive_with_diverse_solutions():
    B, D, N = 8, 10, 50
    z0 = torch.zeros(B, D)
    # Diverse solutions: random binary
    z_n = (torch.rand(N, B, D) > 0.5).float()
    y   = torch.ones(B, D)   # all costs = 1, so obj = number of selected items

    metrics = true_objective_metrics(z_n, z0, y)
    assert metrics["ocv_y"] > 0.0, \
        f"OCV_Y should be positive with diverse solutions, got {metrics['ocv_y']}"


# ---------------------------------------------------------------------------
# FracImproving = 0 when z0 is globally optimal (all z_n have lower true obj)
# ---------------------------------------------------------------------------

def test_frac_improving_zero_when_z0_optimal():
    B, D, N = 4, 5, 30
    # z0 = all-ones (maximises sum(y) for positive y)
    z0 = torch.ones(B, D)
    y  = torch.ones(B, D)   # obj(z0) = D per instance (maximum)

    # z_n always drop at least one item — strictly worse under maximisation
    z_n = torch.zeros(N, B, D)  # obj = 0 < D for all

    metrics = true_objective_metrics(z_n, z0, y)
    assert metrics["frac_improving"] == pytest.approx(0.0, abs=1e-6), \
        f"FracImproving should be 0 when z0 is optimal, got {metrics['frac_improving']}"


# ---------------------------------------------------------------------------
# FracImproving = 1 when every z_n beats z0
# ---------------------------------------------------------------------------

def test_frac_improving_one_when_all_better():
    B, D, N = 4, 5, 20
    z0 = torch.zeros(B, D)          # obj(z0) = 0
    z_n = torch.ones(N, B, D)       # obj(z_n) = D > 0 for positive y
    y   = torch.ones(B, D)

    metrics = true_objective_metrics(z_n, z0, y)
    assert metrics["frac_improving"] == pytest.approx(1.0, abs=1e-6), \
        f"FracImproving should be 1 when all z_n beat z0, got {metrics['frac_improving']}"


# ---------------------------------------------------------------------------
# Return dict has correct keys
# ---------------------------------------------------------------------------

def test_return_keys():
    B, D, N = 2, 4, 5
    z0  = torch.zeros(B, D)
    z_n = torch.rand(N, B, D)
    y   = torch.randn(B, D)
    metrics = true_objective_metrics(z_n, z0, y)
    assert "ocv_y" in metrics
    assert "frac_improving" in metrics


# ---------------------------------------------------------------------------
# eps prevents division by zero when |Y·z_0| = 0
# ---------------------------------------------------------------------------

def test_no_div_zero_when_obj0_is_zero():
    B, D, N = 4, 6, 10
    z0  = torch.zeros(B, D)          # obj(z0) = 0 always
    z_n = (torch.rand(N, B, D) > 0.5).float()
    y   = torch.ones(B, D)

    # Should not raise, result should be finite
    metrics = true_objective_metrics(z_n, z0, y)
    assert torch.isfinite(torch.tensor(metrics["ocv_y"])), \
        "OCV_Y should be finite even when obj_0 = 0"
