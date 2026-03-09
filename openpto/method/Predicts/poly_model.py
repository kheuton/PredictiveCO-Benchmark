"""
Polynomial prediction model for problems with polynomial data-generating processes.

Two modes (auto-detected from input/output dims):
  'basis'       — for scalar-feature-per-item problems (e.g. cubic).
                  Expands each scalar x to [x, x^2, ..., x^poly_deg], then
                  applies a learned linear combination.
                  Input:  (..., 1)  →  Output: (..., 1)

  'composition' — for vector-feature-per-instance problems (e.g. knapsack).
                  Applies a learned linear projection then raises to poly_deg.
                  Input:  (B, num_features)  →  Output: (B, num_targets)
"""

import torch
import torch.nn as nn


class PolyPredModel(nn.Module):
    """
    Correctly-specified polynomial prediction model.

    Args:
        num_features (int): Input feature dimension.
        num_targets (int): Output dimension.
        poly_deg (int): Degree of the polynomial.
        offset (float): Additive offset before raising to poly_deg (composition mode).
            For knapsack DGP this is 3.0.
        poly_mode (str): 'auto', 'basis', or 'composition'. 'auto' chooses
            'basis' when num_features == num_targets == 1, else 'composition'.
    """

    def __init__(
        self,
        num_features: int,
        num_targets: int,
        poly_deg: int = 3,
        offset: float = 0.0,
        poly_mode: str = "auto",
        **kwargs,  # absorb unused args from pred_model_wrapper
    ):
        super().__init__()
        self.poly_deg = poly_deg
        self.offset = offset

        if poly_mode == "auto":
            self.poly_mode = (
                "basis" if (num_features == 1 and num_targets == 1) else "composition"
            )
        else:
            assert poly_mode in ("basis", "composition")
            self.poly_mode = poly_mode

        if self.poly_mode == "basis":
            # Learns a weighted sum of [x, x^2, ..., x^poly_deg]
            # bias=False: cubic DGP y = 10(x^3 - 0.65x) has no constant term
            self.linear = nn.Linear(poly_deg, 1, bias=False)
        else:
            # Learns a linear projection per output, then applies polynomial
            self.linear = nn.Linear(num_features, num_targets)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.poly_mode == "basis":
            # X: (..., 1) — scalar feature per item
            # Expand to polynomial basis: (..., poly_deg)
            features = torch.cat(
                [X ** d for d in range(1, self.poly_deg + 1)], dim=-1
            )
            return self.linear(features)
        else:
            # X: (B, num_features)
            # Linear projection → polynomial
            out = self.linear(X)  # (B, num_targets)
            return (out + self.offset) ** self.poly_deg
