"""One-to-one feature gating layer.

A PyTorch-native reimplementation of scGIST's ``OneToOneLayer``: a single
trainable, per-feature mask (kernel) that element-wise multiplies the input, so
every feature has exactly one scalar weight controlling whether it is selected.

During ``forward`` the input is multiplied by the *continuous* trained weight
(faithful to scGIST): gradients flow to every feature through its own weight, which
is what lets the task loss differentiate the features. The discrete 0/1 mask is
derived only for final reporting (``mask``).
"""

from __future__ import annotations

import torch
from torch import nn


class FeatureGatingLayer(nn.Module):
    """Learns a per-feature 0/1 gate.

    Args:
        n_features: Number of input features. Each gets a single learnable weight
            used to decide whether it is kept or dropped from the panel.
        init: Initial value for each gate weight (default ``0.5``).
    """

    def __init__(self, n_features: int, init: float = 0.5) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be a positive integer.")
        self.n_features = n_features
        self.weight = nn.Parameter(torch.full((n_features,), init))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Multiply ``inputs`` element-wise by the continuous gate weights.

        Args:
            inputs: Input tensor of shape ``(..., n_features)``.

        Returns:
            Masked tensor of the same shape.
        """
        return inputs * self.weight

    @property
    def weights(self) -> torch.Tensor:
        """The raw, unbounded per-feature weights."""
        return self.weight

    @property
    def score(self) -> torch.Tensor:
        """Continuous selection score (the raw weights).

        Used by the ``FeatureSelectorLoss`` regularizer and by ranking for final
        feature selection.
        """
        return self.weight

    @property
    def mask(self) -> torch.Tensor:
        """Detached 0/1 mask for reporting: 1 where ``weight >= 0.5``."""
        return (self.weight.detach() >= 0.5).to(self.weight.dtype)