"""One-to-one feature gating layer.

Maps a PyTorch-native reimplementation of scGIST's ``OneToOneLayer``: a single
trainable, per-feature mask (kernel) that element-wise multiplies the input, so
every feature has exactly one scalar weight controlling whether it is selected.

Implementation detail (Phase 1, core engine): the applied mask is a binarized /
straight-through estimate of the trained kernel so that downstream layers see
hard 0/1 gates while gradients still flow to the kernel during training.
"""

from __future__ import annotations

from torch import nn


class FeatureGatingLayer(nn.Module):
    """Learns a per-feature 0/1 gate.

    Args:
        n_features: Number of input features. Each gets a single learnable weight
            used to decide whether it is kept or dropped from the panel.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features
        raise NotImplementedError("FeatureGatingLayer lands in Phase 1 (core engine).")

    def forward(self, inputs: object) -> object:  # pragma: no cover - Phase 1 placeholder
        raise NotImplementedError("FeatureGatingLayer lands in Phase 1 (core engine).")
