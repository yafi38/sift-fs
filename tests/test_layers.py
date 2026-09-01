"""Tests for the one-to-one feature gating layer."""

from __future__ import annotations

import torch

from siftfs.layers import FeatureGatingLayer


def test_initial_weights_are_half() -> None:
    layer = FeatureGatingLayer(n_features=5)
    assert layer.weight.shape == (5,)
    assert torch.allclose(layer.weight, torch.full((5,), 0.5))


def test_forward_multiplies_input_by_weights() -> None:
    layer = FeatureGatingLayer(n_features=3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.0, 0.0, 0.5]))
    x = torch.tensor([[2.0, 4.0, 8.0]])
    out = layer(x)
    assert torch.allclose(out, torch.tensor([[2.0, 0.0, 4.0]]))


def test_forward_passes_gradient_to_weight() -> None:
    layer = FeatureGatingLayer(n_features=2)
    x = torch.tensor([3.0, 5.0])
    out = layer(x)
    out.sum().backward()
    assert layer.weight.grad is not None
    # d(out_i)/d(w_i) = x_i.
    assert torch.allclose(layer.weight.grad, torch.tensor([3.0, 5.0]))


def test_mask_thresholds_at_half() -> None:
    layer = FeatureGatingLayer(n_features=4)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.9, 0.1, 0.5, -0.3]))
    expected = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 0.5 thresholds to 1
    assert torch.allclose(layer.mask, expected)


def test_negative_init_allows_zero_start() -> None:
    layer = FeatureGatingLayer(n_features=3, init=-0.2)
    assert torch.allclose(layer.mask, torch.zeros(3))


def test_invalid_n_features_raises() -> None:
    try:
        FeatureGatingLayer(n_features=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for n_features=0")