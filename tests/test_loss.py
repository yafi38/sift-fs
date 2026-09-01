"""Tests for the composite feature-selection loss."""

from __future__ import annotations

import torch

from siftfs.layers import FeatureGatingLayer
from siftfs.loss import FeatureSelectorLoss


def _layer(weights: list[float]) -> FeatureGatingLayer:
    layer = FeatureGatingLayer(n_features=len(weights))
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weights))
    return layer


def test_regularization_binarizes_weights() -> None:
    # A weight of exactly 0 or 1 yields zero binarization penalty.
    layer = _layer([0.0, 1.0])
    loss = FeatureSelectorLoss(layer, l1=1.0)
    pen = loss._regularization()
    assert torch.allclose(pen, torch.tensor(0.0))


def test_regularization_scaled_by_l1() -> None:
    layer = _layer([0.5])
    raw = FeatureSelectorLoss(layer, l1=1.0)._regularization()
    scaled = FeatureSelectorLoss(layer, l1=0.01)._regularization()
    assert torch.allclose(scaled, raw * 0.01)


def test_exact_panel_size_is_enforced() -> None:
    layer = _layer([1.0, 1.0, 0.0])
    loss = FeatureSelectorLoss(layer, panel_size=2, alpha=1.0, l1=1.0)
    # Sum |x| = 2 -> exactly at panel size, zero count penalty.
    count_pen = torch.abs(loss.gating_layer.score.sum() - 2)
    assert torch.allclose(count_pen, torch.tensor(0.0))

    layer2 = _layer([1.0, 1.0, 1.0])  # sum = 3, but d = 2
    loss2 = FeatureSelectorLoss(layer2, panel_size=2, alpha=1.0, l1=1.0)
    assert loss2._regularization() > 0


def test_strict_vs_loose_panel_size() -> None:
    # sum=1, d=2: strict penalizes |1-2|, loose penalizes max(1-2,0)=0.
    layer = _layer([1.0, 0.0])
    strict = FeatureSelectorLoss(layer, panel_size=2, alpha=1.0, strict=True, l1=1.0)
    loose = FeatureSelectorLoss(layer, panel_size=2, alpha=1.0, strict=False, l1=1.0)
    assert loose._regularization() == 0.0
    assert strict._regularization() > 0.0


def test_priority_drives_selection() -> None:
    layer = _layer([1.0, 0.5])
    binarize = 1.0 * abs(1.0) * abs(1.0 - 1) + 1.0 * abs(0.5) * abs(0.5 - 1)  # 0 + 0.25
    # s = [1, 0]: only feature 0 is prioritized; it's already selected -> no penalty.
    loss = FeatureSelectorLoss(layer, priority_scores=[1.0, 0.0], beta=1.0, l1=1.0)
    pen = loss._regularization()
    assert torch.allclose(pen, torch.tensor(binarize))

    # s = [1, 1]: feature 1 also prioritized, but it's only at 0.5 -> penalty.
    loss2 = FeatureSelectorLoss(layer, priority_scores=[1.0, 1.0], beta=1.0, l1=1.0)
    pen2 = loss2._regularization()
    priority = 1.0 * (1.0 - 0.5)
    assert torch.allclose(pen2, torch.tensor(binarize + priority))


def test_group_keeps_members_together() -> None:
    # Members equal -> zero group penalty; members differ -> positive.
    same = FeatureSelectorLoss(_layer([0.9, 0.9, 0.1]), groups=[[0, 1]], gamma=1.0, l1=1.0)
    diff = FeatureSelectorLoss(_layer([0.9, 0.1, 0.1]), groups=[[0, 1]], gamma=1.0, l1=1.0)
    assert same._regularization() < diff._regularization()


def test_call_combines_task_loss_and_regularization() -> None:
    layer = _layer([1.0, 0.0])
    loss = FeatureSelectorLoss(layer, panel_size=1)
    comb = loss(torch.tensor(0.5))
    assert torch.allclose(comb, torch.tensor(0.5) + loss._regularization())


def test_priority_scores_length_mismatch_raises() -> None:
    layer = _layer([1.0, 0.0])
    try:
        FeatureSelectorLoss(layer, priority_scores=[1.0, 0.0, 0.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for length mismatch")
