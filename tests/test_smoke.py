"""Placeholder test for the package import surface.

Full unit tests for the gating layer, loss, and selector land alongside the
Phase 1 (core engine) implementation.
"""

from __future__ import annotations

import siftfs


def test_version_is_pep440() -> None:
    assert siftfs.__version__.count(".") >= 2


def test_public_api_is_exported() -> None:
    for name in ("FeatureSelector", "FeatureGatingLayer", "FeatureSelectorLoss"):
        assert hasattr(siftfs, name)
