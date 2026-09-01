"""The user-facing, scikit-learn-like estimator.

``FeatureSelector`` wraps the gating layer and loss into a familiar estimator with
``fit(X, y)`` and ``get_selected_features()``. The internal neural net trains the
per-feature gates so the final panel satisfies the exact feature budget ``d`` while
external downstream classifiers (RF / SVM / KNN) can then be trained on the selected
subset — the workflow used in Phase 3 benchmarking.
"""

from __future__ import annotations


class FeatureSelector:
    """Constrained feature selector with an exact feature budget (Phase 1)."""

    def __init__(self) -> None:
        raise NotImplementedError("FeatureSelector lands in Phase 1 (core engine).")

    def fit(self, X: object, y: object) -> None:  # pragma: no cover - Phase 1 placeholder
        raise NotImplementedError("FeatureSelector lands in Phase 1 (core engine).")

    def get_selected_features(self) -> list[object]:  # pragma: no cover - Phase 1 placeholder
        raise NotImplementedError("FeatureSelector lands in Phase 1 (core engine).")
