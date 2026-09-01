"""Tests for the scikit-learn-like FeatureSelector estimator."""

from __future__ import annotations

import numpy as np
import pytest

from siftfs import FeatureSelector


def make_data(n: int = 400, p: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    # Only the first 4 features are informative.
    signal = X[:, :4]
    y = (signal[:, 0] + signal[:, 1] * 2 - signal[:, 2] + signal[:, 3]).copy()
    # Three classes.
    y = (y > np.quantile(y, 1 / 3)).astype(int) + (y > np.quantile(y, 2 / 3)).astype(int)
    return X, y


def test_fit_selects_exact_budget() -> None:
    X, y = make_data()
    sel = FeatureSelector(panel_size=5, epochs=20, validation_split=0.2)
    sel.fit(X, y)
    feats = sel.get_selected_features()
    assert len(feats) == 5
    assert len(set(feats)) == 5
    assert all(0 <= i < X.shape[1] for i in feats)


def test_selects_informative_features() -> None:
    X, y = make_data()
    sel = FeatureSelector(panel_size=4, epochs=25, validation_split=0.2)
    sel.fit(X, y)
    feats = sel.get_selected_features()
    # The 4 informative features (0..3) should dominate the top-4 selection.
    assert set(feats) == {0, 1, 2, 3}


def test_transform_returns_selected_columns() -> None:
    X, y = make_data()
    sel = FeatureSelector(panel_size=4, epochs=10, validation_split=0.2)
    Xt = sel.fit(X, y).transform(X)
    assert Xt.shape == (X.shape[0], 4)


def test_get_selected_features_before_fit_raises() -> None:
    sel = FeatureSelector(panel_size=3)
    with pytest.raises(RuntimeError):
        sel.get_selected_features()


def test_early_stop_false_trains_fixed_epochs() -> None:
    X, y = make_data()
    sel = FeatureSelector(panel_size=5, epochs=3, early_stop=False, validation_split=0.2)
    sel.fit(X, y)
    feats = sel.get_selected_features()
    assert len(feats) == 5
    assert all(0 <= i < X.shape[1] for i in feats)


def test_panel_size_larger_than_features_raises() -> None:
    X, y = make_data()
    sel = FeatureSelector(panel_size=100)
    with pytest.raises(ValueError):
        sel.fit(X, y)


def test_invalid_panel_size_raises() -> None:
    with pytest.raises(ValueError):
        FeatureSelector(panel_size=0)


def test_high_dim_does_not_collapse_to_undifferentiated_weights() -> None:
    """Regression: early stopping must use the combined (task + reg) loss.

    On high-dimensional data with few informative features, monitoring only the
    task cross-entropy for early stopping restored an early snapshot with uniform
    (undifferentiated) gate weights, producing arbitrary garbage selection. The
    combined loss keeps training so the informative features are ranked on top.
    """
    rng = np.random.RandomState(0)
    n, p = 1500, 2000
    X = rng.randn(n, p)
    sig = X[:, :4]
    y = sig[:, 0] + sig[:, 1] * 2 - sig[:, 2] + sig[:, 3]
    qs = np.quantile(y, [1 / 3, 2 / 3])
    y = (y > qs[0]).astype(int) + (y > qs[1]).astype(int)

    sel = FeatureSelector(panel_size=4, alpha=1.5, epochs=150, seed=1)
    sel.fit(X, y)

    # The informative features must be selected when the budget allows
    # (the bug produced uniform weights -> arbitrary selection).
    assert set(sel.get_selected_features()) == {0, 1, 2, 3}
