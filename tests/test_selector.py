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
    sel = FeatureSelector(panel_size=5, epochs=20)
    sel.fit(X, y)
    feats = sel.get_selected_features()
    assert len(feats) == 5
    assert len(set(feats)) == 5
    assert all(0 <= i < X.shape[1] for i in feats)


def test_selects_informative_features() -> None:
    X, y = make_data()
    sel = FeatureSelector(panel_size=4, epochs=25)
    sel.fit(X, y)
    feats = sel.get_selected_features()
    # The 4 informative features (0..3) should dominate the top-4 selection.
    assert set(feats) == {0, 1, 2, 3}


def test_transform_returns_selected_columns() -> None:
    X, y = make_data()
    sel = FeatureSelector(panel_size=4, epochs=10)
    Xt = sel.fit(X, y).transform(X)
    assert Xt.shape == (X.shape[0], 4)


def test_get_selected_features_before_fit_raises() -> None:
    sel = FeatureSelector(panel_size=3)
    with pytest.raises(RuntimeError):
        sel.get_selected_features()


def test_transform_before_fit_raises() -> None:
    sel = FeatureSelector(panel_size=3)
    with pytest.raises(RuntimeError):
        sel.transform(np.zeros((10, 5)))


def test_non_strict_caps_selection_to_significant_features() -> None:
    X, y = make_data()
    sel = FeatureSelector(panel_size=5, epochs=20, strict=False)
    sel.fit(X, y)
    feats = sel.get_selected_features()
    # Non-strict mode may return fewer than panel_size features, never more.
    assert len(feats) <= 5
    assert all(0 <= i < X.shape[1] for i in feats)


def test_non_strict_high_threshold_yields_empty_selection() -> None:
    """With no feature exceeding the significance threshold, d caps to zero.

    Mirrors scGIST's ``min(panel_size, n_significant) == 0`` behavior.
    """
    X, y = make_data()
    sel = FeatureSelector(
        panel_size=5,
        epochs=20,
        strict=False,
        significance_threshold=100.0,
    )
    sel.fit(X, y)
    assert sel.get_selected_features() == []


def test_device_is_stored_as_given() -> None:
    assert FeatureSelector(panel_size=3).device is None
    assert FeatureSelector(panel_size=3, device="cpu").device == "cpu"


def test_panel_size_larger_than_features_raises() -> None:
    X, y = make_data()
    sel = FeatureSelector(panel_size=100)
    with pytest.raises(ValueError):
        sel.fit(X, y)


def test_invalid_panel_size_raises() -> None:
    with pytest.raises(ValueError):
        FeatureSelector(panel_size=0)


def test_anneal_disabled_by_default() -> None:
    sel = FeatureSelector(panel_size=5)
    assert sel.anneal_l1 is False
    assert sel.anneal_l1_from is None
    assert sel.anneal_l1_epochs is None


def test_anneal_overrides_require_anneal_l1_true() -> None:
    with pytest.raises(ValueError):
        FeatureSelector(panel_size=5, anneal_l1_from=1e-4)
    with pytest.raises(ValueError):
        FeatureSelector(panel_size=5, anneal_l1_epochs=10)


def test_anneal_defaults_stay_unresolved_until_fit() -> None:
    sel = FeatureSelector(panel_size=5, l1=0.01, epochs=100, anneal_l1=True)
    assert sel.anneal_l1_from is None
    assert sel.anneal_l1_epochs is None


def test_anneal_requires_positive_values() -> None:
    with pytest.raises(ValueError):
        FeatureSelector(panel_size=5, anneal_l1=True, anneal_l1_from=0.0)
    with pytest.raises(ValueError):
        FeatureSelector(panel_size=5, anneal_l1=True, anneal_l1_epochs=0)
    with pytest.raises(ValueError):
        FeatureSelector(panel_size=5, l1=0.0, anneal_l1=True)


def test_anneal_from_must_be_below_l1() -> None:
    with pytest.raises(ValueError):
        FeatureSelector(panel_size=5, l1=0.01, anneal_l1=True, anneal_l1_from=0.01)
    with pytest.raises(ValueError):
        FeatureSelector(panel_size=5, l1=0.01, anneal_l1=True, anneal_l1_from=0.1)


def test_anneal_epochs_must_not_exceed_epochs() -> None:
    with pytest.raises(ValueError):
        FeatureSelector(panel_size=5, epochs=100, anneal_l1=True, anneal_l1_epochs=101)
    FeatureSelector(panel_size=5, epochs=100, anneal_l1=True, anneal_l1_epochs=100)


def test_l1_is_fixed_without_annealing() -> None:
    sel = FeatureSelector(panel_size=5, l1=0.01, epochs=100)
    assert all(sel._l1_for_epoch(e) == 0.01 for e in range(100))


def test_anneal_ramps_geometrically_then_holds() -> None:
    sel = FeatureSelector(
        panel_size=5, l1=0.01, epochs=100, anneal_l1=True, anneal_l1_from=1e-5, anneal_l1_epochs=50
    )
    assert sel._l1_for_epoch(0) == pytest.approx(1e-5)
    assert sel._l1_for_epoch(25) == pytest.approx((1e-5 * 0.01) ** 0.5)
    assert sel._l1_for_epoch(50) == pytest.approx(0.01)
    assert sel._l1_for_epoch(99) == pytest.approx(0.01)


def test_anneal_default_schedule() -> None:
    sel = FeatureSelector(panel_size=5, l1=0.01, epochs=100, anneal_l1=True)
    assert sel._l1_for_epoch(0) == pytest.approx(0.01 / 1000)
    assert sel._l1_for_epoch(74) < 0.01
    assert sel._l1_for_epoch(75) == pytest.approx(0.01)


def test_anneal_default_schedule_follows_epochs_changed_after_init() -> None:
    sel = FeatureSelector(panel_size=5, l1=0.01, epochs=100, anneal_l1=True)
    sel.epochs = 200
    assert sel._l1_for_epoch(75) < 0.01
    assert sel._l1_for_epoch(150) == pytest.approx(0.01)


def test_anneal_selects_informative_features() -> None:
    X, y = make_data()
    sel = FeatureSelector(panel_size=4, epochs=25, anneal_l1=True)
    sel.fit(X, y)
    assert set(sel.get_selected_features()) == {0, 1, 2, 3}


def test_repr_includes_anneal_settings() -> None:
    sel = FeatureSelector(panel_size=5, anneal_l1=True, anneal_l1_from=1e-5, anneal_l1_epochs=10)
    text = repr(sel)
    assert "anneal_l1=True" in text
    assert "anneal_l1_from=1e-05" in text
    assert "anneal_l1_epochs=10" in text


def test_high_dim_does_not_collapse_to_undifferentiated_weights() -> None:
    """Regression: gate weights must differentiate on high-dimensional data.

    With few informative features among many, training once ended on a snapshot
    with uniform (undifferentiated) gate weights, producing arbitrary selection.
    The informative features must be ranked on top.
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
