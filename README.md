# sift-fs

A **PyTorch-native, general-purpose constrained feature selection** library for
tabular data, edge computing, and IoT ML.

`sift-fs` adapts the core mathematical engine of
[`scGIST`](https://github.com/yafi38/scGIST) — a one-to-one gating layer, exact
feature-count enforcement, and group-level penalties — into a lightweight framework
aimed at resource-constrained devices.

## Why sift-fs?

Standard feature-selection methods fail edge computing in two critical ways:

1. **Unpredictable feature counts.** `L1` regularization (Lasso) uses a continuous
   penalty hyperparameter (`λ`) that you must tune to *guess* at a feature budget.
   `sift-fs` lets you set an exact target count (`d`) and enforces a hard cap
   programmatically.

2. **Ignoring physical hardware.** Standard algorithms may pick 1 axis of a
   3-axis accelerometer and 1 axis of a gyroscope — both sensors stay powered on,
   so **zero battery is saved**. `sift-fs`'s group penalty forces the model to
   evaluate physical components as unified blocks, keeping or dropping an entire
   sensor so hardware can actually be powered off.

## Status

Early development. The project is scaffolded (package layout, tooling, CI); the
core engine lands in **Phase 1**.

## Roadmap

- **Phase 1 — Core engine:** PyTorch gating layer, exact-`d` loss, group penalty,
  scikit-learn-like API (`fit(X, y)`, `get_selected_features()`).
- **Phase 2 — Dataset setup:** UCI HAR on Kaggle; map 561 statistical features back
  to physical sensor groups.
- **Phase 3 — Benchmarking:** downstream classifiers (RF / SVM / KNN), baselines
  (RFE, Lasso, Mutual Information, Concrete Autoencoders), Pareto curves
  (accuracy vs. `d`), and an active-hardware / battery metric.
- **Phase 4 — Release & paper:** package on PyPI, README with benchmark plots, and a
  workshop paper on the single-cell → edge-hardware transition.

## Installation

```bash
uv sync
```

## Development

```bash
# Lint & format checks
uv run ruff check .

# Type checking
uv run mypy src

# Tests
uv run pytest
```

## License

[MIT](LICENSE)
