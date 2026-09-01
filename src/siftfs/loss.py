"""Feature-selection loss terms.

Conceptually mirrors scGIST's ``FeatureRegularizer``: a composite loss that balances

    * the task objective (e.g., cross-entropy for classification),
    * an exact feature-count penalty enforcing the panel size ``d``,
    * an optional priority term that drives the model to include specific features,
    * an optional group/complex term that keeps or drops whole physical groups
      (e.g., all axes of an accelerometer) together.

All values should be tensors; helpers to assemble the combined scalar loss live here.
"""

from __future__ import annotations


class FeatureSelectorLoss:
    """Assembles the multi-term feature-selection loss (Phase 1)."""

    def __init__(self) -> None:
        raise NotImplementedError("FeatureSelectorLoss lands in Phase 1 (core engine).")
