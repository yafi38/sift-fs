"""sift-fs: A PyTorch-native, general-purpose constrained feature selection library.

The name "sift" mirrors the project's aim: sifting a large feature space down to a
budgeted, hardware-aware subset. It adapts the core mathematical engine of scGIST
(one-to-one gating layer, exact feature-count enforcement, and group-level penalties)
into a lightweight framework for tabular data, edge computing, and IoT ML.
"""

__version__ = "0.1.0"

from siftfs.layers import FeatureGatingLayer
from siftfs.loss import FeatureSelectorLoss
from siftfs.selector import FeatureSelector

__all__ = [
    "FeatureSelector",
    "FeatureGatingLayer",
    "FeatureSelectorLoss",
]
