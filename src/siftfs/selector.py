"""The user-facing, scikit-learn-like estimator.

``FeatureSelector`` wraps the gating layer and composite loss into a familiar
estimator with ``fit(X, y)`` and ``get_selected_features()``. It trains a small MLP
that learns per-feature gates: at the end, the gate weights define which features
are selected (top-``panel_size`` by absolute score), so an exact feature budget is
enforced programmatically with no ``lambda`` tuning.

The selected subset can then be fed to any downstream classifier (RF / SVM / KNN)
in the Phase 3 benchmarking workflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from siftfs.layers import FeatureGatingLayer
from siftfs.loss import FeatureSelectorLoss

Array = npt.NDArray[np.floating[Any]]

DEFAULT_ANNEAL_START_RATIO = 1e-3
DEFAULT_ANNEAL_EPOCH_FRACTION = 0.75


class _MaskedMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dims: tuple[int, ...],
        init: float,
    ) -> None:
        super().__init__()
        self.gating = FeatureGatingLayer(n_features, init=init)
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.task = nn.Sequential(*layers)
        self._l2_weights: list[nn.Parameter] = [w for m in self.task for w in m.parameters() if w.ndim == 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.task(self.gating(x)))

    def l2_penalty(self) -> torch.Tensor:
        """Sum of squared weights of the task MLP (scGIST's l2(0.01))."""
        total = torch.zeros((), device=self.gating.weight.device, dtype=self.gating.weight.dtype)
        for w in self._l2_weights:
            total = total + (w**2).sum()
        return total


class FeatureSelector:
    """Constrained feature selector with an exact feature budget.

    Args:
        panel_size: Exact number of features to select (``d``).
        priority_scores: Per-feature priority in ``[0, 1]`` driving inclusion.
        groups: Index groups whose features are kept or dropped together.
        alpha: Strength of the exact feature-count penalty.
        beta: Strength of the priority penalty.
        gamma: Strength of the group penalty.
        strict: Exact-``d`` (``True``) or at-most-``d`` (``False``) budget.
        significance_threshold: In non-strict mode, gate weights with absolute
            magnitude at or below this value are treated as inactive and excluded,
            and ``d`` is capped to the number of significant features (scGIST: 0.01).
        l1: Overall scaling of the feature-selection regularization (scGIST: 0.01).
            When annealing is enabled (``anneal_l1``), this is the target value
            ``l1`` ramps up to.
        anneal_l1: When ``True``, ``l1`` ramps up exponentially from
            ``anneal_l1_from`` to ``l1`` over ``anneal_l1_epochs`` epochs
            instead of staying fixed at ``l1`` throughout training. Starting
            well below ``l1`` lets the gating layer explore the feature space
            with little selection pressure before the budget and binarization
            penalties kick in. Defaults to ``False`` (fixed ``l1``).
        anneal_l1_from: Starting value for the ``l1`` ramp, in ``(0, l1)``. Only
            used when ``anneal_l1`` is ``True``; ``None`` means ``l1 / 1000``.
        anneal_l1_epochs: Number of epochs over which ``l1`` ramps from
            ``anneal_l1_from`` to ``l1``, in ``[1, epochs]``; held at ``l1`` for
            any epochs beyond this. Only used when ``anneal_l1`` is ``True``;
            ``None`` means ``round(epochs * 0.75)``.
        l2_decay: L2 penalty coefficient on the task MLP weights (scGIST: 0.01).
        hidden_dims: Hidden layer widths of the task MLP.
        init: Initial value of each gate weight.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        batch_size: Minibatch size.
        device: Torch device; when ``None``, CUDA is used if available at fit time.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        panel_size: int,
        priority_scores: Sequence[float] | None = None,
        groups: Sequence[Sequence[int]] | None = None,
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.5,
        strict: bool = True,
        significance_threshold: float = 0.01,
        l1: float = 0.01,
        anneal_l1: bool = False,
        anneal_l1_from: float | None = None,
        anneal_l1_epochs: int | None = None,
        l2_decay: float = 0.01,
        hidden_dims: tuple[int, ...] = (32, 16),
        init: float = 0.5,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str | None = None,
        seed: int = 33,
    ) -> None:
        if panel_size < 1:
            raise ValueError("panel_size must be a positive integer.")
        _validate_anneal(anneal_l1, anneal_l1_from, anneal_l1_epochs, l1, epochs)

        self.panel_size = panel_size
        self.priority_scores = priority_scores
        self.groups = groups
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.strict = strict
        self.significance_threshold = significance_threshold
        self.l1 = l1
        self.anneal_l1 = anneal_l1
        self.anneal_l1_from = anneal_l1_from
        self.anneal_l1_epochs = anneal_l1_epochs
        self.l2_decay = l2_decay
        self.hidden_dims = tuple(hidden_dims)
        self.init = init
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

        self.gating_layer_: FeatureGatingLayer | None = None
        self.selected_indices_: list[int] | None = None
        self.feature_scores_: Array | None = None

    def __repr__(self) -> str:
        return (
            f"FeatureSelector(panel_size={self.panel_size!r}, "
            f"strict={self.strict!r}, alpha={self.alpha!r}, beta={self.beta!r}, "
            f"gamma={self.gamma!r}, l1={self.l1!r}, l2_decay={self.l2_decay!r}, "
            f"anneal_l1={self.anneal_l1!r}, anneal_l1_from={self.anneal_l1_from!r}, "
            f"anneal_l1_epochs={self.anneal_l1_epochs!r}, "
            f"hidden_dims={self.hidden_dims!r}, epochs={self.epochs!r}, "
            f"lr={self.lr!r}, batch_size={self.batch_size!r})"
        )

    def fit(self, X: Array, y: npt.ArrayLike) -> FeatureSelector:
        """Train the gating network and record the selected features.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Integer labels of shape ``(n_samples,)``.

        Returns:
            ``self``.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1-D array of labels.")
        n_features = X.shape[1]

        if self.panel_size > n_features:
            raise ValueError(f"panel_size {self.panel_size} exceeds n_features {n_features}")
        labels = np.unique(y)
        n_classes = int(labels.shape[0])
        encoded = np.searchsorted(labels, y)
        device = self._resolve_device()

        # Balance the classes in the task loss (matches scGIST's balanced
        # class weighting) so minority classes still influence which
        # features get selected.
        class_weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=encoded)
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        torch.manual_seed(self.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
        torch_generator = torch.Generator().manual_seed(self.seed)

        model = _MaskedMLP(n_features, n_classes, self.hidden_dims, self.init).to(device)
        loss_fn = self._make_loss(model.gating)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        train_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(encoded))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, generator=torch_generator)

        model.train()
        for epoch in range(self.epochs):
            loss_fn.l1 = self._l1_for_epoch(epoch)
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                task_loss = nn.functional.cross_entropy(logits, yb, weight=class_weight_tensor)
                total = loss_fn(task_loss) + 0.5 * self.l2_decay * model.l2_penalty()
                total.backward()  # type: ignore[no-untyped-call]  # torch 2.13 stubs omit backward
                optimizer.step()

        self.gating_layer_ = model.gating
        self.feature_scores_ = model.gating.score.detach().cpu().numpy()
        ranked = self._rank_features(self.feature_scores_)
        if self.strict:
            self.selected_indices_ = ranked[: self.panel_size]
        else:
            # Match scGIST's get_markers_indices: non-strict counts only features
            # with a significant absolute weight, then caps d to that count.
            significant = int((np.abs(self.feature_scores_) > self.significance_threshold).sum())
            self.selected_indices_ = ranked[: min(self.panel_size, significant)]
        return self

    def _l1_for_epoch(self, epoch: int) -> float:
        """Feature-selection strength for ``epoch``: fixed, or a geometric ramp to ``l1``."""
        if not self.anneal_l1:
            return self.l1
        start = self._anneal_start_l1()
        progress = min(epoch / self._anneal_epochs(), 1.0)
        return float(start * (self.l1 / start) ** progress)

    def _anneal_start_l1(self) -> float:
        if self.anneal_l1_from is not None:
            return self.anneal_l1_from
        return self.l1 * DEFAULT_ANNEAL_START_RATIO

    def _anneal_epochs(self) -> int:
        if self.anneal_l1_epochs is not None:
            return self.anneal_l1_epochs
        return max(1, round(self.epochs * DEFAULT_ANNEAL_EPOCH_FRACTION))

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _make_loss(self, gating: FeatureGatingLayer) -> FeatureSelectorLoss:
        return FeatureSelectorLoss(
            gating,
            panel_size=self.panel_size,
            priority_scores=self.priority_scores,
            groups=self.groups,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            strict=self.strict,
            l1=self.l1,
        )

    @staticmethod
    def _rank_features(scores: Array) -> list[int]:
        # Rank by absolute weight, matching scGIST's ``get_markers_indices``
        # (which sorts ``abs(weights)`` descending). A large-magnitude weight is
        # influential regardless of its sign, so it must not be dropped just
        # because it trained negative.
        return sorted(range(len(scores)), key=lambda i: abs(scores[i]), reverse=True)

    def get_selected_features(self) -> list[int]:
        """Return the indices of the selected features.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if self.selected_indices_ is None:
            raise RuntimeError("Call fit before get_selected_features.")
        return list(self.selected_indices_)

    def transform(self, X: Array) -> Array:
        """Return only the selected feature columns of ``X``.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if self.selected_indices_ is None:
            raise RuntimeError("Call fit before transform.")
        return np.asarray(X)[:, self.selected_indices_]

    def fit_transform(self, X: Array, y: npt.ArrayLike) -> Array:
        """Fit and return the selected feature columns of ``X``."""
        return self.fit(X, y).transform(X)


def _validate_anneal(
    anneal_l1: bool,
    anneal_l1_from: float | None,
    anneal_l1_epochs: int | None,
    l1: float,
    epochs: int,
) -> None:
    if not anneal_l1:
        if anneal_l1_from is not None or anneal_l1_epochs is not None:
            raise ValueError("anneal_l1_from/anneal_l1_epochs require anneal_l1=True.")
        return
    if l1 <= 0:
        raise ValueError("l1 must be positive when annealing is enabled.")
    if anneal_l1_from is not None and not 0 < anneal_l1_from < l1:
        raise ValueError(f"anneal_l1_from must be in (0, l1={l1}), got {anneal_l1_from}.")
    if anneal_l1_epochs is not None and not 1 <= anneal_l1_epochs <= epochs:
        raise ValueError(f"anneal_l1_epochs must be in [1, epochs={epochs}], got {anneal_l1_epochs}.")
