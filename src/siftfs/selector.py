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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from siftfs.layers import FeatureGatingLayer
from siftfs.loss import FeatureSelectorLoss

Array = npt.NDArray[np.floating[Any]]


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
        self._l2_weights: list[nn.Parameter] = [
            w for m in self.task for w in m.parameters() if w.ndim == 2
        ]

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
        l1: Overall scaling of the feature-selection regularization (scGIST: 0.01).
        l2_decay: L2 penalty coefficient on the task MLP weights (scGIST: 0.01).
        hidden_dims: Hidden layer widths of the task MLP.
        init: Initial value of each gate weight.
        epochs: Number of training epochs.
        lr: Adam learning rate.
        batch_size: Minibatch size.
        validation_split: Fraction of data held out for early stopping.
        early_stop: When ``True``, stops training once validation performance stops
            improving (patience) and restores the best checkpoint. When ``False``,
            trains exactly ``epochs`` epochs and keeps the final weights (matching
            scGIST's fixed-epoch behavior).
        patience: Epochs of no validation improvement before early stopping.
        device: Torch device; defaults to CUDA if available.
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
        l1: float = 0.01,
        l2_decay: float = 0.01,
        hidden_dims: tuple[int, ...] = (32, 16),
        init: float = 0.5,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
        validation_split: float = 0.2,
        early_stop: bool = True,
        patience: int = 10,
        device: str | None = None,
        seed: int = 33,
    ) -> None:
        if panel_size < 1:
            raise ValueError("panel_size must be a positive integer.")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1).")

        self.panel_size = panel_size
        self.priority_scores = priority_scores
        self.groups = groups
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.strict = strict
        self.l1 = l1
        self.l2_decay = l2_decay
        self.hidden_dims = tuple(hidden_dims)
        self.init = init
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stop = early_stop
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        self.gating_layer_: FeatureGatingLayer | None = None
        self.selected_indices_: list[int] | None = None
        self.feature_scores_: Array | None = None

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
            raise ValueError(
                f"panel_size {self.panel_size} exceeds n_features {n_features}"
            )
        labels = np.unique(y)
        n_classes = int(labels.shape[0])
        encoded = np.searchsorted(labels, y)

        X_train, X_val, y_train, y_val = train_test_split(
            X, encoded, test_size=self.validation_split,
            random_state=self.seed, stratify=encoded,
        )

        # Balance the classes in the task loss (matches scGIST's balanced
        # class weighting) so minority cell types still influence which
        # features get selected.
        class_weights = compute_class_weight(
            "balanced", classes=np.arange(n_classes), y=y_train
        )
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        torch.manual_seed(self.seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
        torch_generator = torch.Generator().manual_seed(self.seed)

        model = _MaskedMLP(n_features, n_classes, self.hidden_dims, self.init).to(self.device)
        loss_fn = self._make_loss(model.gating)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, generator=torch_generator)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_val = float("inf")
        best_state: dict[str, object] | None = None
        stall = 0

        for _ in range(self.epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                task_loss = nn.functional.cross_entropy(logits, yb, weight=class_weight_tensor)
                total = loss_fn(task_loss) + 0.5 * self.l2_decay * model.l2_penalty()
                total.backward()  # type: ignore[no-untyped-call]  # torch 2.13 stubs omit backward
                optimizer.step()

            if not self.early_stop:
                continue

            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = model(xb)
                    task_loss = nn.functional.cross_entropy(logits, yb, weight=class_weight_tensor)
                    combined = loss_fn(task_loss) + 0.5 * self.l2_decay * model.l2_penalty()
                    val_loss += combined.item() * xb.size(0)
            val_loss /= max(len(val_ds), 1)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= self.patience:
                    break

        if self.early_stop and best_state is not None:
            model.load_state_dict(best_state)

        self.gating_layer_ = model.gating
        self.feature_scores_ = model.gating.score.detach().cpu().numpy()
        self.selected_indices_ = self._rank_features(self.feature_scores_)[: self.panel_size]
        return self

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
        """Return only the selected feature columns of ``X``."""
        return np.asarray(X)[:, self.selected_indices_]

    def fit_transform(self, X: Array, y: npt.ArrayLike) -> Array:
        """Fit and return the selected feature columns of ``X``."""
        return self.fit(X, y).transform(X)
