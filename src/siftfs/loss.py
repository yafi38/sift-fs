"""Feature-selection loss terms.

Ports scGIST's ``FeatureRegularizer`` to PyTorch. The composite loss balances

    * the task objective (e.g., cross-entropy for classification),
    * an exact feature-count penalty enforcing the panel size ``d``,
    * an optional priority term that drives the model to include specific features,
    * an optional group term that keeps or drops whole physical groups together.

All penalties operate on the gating layer's continuous ``score`` (its raw weights).
The signs of the penalty coefficients are folded in so users pass ``lambda_``,
``alpha``, ``beta``, ``gamma`` as the *strength* of each term.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from siftfs.layers import FeatureGatingLayer


class FeatureSelectorLoss:
    """Assembles the multi-term feature-selection loss.

    Args:
        gating_layer: The trained ``FeatureGatingLayer`` whose weights are
            regularized.
        panel_size: Exact target number of selected features ``d``. When ``None``,
            the feature-count term is disabled.
        priority_scores: Per-feature priority ``s_i`` in ``[0, 1]``. Higher is more
            strongly pushed toward inclusion. ``None`` disables the priority term.
        groups: Sequence of index groups. All features in a group are driven to be
            kept or dropped together. ``None`` disables the group term.
        alpha: Strength of the exact feature-count penalty.
        beta: Strength of the priority penalty.
        gamma: Strength of the group penalty.
        strict: When ``True``, the count term penalizes any deviation from ``d``
            (exact budget). When ``False``, it only penalizes exceeding ``d``.
        l1: Overall scaling factor applied to the entire feature-selection
            regularization (scGIST uses ``0.01``).
    """

    def __init__(
        self,
        gating_layer: FeatureGatingLayer,
        panel_size: int | None = None,
        priority_scores: Sequence[float] | None = None,
        groups: Sequence[Sequence[int]] | None = None,
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.5,
        strict: bool = True,
        l1: float = 0.01,
    ) -> None:
        self.gating_layer = gating_layer
        self.panel_size = panel_size
        self.strict = strict
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.l1 = float(l1)

        n = gating_layer.n_features
        self.s: Tensor | None = None
        if priority_scores is not None:
            if len(priority_scores) != n:
                raise ValueError(
                    f"priority_scores length {len(priority_scores)} != n_features {n}"
                )
            self.s = torch.tensor(priority_scores, dtype=torch.float32)

        self.groups = groups

    def __call__(self, task_loss: Tensor, *, output: Tensor | None = None) -> Tensor:
        """Combine a task loss with the feature-selection penalties.

        Args:
            task_loss: The scalar task loss (e.g., cross-entropy) to keep.
            output: Pass-through slot for future auxiliary outputs. Currently
                unused; retained for API stability.

        Returns:
            The combined scalar loss.
        """
        del output  # reserved
        return task_loss + self._regularization()

    def _regularization(self) -> Tensor:
        x = self.gating_layer.score
        abs_x = torch.abs(x)
        reg = torch.zeros((), dtype=x.dtype, device=x.device)
        # Force weights toward either 0 or 1.
        reg = reg + torch.sum(abs_x * torch.abs(x - 1))

        # Enforce the feature budget (exact d, or at-most d).
        if self.panel_size is not None:
            count = torch.sum(abs_x)
            if self.strict:
                reg = reg + torch.abs(count - self.panel_size) * self.alpha
            else:
                reg = reg + torch.clamp_min(count - self.panel_size, 0.0) * self.alpha

        # Reward including prioritized genes.
        if self.s is not None:
            one = torch.ones_like(x)
            reg = reg + torch.sum((one - torch.minimum(x, one)) * self.s) * self.beta

        # Keep group members together: sum of pairwise absolute differences within
        # each group. Minimizing this drives all members of a group to the same
        # gate, so a physical sensor is kept or dropped as a unit.
        if self.groups:
            pair_sum = torch.zeros((), dtype=x.dtype, device=x.device)
            for group in self.groups:
                members = abs_x[list(group)]
                pair_sum = pair_sum + torch.sum(
                    torch.abs(members.unsqueeze(1) - members.unsqueeze(0))
                ) / 2
            reg = reg + pair_sum * self.gamma

        return self.l1 * reg
