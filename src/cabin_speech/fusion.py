from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MultimodalFusionNet(nn.Module):
    """Compact late-fusion baseline for cabin active-speaker classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (96, 48),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError("invalid model dimensions")

        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, int(hidden)),
                    nn.LayerNorm(int(hidden)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = int(hidden)
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, feature_dim]")
        return self.network(features)
