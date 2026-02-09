import torch
from torch import Tensor, nn


class FeatureRenderer(nn.Module):
    """Just a weighted sum."""

    @classmethod
    def forward(
        cls,
        features: Tensor,
        weights: Tensor,
    ) -> Tensor:
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.sum(weights * features, dim=-2)


class ScalarRenderer(nn.Module):
    """Weighted sum for scalar or multi-channel fields."""

    @classmethod
    def forward(
        cls,
        values: Tensor,
        weights: Tensor,
    ) -> Tensor:
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.sum(weights * values, dim=-2)
