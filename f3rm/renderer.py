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
