import torch
from jaxtyping import Float
from torch import Tensor, nn


class FeatureRenderer(nn.Module):
    """Just a weighted sum."""

    @classmethod
    def forward(
        cls,
        features: Float[Tensor, "*bs num_samples num_channels"],
        weights: Float[Tensor, "*bs num_samples 1"],
    ) -> Float[Tensor, "*bs num_channels"]:
        output = torch.sum(weights * features, dim=-2)
        return output


class CentroidRenderer(nn.Module):
    """Weighted sum for 3D centroid-offset or centroid vectors."""

    @classmethod
    def forward(
        cls,
        values: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
    ) -> Float[Tensor, "*bs 3"]:
        return torch.sum(weights * values, dim=-2)
