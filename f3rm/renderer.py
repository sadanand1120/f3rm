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
        # Handle non-finite values that can occur with gradient flow instability
        if not torch.isfinite(features).all():
            features = torch.where(torch.isfinite(features), features, torch.zeros_like(features))
        output = torch.sum(weights * features, dim=-2)
        return output


class CentroidRenderer(nn.Module):
    """Weighted sum for 3D centroid vectors."""

    @classmethod
    def forward(
        cls,
        values: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
    ) -> Float[Tensor, "*bs 3"]:
        # Handle non-finite values that can occur with gradient flow instability
        if not torch.isfinite(values).all():
            values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        return torch.sum(weights * values, dim=-2)


class ScalarRenderer(nn.Module):
    """Weighted sum for scalar or multi-channel fields (e.g., centroid spread)."""

    @classmethod
    def forward(
        cls,
        values: Float[Tensor, "*bs num_samples c"],
        weights: Float[Tensor, "*bs num_samples 1"],
    ) -> Float[Tensor, "*bs c"]:
        # Handle non-finite values that can occur with gradient flow instability
        if not torch.isfinite(values).all():
            values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        return torch.sum(weights * values, dim=-2)


class ClassProbRenderer(nn.Module):
    """Render per-sample class logits into per-ray class probabilities.

    For each sample along a ray, convert logits -> softmax probabilities, then
    integrate along the ray with transmittance weights to get per-ray class probs.
    """

    @classmethod
    def forward(
        cls,
        values: Float[Tensor, "*bs num_samples c"],
        weights: Float[Tensor, "*bs num_samples 1"],
    ) -> Float[Tensor, "*bs c"]:
        # Handle non-finite values that can occur with gradient flow instability
        if not torch.isfinite(values).all():
            values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        probs = torch.softmax(values, dim=-1)
        return torch.sum(weights * probs, dim=-2)


class VectorRenderer(nn.Module):
    """Render unit vectors along the ray, similar to NormalsRenderer."""

    @classmethod
    def forward(
        cls,
        vectors: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        normalize: bool = True,
    ) -> Float[Tensor, "*bs 3"]:
        """Calculate vectors along the ray.

        Args:
            vectors: Unit vectors for each sample.
            weights: Weights of each sample.
            normalize: Normalize final vectors.
        """
        v = torch.sum(weights * vectors, dim=-2)
        if normalize:
            from nerfstudio.utils.math import safe_normalize
            v = safe_normalize(v)
        return v
