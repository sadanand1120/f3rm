import torch
from torch import nn, Tensor


class ProbFromProbsShader(nn.Module):
    """Visualize probabilities (already in [0,1]) as grayscale without extra sigmoid."""

    def __init__(self):
        super().__init__()

    def forward(self, probs: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        p = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        p = torch.clamp(p, 0.0, 1.0)
        rgb = p.expand(*p.shape[:-1], 3)
        if valid_mask is not None:
            rgb = rgb * valid_mask.to(rgb.device)
        return rgb.to(torch.float16)


class VectorShader(nn.Module):
    """Visualize unit vectors similar to NormalsShader."""

    def __init__(self):
        super().__init__()

    def forward(self, vectors: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        """Map vectors from [-1, 1] to RGB in [0, 1]."""
        rgb = torch.clamp((vectors + 1.0) / 2.0, 0.0, 1.0)

        if valid_mask is not None:
            rgb = rgb * valid_mask.to(rgb.device)

        return rgb.to(torch.float16)


class SceneBoxCoordinateShader(nn.Module):
    """Map scene coordinates into RGB using the scene box bounds."""

    def __init__(self):
        super().__init__()

    def forward(self, coordinates: Tensor, aabb: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        aabb = aabb.to(device=coordinates.device, dtype=coordinates.dtype)
        coord_min = aabb[0]
        coord_max = aabb[1]
        rgb = (coordinates - coord_min) / (coord_max - coord_min)
        rgb = torch.clamp(torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        if valid_mask is not None:
            rgb = rgb * valid_mask.to(device=rgb.device, dtype=rgb.dtype)
        return rgb.to(torch.float16)
