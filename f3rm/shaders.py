import torch
from torch import nn, Tensor
from jaxtyping import Float
from nerfstudio.field_components.spatial_distortions import SpatialDistortion


class CentroidShader(nn.Module):
    """Map world-space centroids to RGB in [0,1] with enhanced local contrast.

    We apply scene contraction, then per-batch min-max normalization across the
    current set of centroids to spread colors and improve discernibility when
    points lie in a small region. NaNs/Infs are sanitized and an optional
    valid_mask can zero-out invalid pixels.
    """

    def __init__(self, spatial_distortion: SpatialDistortion) -> None:
        super().__init__()
        self.spatial_distortion = spatial_distortion

    def contract(self, centroids_world: Tensor) -> Tensor:
        contracted = self.spatial_distortion(centroids_world)
        return torch.nan_to_num(contracted, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        centroids_world: Float[Tensor, "*bs 3"],
        valid_mask: Float[Tensor, "*bs 1"] | None = None,
        global_min: Tensor | None = None,
        global_max: Tensor | None = None,
    ) -> Float[Tensor, "*bs 3"]:
        contracted = self.contract(centroids_world)
        # vis = (contracted + 2.0) / 4.0  (legacy, if-else below replaces this)
        if global_min is not None and global_max is not None:
            # Use supplied global range for consistent coloring across aggregated sets
            span = (global_max - global_min).clamp_min(1e-6)
            vis = (contracted - global_min) / span
        else:
            # Fallback to per-batch min-max to maximize local contrast
            reduce_dims = tuple(range(contracted.ndim - 1))  # all spatial dims
            min_v = contracted.amin(dim=reduce_dims, keepdim=True)
            max_v = contracted.amax(dim=reduce_dims, keepdim=True)
            span = (max_v - min_v).clamp_min(1e-6)
            vis = (contracted - min_v) / span
        vis = torch.clamp(vis, 0.0, 1.0)

        if valid_mask is not None:
            vis = vis * valid_mask.to(vis.device)
        return vis


class ScalarShader(nn.Module):
    """Map non-negative scalar field (e.g., L2 error) to grayscale.

    Centroids are contracted to [0,1]^3, so the max L2 distance is sqrt(3).
    We clamp to [0, sqrt(3)] and scale to [0,1].
    """

    def __init__(self, max_value: float = 3.0 ** 0.5):
        super().__init__()
        self.max_value = float(max_value)

    def forward(self, scalar: Float[Tensor, "*bs 1"], valid_mask: Float[Tensor, "*bs 1"] | None = None) -> Float[Tensor, "*bs 3"]:
        # Sanitize
        x = torch.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0).squeeze(-1)
        # Clamp to [0, max] and scale to [0,1]
        x = torch.clamp(x, 0.0, self.max_value) / self.max_value
        rgb = x.unsqueeze(-1).expand(*x.shape, 3)
        if valid_mask is not None:
            rgb = rgb * valid_mask.to(rgb.device)
        return rgb


class ProbShader(nn.Module):
    """Visualize logits as grayscale probability via sigmoid in [0,1]."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: Float[Tensor, "*bs 1"], valid_mask: Float[Tensor, "*bs 1"] | None = None) -> Float[Tensor, "*bs 3"]:
        x = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.sigmoid(x)
        rgb = probs.expand(*probs.shape[:-1], 3)
        if valid_mask is not None:
            rgb = rgb * valid_mask.to(rgb.device)
        return rgb


class ProbFromProbsShader(nn.Module):
    """Visualize probabilities (already in [0,1]) as grayscale without extra sigmoid."""

    def __init__(self):
        super().__init__()

    def forward(self, probs: Float[Tensor, "*bs 1"], valid_mask: Float[Tensor, "*bs 1"] | None = None) -> Float[Tensor, "*bs 3"]:
        p = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        p = torch.clamp(p, 0.0, 1.0)
        rgb = p.expand(*p.shape[:-1], 3)
        if valid_mask is not None:
            rgb = rgb * valid_mask.to(rgb.device)
        return rgb
