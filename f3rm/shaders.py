import torch
from torch import nn, Tensor
from jaxtyping import Float
from nerfstudio.field_components.spatial_distortions import SpatialDistortion


class CentroidShader(nn.Module):
    """Map world-space centroid coordinates to RGB in [0,1] for visualization.

    Applies scene contraction, normalizes to [0,1], clamps, and sanitizes NaNs/Infs.
    Optionally masks out invalid pixels.
    """

    def __init__(self, spatial_distortion: SpatialDistortion) -> None:
        super().__init__()
        self.spatial_distortion = spatial_distortion

    def forward(
        self,
        centroids_world: Float[Tensor, "*bs 3"],
        valid_mask: Float[Tensor, "*bs 1"] | None = None,
    ) -> Float[Tensor, "*bs 3"]:
        contracted = self.spatial_distortion(centroids_world)
        vis = (contracted + 2.0) / 4.0
        vis = torch.clamp(vis, 0.0, 1.0)
        vis = torch.nan_to_num(vis, nan=0.0, posinf=0.0, neginf=0.0)
        if valid_mask is not None:
            # broadcast-last-dim
            vis = vis * valid_mask.to(vis.device)
        return vis
