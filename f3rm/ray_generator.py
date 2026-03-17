from torch import Tensor, nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle


class FeatureRayGenerator(nn.Module):
    """Ray generator specialized for sampled per-ray indices."""

    image_coords: Tensor

    def __init__(self, cameras: Cameras) -> None:
        super().__init__()
        self.cameras = cameras
        self.register_buffer("image_coords", cameras.get_image_coords().to(cameras.device), persistent=False)

    def forward(self, ray_indices: Tensor) -> RayBundle:
        camera_indices = ray_indices[:, :1].long()
        coords = self.image_coords[ray_indices[:, 1].long(), ray_indices[:, 2].long()]
        return self.cameras._generate_rays_from_coords(camera_indices, coords)
