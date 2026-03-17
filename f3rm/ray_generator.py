from torch import Tensor, nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle


class FeatureRayGenerator(nn.Module):
    """Ray generator specialized for sampled per-ray indices."""

    image_coords: Tensor
    flat_image_coords: Tensor

    def __init__(self, cameras: Cameras) -> None:
        super().__init__()
        self.cameras = cameras
        image_coords = cameras.get_image_coords().to(cameras.device)
        self.register_buffer("image_coords", image_coords, persistent=False)
        self.register_buffer("flat_image_coords", image_coords.view(-1, 2), persistent=False)
        self.image_width = image_coords.shape[1]

    def forward(self, ray_indices: Tensor) -> RayBundle:
        camera_indices = ray_indices[:, :1].long()
        flat_indices = ray_indices[:, 1].long() * self.image_width + ray_indices[:, 2].long()
        coords = self.flat_image_coords.index_select(0, flat_indices)
        return self.cameras._generate_rays_from_coords(camera_indices, coords)
