import torch
from torch import Tensor, nn

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CameraType, Cameras
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
        self.use_precomputed = self._can_precompute_perspective()
        if self.use_precomputed:
            self._init_precomputed_perspective_rays()

    @staticmethod
    def _all_rows_equal(values: Tensor) -> bool:
        rows = values.reshape(-1, values.shape[-1])
        return torch.equal(rows, rows[:1].expand_as(rows))

    def _can_precompute_perspective(self) -> bool:
        return (
            len(self.cameras.shape) == 1
            and not self.cameras.is_jagged
            and torch.all(self.cameras.camera_type == CameraType.PERSPECTIVE.value)
            and self._all_rows_equal(self.cameras.fx)
            and self._all_rows_equal(self.cameras.fy)
            and self._all_rows_equal(self.cameras.cx)
            and self._all_rows_equal(self.cameras.cy)
            and (
                self.cameras.distortion_params is None
                or self._all_rows_equal(self.cameras.distortion_params)
            )
        )

    def _init_precomputed_perspective_rays(self) -> None:
        y = self.flat_image_coords[:, 0]
        x = self.flat_image_coords[:, 1]
        fx = self.cameras.fx.reshape(-1, 1)[0, 0]
        fy = self.cameras.fy.reshape(-1, 1)[0, 0]
        cx = self.cameras.cx.reshape(-1, 1)[0, 0]
        cy = self.cameras.cy.reshape(-1, 1)[0, 0]

        coord = torch.stack([(x - cx) / fx, (y - cy) / fy], dim=-1)
        coord_x_offset = torch.stack([(x - cx + 1) / fx, (y - cy) / fy], dim=-1)
        coord_y_offset = torch.stack([(x - cx) / fx, (y - cy + 1) / fy], dim=-1)
        coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)

        if self.cameras.distortion_params is not None:
            distortion = self.cameras.distortion_params.reshape(-1, self.cameras.distortion_params.shape[-1])[0]
            if distortion.any():
                distortion = distortion.unsqueeze(0).expand(coord_stack.shape[0] * coord_stack.shape[1], -1)
                coord_stack = camera_utils.radial_and_tangential_undistort(
                    coord_stack.reshape(-1, 2), distortion
                ).view_as(coord_stack)

        coord_stack[..., 1] *= -1
        directions_stack = torch.empty((3, coord_stack.shape[1], 3), device=coord_stack.device)
        directions_stack[..., 0] = coord_stack[..., 0]
        directions_stack[..., 1] = coord_stack[..., 1]
        directions_stack[..., 2] = -1.0

        directions_stack, directions_norm = camera_utils.normalize_with_norm(directions_stack, -1)
        dx = torch.sqrt(torch.sum((directions_stack[0] - directions_stack[1]) ** 2, dim=-1, keepdim=True))
        dy = torch.sqrt(torch.sum((directions_stack[0] - directions_stack[2]) ** 2, dim=-1, keepdim=True))

        self.register_buffer("precomputed_directions", directions_stack[0], persistent=False)
        self.register_buffer("precomputed_pixel_area", dx * dy, persistent=False)
        self.register_buffer("precomputed_directions_norm", directions_norm[0].detach(), persistent=False)

    def _forward_precomputed(self, camera_indices: Tensor, flat_indices: Tensor) -> RayBundle:
        camera_indices = camera_indices.to(self.cameras.device)
        flat_indices = flat_indices.to(self.precomputed_directions.device)
        camera_ids = camera_indices[:, 0]
        c2w = self.cameras.camera_to_worlds[camera_ids]
        rotation = c2w[:, :3, :3]
        origins = c2w[:, :3, 3]
        directions = self.precomputed_directions.index_select(0, flat_indices)
        directions = torch.sum(directions[:, None, :] * rotation, dim=-1)
        pixel_area = self.precomputed_pixel_area.index_select(0, flat_indices)
        directions_norm = self.precomputed_directions_norm.index_select(0, flat_indices)

        metadata = (
            self.cameras._apply_fn_to_dict(self.cameras.metadata, lambda x: x[camera_ids])
            if self.cameras.metadata is not None
            else {}
        )
        metadata["directions_norm"] = directions_norm
        times = self.cameras.times[camera_ids, 0:1] if self.cameras.times is not None else None
        return RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            times=times,
            metadata=metadata,
        )

    def forward(self, ray_indices: Tensor) -> RayBundle:
        camera_indices = ray_indices[:, :1].to(device=self.cameras.device, dtype=torch.long)
        flat_indices = ray_indices[:, 1].long() * self.image_width + ray_indices[:, 2].long()
        if self.use_precomputed:
            return self._forward_precomputed(camera_indices, flat_indices)
        coords = self.flat_image_coords.index_select(0, flat_indices.to(self.flat_image_coords.device))
        return self.cameras._generate_rays_from_coords(camera_indices, coords)
