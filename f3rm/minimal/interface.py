from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from f3rm.minimal.homography import Homography
from nerfstudio.cameras.cameras import Cameras, CameraType
np.set_printoptions(precision=3, suppress=True)
os.environ["TORCHDYNAMO_DISABLE"] = "1"


class NERFinterface:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        _, pipeline, *_ = eval_setup(self.config_path)
        self.pipeline = pipeline
        self.model = pipeline.model
        self.device = self.model.device

    def get_custom_camera_outputs(self, fx: float, fy: float, width: int, height: int, c2w: np.ndarray,
                                  cx: float = None, cy: float = None, cam_type: CameraType = CameraType.PERSPECTIVE, dist_params: list = None,
                                  fars: float = None, nears: float = None) -> Cameras:
        """Get a custom camera's outputs with the given parameters. c2w is camera-to-nerf-world 3x4 matrix."""
        c2w = torch.tensor(c2w).cpu()
        if dist_params is None:
            dist_params = torch.zeros(6).cpu()
        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2
        custom_camera = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            camera_to_worlds=c2w,
            camera_type=cam_type,
            distortion_params=dist_params,
        ).to(self.device)
        ray_bundle = custom_camera.generate_rays(camera_indices=0).to(self.device)
        if fars is not None:
            ray_bundle.fars = torch.ones_like(ray_bundle.pixel_area) * fars
        if nears is not None:
            ray_bundle.nears = torch.ones_like(ray_bundle.pixel_area) * nears
        outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        return outputs

    def get_c2w_from_viewer(self, viewer_mat: list):
        """Get the camera-to-world matrix 3x4 from the viewer matrix."""
        c2w_viewer = np.array(viewer_mat).reshape(4, 4).T[:3, :4]
        c2w_viewer = torch.from_numpy(c2w_viewer).float().cpu()
        return c2w_viewer
