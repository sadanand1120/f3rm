from nerfstudio.utils.eval_utils import eval_load_checkpoint, TrainerConfig, Pipeline, VanillaDataManagerConfig, all_methods
import yaml
from pathlib import Path
import os
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
from typing import Tuple, Optional, Literal
import torch
import numpy as np
from f3rm.minimal.homography import Homography
from nerfstudio.cameras.cameras import Cameras, CameraType
np.set_printoptions(precision=3, suppress=True)
os.environ["TORCHDYNAMO_DISABLE"] = "1"


class NERFinterface:
    def __init__(self, config_path: str, device: Optional[torch.device] = None):
        self.config_path = Path(config_path)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config, pipeline, *_ = self.eval_setup(config_path=self.config_path, test_mode="inference")
        self.config = config
        self.pipeline = pipeline
        self.model = pipeline.model

    @torch.inference_mode()
    def eval_setup(self, config_path: Path, eval_num_rays_per_chunk: Optional[int] = None, test_mode: Literal["test", "val", "inference"] = "test") -> Tuple[TrainerConfig, Pipeline, Path, int]:
        """Shared setup for loading a saved pipeline for evaluation. Modified from nerfstudio.utils.eval_utils.eval_setup.
        Args:
            config_path: Path to config YAML file.
            eval_num_rays_per_chunk: Number of rays per forward pass
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test dataset into memory
                'inference': does not load any dataset into memory
        Returns:
            Loaded config, pipeline module, corresponding checkpoint, and step
        """
        # load save config
        config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
        assert isinstance(config, TrainerConfig)
        config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
        if eval_num_rays_per_chunk:
            config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk
        config.load_dir = config.get_checkpoint_dir()
        if isinstance(config.pipeline.datamanager, VanillaDataManagerConfig):
            config.pipeline.datamanager.eval_image_indices = None

        # setup pipeline (which includes the DataManager)
        pipeline = config.pipeline.setup(device=self.device, test_mode=test_mode)
        assert isinstance(pipeline, Pipeline)
        pipeline.eval()

        # load checkpointed information
        checkpoint_path, step = eval_load_checkpoint(config, pipeline)
        return config, pipeline, checkpoint_path, step

    @staticmethod
    def get_nerf_ccs_to_normal_ccs_T(device=None):
        """Get the transformation matrix from NeRF CCS to normal CCS."""
        T = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        if device is not None:
            return torch.from_numpy(T).to(device)
        return T

    @staticmethod
    def get_cam_intrinsics_nodist(fx: float, fy: float, width: int, height: int):
        """Get camera intrinsics matrix without distortion.
        Returns a 3x3 matrix
        """
        return np.array([
            [fx, 0, width / 2],
            [0, fy, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)

    @torch.inference_mode()
    def get_custom_camera_outputs(self, fx: float, fy: float, width: int, height: int, c2w: np.ndarray,
                                  cx: Optional[float] = None, cy: Optional[float] = None, cam_type: CameraType = CameraType.PERSPECTIVE, dist_params: Optional[list] = None,
                                  fars: Optional[float] = None, nears: Optional[float] = None, return_rays: bool = False) -> Cameras:
        """Get a custom camera's outputs with the given parameters. c2w is camera-to-nerf-world 3x4 matrix."""
        c2w = torch.as_tensor(c2w, device=self.device)
        dist_params = torch.zeros(6, device=self.device) if dist_params is None else torch.as_tensor(dist_params, device=self.device)
        cx = width / 2 if cx is None else cx
        cy = height / 2 if cy is None else cy
        custom_camera = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            camera_to_worlds=c2w.cpu(),   # it needs cpu
            camera_type=cam_type,
            distortion_params=dist_params.cpu(),   # it needs cpu
        ).to(self.device)
        ray_bundle = custom_camera.generate_rays(camera_indices=0).to(self.device)
        ray_bundle.fars = torch.ones_like(ray_bundle.pixel_area) * fars if fars is not None else None
        ray_bundle.nears = torch.ones_like(ray_bundle.pixel_area) * nears if nears is not None else None
        outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        outputs = {k: v.to(self.device) for k, v in outputs.items()}
        return (outputs, ray_bundle) if return_rays else outputs

    @staticmethod
    def get_c2w_from_viewer(viewer_mat: list, device=None) -> np.ndarray:
        """Get the camera-to-world matrix 3x4 from the viewer matrix."""
        if device is not None:
            viewer_tensor = torch.tensor(viewer_mat, dtype=torch.float32, device=device)
            return viewer_tensor.reshape(4, 4).T[:3, :4]
        return np.array(viewer_mat, dtype=np.float32).reshape(4, 4).T[:3, :4]

    @torch.inference_mode()
    def pointcloud_from_camera(
        self,
        fx: float,
        fy: float,
        width: int,
        height: int,
        c2w: np.ndarray,               # 3x4 or 4x4 camera-to-NeRF
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        cam_type: CameraType = CameraType.PERSPECTIVE,
        dist_params: Optional[list] = None,
        fars: Optional[float] = None,
        nears: Optional[float] = None,
        nerf_to_world: Optional[np.ndarray] = None,  # 4x4; pass if you want world coords
        bbox_min: Tuple[float, float, float] = (-1, -1, -1),
        bbox_max: Tuple[float, float, float] = (1, 1, 1),
        use_bbox: bool = False,
        save_feature_pca: bool = False,
    ) -> o3d.geometry.PointCloud:
        """
        Render once from a custom camera and return the visible surface as an Open3D point cloud.
        Coordinates are in the NeRF frame unless `nerf_to_world` is provided.
        """
        # 1. render
        outputs, ray_bundle = self.get_custom_camera_outputs(fx, fy, width, height, c2w, cx, cy, cam_type,
                                                             dist_params, fars, nears, return_rays=True)
        depth = outputs["depth"].squeeze(-1)        # (H,W)
        if save_feature_pca:
            rgb = outputs["feature_pca"]                      # (H,W,3)
        else:
            rgb = outputs["rgb"]                      # (H,W,3)

        # 2. lift depth to 3-D points in NeRF space
        origins = ray_bundle.origins.squeeze(0)      # (H,W,3)
        directions = ray_bundle.directions.squeeze(0)   # (H,W,3)
        pts_nerf = origins + directions * depth.unsqueeze(-1)

        # 3. (optional) bbox filter
        if use_bbox:
            comp_l = torch.as_tensor(bbox_min, device=self.device)
            comp_m = torch.as_tensor(bbox_max, device=self.device)
            keep = torch.all(torch.cat([pts_nerf > comp_l,
                                        pts_nerf < comp_m], -1), -1)
            pts_nerf = pts_nerf[keep]
            rgb = rgb[keep]

        # 4. (optional) convert to world
        if nerf_to_world is not None:
            R = torch.as_tensor(nerf_to_world[:3, :3], device=self.device)
            t = torch.as_tensor(nerf_to_world[:3, 3], device=self.device)
            pts = (R @ pts_nerf.T).T + t
        else:
            pts = pts_nerf

        pts = pts.reshape(-1, 3).cpu().numpy().astype(np.float64)   # (N,3)
        rgb = rgb.reshape(-1, 3).cpu().numpy().astype(np.float64)   # (N,3)

        # 5. build Open3D cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        return pcd
