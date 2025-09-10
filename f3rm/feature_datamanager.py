import gc
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Optional, List

import numpy as np
from pathlib import Path
import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from f3rm.features.extract_features_standalone import extract_features_for_dataset
from f3rm.features.utils import BatchFeatureLoader


@dataclass
class FeatureDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: FeatureDataManager)
    feature_type: Literal["CLIP", "DINO"] = "CLIP"
    """Feature type to extract."""
    sam2_feature_type: str = "SAM2"
    """SAM2 feature type for centroid supervision (SAM2, CLIPSAM_book, CLIPSAM_, etc.)."""
    foreground_feature_type: str = "FOREGROUND_"
    orientany_feature_type: str = "ORIENTANY_"
    """OrientAny feature type for separate orientation head (ORIENTANY_*, e.g., ORIENTANY_book)."""
    enable_cache: bool = True
    """Whether to cache extracted features."""
    cpu_feature_cache_images: int = 128
    gpu_feature_cache_images: int = 16


class FeatureDataManager(VanillaDataManager):
    config: FeatureDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        self.eval_offset = len(self.train_dataset)
        self.feature_loader = extract_features_for_dataset(
            image_fnames=image_fnames,
            data_dir=self.config.dataparser.data,
            feature_type=self.config.feature_type,
            device=self.device,
            batch_size=128,
            enable_cache=self.config.enable_cache,
            force=False,
            max_cpu_images=self.config.cpu_feature_cache_images,
            max_gpu_images=self.config.gpu_feature_cache_images,
        )
        CONSOLE.print(f"Created batch loader for {self.config.feature_type} features")
        self.sam2_loader = extract_features_for_dataset(
            image_fnames=image_fnames,
            data_dir=self.config.dataparser.data,
            feature_type=self.config.sam2_feature_type,
            device=self.device,
            batch_size=128,
            enable_cache=self.config.enable_cache,
            force=False,
            max_cpu_images=self.config.cpu_feature_cache_images,
            max_gpu_images=self.config.gpu_feature_cache_images,
        )
        CONSOLE.print(f"Created batch loader for {self.config.sam2_feature_type} auto-masks")
        self.fg_loader = extract_features_for_dataset(
            image_fnames=image_fnames,
            data_dir=self.config.dataparser.data,
            feature_type=self.config.foreground_feature_type,
            device=self.device,
            batch_size=128,
            enable_cache=self.config.enable_cache,
            force=False,
            max_cpu_images=self.config.cpu_feature_cache_images,
            max_gpu_images=self.config.gpu_feature_cache_images,
        )
        CONSOLE.print(f"Created batch loader for {self.config.foreground_feature_type} maps")
        self.orientany_loader = extract_features_for_dataset(
            image_fnames=image_fnames,
            data_dir=self.config.dataparser.data,
            feature_type=self.config.orientany_feature_type,
            device=self.device,
            batch_size=128,
            enable_cache=self.config.enable_cache,
            force=False,
            max_cpu_images=self.config.cpu_feature_cache_images,
            max_gpu_images=self.config.gpu_feature_cache_images,
        )
        CONSOLE.print(f"Created batch loader for {self.config.orientany_feature_type} maps")
        feat_dim = self.feature_loader.C
        self.train_dataset.metadata["feature_type"] = self.config.feature_type
        self.train_dataset.metadata["feature_dim"] = feat_dim
        self.train_dataset.metadata["has_foreground"] = True
        self.train_dataset.metadata["has_orientany"] = True
        feat_h, feat_w = self.feature_loader.H, self.feature_loader.W
        im_h = set(self.train_dataset.cameras.image_height.squeeze().tolist())
        im_w = set(self.train_dataset.cameras.image_width.squeeze().tolist())
        assert len(im_h) == 1, "All images must have the same height"
        assert len(im_w) == 1, "All images must have the same width"
        im_h, im_w = im_h.pop(), im_w.pop()
        self.feat_scale_h = feat_h / im_h
        self.feat_scale_w = feat_w / im_w
        fg_h, fg_w = self.fg_loader.H, self.fg_loader.W
        self.fg_scale_h = fg_h / im_h
        self.fg_scale_w = fg_w / im_w
        orientany_h, orientany_w = self.orientany_loader.H, self.orientany_loader.W
        self.orientany_scale_h = orientany_h / im_h
        self.orientany_scale_w = orientany_w / im_w
        CONSOLE.print(f"Feat h: {feat_h}, Feat w: {feat_w}, Feat c: {feat_dim}, Im h: {im_h}, Im w: {im_w}")
        CONSOLE.print(f"Feat scale h: {self.feat_scale_h}, Feat scale w: {self.feat_scale_w}")
        CONSOLE.print(f"FG scale h: {self.fg_scale_h}, FG scale w: {self.fg_scale_w}")
        CONSOLE.print(f"OrientAny scale h: {self.orientany_scale_h}, OrientAny scale w: {self.orientany_scale_w}")
        self.current_batch_features = {}
        self.current_batch_fg = {}
        self.current_batch_sam2 = {}
        self.current_batch_orientany = {}
        torch.cuda.empty_cache()
        gc.collect()

    def _clear_batch_features(self):
        """Clear current batch features to prevent VRAM accumulation."""
        self.current_batch_features.clear()
        self.current_batch_fg.clear()
        self.current_batch_sam2.clear()
        self.current_batch_orientany.clear()
        # torch.cuda.empty_cache()

    def _load_batch_features(self, camera_indices: torch.Tensor):
        """Load features for unique cameras in the current batch."""
        unique_cameras = torch.unique(camera_indices)
        self.current_batch_features = self.feature_loader.load_batch_images(unique_cameras)
        self.current_batch_fg = self.fg_loader.load_batch_images(unique_cameras)
        self.current_batch_sam2 = self.sam2_loader.load_batch_images(unique_cameras)
        self.current_batch_orientany = self.orientany_loader.load_batch_images(unique_cameras)

    def _gather_feats_from_batch(self, batch_features: dict, camera_idx: torch.Tensor, y_idx: torch.Tensor, x_idx: torch.Tensor) -> torch.Tensor:
        """Gather features from batch-loaded feature dictionary. Vectorized for tensor features to avoid per-ray Python loops.
        """
        batch_size = len(camera_idx)
        if len(batch_features) == 0:
            return torch.zeros(batch_size, 0, device=self.device, dtype=torch.float16)
        first_cam = next(iter(batch_features.keys()))
        first_features = batch_features[first_cam]
        if isinstance(first_features, torch.Tensor):
            # Regular tensor features (CLIP, DINO, FOREGROUND, ORIENTANY)
            # Build a stacked tensor for the unique cameras in the batch and index in one shot.
            unique_cams, inverse = torch.unique(camera_idx, sorted=True, return_inverse=True)
            stacked = [batch_features[int(cam.item())] for cam in unique_cams]
            features_u = torch.stack(stacked, dim=0)  # [U, H, W, C]
            output = features_u[inverse, y_idx, x_idx, :]
        elif isinstance(first_features, list):
            # SAM2/CLIPSAM masks - return as list for now
            # This would need special handling in the model
            output = []
            for i, (ci, yi, xi) in enumerate(zip(camera_idx, y_idx, x_idx)):
                cam_idx_int = int(ci.item())
                if cam_idx_int in batch_features:
                    output.append(batch_features[cam_idx_int])
                else:
                    output.append([])
            return output
        else:
            # TEXT features - return as list
            output = []
            for i, (ci, yi, xi) in enumerate(zip(camera_idx, y_idx, x_idx)):
                cam_idx_int = int(ci.item())
                if cam_idx_int in batch_features:
                    output.append(batch_features[cam_idx_int])
                else:
                    output.append([])
            return output
        return output

    def _index_triplet(self, batch, scale_h, scale_w):
        ray_indices = batch["indices"]
        camera_idx = ray_indices[:, 0]
        y_idx = (ray_indices[:, 1] * scale_h).long()
        x_idx = (ray_indices[:, 2] * scale_w).long()
        return camera_idx, y_idx, x_idx

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)
        batch["image"] = batch["image"].to(self.device, non_blocking=True)
        self._clear_batch_features()
        camera_idx, y_feat, x_feat = self._index_triplet(batch, self.feat_scale_h, self.feat_scale_w)
        self._load_batch_features(camera_idx)
        batch["feature"] = self._gather_feats_from_batch(self.current_batch_features, camera_idx, y_feat, x_feat)
        cam_fg, y_fg, x_fg = self._index_triplet(batch, self.fg_scale_h, self.fg_scale_w)
        batch["foreground"] = self._gather_feats_from_batch(self.current_batch_fg, cam_fg, y_fg, x_fg)
        cam_orientany, y_orientany, x_orientany = self._index_triplet(batch, self.orientany_scale_h, self.orientany_scale_w)
        batch["orientany"] = self._gather_feats_from_batch(self.current_batch_orientany, cam_orientany, y_orientany, x_orientany)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_eval(step)
        batch["image"] = batch["image"].to(self.device, non_blocking=True)
        self._clear_batch_features()
        camera_idx, y_feat, x_feat = self._index_triplet(batch, self.feat_scale_h, self.feat_scale_w)
        camera_idx_global = camera_idx + self.eval_offset
        self._load_batch_features(camera_idx_global)
        batch["feature"] = self._gather_feats_from_batch(self.current_batch_features, camera_idx_global, y_feat, x_feat)
        cam_fg, y_fg, x_fg = self._index_triplet(batch, self.fg_scale_h, self.fg_scale_w)
        cam_fg_global = cam_fg + self.eval_offset
        batch["foreground"] = self._gather_feats_from_batch(self.current_batch_fg, cam_fg_global, y_fg, x_fg)
        cam_orientany, y_orientany, x_orientany = self._index_triplet(batch, self.orientany_scale_h, self.orientany_scale_w)
        cam_orientany_global = cam_orientany + self.eval_offset
        batch["orientany"] = self._gather_feats_from_batch(self.current_batch_orientany, cam_orientany_global, y_orientany, x_orientany)
        return ray_bundle, batch
