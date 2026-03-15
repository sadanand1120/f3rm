import gc
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from f3rm.features.extract_features_standalone import extract_features_for_dataset


@dataclass
class FeatureDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: FeatureDataManager)
    feature_type: Literal["CLIP"] = "CLIP"
    foreground_feature_type: str = "FOREGROUND_"
    enable_cache: bool = True
    """Whether to cache extracted features."""
    cpu_feature_cache_images: int = 128
    gpu_feature_cache_images: int = 16


class FeatureDataManager(VanillaDataManager):
    config: FeatureDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Dataset and device setup
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.eval_offset = len(self.train_dataset)
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        loader_kwargs = {
            "image_fnames": image_fnames,
            "data_dir": self.config.dataparser.data,
            "device": self.device,
            "batch_size": 128,  # TODO: why hardcoded 128?
            "enable_cache": self.config.enable_cache,
            "force": False,
            "max_cpu_images": self.config.cpu_feature_cache_images,
            "max_gpu_images": self.config.gpu_feature_cache_images,
        }

        # Feature loaders
        self.feature_loader = extract_features_for_dataset(
            feature_type=self.config.feature_type,
            **loader_kwargs,
        )
        CONSOLE.print(f"Created batch loader for {self.config.feature_type} features")

        self.fg_loader = extract_features_for_dataset(
            feature_type=self.config.foreground_feature_type,
            **loader_kwargs,
        )
        CONSOLE.print(f"Created batch loader for {self.config.foreground_feature_type} maps")

        # Metadata required by downstream model construction
        self.train_dataset.metadata["feature_type"] = self.config.feature_type
        self.train_dataset.metadata["feature_dim"] = self.feature_loader.C

        # Validate camera dimensions and compute scaling into feature grids
        feat_h, feat_w = self.feature_loader.H, self.feature_loader.W
        fg_h, fg_w = self.fg_loader.H, self.fg_loader.W
        im_h = set(self.train_dataset.cameras.image_height.squeeze().tolist())
        im_w = set(self.train_dataset.cameras.image_width.squeeze().tolist())
        assert len(im_h) == 1, "All images must have the same height"
        assert len(im_w) == 1, "All images must have the same width"
        im_h, im_w = im_h.pop(), im_w.pop()
        self.feat_scale_h = feat_h / im_h
        self.feat_scale_w = feat_w / im_w
        self.fg_scale_h = fg_h / im_h
        self.fg_scale_w = fg_w / im_w
        CONSOLE.print(
            f"Feat h: {feat_h}, Feat w: {feat_w}, Feat c: {self.feature_loader.C}, Im h: {im_h}, Im w: {im_w}"
        )
        CONSOLE.print(f"Feat scale h: {self.feat_scale_h}, Feat scale w: {self.feat_scale_w}")
        CONSOLE.print(f"FG scale h: {self.fg_scale_h}, FG scale w: {self.fg_scale_w}")

        # Per-batch runtime caches
        self.current_batch_features = {}
        self.current_batch_fg = {}

        torch.cuda.empty_cache()
        gc.collect()

    def _clear_batch_features(self):
        """Clear current batch features to prevent VRAM accumulation."""
        self.current_batch_features.clear()
        self.current_batch_fg.clear()

    def _load_batch_features(self, camera_indices: torch.Tensor):
        """Load features for unique cameras in the current batch."""
        unique_cameras = torch.unique(camera_indices)
        self.current_batch_features = self.feature_loader.load_batch_images(unique_cameras)
        self.current_batch_fg = self.fg_loader.load_batch_images(unique_cameras)

    def _gather_feats_from_batch(
        self, batch_features: dict, camera_idx: torch.Tensor, y_idx: torch.Tensor, x_idx: torch.Tensor
    ) -> torch.Tensor:
        """Gather features from batch-loaded feature dictionary. Vectorized for tensor features to avoid per-ray Python loops.
        """
        batch_size = len(camera_idx)
        if not batch_features:
            return torch.zeros(batch_size, 0, device=self.device, dtype=torch.float16)
        # Regular tensor features (CLIP, FOREGROUND)
        # Build a stacked tensor for the unique cameras in the batch and index in one shot.
        unique_cams, inverse = torch.unique(camera_idx, sorted=True, return_inverse=True)
        stacked = [batch_features[int(cam.item())] for cam in unique_cams]
        features_u = torch.stack(stacked, dim=0)  # [U, H, W, C]
        return features_u[inverse, y_idx, x_idx, :]

    def _index_triplet(self, batch, scale_h, scale_w):
        ray_indices = batch["indices"]
        camera_idx = ray_indices[:, 0]
        y_idx = (ray_indices[:, 1] * scale_h).long()
        x_idx = (ray_indices[:, 2] * scale_w).long()
        return camera_idx, y_idx, x_idx

    def _populate_batch_features(self, batch: Dict, is_eval: bool) -> None:
        self._clear_batch_features()
        camera_idx, y_feat, x_feat = self._index_triplet(batch, self.feat_scale_h, self.feat_scale_w)
        cam_fg, y_fg, x_fg = self._index_triplet(batch, self.fg_scale_h, self.fg_scale_w)

        if is_eval:
            camera_idx = camera_idx + self.eval_offset
            cam_fg = cam_fg + self.eval_offset

        self._load_batch_features(camera_idx)
        batch["feature"] = self._gather_feats_from_batch(self.current_batch_features, camera_idx, y_feat, x_feat)
        batch["foreground"] = self._gather_feats_from_batch(self.current_batch_fg, cam_fg, y_fg, x_fg)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)
        batch["image"] = batch["image"].to(self.device, non_blocking=True)
        self._populate_batch_features(batch, is_eval=False)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_eval(step)
        batch["image"] = batch["image"].to(self.device, non_blocking=True)
        self._populate_batch_features(batch, is_eval=True)
        return ray_bundle, batch
