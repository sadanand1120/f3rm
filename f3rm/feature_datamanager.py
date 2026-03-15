import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple, Type

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from f3rm.features.extract_features_standalone import extract_features_for_dataset
from f3rm.features.centroid_extract import CENTROID_FILL_VALUE, infer_sam3d_feature_type, parse_centroid_feature_type


@dataclass
class FeatureDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: FeatureDataManager)
    feature_type: Literal["CLIP", "DINO"] = "CLIP"
    foreground_feature_type: str = "FOREGROUND_"
    centroid_feature_type: str = "CENTROID_"
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
        self.all_image_paths = [str(Path(image_path).resolve()) for image_path in image_fnames]
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

        self.centroid_loader = extract_features_for_dataset(
            feature_type=self.config.centroid_feature_type,
            **loader_kwargs,
        )
        CONSOLE.print(f"Created batch loader for {self.config.centroid_feature_type} maps")

        # Metadata required by downstream model construction
        self.train_dataset.metadata["feature_type"] = self.config.feature_type
        self.train_dataset.metadata["feature_dim"] = self.feature_loader.C

        # Validate camera dimensions and compute scaling into feature grids
        feat_h, feat_w = self.feature_loader.H, self.feature_loader.W
        fg_h, fg_w = self.fg_loader.H, self.fg_loader.W
        centroid_h, centroid_w = self.centroid_loader.H, self.centroid_loader.W
        im_h = set(self.train_dataset.cameras.image_height.squeeze().tolist())
        im_w = set(self.train_dataset.cameras.image_width.squeeze().tolist())
        assert len(im_h) == 1, "All images must have the same height"
        assert len(im_w) == 1, "All images must have the same width"
        im_h, im_w = im_h.pop(), im_w.pop()
        self.feat_scale_h = feat_h / im_h
        self.feat_scale_w = feat_w / im_w
        self.fg_scale_h = fg_h / im_h
        self.fg_scale_w = fg_w / im_w
        self.centroid_scale_h = centroid_h / im_h
        self.centroid_scale_w = centroid_w / im_w
        CONSOLE.print(
            f"Feat h: {feat_h}, Feat w: {feat_w}, Feat c: {self.feature_loader.C}, Im h: {im_h}, Im w: {im_w}"
        )
        CONSOLE.print(f"Feat scale h: {self.feat_scale_h}, Feat scale w: {self.feat_scale_w}")
        CONSOLE.print(f"FG scale h: {self.fg_scale_h}, FG scale w: {self.fg_scale_w}")
        CONSOLE.print(f"CENTROID scale h: {self.centroid_scale_h}, CENTROID scale w: {self.centroid_scale_w}")

        # Per-batch runtime caches
        self.current_batch_features = {}
        self.current_batch_fg = {}
        self.current_batch_centroid = {}
        self.centroid_scale_results: Dict[str, Dict[int, Dict]] = {}
        centroid_prompts = parse_centroid_feature_type(self.config.centroid_feature_type)
        centroid_prompt_arg = None if not centroid_prompts else centroid_prompts
        self.centroid_sam3d_feature_name = infer_sam3d_feature_type(centroid_prompt_arg).lower()

        torch.cuda.empty_cache()
        gc.collect()

    def _clear_batch_features(self):
        """Clear current batch features to prevent VRAM accumulation."""
        self.current_batch_features.clear()
        self.current_batch_fg.clear()
        self.current_batch_centroid.clear()

    def _load_batch_features(self, camera_indices: torch.Tensor):
        """Load features for unique cameras in the current batch."""
        unique_cameras = torch.unique(camera_indices)
        self.current_batch_features = self.feature_loader.load_batch_images(unique_cameras)
        self.current_batch_fg = self.fg_loader.load_batch_images(unique_cameras)
        self.current_batch_centroid = self.centroid_loader.load_batch_images(unique_cameras)

    def _gather_feats_from_batch(
        self, batch_features: dict, camera_idx: torch.Tensor, y_idx: torch.Tensor, x_idx: torch.Tensor
    ) -> torch.Tensor:
        """Gather features from batch-loaded feature dictionary. Vectorized for tensor features to avoid per-ray Python loops.
        """
        batch_size = len(camera_idx)
        if not batch_features:
            return torch.zeros(batch_size, 0, device=self.device, dtype=torch.float16)
        # Regular tensor features (CLIP, DINO, FOREGROUND)
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

    def set_centroid_scale_results(self, scale_results: Dict[str, Dict[int, Dict]]) -> None:
        self.centroid_scale_results = scale_results

    def _scale_centroid_targets(self, camera_idx: torch.Tensor, centroid_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        camera_idx = camera_idx.to(centroid_raw.device, non_blocking=True)
        centroid_target = torch.zeros((len(camera_idx), 3), device=centroid_raw.device, dtype=torch.float32)
        fill_value = float(CENTROID_FILL_VALUE[0])
        centroid_valid = (centroid_raw[:, 0] != fill_value) & torch.all(centroid_raw[:, 1:4] != fill_value, dim=-1)
        if not self.centroid_scale_results:
            return centroid_target, centroid_valid & False

        mask_indices = centroid_raw[:, 0].round().long()
        for camera in torch.unique(camera_idx):
            camera_mask = camera_idx == camera
            image_scales = self.centroid_scale_results.get(self.all_image_paths[int(camera.item())])
            if image_scales is None:
                centroid_valid[camera_mask] = False
                continue

            camera_mask_indices = torch.unique(mask_indices[camera_mask & centroid_valid])
            for mask_index in camera_mask_indices:
                mask_value = int(mask_index.item())
                scale_meta = image_scales.get(mask_value)
                full_mask = camera_mask & centroid_valid & (mask_indices == mask_value)
                if scale_meta is None:
                    centroid_valid[full_mask] = False
                    continue
                centroid_target[full_mask] = float(scale_meta["sam3d_to_nerf_scale"]) * centroid_raw[full_mask, 1:4].float()
        return centroid_target, centroid_valid

    def get_centroid_image(self, camera_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        centroid_raw = self.centroid_loader[camera_idx]
        h, w = centroid_raw.shape[:2]
        camera_idx_tensor = torch.full((h * w,), camera_idx, device=self.device, dtype=torch.long)
        centroid_target, centroid_valid = self._scale_centroid_targets(camera_idx_tensor, centroid_raw.view(-1, 4))
        return centroid_target.view(h, w, 3), centroid_valid.view(h, w, 1)

    def _populate_batch_features(self, batch: Dict, is_eval: bool) -> None:
        self._clear_batch_features()
        camera_idx, y_feat, x_feat = self._index_triplet(batch, self.feat_scale_h, self.feat_scale_w)
        cam_fg, y_fg, x_fg = self._index_triplet(batch, self.fg_scale_h, self.fg_scale_w)
        cam_centroid, y_centroid, x_centroid = self._index_triplet(batch, self.centroid_scale_h, self.centroid_scale_w)

        if is_eval:
            camera_idx = camera_idx + self.eval_offset
            cam_fg = cam_fg + self.eval_offset
            cam_centroid = cam_centroid + self.eval_offset

        self._load_batch_features(camera_idx)
        batch["feature"] = self._gather_feats_from_batch(self.current_batch_features, camera_idx, y_feat, x_feat)
        batch["foreground"] = self._gather_feats_from_batch(self.current_batch_fg, cam_fg, y_fg, x_fg)
        centroid_raw = self._gather_feats_from_batch(self.current_batch_centroid, cam_centroid, y_centroid, x_centroid)
        batch["centroid"], batch["centroid_valid"] = self._scale_centroid_targets(cam_centroid, centroid_raw)

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
