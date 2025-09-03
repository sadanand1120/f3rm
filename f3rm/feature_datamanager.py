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

from f3rm.features.extract_features_standalone import (
    extract_features_for_dataset,
    LazyFeatures
)

# Stream for async CUDA operations - use higher priority for faster transfers
_copy_stream = torch.cuda.Stream(priority=-1)
# Preallocated tensor cache for batch operations
_tensor_cache = {}


def _async_to_cuda(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Optimized pin + async-copy with tensor reuse."""
    if t.is_cuda:
        return t

    # Try to reuse pinned memory for same shape tensors
    shape_key = tuple(t.shape)
    if shape_key in _tensor_cache and _tensor_cache[shape_key].dtype == t.dtype:
        cached_tensor = _tensor_cache[shape_key]
        if cached_tensor.numel() == t.numel():
            cached_tensor.copy_(t.view_as(cached_tensor))
            with torch.cuda.stream(_copy_stream):
                gpu = cached_tensor.to(device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(_copy_stream)
            return gpu

    # Fallback to standard pinning
    pin = t.pin_memory()
    # Cache pinned tensor for reuse (limit cache size to prevent memory growth)
    if len(_tensor_cache) < 20:  # Reasonable cache size limit
        _tensor_cache[shape_key] = pin

    with torch.cuda.stream(_copy_stream):
        gpu = pin.to(device, non_blocking=True)
    torch.cuda.current_stream().wait_stream(_copy_stream)
    return gpu


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


class FeatureDataManager(VanillaDataManager):
    config: FeatureDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert device string to torch.device object if needed
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        # Extract features
        features: Float[torch.Tensor, "n h w c"] = self.extract_features_sharded()

        # Split into train and eval features
        self.features = features         # ≤-1 GB LazyFeatures or a real tensor
        self.eval_offset = len(self.train_dataset)

        # Load SAM2 auto masks lazily (for centroid cache GT); uses the same image order
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        self.sam2_masks = extract_features_for_dataset(
            image_fnames=image_fnames,
            data_dir=self.config.dataparser.data,
            feature_type=self.config.sam2_feature_type,
            device=self.device,
            shard_size=64,
            enable_cache=self.config.enable_cache,
            force=False,
        )
        CONSOLE.print("Loaded SAM2 lazy auto-masks for centroid supervision cache")

        # Load FOREGROUND shards lazily (2-dim one-hot per pixel), same image order
        self.fg_maps = extract_features_for_dataset(
            image_fnames=image_fnames,
            data_dir=self.config.dataparser.data,
            feature_type=self.config.foreground_feature_type,
            device=self.device,
            shard_size=64,
            enable_cache=self.config.enable_cache,
            force=False,
        )
        CONSOLE.print("Loaded FOREGROUND lazy maps for foreground head supervision")

        # Load ORIENTANY shards lazily (10-dim compact features per pixel), same image order
        self.orientany_maps = extract_features_for_dataset(
            image_fnames=image_fnames,
            data_dir=self.config.dataparser.data,
            feature_type=self.config.orientany_feature_type,
            device=self.device,
            shard_size=64,
            enable_cache=self.config.enable_cache,
            force=False,
        )
        CONSOLE.print("Loaded ORIENTANY lazy maps for OrientAny head supervision")

        # Set metadata, so we can initialize model with feature dimensionality
        feat_dim = (self.features.C if isinstance(self.features, LazyFeatures) else self.features.shape[-1])
        self.train_dataset.metadata["feature_type"] = self.config.feature_type
        self.train_dataset.metadata["feature_dim"] = feat_dim
        # Expose presence of foreground maps via metadata
        self.train_dataset.metadata["has_foreground"] = True
        self.train_dataset.metadata["has_orientany"] = True

        # Determine scaling factors for nearest neighbor interpolation
        feat_h, feat_w = (features.H, features.W) if isinstance(features, LazyFeatures) else features.shape[1:3]
        im_h = set(self.train_dataset.cameras.image_height.squeeze().tolist())
        im_w = set(self.train_dataset.cameras.image_width.squeeze().tolist())
        assert len(im_h) == 1, "All images must have the same height"
        assert len(im_w) == 1, "All images must have the same width"
        im_h, im_w = im_h.pop(), im_w.pop()
        # Feature scaling (for CLIP/DINO features)
        self.feat_scale_h = feat_h / im_h
        self.feat_scale_w = feat_w / im_w
        # Foreground scaling (infer from actual FG map dimensions)
        fg_h, fg_w = (self.fg_maps.H, self.fg_maps.W) if isinstance(self.fg_maps, LazyFeatures) else self.fg_maps.shape[1:3]
        self.fg_scale_h = fg_h / im_h
        self.fg_scale_w = fg_w / im_w
        # OrientAny scaling (infer from actual OrientAny map dimensions)
        sample_features = self.orientany_maps[0]  # Load first image to get dimensions
        orientany_h, orientany_w = sample_features.shape[:2]
        self.orientany_scale_h = orientany_h / im_h
        self.orientany_scale_w = orientany_w / im_w
        CONSOLE.print(f"Feat h: {feat_h}, Feat w: {feat_w}, Feat c: {feat_dim}, Im h: {im_h}, Im w: {im_w}")
        CONSOLE.print(f"Feat scale h: {self.feat_scale_h}, Feat scale w: {self.feat_scale_w}")
        CONSOLE.print(f"FG scale h: {self.fg_scale_h}, FG scale w: {self.fg_scale_w}")
        CONSOLE.print(f"OrientAny scale h: {self.orientany_scale_h}, OrientAny scale w: {self.orientany_scale_w}")
        # assert np.isclose(
        #    self.scale_h, self.scale_w, atol=1.5e-3
        # ), f"Scales must be similar, got h={self.scale_h} and w={self.scale_w}"

        # Garbage collect
        torch.cuda.empty_cache()
        gc.collect()

    def extract_features_sharded(self,) -> Float[torch.Tensor, "n h w c"]:
        """Extract features (with caching where supported)."""
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames

        # Use the shared feature extraction function
        return extract_features_for_dataset(
            image_fnames=image_fnames,
            data_dir=self.config.dataparser.data,
            feature_type=self.config.feature_type,
            device=self.device,
            shard_size=64,  # default shard size
            enable_cache=self.config.enable_cache,
            force=False,  # don't force re-extraction during training
        )

    # Nearest-neighbor sampling helpers
    def _gather_feats(self, feats, camera_idx, y_idx, x_idx):
        if isinstance(feats, LazyFeatures):
            out = [feats[int(ci), int(yi), int(xi)] for ci, yi, xi in zip(camera_idx, y_idx, x_idx)]
            cpu_batch = torch.stack(out, dim=0)
        elif hasattr(feats, '__class__') and 'ORIENTANYLazyFeatures' in str(feats.__class__):
            cpu_batch = self._gather_orientany_batch(feats, camera_idx, y_idx, x_idx)
        else:
            cpu_batch = feats[camera_idx, y_idx, x_idx]
        return _async_to_cuda(cpu_batch, self.device)

    def _gather_orientany_batch(self, feats, camera_idx, y_idx, x_idx):
        """Highly optimized batch loading for ORIENTANYLazyFeatures with vectorized operations."""
        # Convert to CPU tensors for faster indexing
        camera_idx_cpu = camera_idx.cpu() if camera_idx.is_cuda else camera_idx
        y_idx_cpu = y_idx.cpu() if y_idx.is_cuda else y_idx
        x_idx_cpu = x_idx.cpu() if x_idx.is_cuda else x_idx

        # Group by unique cameras to minimize loading
        unique_cams, inverse_indices = torch.unique(camera_idx_cpu, return_inverse=True)
        cam_data = {}

        # Load all unique camera data once (this is cached in ORIENTANYLazyFeatures)
        for ci in unique_cams.tolist():
            cam_data[ci] = feats[ci]  # Load full (H,W,10) array

        # Vectorized feature extraction
        batch_size = len(camera_idx)
        batch_features = np.zeros((batch_size, 10), dtype=np.float32)

        # Process each unique camera's pixels in batch
        for i, ci in enumerate(unique_cams.tolist()):
            mask = (inverse_indices == i)
            if not mask.any():
                continue

            indices = torch.nonzero(mask, as_tuple=True)[0]
            y_batch = y_idx_cpu[indices].long()
            x_batch = x_idx_cpu[indices].long()

            # Vectorized indexing for this camera
            cam_features = cam_data[ci]  # (H, W, 10)
            pixel_feats = cam_features[y_batch, x_batch]  # (N, 10)
            batch_features[indices] = pixel_feats

        return torch.from_numpy(batch_features)

    # helper
    def _index_triplet(self, batch, scale_h, scale_w):
        ray_indices = batch["indices"]
        camera_idx = ray_indices[:, 0]
        y_idx = (ray_indices[:, 1] * scale_h).long()
        x_idx = (ray_indices[:, 2] * scale_w).long()
        return camera_idx, y_idx, x_idx

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)
        batch["image"] = _async_to_cuda(batch["image"], self.device)
        # Features sampled on feature grid (scaled)
        camera_idx, y_feat, x_feat = self._index_triplet(batch, self.feat_scale_h, self.feat_scale_w)
        batch["feature"] = self._gather_feats(self.features, camera_idx, y_feat, x_feat)
        # Foreground maps at image resolution (no scaling)
        cam_fg, y_fg, x_fg = self._index_triplet(batch, self.fg_scale_h, self.fg_scale_w)
        batch["foreground"] = self._gather_feats(self.fg_maps, cam_fg, y_fg, x_fg)
        # OrientAny maps at image resolution (no scaling)
        cam_orientany, y_orientany, x_orientany = self._index_triplet(batch, self.orientany_scale_h, self.orientany_scale_w)
        batch["orientany"] = self._gather_feats(self.orientany_maps, cam_orientany, y_orientany, x_orientany)
        # Less frequent cache clearing for better speed
        if step % 500 == 0:
            torch.cuda.empty_cache()
            # Occasionally clear feature caches to prevent memory buildup
            if hasattr(self.orientany_maps, 'clear_cache'):
                self.orientany_maps.clear_cache()
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_eval(step)
        batch["image"] = _async_to_cuda(batch["image"], self.device)
        # Features on feature grid (scaled) with eval offset
        camera_idx, y_feat, x_feat = self._index_triplet(batch, self.feat_scale_h, self.feat_scale_w)
        camera_idx_global = camera_idx + self.eval_offset
        batch["feature"] = self._gather_feats(self.features, camera_idx_global, y_feat, x_feat)
        # Foreground maps at image resolution (no scaling) with eval offset
        cam_fg, y_fg, x_fg = self._index_triplet(batch, self.fg_scale_h, self.fg_scale_w)
        cam_fg_global = cam_fg + self.eval_offset
        batch["foreground"] = self._gather_feats(self.fg_maps, cam_fg_global, y_fg, x_fg)
        # OrientAny maps at image resolution (no scaling) with eval offset
        cam_orientany, y_orientany, x_orientany = self._index_triplet(batch, self.orientany_scale_h, self.orientany_scale_w)
        cam_orientany_global = cam_orientany + self.eval_offset
        batch["orientany"] = self._gather_feats(self.orientany_maps, cam_orientany_global, y_orientany, x_orientany)
        return ray_bundle, batch
