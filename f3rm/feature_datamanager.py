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

from f3rm.scripts.extract_features_standalone import (
    extract_features_for_dataset,
    LazyFeatures
)

# Stream for async CUDA operations
_copy_stream = torch.cuda.Stream()


def _async_to_cuda(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Pin + async-copy on a dedicated stream."""
    if t.is_cuda:
        return t
    pin = t.pin_memory()
    with torch.cuda.stream(_copy_stream):
        gpu = pin.to(device, non_blocking=True)
    torch.cuda.current_stream().wait_stream(_copy_stream)
    return gpu


@dataclass
class FeatureDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: FeatureDataManager)
    feature_type: Literal["CLIP", "DINOCLIP", "DINO", "ROBOPOINTproj", "ROBOPOINTnoproj"] = "CLIP"
    """Feature type to extract."""
    enable_cache: bool = True
    """Whether to cache extracted features."""


class FeatureDataManager(VanillaDataManager):
    config: FeatureDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract features
        features: Union[
            Float[torch.Tensor, "n h w c"],
            Dict[str, Float[torch.Tensor, "n h w c"]],  # For DINOCLIP
        ] = self.extract_features_sharded()

        # Split into train and eval features
        if self.config.feature_type == "DINOCLIP":
            # self.config.enable_cache = False    # Disable cache for DINOCLIP
            self.features_dino = features["dino"]
            self.features_clip = features["clip"]
            self.eval_offset = len(self.train_dataset)        # shared for both

            d_dim = (self.features_dino.C if isinstance(self.features_dino, LazyFeatures) else self.features_dino.shape[-1])
            c_dim = (self.features_clip.C if isinstance(self.features_clip, LazyFeatures) else self.features_clip.shape[-1])
            self.train_dataset.metadata.update(
                {
                    "feature_type": "DINOCLIP",
                    "feature_dim_dino": d_dim,
                    "feature_dim_clip": c_dim,
                }
            )

            feat_h_clip, feat_w_clip = (
                (self.features_clip.H, self.features_clip.W)
                if isinstance(self.features_clip, LazyFeatures)
                else self.features_clip.shape[1:3]
            )
            feat_h_dino, feat_w_dino = (
                (self.features_dino.H, self.features_dino.W)
                if isinstance(self.features_dino, LazyFeatures)
                else self.features_dino.shape[1:3]
            )
        else:
            self.features = features         # ≤-1 GB LazyFeatures or a real tensor
            self.eval_offset = len(self.train_dataset)

            # Set metadata, so we can initialize model with feature dimensionality
            feat_dim = (self.features.C if isinstance(self.features, LazyFeatures) else self.features.shape[-1])
            self.train_dataset.metadata["feature_type"] = self.config.feature_type
            self.train_dataset.metadata["feature_dim"] = feat_dim

            # Determine scaling factors for nearest neighbor interpolation
            feat_h, feat_w = (features.H, features.W) if isinstance(features, LazyFeatures) else features.shape[1:3]
        im_h = set(self.train_dataset.cameras.image_height.squeeze().tolist())
        im_w = set(self.train_dataset.cameras.image_width.squeeze().tolist())
        assert len(im_h) == 1, "All images must have the same height"
        assert len(im_w) == 1, "All images must have the same width"
        im_h, im_w = im_h.pop(), im_w.pop()
        if self.config.feature_type == "DINOCLIP":
            self.scale_h_dino = feat_h_dino / im_h
            self.scale_w_dino = feat_w_dino / im_w
            self.scale_h_clip = feat_h_clip / im_h
            self.scale_w_clip = feat_w_clip / im_w
        else:
            self.scale_h = feat_h / im_h
            self.scale_w = feat_w / im_w
        # assert np.isclose(
        #    self.scale_h, self.scale_w, atol=1.5e-3
        # ), f"Scales must be similar, got h={self.scale_h} and w={self.scale_w}"

        # Garbage collect
        torch.cuda.empty_cache()
        gc.collect()

    def extract_features_sharded(self,) -> Union[Float[torch.Tensor, "n h w c"], Dict[str, Float[torch.Tensor, "n h w c"]]]:
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
        else:
            cpu_batch = feats[camera_idx, y_idx, x_idx]
        return _async_to_cuda(cpu_batch, self.device)

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
        if self.config.feature_type == "DINOCLIP":
            camera_idx, y_idx, x_idx = self._index_triplet(batch, self.scale_h_dino, self.scale_w_dino)
            batch["feature_dino"] = self._gather_feats(self.features_dino, camera_idx, y_idx, x_idx)
            camera_idx, y_idx, x_idx = self._index_triplet(batch, self.scale_h_clip, self.scale_w_clip)
            batch["feature_clip"] = self._gather_feats(self.features_clip, camera_idx, y_idx, x_idx)
        else:
            camera_idx, y_idx, x_idx = self._index_triplet(batch, self.scale_h, self.scale_w)
            batch["feature"] = self._gather_feats(self.features, camera_idx, y_idx, x_idx)
        if step % 100 == 0:
            torch.cuda.empty_cache()
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_eval(step)
        batch["image"] = _async_to_cuda(batch["image"], self.device)
        if self.config.feature_type == "DINOCLIP":
            camera_idx, y_idx, x_idx = self._index_triplet(batch, self.scale_h_dino, self.scale_w_dino)
            batch["feature_dino"] = self._gather_feats(self.features_dino, camera_idx + self.eval_offset, y_idx, x_idx)
            camera_idx, y_idx, x_idx = self._index_triplet(batch, self.scale_h_clip, self.scale_w_clip)
            batch["feature_clip"] = self._gather_feats(self.features_clip, camera_idx + self.eval_offset, y_idx, x_idx)
        else:
            camera_idx, y_idx, x_idx = self._index_triplet(batch, self.scale_h, self.scale_w)
            camera_idx_global = camera_idx + self.eval_offset
            batch["feature"] = self._gather_feats(self.features, camera_idx_global, y_idx, x_idx)
        return ray_bundle, batch
