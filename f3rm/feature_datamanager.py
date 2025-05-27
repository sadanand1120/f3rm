import gc
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Optional, List, Callable

import numpy as np
import math
from pathlib import Path
from tqdm.auto import tqdm
import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from f3rm.features.clip_extract import CLIPArgs, extract_clip_features
from f3rm.features.dino_extract import DINOArgs, extract_dino_features
from f3rm.features.robopoint_extract import ROBOPOINTArgs, extract_robopoint_proj_features, extract_robopoint_noproj_features

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


class LazyFeatures:
    """Memory-mapped shards with O(1) random access.

    Exposes `feat[idx_img, y, x] → torch.Tensor(C)` and keeps each shard
    mapped only once (OS handles paging).  Nothing is ever `torch.cat`-ed.
    """

    def __init__(self, shard_paths: List[Path], device: torch.device):
        self.paths = shard_paths
        self.device = device
        self.mmaps = [None] * len(shard_paths)          # lazy mmap
        self.lengths = []
        for p in shard_paths:
            arr = np.load(p, mmap_mode="r")
            self.lengths.append(arr.shape[0])
            if self.mmaps[0] is None:                     # keep dims
                self.H, self.W, self.C = arr.shape[1:]
        self.cum = np.cumsum([0] + self.lengths)          # prefix-sum

    # helper → shard id, local idx
    def _loc(self, idx_img: int) -> Tuple[int, int]:
        sid = int(np.searchsorted(self.cum, idx_img, side="right") - 1)
        return sid, idx_img - self.cum[sid]

    def _get_shard(self, sid: int):
        if self.mmaps[sid] is None:
            self.mmaps[sid] = np.load(self.paths[sid], mmap_mode="r", allow_pickle=False)
        return self.mmaps[sid]

    # single triple access
    def __getitem__(self, triple):
        idx_img, y, x = triple
        sid, loc = self._loc(int(idx_img))
        feat = self._get_shard(sid)[loc, int(y), int(x)]        # numpy view
        return torch.from_numpy(feat)


def _cache_paths(data_dir: Path, feature_type: str) -> Tuple[Path, Path]:
    root = data_dir / "features" / feature_type.lower()
    return root, root / "meta.pt"


def feature_loader(image_fnames, extract_args, data_dir: Path, feature_type: str, device) -> Optional[LazyFeatures]:
    root, meta = _cache_paths(data_dir, feature_type)
    if not meta.exists():
        return None
    md = torch.load(meta)
    if (md.get("image_fnames") != image_fnames or md.get("args") != extract_args.id_dict()):
        return None
    shard_paths = sorted(root.glob("chunk_*.npy"))
    if not shard_paths:
        return None
    CONSOLE.print(f"Using mmap cache: {feature_type} ({len(shard_paths)} shards)")
    return LazyFeatures(shard_paths, device)


def feature_saver(image_fnames: List[str], extract_fn: Callable[[List[str], torch.device], torch.Tensor],
                  extract_args, data_dir: Path, feature_type: str, device, shard_sz: int = 64,) -> None:
    root, meta = _cache_paths(data_dir, feature_type)
    root.mkdir(parents=True, exist_ok=True)
    n_imgs = len(image_fnames)
    n_shards = math.ceil(n_imgs / shard_sz)
    for i in tqdm(range(n_shards), desc=f"{feature_type}: extract→save"):
        s, e = i * shard_sz, min((i + 1) * shard_sz, n_imgs)
        feats = extract_fn(image_fnames[s:e], device).cpu()
        np.save(root / f"chunk_{i:04d}.npy", feats.numpy(), allow_pickle=False)
        del feats
        torch.cuda.empty_cache()
    torch.save({"args": extract_args.id_dict(), "image_fnames": image_fnames}, meta)
    CONSOLE.print(f"Saved {feature_type} shards → {root}")


@dataclass
class FeatureDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: FeatureDataManager)
    feature_type: Literal["CLIP", "DINOCLIP", "DINO", "ROBOPOINTproj", "ROBOPOINTnoproj"] = "CLIP"
    """Feature type to extract."""
    enable_cache: bool = True
    """Whether to cache extracted features."""


feat_type_to_extract_fn = {
    "CLIP": extract_clip_features,
    "DINO": extract_dino_features,
    "ROBOPOINTproj": extract_robopoint_proj_features,
    "ROBOPOINTnoproj": extract_robopoint_noproj_features,
}

feat_type_to_args = {
    "CLIP": CLIPArgs,
    "DINO": DINOArgs,
    "ROBOPOINTproj": ROBOPOINTArgs,
    "ROBOPOINTnoproj": ROBOPOINTArgs,
}


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
        if self.config.feature_type == "DINOCLIP":
            cache_dir = self.config.dataparser.data
            out: Dict[str, LazyFeatures] = {}
            for sub in ("DINO", "CLIP"):
                fn, args = feat_type_to_extract_fn[sub], feat_type_to_args[sub]
                feats = (
                    feature_loader(image_fnames, args, cache_dir, sub, self.device)
                    if self.config.enable_cache
                    else None
                )
                if feats is None:
                    CONSOLE.print(f"{sub}: cache miss → extracting…")
                    if self.config.enable_cache:
                        feature_saver(image_fnames, fn, args, cache_dir, sub, self.device)
                        feats = feature_loader(image_fnames, args, cache_dir, sub, self.device)
                    else:
                        # fall-back: immediate extraction, still lazy in one shard
                        tmp = fn(image_fnames, self.device).cpu().numpy()
                        np.save(cache_dir / f"tmp.npy", tmp, allow_pickle=False)
                        del tmp
                        feats = LazyFeatures([cache_dir / "tmp.npy"], self.device)
                out[sub.lower()] = feats
            return {"dino": out["dino"], "clip": out["clip"]}

        # === original single-type code path ===
        if self.config.feature_type not in feat_type_to_extract_fn:
            raise ValueError(f"Unknown feature type {self.config.feature_type}")
        fn, args = feat_type_to_extract_fn[self.config.feature_type], feat_type_to_args[self.config.feature_type]

        feats = (
            feature_loader(image_fnames, args, self.config.dataparser.data, self.config.feature_type, self.device)
            if self.config.enable_cache
            else None
        )
        if feats is None:
            CONSOLE.print("Cache miss → extracting…")
            if self.config.enable_cache:
                feature_saver(image_fnames, fn, args, self.config.dataparser.data, self.config.feature_type, self.device)
                feats = feature_loader(image_fnames, args, self.config.dataparser.data, self.config.feature_type, self.device)
            else:
                tmp = fn(image_fnames, self.device).cpu().numpy()
                np.save(self.config.dataparser.data / "tmp.npy", tmp, allow_pickle=False)
                del tmp
                feats = LazyFeatures([self.config.dataparser.data / "tmp.npy"], self.device)
        return feats

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
