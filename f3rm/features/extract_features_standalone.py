#!/usr/bin/env python3
"""
Standalone Feature Extraction Script

This script extracts features (CLIP, DINO, or SAM2) for a dataset
and caches them in shards at the exact same location and format as expected by
the training pipeline. This allows pre-processing features independently of training.

Usage:
    python f3rm/features/extract_features_standalone.py \
        --data datasets/f3rm/custom/scene001 \
        --feature-type CLIP \
        --shard-size 64

Supported feature types: CLIP, DINO, SAM2
"""

import argparse
import gc
import math
from pathlib import Path
from typing import Dict, List, Literal, Union, Optional, Callable, Tuple

import torch
import numpy as np
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm.auto import tqdm

from f3rm.features.clip_extract import CLIPArgs, extract_clip_features
from f3rm.features.dino_extract import DINOArgs, extract_dino_features
from f3rm.features.sam2_extract import SAM2Args, extract_sam2_masks


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


# Map feature types to extract functions and args
FEAT_TYPE_TO_EXTRACT_FN = {
    "CLIP": extract_clip_features,
    "DINO": extract_dino_features,
    "SAM2": extract_sam2_masks,
}

FEAT_TYPE_TO_ARGS = {
    "CLIP": CLIPArgs,
    "DINO": DINOArgs,
    "SAM2": SAM2Args,
}


def get_cache_paths(data_dir: Path, feature_type: str) -> Tuple[Path, Path]:
    """Get cache directory and metadata paths for a feature type."""
    root = data_dir / "features" / feature_type.lower()
    return root, root / "meta.pt"


def feature_saver(
    image_fnames: List[str],
    extract_fn: Callable,
    extract_args,
    data_dir: Path,
    feature_type: str,
    device: torch.device,
    shard_sz: int = 64,
) -> None:
    """Extract features and save them as shards."""
    root, meta = get_cache_paths(data_dir, feature_type)
    root.mkdir(parents=True, exist_ok=True)

    n_imgs = len(image_fnames)
    n_shards = math.ceil(n_imgs / shard_sz)

    CONSOLE.print(f"Extracting {feature_type} features for {n_imgs} images in {n_shards} shards...")

    for i in tqdm(range(n_shards), desc=f"{feature_type}: extract→save"):
        s, e = i * shard_sz, min((i + 1) * shard_sz, n_imgs)

        if feature_type == "SAM2":
            # SAM2 returns numpy arrays directly
            feats = extract_fn(image_fnames[s:e], verbose=False)
        else:
            # CLIP/DINO return torch tensors
            feats = extract_fn(image_fnames[s:e], device).cpu().numpy()

        np.save(root / f"chunk_{i:04d}.npy", feats, allow_pickle=False)
        del feats
        torch.cuda.empty_cache()

    # Save metadata
    torch.save({"args": extract_args.id_dict(), "image_fnames": image_fnames}, meta)
    CONSOLE.print(f"Saved {feature_type} shards → {root}")


def feature_loader(image_fnames: List[str], extract_args, data_dir: Path, feature_type: str, device: torch.device) -> Optional[LazyFeatures]:
    """Load cached features if they exist and match the expected configuration."""
    root, meta = get_cache_paths(data_dir, feature_type)

    CONSOLE.print(f"[DEBUG] {feature_type}: Checking cache at {root}")
    CONSOLE.print(f"[DEBUG] {feature_type}: Looking for metadata at {meta}")

    if not meta.exists():
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - Metadata file does not exist: {meta}")
        return None

    CONSOLE.print(f"[DEBUG] {feature_type}: Metadata file exists, loading...")
    md = torch.load(meta)

    # Check args match
    current_args = extract_args.id_dict()
    cached_args = md.get("args")
    args_match = cached_args == current_args

    CONSOLE.print(f"[DEBUG] {feature_type}: Args comparison:")
    CONSOLE.print(f"[DEBUG] {feature_type}: Current args: {current_args}")
    CONSOLE.print(f"[DEBUG] {feature_type}: Cached args:  {cached_args}")
    CONSOLE.print(f"[DEBUG] {feature_type}: Args match: {args_match}")

    # Check image filenames match - CONVERT TO STRINGS to handle Path vs string mismatch
    cached_fnames = md.get("image_fnames")
    # Convert both to strings to ensure consistent comparison
    current_fnames_str = [str(fname) for fname in image_fnames]
    cached_fnames_str = [str(fname) for fname in cached_fnames] if cached_fnames else None
    fnames_match = cached_fnames_str == current_fnames_str

    CONSOLE.print(f"[DEBUG] {feature_type}: Image filenames comparison:")
    CONSOLE.print(f"[DEBUG] {feature_type}: Current count: {len(current_fnames_str)}")
    CONSOLE.print(f"[DEBUG] {feature_type}: Cached count:  {len(cached_fnames_str) if cached_fnames_str else 'None'}")

    if cached_fnames_str and len(cached_fnames_str) == len(current_fnames_str):
        # Check first and last few for debugging
        CONSOLE.print(f"[DEBUG] {feature_type}: First 3 current: {current_fnames_str[:3]}")
        CONSOLE.print(f"[DEBUG] {feature_type}: First 3 cached:  {cached_fnames_str[:3]}")
        CONSOLE.print(f"[DEBUG] {feature_type}: Last 3 current:  {current_fnames_str[-3:]}")
        CONSOLE.print(f"[DEBUG] {feature_type}: Last 3 cached:   {cached_fnames_str[-3:]}")

        # Find first difference if any
        if not fnames_match:
            for i, (curr, cached) in enumerate(zip(current_fnames_str, cached_fnames_str)):
                if curr != cached:
                    CONSOLE.print(f"[DEBUG] {feature_type}: First difference at index {i}:")
                    CONSOLE.print(f"[DEBUG] {feature_type}: Current: {curr}")
                    CONSOLE.print(f"[DEBUG] {feature_type}: Cached:  {cached}")
                    break

    CONSOLE.print(f"[DEBUG] {feature_type}: Filenames match: {fnames_match}")

    if not args_match or not fnames_match:
        if not args_match:
            CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - Args don't match")
        if not fnames_match:
            CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - Image filenames don't match")
        return None

    # Check shard files exist
    shard_paths = sorted(root.glob("chunk_*.npy"))
    if not shard_paths:
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - No shard files found")
        return None

    CONSOLE.print(f"[DEBUG] {feature_type}: CACHE HIT - Using mmap cache with {len(shard_paths)} shards")
    return LazyFeatures(shard_paths, device)


def get_image_filenames_from_dataparser(data_dir: Path) -> List[str]:
    """Get image filenames in the same order as the training pipeline."""
    # Use the EXACT same dataparser configuration as the training pipeline in f3rm/f3rm_config.py
    dataparser_config = NerfstudioDataParserConfig(
        data=data_dir,
        train_split_fraction=0.95,
    )
    dataparser = dataparser_config.setup()

    # Parse train and test datasets
    train_dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
    test_dataparser_outputs = dataparser.get_dataparser_outputs(split="val")

    # Combine image filenames in the same order as feature_datamanager
    train_image_filenames = [str(path) for path in train_dataparser_outputs.image_filenames]
    test_image_filenames = [str(path) for path in test_dataparser_outputs.image_filenames]

    all_image_filenames = train_image_filenames + test_image_filenames

    CONSOLE.print(f"Found {len(train_image_filenames)} train images and {len(test_image_filenames)} test images")
    return all_image_filenames


def extract_features_for_dataset(
    image_fnames: List[str],
    data_dir: Path,
    feature_type: Literal["CLIP", "DINO", "SAM2"],
    device: torch.device,
    shard_size: int = 64,
    enable_cache: bool = True,
    force: bool = False,
) -> Union[torch.Tensor, LazyFeatures]:
    """
    Extract features for a dataset (same logic as FeatureDataManager.extract_features_sharded).

    Returns:
        - LazyFeatures or torch.Tensor
    """
    fn, args = FEAT_TYPE_TO_EXTRACT_FN[feature_type], FEAT_TYPE_TO_ARGS[feature_type]

    CONSOLE.print(f"[DEBUG] {feature_type}: enable_cache={enable_cache}, checking for cached features...")
    feats = (
        feature_loader(image_fnames, args, data_dir, feature_type, device)
        if enable_cache
        else None
    )

    if feats is None:
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS → extracting features")
        CONSOLE.print("Cache miss → extracting…")
        if enable_cache and not force:
            feature_saver(image_fnames, fn, args, data_dir, feature_type, device, shard_size)
            feats = feature_loader(image_fnames, args, data_dir, feature_type, device)
        else:
            tmp = fn(image_fnames, device).cpu().numpy()
            np.save(data_dir / "tmp.npy", tmp, allow_pickle=False)
            del tmp
            feats = LazyFeatures([data_dir / "tmp.npy"], device)
    else:
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE HIT → reusing cached features")

    return feats


def extract_features_standalone(
    data_dir: Path,
    feature_type: Literal["CLIP", "DINO", "SAM2"],
    shard_size: int = 64,
    device: str = "auto",
    force: bool = False,
) -> None:
    """Extract features standalone."""

    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    CONSOLE.print(f"Using device: {device}")

    # Get image filenames in the same order as training pipeline
    image_fnames = get_image_filenames_from_dataparser(data_dir)

    # Extract features
    extract_features_for_dataset(
        image_fnames=image_fnames,
        data_dir=data_dir,
        feature_type=feature_type,
        device=device,
        shard_size=shard_size,
        enable_cache=True,
        force=force,
    )

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    CONSOLE.print("Feature extraction completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features for F3RM training independently of the training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the dataset directory (same as used in training)"
    )

    parser.add_argument(
        "--feature-type",
        choices=["CLIP", "DINO", "SAM2"],
        default="CLIP",
        help="Feature type to extract"
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=64,
        help="Number of images per shard"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu, cuda:0, etc.)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cache exists"
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data}")

    # Check for transforms.json (nerfstudio format)
    transforms_json = args.data / "transforms.json"
    if not transforms_json.exists():
        raise FileNotFoundError(f"transforms.json not found in {args.data}. Make sure this is a processed nerfstudio dataset.")

    CONSOLE.print(f"Extracting {args.feature_type} features from {args.data}")

    # Extract features
    extract_features_standalone(
        data_dir=args.data,
        feature_type=args.feature_type,
        shard_size=args.shard_size,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
