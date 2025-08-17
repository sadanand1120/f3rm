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
import asyncio
import gc
import math
from pathlib import Path
from typing import Dict, List, Literal, Union, Optional, Tuple, Callable, Any

import torch
import numpy as np
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm.auto import tqdm

from f3rm.features.clip_extract import CLIPArgs, make_clip_extractor
from f3rm.features.dino_extract import DINOArgs, make_dino_extractor
from f3rm.features.sam2_extract import SAM2Args, make_sam2_extractor
from f3rm.features.utils import run_async_in_any_context, SAM2LazyAutoMasks, LazyFeatures, pack_auto_masks, pack_batch_auto_masks


FEAT_TYPE_TO_ARGS = {
    "CLIP": CLIPArgs,
    "DINO": DINOArgs,
    "SAM2": SAM2Args,
}

FEAT_TYPE_TO_MAKE_EXTRACTOR: Dict[str, Callable[[torch.device, bool], Any]] = {
    "CLIP": make_clip_extractor,
    "DINO": make_dino_extractor,
    "SAM2": make_sam2_extractor,
}


def get_cache_paths(data_dir: Path, feature_type: str) -> Tuple[Path, Path]:
    """Get cache directory and metadata paths for a feature type."""
    root = data_dir / "features" / feature_type.lower()
    return root, root / "meta.pt"


async def _save_shards_generic(
    image_fnames: List[str],
    data_dir: Path,
    feature_type: str,
    extractor_factory: Callable[[torch.device, bool], Any],
    args_cls: Any,
    device: torch.device,
    shard_sz: int,
):
    root, meta = get_cache_paths(data_dir, feature_type)
    root.mkdir(parents=True, exist_ok=True)
    n_imgs = len(image_fnames)
    n_shards = math.ceil(n_imgs / shard_sz)
    extractor = extractor_factory(device, verbose=True)
    for i in tqdm(range(n_shards), desc=f"{feature_type}: shards", position=0):
        s, e = i * shard_sz, min((i + 1) * shard_sz, n_imgs)
        batch_paths = image_fnames[s:e]
        data = await extractor.extract_batch_async(batch_paths)
        if feature_type in ("CLIP", "DINO"):
            np.save(root / f"chunk_{i:04d}.npy", data.cpu().numpy(), allow_pickle=False)
        elif feature_type == "SAM2":
            # Pack SAM2 data into memory-mappable format using consolidated function
            packed_batch = pack_batch_auto_masks(data)

            # Store as single concatenated arrays for memory mapping
            np.savez_compressed(
                root / f"chunk_{i:04d}.npz",
                **packed_batch
            )
        del data
        torch.cuda.empty_cache()
        gc.collect()
    torch.save({"args": args_cls.id_dict(), "image_fnames": image_fnames}, meta)
    CONSOLE.print(f"Saved {feature_type} shards → {root}")


def feature_loader(image_fnames: List[str], extract_args, data_dir: Path, feature_type: str):
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
    if feature_type == "SAM2":
        shard_paths = sorted(root.glob("chunk_*.npz"))
    else:
        shard_paths = sorted(root.glob("chunk_*.npy"))
    if not shard_paths:
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - No shard files found")
        return None

    if feature_type == "SAM2":
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE HIT - Using SAM2 lazy cache with {len(shard_paths)} shards")
        return SAM2LazyAutoMasks(shard_paths)

    CONSOLE.print(f"[DEBUG] {feature_type}: CACHE HIT - Using mmap cache with {len(shard_paths)} shards")
    return LazyFeatures(shard_paths)


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
) -> Union[torch.Tensor, LazyFeatures, SAM2LazyAutoMasks]:
    """
    Extract features for a dataset (same logic as FeatureDataManager.extract_features_sharded).

    Returns:
        - LazyFeatures or torch.Tensor
    """
    args = FEAT_TYPE_TO_ARGS[feature_type]

    CONSOLE.print(f"[DEBUG] {feature_type}: enable_cache={enable_cache}, checking for cached features...")
    feats = (
        feature_loader(image_fnames, args, data_dir, feature_type)
        if enable_cache and not force
        else None
    )

    if feats is not None:
        CONSOLE.print(f"[{feature_type}] Using cached features")
        return feats

    CONSOLE.print(f"[{feature_type}] Extracting features...")

    async def _run():
        await _save_shards_generic(
            image_fnames=image_fnames,
            data_dir=data_dir,
            feature_type=feature_type,
            extractor_factory=FEAT_TYPE_TO_MAKE_EXTRACTOR[feature_type],
            args_cls=args,
            device=device,
            shard_sz=shard_size,
        )

    run_async_in_any_context(_run)

    loaded = feature_loader(image_fnames, args, data_dir, feature_type)
    return loaded if loaded is not None else torch.empty(0)


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

    # needed to decrease from 64 to 16 (can even do 8 if needed) for reasonable train speed with 2048 load_size for features
    parser.add_argument(
        "--shard-size",
        type=int,
        default=16,
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
