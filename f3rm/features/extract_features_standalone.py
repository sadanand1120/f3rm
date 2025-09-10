#!/usr/bin/env python3
"""
Standalone Feature Extraction Script

Extracts features (CLIP, DINO, SAM2, TEXT, CLIPSAM_*, FOREGROUND_*, ORIENTANY_*) for a dataset
and saves them as individual per-image files for efficient batch loading during training.

Features are processed in batches for memory efficiency during extraction, but each image's
features are saved as separate files (e.g., image_000000.npy, image_000001.npy, etc.).

Usage:
    python f3rm/features/extract_features_standalone.py \
        --data datasets/f3rm/custom/scene001 \
        --feature-type CLIP \
        --batch-size 64

Supported feature types: CLIP, DINO, SAM2, TEXT, CLIPSAM_*, FOREGROUND_*, ORIENTANY_*
"""

import argparse
import asyncio
import gc
import json
import math
from pathlib import Path
from typing import Dict, List, Literal, Union, Optional, Tuple, Callable, Any, Type

import torch
import numpy as np
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm.auto import tqdm

from f3rm.features.clip_extract import CLIPArgs, CLIPExtractor
from f3rm.features.dino_extract import DINOArgs, DINOExtractor
from f3rm.features.sam2_extract import SAM2Args, SAM2Extractor
from f3rm.features.clipsam_extract import CLIPSAMArgs, CLIPSAMExtractor, parse_clipsam_feature_type
from f3rm.features.foreground_extract import FOREGROUNDArgs, FOREGROUNDExtractor, parse_foreground_feature_type
from f3rm.features.orientany_extract import ORIENTANYArgs, ORIENTANYExtractor, parse_orientany_feature_type
from f3rm.features.text_extract import TextArgs, TextExtractor
from f3rm.features.utils import run_async_in_any_context, pack_auto_masks, unpack_auto_masks, BatchFeatureLoader, get_cache_paths


FEAT_TYPE_TO_ARGS = {
    "CLIP": CLIPArgs,
    "DINO": DINOArgs,
    "SAM2": SAM2Args,
    "CLIPSAM": CLIPSAMArgs,
    "TEXT": TextArgs,
    "FOREGROUND": FOREGROUNDArgs,
    "ORIENTANY": ORIENTANYArgs,
}

FEAT_TYPE_TO_EXTRACTOR_CLASS: Dict[str, Type] = {
    "CLIP": CLIPExtractor,
    "DINO": DINOExtractor,
    "SAM2": SAM2Extractor,
    "TEXT": TextExtractor,
}


async def _save_per_image_generic(
    image_fnames: List[str],
    data_dir: Path,
    feature_type: str,
    extractor_class: Type,
    args_cls: Any,
    device: torch.device,
    batch_size: int,
):
    root, meta = get_cache_paths(data_dir, feature_type)
    root.mkdir(parents=True, exist_ok=True)
    n_imgs = len(image_fnames)
    n_batches = math.ceil(n_imgs / batch_size)

    if feature_type.startswith("CLIPSAM_"):
        parsed_prompts = parse_clipsam_feature_type(feature_type)
        text_prompts_arg = None if (parsed_prompts is not None and len(parsed_prompts) == 0) else parsed_prompts
        CONSOLE.print(f"CLIPSAM parsed text prompts: {parsed_prompts} -> using {'TEXT shards' if text_prompts_arg is None else 'global prompts'}")
        extractor = CLIPSAMExtractor(device=device, data_dir=data_dir, text_prompts=text_prompts_arg, verbose=True)
    elif feature_type.startswith("FOREGROUND_"):
        parsed_prompts = parse_foreground_feature_type(feature_type)
        text_prompts_arg = None if (parsed_prompts is not None and len(parsed_prompts) == 0) else parsed_prompts
        CONSOLE.print(f"FOREGROUND parsed text prompts: {parsed_prompts} -> using {'TEXT shards' if text_prompts_arg is None else 'global prompts'}")
        extractor = FOREGROUNDExtractor(device=device, data_dir=data_dir, text_prompts=text_prompts_arg, verbose=True)
    elif feature_type.startswith("ORIENTANY_"):
        parsed_prompts = parse_orientany_feature_type(feature_type)
        text_prompts_arg = None if (parsed_prompts is not None and len(parsed_prompts) == 0) else parsed_prompts
        CONSOLE.print(f"ORIENTANY parsed text prompts: {parsed_prompts} -> using {'TEXT shards' if text_prompts_arg is None else 'global prompts'}")
        extractor = ORIENTANYExtractor(device=device, data_dir=data_dir, text_prompts=text_prompts_arg, verbose=True)
    elif feature_type == "TEXT":
        extractor = TextExtractor(device=device, verbose=True, data_dir=data_dir)
    else:
        if extractor_class is None:
            raise ValueError(f"No extractor class found for feature type: {feature_type}")
        extractor = extractor_class(device=device, verbose=True)

    # Extract features in batches, then save per-image files
    for i in tqdm(range(n_batches), desc=f"{feature_type}: extracting", position=0):
        s, e = i * batch_size, min((i + 1) * batch_size, n_imgs)
        batch_paths = image_fnames[s:e]
        data = await extractor.extract_batch_async(batch_paths)

        # Save each image's features individually
        for j, img_path in enumerate(batch_paths):
            img_idx = s + j

            if feature_type in ("CLIP", "DINO"):
                img_data = data[j].cpu().half()
                np.save(root / f"image_{img_idx:06d}.npy", img_data.numpy(), allow_pickle=False)
            elif feature_type.startswith("FOREGROUND_"):
                img_data = data[j].astype(np.float16)
                np.save(root / f"image_{img_idx:06d}.npy", img_data, allow_pickle=False)
            elif feature_type.startswith("ORIENTANY_"):
                pixel_data = data[j]['pixel_data'].astype(np.float16)
                instance_features = data[j]['instance_features']
                np.save(root / f"image_{img_idx:06d}_pixel.npy", pixel_data, allow_pickle=False)
                with open(root / f"image_{img_idx:06d}_instances.json", 'w') as f:
                    json.dump(instance_features, f, indent=2)
            elif feature_type.startswith("CLIPSAM_") or feature_type == "SAM2":
                packed_img = pack_auto_masks(data[j])
                np.savez_compressed(
                    root / f"image_{img_idx:06d}.npz",
                    **packed_img
                )
            elif feature_type == "TEXT":
                with open(root / f"image_{img_idx:06d}.json", 'w') as f:
                    json.dump(data[j], f, indent=2)

        del data
        torch.cuda.empty_cache()
        gc.collect()

    torch.save({"args": args_cls.id_dict(), "image_fnames": image_fnames}, meta)
    CONSOLE.print(f"Saved {feature_type} per-image features → {root}")


def feature_loader(image_fnames: List[str], extract_args, data_dir: Path, feature_type: str) -> Optional[BatchFeatureLoader]:
    """Check if cached features exist and return a BatchFeatureLoader."""
    root, meta = get_cache_paths(data_dir, feature_type)

    if not meta.exists():
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - Metadata file does not exist")
        return None

    md = torch.load(meta)

    # Check args match
    current_args = extract_args.id_dict()
    cached_args = md.get("args")
    args_match = cached_args == current_args

    # Check image filenames match
    cached_fnames = md.get("image_fnames")
    current_fnames_str = [str(fname) for fname in image_fnames]
    cached_fnames_str = [str(fname) for fname in cached_fnames] if cached_fnames else None
    fnames_match = cached_fnames_str == current_fnames_str

    if not args_match or not fnames_match:
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - {'Args' if not args_match else 'Filenames'} don't match")
        return None

    # Check per-image files exist
    sample_paths = []
    if feature_type.startswith("CLIPSAM_") or feature_type == "SAM2":
        sample_paths = [root / f"image_{i:06d}.npz" for i in range(min(3, len(image_fnames)))]
    elif feature_type == "TEXT":
        sample_paths = [root / f"image_{i:06d}.json" for i in range(min(3, len(image_fnames)))]
    elif feature_type.startswith("ORIENTANY_"):
        sample_paths = [root / f"image_{i:06d}_pixel.npy" for i in range(min(3, len(image_fnames)))]
    else:
        sample_paths = [root / f"image_{i:06d}.npy" for i in range(min(3, len(image_fnames)))]

    if not all(p.exists() for p in sample_paths):
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - Per-image files not found")
        return None

    CONSOLE.print(f"[DEBUG] {feature_type}: CACHE HIT - Using batch feature loader")
    return "BATCH_LOADER"  # Placeholder - will be replaced with actual loader in extract_features_for_dataset


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
    feature_type: Literal["CLIP", "DINO", "SAM2", "TEXT", "CLIPSAM_*", "FOREGROUND_*", "ORIENTANY_*"],
    device: torch.device,
    batch_size: int = 64,
    enable_cache: bool = True,
    force: bool = False,
    max_cpu_images: int = 128,
    max_gpu_images: int = 16,
) -> BatchFeatureLoader:
    """
    Extract features for a dataset with per-image storage system.

    Features are extracted in batches for efficiency, but stored as individual files per image.
    This allows for efficient batch loading during training while maintaining per-image granularity.

    Returns:
        BatchFeatureLoader for efficient batch loading during training
    """
    # Get args class - for CLIPSAM, always use CLIPSAMArgs regardless of specific feature type
    if feature_type.startswith("CLIPSAM_"):
        args = CLIPSAMArgs
    elif feature_type.startswith("FOREGROUND_"):
        args = FOREGROUNDArgs
    elif feature_type.startswith("ORIENTANY_"):
        args = ORIENTANYArgs
    else:
        args = FEAT_TYPE_TO_ARGS[feature_type]

    CONSOLE.print(f"[DEBUG] {feature_type}: enable_cache={enable_cache}, checking for cached features...")
    loader_result = (
        feature_loader(image_fnames, args, data_dir, feature_type)
        if enable_cache and not force
        else None
    )

    if loader_result is not None:
        CONSOLE.print(f"[{feature_type}] Using cached features")
        return BatchFeatureLoader(data_dir, feature_type, image_fnames, device, max_cpu_images=max_cpu_images, max_gpu_images=max_gpu_images)

    CONSOLE.print(f"[{feature_type}] Extracting features...")

    async def _run():
        await _save_per_image_generic(
            image_fnames=image_fnames,
            data_dir=data_dir,
            feature_type=feature_type,
            extractor_class=FEAT_TYPE_TO_EXTRACTOR_CLASS.get(feature_type, None),
            args_cls=args,
            device=device,
            batch_size=batch_size,
        )

    run_async_in_any_context(_run)

    # Return the batch loader
    return BatchFeatureLoader(data_dir, feature_type, image_fnames, device, max_cpu_images=max_cpu_images, max_gpu_images=max_gpu_images)


def extract_features_standalone(
    data_dir: Path,
    feature_type: Literal["CLIP", "DINO", "SAM2", "TEXT", "CLIPSAM_*", "FOREGROUND_*", "ORIENTANY_*"],
    batch_size: int = 64,
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
    batch_loader = extract_features_for_dataset(
        image_fnames=image_fnames,
        data_dir=data_dir,
        feature_type=feature_type,
        device=device,
        batch_size=batch_size,
        enable_cache=True,
        force=force,
    )

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    CONSOLE.print("Feature extraction completed!")
    return batch_loader  # Return for potential use in testing


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
        type=str,
        default="CLIP",
        help=(
            "Feature type to extract.\n"
            "- CLIPSAM: 'CLIPSAM_book' or 'CLIPSAM_book_pen' (global prompts), 'CLIPSAM_' (use TEXT shards).\n"
            "- FOREGROUND: 'FOREGROUND_book' or 'FOREGROUND_book_pen' (global), 'FOREGROUND_' (use TEXT shards).\n"
            "- ORIENTANY: 'ORIENTANY_book' or 'ORIENTANY_book_pen' (global), 'ORIENTANY_' (use TEXT shards).\n"
            "Examples: CLIP, DINO, SAM2, TEXT, CLIPSAM_book, CLIPSAM_, FOREGROUND_book, FOREGROUND_, ORIENTANY_book, ORIENTANY_."
        )
    )

    # Batch size for extraction processing (not storage sharding - features are saved per-image)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images to process in each extraction batch (for memory efficiency during extraction)"
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
        batch_size=args.batch_size,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
