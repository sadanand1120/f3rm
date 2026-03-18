#!/usr/bin/env python3
"""
Standalone Feature Extraction Script

Extracts CLIP features for a dataset and saves them as per-image files for efficient
batch loading during training.
"""

import gc
import math
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from nerfstudio.utils.rich_utils import CONSOLE

from f3rm.features.clip_extract import CLIPArgs, CLIPExtractor, examine_saved
from f3rm.features.utils import BatchFeatureLoader, get_cache_paths


def _current_feature_args_id(feature_type: str) -> dict[str, Any]:
    if feature_type != "CLIP":
        raise ValueError(f"Unsupported feature type: {feature_type}")
    return {
        "model_name": CLIPArgs.model_name,
        "model_pretrained": CLIPArgs.model_pretrained,
        "load_size": CLIPArgs.load_size,
        "skip_center_crop": CLIPArgs.skip_center_crop,
        "agg_scales": CLIPArgs.agg_scales,
        "agg_weights": CLIPArgs.agg_weights,
    }


def create_feature_visualization(data_dir: Path, feature_type: str):
    if feature_type != "CLIP":
        raise ValueError(f"Unsupported feature type: {feature_type}")

    feat_dir = data_dir / "features" / feature_type.lower()
    if not feat_dir.exists():
        CONSOLE.print(f"[yellow]Feature directory not found: {feat_dir}")
        return

    CONSOLE.print(f"[blue]Creating visualization video for {feature_type}...")
    examine_saved(str(feat_dir))
    CONSOLE.print(f"[green]Video saved: {feat_dir}/features_viz.mp4")


def _save_per_image_clip(
    image_fnames: List[str],
    data_dir: Path,
    device: torch.device,
    batch_size: int,
):
    from tqdm.auto import tqdm

    root, meta = get_cache_paths(data_dir, "CLIP")
    root.mkdir(parents=True, exist_ok=True)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    extractor = CLIPExtractor(device=device, verbose=True)
    n_imgs = len(image_fnames)
    n_batches = math.ceil(n_imgs / batch_size)

    for i in tqdm(range(n_batches), desc="CLIP: extracting", position=0):
        s, e = i * batch_size, min((i + 1) * batch_size, n_imgs)
        batch_paths = image_fnames[s:e]
        data = extractor.extract_batch(batch_paths)

        for j in range(len(batch_paths)):
            img_idx = s + j
            img_data = data[j].cpu().half()
            np.save(root / f"image_{img_idx:06d}.npy", img_data.numpy(), allow_pickle=False)

        del data
        if torch.cuda.is_available() and ((i + 1) % 10 == 0 or i == n_batches - 1):
            torch.cuda.empty_cache()
            gc.collect()

    normalized_image_fnames = [_normalize_image_path_for_cache(str(fname), data_dir) for fname in image_fnames]
    torch.save(
        {
            "args": CLIPArgs.id_dict(),
            "image_fnames": image_fnames,
            "normalized_image_fnames": normalized_image_fnames,
        },
        meta,
    )
    CONSOLE.print(f"Saved CLIP per-image features -> {root}")


def _cache_file_count_matches(root: Path, feature_type: str, num_images: int) -> bool:
    if feature_type != "CLIP":
        raise ValueError(f"Unsupported feature type: {feature_type}")
    if num_images == 0:
        return True
    return len(list(root.glob("image_*.npy"))) == num_images


def _normalize_image_path_for_cache(path_like: str, data_dir: Path) -> str:
    p = Path(str(path_like))
    if p.is_absolute():
        return str(p.resolve())

    bases = [Path.cwd(), data_dir, *data_dir.parents]
    for base in bases:
        candidate = base / p
        if candidate.exists():
            return str(candidate.resolve())
    return str((Path.cwd() / p).resolve())


def feature_loader(image_fnames: List[str], current_args: dict[str, Any], data_dir: Path, feature_type: str) -> bool:
    if feature_type != "CLIP":
        raise ValueError(f"Unsupported feature type: {feature_type}")

    root, meta = get_cache_paths(data_dir, feature_type)
    if not meta.exists():
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - Metadata file does not exist")
        return False

    try:
        md = torch.load(meta, map_location="cpu")
    except Exception as exc:
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - Failed reading metadata ({exc})")
        return False

    cached_args = md.get("args")
    args_match = cached_args == current_args

    cached_fnames = md.get("image_fnames")
    current_fnames_raw = [str(fname) for fname in image_fnames]
    if cached_fnames == current_fnames_raw:
        fnames_match = True
    else:
        current_fnames_str = [_normalize_image_path_for_cache(fname, data_dir) for fname in current_fnames_raw]
        cached_fnames_str = md.get("normalized_image_fnames")
        if cached_fnames_str is None and cached_fnames:
            cached_fnames_str = [_normalize_image_path_for_cache(str(fname), data_dir) for fname in cached_fnames]
        fnames_match = cached_fnames_str == current_fnames_str

    if not args_match or not fnames_match:
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - {'Args' if not args_match else 'Filenames'} don't match")
        return False

    if not _cache_file_count_matches(root, feature_type, len(image_fnames)):
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - Per-image file count mismatch")
        return False

    CONSOLE.print(f"[DEBUG] {feature_type}: CACHE HIT - Using batch feature loader")
    return True


def get_image_filenames_from_dataparser(data_dir: Path) -> List[str]:
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

    dataparser_config = NerfstudioDataParserConfig(
        data=data_dir,
        train_split_fraction=0.95,
    )
    dataparser = dataparser_config.setup()

    train_dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
    test_dataparser_outputs = dataparser.get_dataparser_outputs(split="val")

    train_image_filenames = [str(path) for path in train_dataparser_outputs.image_filenames]
    test_image_filenames = [str(path) for path in test_dataparser_outputs.image_filenames]
    all_image_filenames = train_image_filenames + test_image_filenames

    CONSOLE.print(f"Found {len(train_image_filenames)} train images and {len(test_image_filenames)} test images")
    return all_image_filenames


def extract_features_for_dataset(
    image_fnames: List[str],
    data_dir: Path,
    feature_type: str,
    device: torch.device,
    batch_size: int = 64,
    enable_cache: bool = True,
    pin_cpu_tensors: bool = True,
    force: bool = False,
    max_cpu_images: int = 128,
    max_gpu_images: int = 16,
) -> BatchFeatureLoader:
    if feature_type != "CLIP":
        raise ValueError(f"Unsupported feature type: {feature_type}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    CONSOLE.print(f"[DEBUG] {feature_type}: enable_cache={enable_cache}, checking for cached features...")
    current_args = _current_feature_args_id(feature_type)
    cache_hit = feature_loader(image_fnames, current_args, data_dir, feature_type) if enable_cache and not force else False

    if cache_hit:
        CONSOLE.print(f"[{feature_type}] Using cached features")
        return BatchFeatureLoader(
            data_dir,
            feature_type,
            image_fnames,
            device,
            max_cpu_images=max_cpu_images,
            max_gpu_images=max_gpu_images,
            pin_cpu_tensors=pin_cpu_tensors,
        )

    CONSOLE.print(f"[{feature_type}] Extracting features...")
    _save_per_image_clip(
        image_fnames=image_fnames,
        data_dir=data_dir,
        device=device,
        batch_size=batch_size,
    )

    return BatchFeatureLoader(
        data_dir,
        feature_type,
        image_fnames,
        device,
        max_cpu_images=max_cpu_images,
        max_gpu_images=max_gpu_images,
        pin_cpu_tensors=pin_cpu_tensors,
    )


def extract_features_standalone(
    data_dir: Path,
    feature_type: str = "CLIP",
    batch_size: int = 64,
    device: str = "auto",
    force: bool = False,
) -> BatchFeatureLoader:
    if feature_type != "CLIP":
        raise ValueError(f"Unsupported feature type: {feature_type}")

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    CONSOLE.print(f"Using device: {device}")
    image_fnames = get_image_filenames_from_dataparser(data_dir)

    batch_loader = extract_features_for_dataset(
        image_fnames=image_fnames,
        data_dir=data_dir,
        feature_type=feature_type,
        device=device,
        batch_size=batch_size,
        enable_cache=True,
        force=force,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    CONSOLE.print("Feature extraction completed!")
    return batch_loader


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract CLIP features for F3RM training independently of the training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the dataset directory (same as used in training)",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="CLIP",
        help="Feature type to extract. Only CLIP is supported.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images to process in each extraction batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu, cuda:0, etc.)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cache exists",
    )

    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data}")

    transforms_json = args.data / "transforms.json"
    if not transforms_json.exists():
        raise FileNotFoundError(f"transforms.json not found in {args.data}. Make sure this is a processed nerfstudio dataset.")

    CONSOLE.print(f"Extracting {args.feature_type} features from {args.data}")

    extract_features_standalone(
        data_dir=args.data,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        device=args.device,
        force=args.force,
    )

    CONSOLE.print("Creating visualization video...")
    create_feature_visualization(args.data, args.feature_type)


if __name__ == "__main__":
    main()
