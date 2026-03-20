#!/usr/bin/env python3
"""
Standalone feature extraction entrypoint.

This file remains the dataset-level abstraction boundary so new extractor backends
can be added without changing training-time loading code.
"""

import gc
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from nerfstudio.utils.rich_utils import CONSOLE

from f3rm.features.clip_extract import CLIPArgs, CLIPExtractor, examine_saved as examine_saved_clip
from f3rm.features.sam_extract import SAMArgs, SAMExtractor, examine_saved as examine_saved_sam
from f3rm.features.utils import BatchFeatureLoader, get_cache_paths


@dataclass(frozen=True)
class FeatureSpec:
    args_cls: Any
    extractor_cls: Any
    examine_saved: Callable[[str], None]
    output_dtype: Optional[torch.dtype] = None


FEATURE_SPECS = {
    "CLIP": FeatureSpec(
        args_cls=CLIPArgs,
        extractor_cls=CLIPExtractor,
        examine_saved=examine_saved_clip,
        output_dtype=torch.float16,
    ),
    "SAM": FeatureSpec(
        args_cls=SAMArgs,
        extractor_cls=SAMExtractor,
        examine_saved=examine_saved_sam,
    ),
}


def _resolve_feature_spec(feature_type: str) -> tuple[str, FeatureSpec]:
    normalized_feature_type = feature_type.upper()
    spec = FEATURE_SPECS.get(normalized_feature_type)
    if spec is None:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    return normalized_feature_type, spec


def _save_per_image_features(
    image_fnames: List[str],
    data_dir: Path,
    feature_type: str,
    device: torch.device,
    batch_size: int,
    write_threads: int = 1,
) -> None:
    import concurrent.futures
    from tqdm.auto import tqdm

    feature_type, spec = _resolve_feature_spec(feature_type)
    root, meta = get_cache_paths(data_dir, feature_type)
    root.mkdir(parents=True, exist_ok=True)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if write_threads <= 0:
        raise ValueError(f"write_threads must be > 0, got {write_threads}")

    n_imgs = len(image_fnames)
    n_batches = math.ceil(n_imgs / batch_size)
    extractor = spec.extractor_cls(device=device, verbose=True)

    def _save_one_image(img_idx: int, img_tensor: torch.Tensor) -> None:
        np.save(root / f"image_{img_idx:06d}.npy", img_tensor.numpy(), allow_pickle=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=write_threads) as executor:
        for batch_idx in tqdm(range(n_batches), desc=f"{feature_type}: extracting", position=0):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_imgs)
            batch_paths = image_fnames[start:end]
            pending_writes = []

            def _submit_write(local_idx: int, img_tensor: torch.Tensor) -> None:
                pending_writes.append(executor.submit(_save_one_image, start + local_idx, img_tensor))

            extractor.stream_batch(batch_paths, on_result=_submit_write, output_dtype=spec.output_dtype)
            for future in pending_writes:
                future.result()

            if torch.cuda.is_available() and ((batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1):
                torch.cuda.empty_cache()
                gc.collect()

    normalized_image_fnames = [_normalize_image_path_for_cache(str(fname), data_dir) for fname in image_fnames]
    torch.save(
        {
            "args": spec.args_cls.id_dict(),
            "image_fnames": image_fnames,
            "normalized_image_fnames": normalized_image_fnames,
        },
        meta,
    )
    CONSOLE.print(f"Saved {feature_type} per-image features -> {root}")


def _cache_file_count_matches(root: Path, num_images: int) -> bool:
    return num_images == 0 or len(list(root.glob("image_*.npy"))) == num_images


def create_feature_visualization(data_dir: Path, feature_type: str) -> None:
    feature_type, spec = _resolve_feature_spec(feature_type)
    feat_dir = data_dir / "features" / feature_type.lower()
    if not feat_dir.exists():
        CONSOLE.print(f"[yellow]Feature directory not found: {feat_dir}")
        return
    CONSOLE.print(f"[blue]Creating visualization video for {feature_type}...")
    spec.examine_saved(str(feat_dir))
    CONSOLE.print(f"[green]Video saved: {feat_dir / 'features_viz.mp4'}")


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


def feature_cache_matches(image_fnames: List[str], current_args: dict[str, Any], data_dir: Path, feature_type: str) -> bool:
    feature_type, _ = _resolve_feature_spec(feature_type)
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
    if not _cache_file_count_matches(root, len(image_fnames)):
        CONSOLE.print(f"[DEBUG] {feature_type}: CACHE MISS - Per-image file count mismatch")
        return False

    CONSOLE.print(f"[DEBUG] {feature_type}: CACHE HIT - Using batch feature loader")
    return True


def get_image_filenames_from_dataparser(data_dir: Path) -> List[str]:
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

    dataparser_config = NerfstudioDataParserConfig(data=data_dir, train_split_fraction=0.95)
    dataparser = dataparser_config.setup()

    train_dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
    test_dataparser_outputs = dataparser.get_dataparser_outputs(split="val")

    train_image_filenames = [str(path) for path in train_dataparser_outputs.image_filenames]
    test_image_filenames = [str(path) for path in test_dataparser_outputs.image_filenames]
    CONSOLE.print(f"Found {len(train_image_filenames)} train images and {len(test_image_filenames)} test images")
    return train_image_filenames + test_image_filenames


def extract_features_for_dataset(
    image_fnames: List[str],
    data_dir: Path,
    feature_type: str,
    device: torch.device,
    batch_size: int = 64,
    write_threads: int = 1,
    enable_cache: bool = True,
    pin_cpu_tensors: bool = True,
    force: bool = False,
    max_cpu_images: int = 128,
    max_gpu_images: int = 16,
) -> BatchFeatureLoader:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    feature_type, spec = _resolve_feature_spec(feature_type)
    CONSOLE.print(f"[DEBUG] {feature_type}: enable_cache={enable_cache}, checking for cached features...")
    current_args = spec.args_cls.id_dict()
    cache_hit = feature_cache_matches(image_fnames, current_args, data_dir, feature_type) if enable_cache and not force else False

    if not cache_hit:
        CONSOLE.print(f"[{feature_type}] Extracting features...")
        _save_per_image_features(
            image_fnames=image_fnames,
            data_dir=data_dir,
            feature_type=feature_type,
            device=device,
            batch_size=batch_size,
            write_threads=write_threads,
        )
    else:
        CONSOLE.print(f"[{feature_type}] Using cached features")

    return BatchFeatureLoader(
        data_dir,
        feature_type,
        device,
        max_cpu_images=max_cpu_images,
        max_gpu_images=max_gpu_images,
        pin_cpu_tensors=pin_cpu_tensors,
    )


def extract_features_standalone(
    data_dir: Path,
    feature_type: str = "CLIP",
    batch_size: int = 64,
    write_threads: int = 1,
    device: str = "auto",
    force: bool = False,
    skip_visualization: bool = False,
) -> BatchFeatureLoader:
    feature_type, _ = _resolve_feature_spec(feature_type)
    resolved_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)

    CONSOLE.print(f"Using device: {resolved_device}")
    image_fnames = get_image_filenames_from_dataparser(data_dir)
    batch_loader = extract_features_for_dataset(
        image_fnames=image_fnames,
        data_dir=data_dir,
        feature_type=feature_type,
        device=resolved_device,
        batch_size=batch_size,
        write_threads=write_threads,
        enable_cache=True,
        force=force,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    if not skip_visualization:
        create_feature_visualization(data_dir, feature_type)
    CONSOLE.print("Feature extraction completed!")
    return batch_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract dense features for F3RM training independently of the training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset directory (same as used in training)")
    parser.add_argument("--feature-type", type=str, default="CLIP", choices=sorted(FEATURE_SPECS), help="Feature type to extract.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of images to process in each extraction batch.")
    parser.add_argument("--write-threads", type=int, default=1, help="Number of CPU threads used to write extracted feature files.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu, cuda:0, etc.)")
    parser.add_argument("--force", action="store_true", help="Force re-extraction even if cache exists")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip feature visualization video generation.")
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
        write_threads=args.write_threads,
        device=args.device,
        force=args.force,
        skip_visualization=args.skip_visualization,
    )
