import gc
import asyncio
import os
from typing import List, Optional

import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sam2.features.utils import AsyncMultiWrapper

from f3rm.features.utils import (
    BatchFeatureLoader,
    ensure_sam3_feature_cache,
    infer_sam3_feature_type,
    load_sam3_image_fnames,
    normalize_sam3_masks,
    parse_prefixed_feature_type,
    resolve_devices_and_workers,
    run_async_in_any_context,
)


class FOREGROUNDArgs:
    min_instance_percent: float = 1.0
    batch_size_per_gpu: int = 8

    @classmethod
    def id_dict(cls):
        return {
            "min_instance_percent": float(cls.min_instance_percent),
        }


def parse_foreground_feature_type(feature_type: str) -> List[str]:
    return parse_prefixed_feature_type(feature_type, "FOREGROUND_")


class FOREGROUNDWorker:
    def __init__(
        self,
        device: torch.device,
        data_dir: Path,
        sam3_feature_type: str = "SAM3_",
    ):
        self.device = device
        self.data_dir = Path(data_dir)
        self.sam3_feature_type = sam3_feature_type

        self.feat_image_fnames = load_sam3_image_fnames(self.data_dir, self.sam3_feature_type)
        sam3_image_fnames = self.feat_image_fnames
        self._image_to_index = {fname: i for i, fname in enumerate(self.feat_image_fnames)}
        self.sam3_loader = BatchFeatureLoader(self.data_dir, self.sam3_feature_type, sam3_image_fnames, device)

    @staticmethod
    def _filter_masks_by_area(masks: np.ndarray, min_instance_percent: float) -> np.ndarray:
        if masks.size == 0:
            return masks
        h, w = masks.shape[-2:]
        total_pixels = max(1, h * w)
        keep = []
        for mask in masks:
            if mask.shape != (h, w):
                continue
            percent = (float(mask.sum()) / float(total_pixels)) * 100.0
            if percent >= min_instance_percent:
                keep.append(mask)
        if not keep:
            return np.zeros((0, h, w), dtype=bool)
        return np.stack(keep, axis=0).astype(bool)

    def _compute_foreground_for_image(self, image_path: str) -> np.ndarray:
        try:
            idx = self._image_to_index[str(image_path)]
        except KeyError as exc:
            raise ValueError(f"Image path not found in SAM3 meta order: {image_path}") from exc

        raw_masks = self.sam3_loader[idx]
        masks = normalize_sam3_masks(raw_masks, image_path=image_path)
        masks = self._filter_masks_by_area(masks, FOREGROUNDArgs.min_instance_percent)

        if masks.size == 0:
            with Image.open(image_path) as img:
                h, w = img.height, img.width
            fg = np.zeros((h, w), dtype=bool)
            one_hot = np.stack([~fg, fg], axis=-1).astype(np.float16)
            return one_hot

        fg_mask = np.logical_or.reduce(masks, axis=0)
        one_hot = np.stack([~fg_mask, fg_mask], axis=-1).astype(np.float16)
        return one_hot

    async def compute_foreground_for_image_async(self, image_path: str) -> np.ndarray:
        return await asyncio.to_thread(self._compute_foreground_for_image, image_path)


class FOREGROUNDExtractor:
    def __init__(
        self,
        device: torch.device,
        data_dir: Optional[Path] = None,
        text_prompts: Optional[List[str]] = None,
        sam3_feature_type: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.verbose = verbose
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.sam3_feature_type = sam3_feature_type or infer_sam3_feature_type(text_prompts)

        if self.data_dir is None:
            raise ValueError("FOREGROUNDExtractor requires data_dir to locate precomputed SAM3 shards")

        ensure_sam3_feature_cache(self.data_dir, self.sam3_feature_type, consumer="FOREGROUND", require_npz=True)

        devices_param, num_workers = resolve_devices_and_workers(device, FOREGROUNDArgs.batch_size_per_gpu)
        if verbose:
            print(f"Initializing FOREGROUND workers (using {self.sam3_feature_type} masks)")
        self.client = AsyncMultiWrapper(
            FOREGROUNDWorker,
            num_objects=num_workers,
            devices=devices_param,
            data_dir=self.data_dir,
            sam3_feature_type=self.sam3_feature_type,
        )
        self.num_workers = num_workers

    async def extract_batch_async(self, image_paths: List[str]) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting FOREGROUND maps", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_foreground_async(path, self.client) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="FOREGROUND", leave=False)
            results.extend(batch_results)
            gc.collect()
        return results


async def extract_foreground_batch(image_paths: List[str], device: torch.device, data_dir: Path, verbose: bool = False, text_prompts: Optional[List[str]] = None):
    extractor = FOREGROUNDExtractor(device=device, data_dir=data_dir, text_prompts=text_prompts, verbose=verbose)
    return await extractor.extract_batch_async(image_paths)


async def process_single_image_foreground_async(image_path: str, fg_client: AsyncMultiWrapper) -> np.ndarray:
    return await fg_client.compute_foreground_for_image_async(image_path)


def examine_saved(foreground_feat_dir: str):
    """Create .mp4 video of saved FOREGROUND features with visualization."""
    meta_path = os.path.join(foreground_feat_dir, "meta.pt")
    assert os.path.exists(meta_path), f"FOREGROUND meta not found at {meta_path}"

    meta = torch.load(meta_path)
    image_fnames = meta["image_fnames"]
    n_images = len(image_fnames)

    # Load first image to get dimensions
    first_feat = np.load(os.path.join(foreground_feat_dir, "image_000000.npy"))
    H, W = first_feat.shape[:2]

    video_path = os.path.join(foreground_feat_dir, "features_viz.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 2.0, (W, H))

    for i in tqdm(range(n_images), desc="Creating FOREGROUND features video"):
        feat_path = os.path.join(foreground_feat_dir, f"image_{i:06d}.npy")
        feat = np.load(feat_path)  # Shape: (H, W, 2) - background/foreground one-hot

        # Visualize foreground channel (index 1)
        fg_map = feat[..., 1]  # Foreground channel
        frame = (fg_map * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()
    assert os.path.exists(video_path), f"Video not created at {video_path}"


if __name__ == "__main__":
    # examine_saved("datasets/f3rm/opt/objaverse/car2/features/foreground_")
    # examine_saved("datasets/f3rm/opt/objaverse/car2/features/foreground_car")

    data_root = Path("datasets/f3rm/opt/betaipad/small")
    image_dir = data_root / "images"
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    image_paths = [str(p) for p in image_paths[:8]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mode 1: use SAM3_ masks (derived from FOREGROUND_)
    extractor = FOREGROUNDExtractor(device=device, data_dir=data_root, text_prompts=None, verbose=True)
    maps = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))
    print(f"Mode 1 (SAM3_ masks): Extracted {len(maps)} maps, sample shape: {maps[0].shape if maps else None}")

    # Mode 2: use SAM3_<prompts> masks (e.g., SAM3_laptop inferred from FOREGROUND prompts)
    extractor2 = FOREGROUNDExtractor(device=device, data_dir=data_root, text_prompts=["laptop"], verbose=True)
    maps2 = run_async_in_any_context(lambda: extractor2.extract_batch_async(image_paths))
    print(f"Mode 2 (SAM3 prompt masks): Extracted {len(maps2)} maps, sample shape: {maps2[0].shape if maps2 else None}")

    # Visualize a few results (RGB, Mode1 FG, Mode2 FG)
    vis_count = min(3, len(image_paths))
    fig, axes = plt.subplots(3, vis_count, figsize=(4 * vis_count, 10))
    if vis_count == 1:
        axes = axes.reshape(3, 1)
    for i in range(vis_count):
        rgb = Image.open(image_paths[i]).convert("RGB")
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"RGB {i+1}")
        axes[0, i].axis('off')

        fg1 = maps[i][..., 1] if i < len(maps) else None
        fg2 = maps2[i][..., 1] if i < len(maps2) else None
        if fg1 is not None:
            axes[1, i].imshow(fg1, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title("Mode1 FG (SAM3_)")
        axes[1, i].axis('off')
        if fg2 is not None:
            axes[2, i].imshow(fg2, cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_title("Mode2 FG (SAM3 prompt)")
        axes[2, i].axis('off')
    plt.tight_layout()
    plt.show()
