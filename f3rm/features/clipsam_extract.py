import asyncio
import gc
import os
from typing import List, Optional

import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import glob
import math
import shutil
from tqdm.auto import tqdm

from sam2.features.utils import SAM2utils
from sam2.features.clip_main import CLIPfeatures
from f3rm.features.utils import run_async_in_any_context, resolve_devices_and_workers, visualize_auto_masks_demo, pack_auto_masks, unpack_auto_masks, BatchFeatureLoader
from f3rm.features.sam2_extract import SAM2Args
from sam2.features.utils import AsyncMultiWrapper


class CLIPSAMArgs:
    # CLIP x SAM2 filtering hyperparameters
    negative_texts: List[str] = ["object", "floor", "wall"]
    softmax_temp: float = 0.01
    top_mean_percent: float = 15.0
    sim_thresh: float = 0.7
    min_instance_percent: float = 1.0
    batch_size_per_gpu: int = 8

    @classmethod
    def id_dict(cls):
        # Persist configuration to meta for cache validation
        return {
            "negative_texts": list(cls.negative_texts),
            "softmax_temp": float(cls.softmax_temp),
            "top_mean_percent": float(cls.top_mean_percent),
            "sim_thresh": float(cls.sim_thresh),
            "min_instance_percent": float(cls.min_instance_percent),
        }


def parse_clipsam_feature_type(feature_type: str) -> List[str]:
    """Parse text prompts from feature type like 'CLIPSAM_book_pen' -> ['book', 'pen'].

    If feature_type is just 'CLIPSAM_', returns empty list to indicate use TEXT shards.
    """
    if not feature_type.startswith("CLIPSAM_"):
        raise ValueError(f"Invalid CLIPSAM feature type: {feature_type}. Must start with 'CLIPSAM_'")

    # Extract prompts after CLIPSAM_
    prompts_part = feature_type[8:]  # Remove "CLIPSAM_"
    if not prompts_part:
        return []  # Empty list means use TEXT shards

    # Split by underscore, force lowercase, and filter out empty strings
    text_prompts = [word.lower() for word in prompts_part.split("_") if word.strip()]
    return text_prompts


class CLIPSAMWorker:
    """Worker class for async CLIPSAM processing."""

    def __init__(self, device: torch.device, data_dir: Path, text_prompts: Optional[List[str]] = None):
        self.device = device
        self.data_dir = Path(data_dir)
        self.text_prompts = text_prompts  # If None, will load from TEXT shards per-image

        # Load precomputed features
        feat_root = self.data_dir / "features"
        clip_root = feat_root / "clip"
        sam2_root = feat_root / "sam2"
        text_root = feat_root / "text"

        # Assert that required shards exist
        clip_meta = torch.load(clip_root / "meta.pt")
        self.feat_image_fnames = [str(p) for p in clip_meta["image_fnames"]]

        # Load CLIP features
        # Create batch feature loaders for the new per-image system
        self.clip_loader = BatchFeatureLoader(self.data_dir, "CLIP", self.feat_image_fnames, device)
        self.sam2_loader = BatchFeatureLoader(self.data_dir, "SAM2", self.feat_image_fnames, device)

        # Load TEXT features only if we'll use them (when text_prompts is None)
        if self.text_prompts is None:
            self.text_loader = BatchFeatureLoader(self.data_dir, "TEXT", self.feat_image_fnames, device)
        else:
            self.text_loader = None

        self.clip_model = CLIPfeatures(device=self.device)

    def _filter_sam2_inst_mask(self, inst_mask: np.ndarray, min_instance_percent: float) -> np.ndarray:
        unique_ids = np.unique(inst_mask)
        total_pixels = inst_mask.size
        for inst_id in unique_ids:
            if inst_id > 0:
                seg = (inst_mask == inst_id)
                percent = (np.sum(seg) / total_pixels) * 100.0
                if percent < min_instance_percent:
                    inst_mask[seg] = 0
        return inst_mask

    def _pack_inst_mask_to_auto_masks(self, inst_mask: np.ndarray) -> list:
        packed = []
        for inst_id in np.unique(inst_mask):
            if inst_id <= 0:
                continue
            seg = (inst_mask == inst_id)
            if not np.any(seg):
                continue
            ys, xs = np.where(seg)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
            packed.append({
                "segmentation": seg,
                "bbox": bbox,
                "predicted_iou": np.float16("inf"),  # Use fp16 for VRAM efficiency
                "area": np.float16("inf"),           # Use fp16 for VRAM efficiency
            })
        return packed

    def _filter_auto_masks_for_image(self, image_path: str) -> List[dict]:
        # Map image → index (consistent with CLIP shard order)
        try:
            feat_image_index = self.feat_image_fnames.index(str(image_path))
        except ValueError:
            raise ValueError(f"Image path not found in CLIP meta order: {image_path}")

        clip_patch_feats = self.clip_loader[feat_image_index]
        raw_auto_masks = self.sam2_loader[feat_image_index]

        # Get text prompts - either from TEXT per-image files or from global list
        if self.text_prompts is None:
            # Load text prompts from pre-extracted TEXT features (per-image)
            text_prompts = self.text_loader[feat_image_index]
        else:
            # Use globally supplied text prompts
            text_prompts = self.text_prompts

        if not raw_auto_masks or not text_prompts:
            return []

        # Convert to instance mask, filter by min-instance-percent, and pack back
        inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
            raw_auto_masks,
            min_iou=float(SAM2Args.pred_iou_thresh) if SAM2Args.pred_iou_thresh is not None else 0.0,
            min_area=float(SAM2Args.min_mask_region_area) if SAM2Args.min_mask_region_area is not None else 0.0,
            assign_by="area",
            start_from="low",
        )
        if inst_mask is None:
            # No valid masks found, create empty instance mask
            if raw_auto_masks:
                h, w = raw_auto_masks[0]['segmentation'].shape
            else:
                # Use actual image dimensions as fallback
                img = Image.open(image_path)
                h, w = img.height, img.width
            inst_mask = np.zeros((h, w), dtype=np.uint16)
        inst_mask = self._filter_sam2_inst_mask(inst_mask, CLIPSAMArgs.min_instance_percent)
        auto_masks = self._pack_inst_mask_to_auto_masks(inst_mask)
        if not auto_masks:
            return []

        h, w = auto_masks[0]["segmentation"].shape

        # Per-prompt segment similarity maps (mean of top-K% pixels per mask)
        segment_sim_maps: List[np.ndarray] = []
        for text in text_prompts:
            text_emb = self.clip_model.encode_text(text).half()
            neg_text_embs = torch.stack([self.clip_model.encode_text(neg).half() for neg in CLIPSAMArgs.negative_texts], dim=0)
            sim_map = self.clip_model.compute_similarity(
                clip_patch_feats,
                text_emb,
                neg_text_embs=neg_text_embs,
                softmax_temp=CLIPSAMArgs.softmax_temp,
                normalize=True,
            )
            sim_map_up = np.array(Image.fromarray(sim_map.cpu().float().numpy()).resize((w, h), Image.BILINEAR)).astype(np.float16)

            seg_map = np.zeros_like(sim_map_up)
            for m in auto_masks:
                seg = m["segmentation"]
                if seg.shape != (h, w):
                    continue
                vals = sim_map_up[seg]
                k = max(1, int(len(vals) * CLIPSAMArgs.top_mean_percent / 100.0))
                seg_map[seg] = float(np.mean(np.sort(vals)[-k:]))
            segment_sim_maps.append(seg_map)

        combined_sim = np.maximum.reduce(segment_sim_maps) if len(segment_sim_maps) > 0 else np.zeros((h, w))

        # Similarity thresholding (keep masks where any pixel exceeds threshold)
        filtered: List[dict] = []
        for m in auto_masks:
            seg = m["segmentation"]
            if seg.shape != (h, w):
                continue
            if np.any(combined_sim[seg] > CLIPSAMArgs.sim_thresh):
                filtered.append({
                    "segmentation": seg,
                    "bbox": m["bbox"],
                    "predicted_iou": np.float16(m.get("predicted_iou", float("inf"))),  # Use fp16 for VRAM efficiency
                    "area": np.float16(m.get("area", float("inf"))),                   # Use fp16 for VRAM efficiency
                })

        return filtered

    async def filter_auto_masks_for_image_async(self, image_path: str) -> List[dict]:
        return await asyncio.to_thread(self._filter_auto_masks_for_image, image_path)


class CLIPSAMExtractor:
    def __init__(self, device: torch.device, data_dir: Optional[Path] = None, text_prompts: Optional[List[str]] = None, verbose: bool = False) -> None:
        self.device = device
        self.verbose = verbose
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.text_prompts = text_prompts

        if self.data_dir is None:
            raise ValueError("CLIPSAMExtractor requires data_dir to locate precomputed CLIP and SAM2 shards")

        # Assert prerequisites exist
        feat_root = self.data_dir / "features"
        clip_root = feat_root / "clip"
        sam2_root = feat_root / "sam2"
        text_root = feat_root / "text"

        clip_meta_path = clip_root / "meta.pt"
        if not clip_meta_path.exists():
            raise FileNotFoundError(f"Missing CLIP meta: {clip_meta_path}")
        if not list(clip_root.glob("image_*.npy")):
            raise FileNotFoundError(f"Missing CLIP per-image features under {clip_root}")
        if not list(sam2_root.glob("image_*.npz")):
            raise FileNotFoundError(f"Missing SAM2 per-image features under {sam2_root}")

        # TEXT per-image files only required when using per-image text prompts (text_prompts is None)
        if text_prompts is None and not list(text_root.glob("image_*.json")):
            raise FileNotFoundError(f"Missing TEXT per-image features under {text_root} (required when using per-image text prompts)")

        # Multi-GPU setup
        devices_param, num_workers = resolve_devices_and_workers(device, CLIPSAMArgs.batch_size_per_gpu)
        if verbose:
            print("Initializing CLIPSAM workers")
        self.client = AsyncMultiWrapper(CLIPSAMWorker, num_objects=num_workers, devices=devices_param, data_dir=self.data_dir, text_prompts=self.text_prompts)
        self.num_workers = num_workers

        # Warm up workers with a proper test
        if verbose:
            print("Warming up CLIPSAM workers...")
        # Load meta to get a valid image path for warm-up
        clip_meta = torch.load(clip_meta_path)
        if clip_meta["image_fnames"]:
            warmup_path = str(clip_meta["image_fnames"][0])
            for _ in range(self.num_workers):
                try:
                    # Use synchronous warm-up since we're not in async context
                    _ = self.client._filter_auto_masks_for_image(warmup_path)
                except Exception:
                    pass  # Warm-up may fail, that's okay

    async def extract_batch_async(self, image_paths: List[str]) -> List[List[dict]]:
        results: List[List[dict]] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Processing & extracting CLIPSAM auto-masks", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_clipsam_async(path, self.client) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="CLIPSAM auto_mask", leave=False)
            results.extend(batch_results)
            gc.collect()
        return results


async def extract_clipsam_batch(image_paths: List[str], device: torch.device, data_dir: Path, verbose: bool = False, text_prompts: Optional[List[str]] = None):
    extractor = CLIPSAMExtractor(device=device, data_dir=data_dir, text_prompts=text_prompts, verbose=verbose)
    return await extractor.extract_batch_async(image_paths)


async def process_single_image_clipsam_async(image_path: str, clipsam_client: AsyncMultiWrapper) -> List[dict]:
    """Async wrapper for single image CLIPSAM processing."""
    return await clipsam_client.filter_auto_masks_for_image_async(image_path)


def examine_saved(clipsam_feat_dir: str):
    """Create .mp4 video of saved CLIPSAM features with mask visualization."""
    meta_path = os.path.join(clipsam_feat_dir, "meta.pt")
    assert os.path.exists(meta_path), f"CLIPSAM meta not found at {meta_path}"

    meta = torch.load(meta_path)
    image_fnames = meta["image_fnames"]
    n_images = len(image_fnames)

    # Load first image to get dimensions
    first_data = np.load(os.path.join(clipsam_feat_dir, "image_000000.npz"))
    packed_data = {
        "num_masks": int(first_data['num_masks']),
        "mask_data": first_data['mask_data'],
        "mask_shapes": first_data['mask_shapes'],
        "bbox_data": first_data['bbox_data'],
        "pred_iou_data": first_data['pred_iou_data'],
        "area_data": first_data['area_data']
    }
    first_masks = unpack_auto_masks(packed_data)
    H, W = first_masks[0]["segmentation"].shape

    video_path = os.path.join(clipsam_feat_dir, "features_viz.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 2.0, (W, H))

    for i in tqdm(range(n_images), desc="Creating CLIPSAM features video"):
        feat_path = os.path.join(clipsam_feat_dir, f"image_{i:06d}.npz")
        data = np.load(feat_path)
        packed_data = {
            "num_masks": int(data['num_masks']),
            "mask_data": data['mask_data'],
            "mask_shapes": data['mask_shapes'],
            "bbox_data": data['bbox_data'],
            "pred_iou_data": data['pred_iou_data'],
            "area_data": data['area_data']
        }
        auto_masks = unpack_auto_masks(packed_data)

        # Create visualization
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(auto_masks))))
        for j, mask in enumerate(auto_masks):
            seg = mask["segmentation"]
            if seg.shape == (H, W):
                color = (colors[j][:3] * 255).astype(np.uint8)  # Take only RGB, ignore alpha
                frame[seg] = color

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    assert os.path.exists(video_path), f"Video not created at {video_path}"


if __name__ == "__main__":
    # examine_saved("datasets/f3rm/opt/objaverse/car2/features/clipsam_")
    # examine_saved("datasets/f3rm/opt/objaverse/car2/features/clipsam_car")

    # Demo: Full pipeline (extract -> save -> load -> visualize) for CLIPSAM
    data_root = Path("datasets/f3rm/opt/caterpillar")
    image_dir = data_root / "images"
    image_paths = sorted(glob.glob(str(image_dir / "*.jpg")) + glob.glob(str(image_dir / "*.png")))
    image_paths = image_paths[:10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Found {len(image_paths)} images in {image_dir}")
    print("Extracting CLIPSAM features with async multi-GPU processing...")

    # Demo both modes:
    # Mode 1: Use pre-extracted TEXT features (CLIPSAM_)
    print("Mode 1: Using pre-extracted TEXT features")
    extractor = CLIPSAMExtractor(device=device, data_dir=data_root, text_prompts=None, verbose=True)
    batch_masks = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))

    # Mode 2: Use hardcoded prompts (CLIPSAM_book)
    print("Mode 2: Using hardcoded prompts")
    extractor2 = CLIPSAMExtractor(device=device, data_dir=data_root, text_prompts=["toy"], verbose=True)
    batch_masks2 = run_async_in_any_context(lambda: extractor2.extract_batch_async(image_paths))

    # Use the first mode for visualization
    batch_masks = batch_masks2

    print(f"Extracted CLIPSAM auto-masks for {len(batch_masks)} images. Visualizing results...")
    visualize_auto_masks_demo(batch_masks, image_paths, "CLIPSAM", max_vis=4, pred_iou_thresh=SAM2Args.pred_iou_thresh, min_mask_region_area=SAM2Args.min_mask_region_area)

    # Demo 2: Test the full pipeline (extract -> save -> load -> visualize)
    print("\n" + "=" * 60)
    print("DEMO 2: Full pipeline test (extract -> save -> load -> visualize)")
    print("=" * 60)

    # Setup paths
    test_dir = Path("test_clipsam_pipeline")
    test_dir.mkdir(exist_ok=True)

    # Save per-image files (following new per-image system)
    print("Saving per-image files...")
    for i, auto_masks in enumerate(batch_masks):
        # Pack data using consolidated function
        packed_data = pack_auto_masks(auto_masks)

        # Save per-image file
        np.savez_compressed(
            test_dir / f"image_{i:06d}.npz",
            **packed_data
        )

    # Load per-image files using new system
    print("Loading per-image files...")
    loaded_masks = []
    for i in range(len(batch_masks)):
        image_file = test_dir / f"image_{i:06d}.npz"
        if image_file.exists():
            data = np.load(image_file)
            # Use the same unpacking function as the main system
            packed_data = {
                "num_masks": int(data['num_masks']),
                "mask_data": data['mask_data'],
                "mask_shapes": data['mask_shapes'],
                "bbox_data": data['bbox_data'],
                "pred_iou_data": data['pred_iou_data'],
                "area_data": data['area_data']
            }
            auto_masks = unpack_auto_masks(packed_data)
            loaded_masks.append(auto_masks)
        else:
            loaded_masks.append([])

    print(f"Loaded {len(loaded_masks)} images from per-image files")

    # Visualize loaded data
    print("Visualizing loaded data...")
    visualize_auto_masks_demo(loaded_masks, image_paths, "Loaded CLIPSAM", max_vis=3, pred_iou_thresh=SAM2Args.pred_iou_thresh, min_mask_region_area=SAM2Args.min_mask_region_area)

    # Print summary
    print("\nExtracted auto-masks for demo:")
    for i in range(min(3, len(batch_masks))):
        original_count = len(batch_masks[i])
        loaded_count = len(loaded_masks[i]) if i < len(loaded_masks) else 0
        print(f"Image {i}: {original_count} original masks, {loaded_count} loaded masks")

    # Cleanup
    plt.close('all')
    gc.collect()
    shutil.rmtree(test_dir, ignore_errors=True)
    print(f"Cleaned up test directory: {test_dir}")
