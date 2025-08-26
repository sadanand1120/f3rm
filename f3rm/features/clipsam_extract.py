import asyncio
import gc
from typing import List, Optional

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import glob
import math
import shutil
from tqdm.auto import tqdm

from sam2.features.utils import SAM2utils
from sam2.features.clip_main import CLIPfeatures
from f3rm.features.utils import LazyFeatures, SAM2LazyAutoMasks, TextLazyFeatures, pack_batch_auto_masks, run_async_in_any_context, resolve_devices_and_workers
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
        clip_shards = sorted(clip_root.glob("chunk_*.npy"))
        assert clip_shards, f"No CLIP shards found in {clip_root}"
        self.clip_features = LazyFeatures(clip_shards)

        # Load SAM2 masks
        sam2_shards = sorted(sam2_root.glob("chunk_*.npz"))
        assert sam2_shards, f"No SAM2 shards found in {sam2_root}"
        self.sam2_masks = SAM2LazyAutoMasks(sam2_shards)

        # Load TEXT features only if we'll use them (when text_prompts is None)
        if self.text_prompts is None:
            text_shards = sorted(text_root.glob("chunk_*.json"))
            assert text_shards, f"No TEXT shards found in {text_root}"
            self.text_features = TextLazyFeatures(text_shards)
        else:
            self.text_features = None

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
                "predicted_iou": float("inf"),
                "area": float("inf"),
            })
        return packed

    async def filter_auto_masks_for_image_async(self, image_path: str) -> List[dict]:
        """Async version for worker."""
        # Map image → index (consistent with CLIP shard order)
        try:
            feat_image_index = self.feat_image_fnames.index(str(image_path))
        except ValueError:
            raise ValueError(f"Image path not found in CLIP meta order: {image_path}")

        clip_patch_feats = self.clip_features[feat_image_index].to(self.device)
        raw_auto_masks = self.sam2_masks[feat_image_index]

        # Get text prompts - either from TEXT shards (per-image) or from global list
        if self.text_prompts is None:
            # Load text prompts from pre-extracted TEXT features (per-image)
            text_prompts = self.text_features[feat_image_index]
        else:
            # Use globally supplied text prompts
            text_prompts = self.text_prompts

        if not raw_auto_masks or not text_prompts:
            return []

        # Convert to instance mask, filter by min-instance-percent, and pack back
        inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
            raw_auto_masks,
            min_iou=float(SAM2Args.pred_iou_thresh),
            min_area=float(SAM2Args.min_mask_region_area),
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
            text_emb = self.clip_model.encode_text(text)
            neg_text_embs = torch.stack([self.clip_model.encode_text(neg) for neg in CLIPSAMArgs.negative_texts], dim=0)
            sim_map = self.clip_model.compute_similarity(
                clip_patch_feats,
                text_emb,
                neg_text_embs=neg_text_embs,
                softmax_temp=CLIPSAMArgs.softmax_temp,
                normalize=True,
            )
            sim_map_up = np.array(Image.fromarray(sim_map.cpu().numpy()).resize((w, h), Image.BILINEAR))

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
                    "predicted_iou": m.get("predicted_iou", float("inf")),
                    "area": m.get("area", float("inf")),
                })

        return filtered


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
        if not list(clip_root.glob("chunk_*.npy")):
            raise FileNotFoundError(f"Missing CLIP shards under {clip_root}")
        if not list(sam2_root.glob("chunk_*.npz")):
            raise FileNotFoundError(f"Missing SAM2 shards under {sam2_root}")

        # TEXT shards only required when using per-image text prompts (text_prompts is None)
        if text_prompts is None and not list(text_root.glob("chunk_*.json")):
            raise FileNotFoundError(f"Missing TEXT shards under {text_root} (required when using per-image text prompts)")

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
                    _ = self.client.filter_auto_masks_for_image_async(warmup_path)
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


def make_clipsam_extractor(device: torch.device, verbose: bool = False, data_dir: Optional[Path] = None, text_prompts: Optional[List[str]] = None) -> "CLIPSAMExtractor":
    if data_dir is None:
        raise ValueError("make_clipsam_extractor requires data_dir")
    return CLIPSAMExtractor(device=device, data_dir=data_dir, text_prompts=text_prompts, verbose=verbose)


async def extract_clipsam_batch(image_paths: List[str], device: torch.device, data_dir: Path, verbose: bool = False, text_prompts: Optional[List[str]] = None):
    extractor = make_clipsam_extractor(device=device, verbose=verbose, data_dir=data_dir, text_prompts=text_prompts)
    return await extractor.extract_batch_async(image_paths)


def pack_clipsam_batch(batch_masks: List[List[dict]]) -> dict:
    # Pack in the exact same format used by SAM2 shards
    return pack_batch_auto_masks(batch_masks)


async def process_single_image_clipsam_async(image_path: str, clipsam_client: AsyncMultiWrapper) -> List[dict]:
    """Async wrapper for single image CLIPSAM processing."""
    return await clipsam_client.filter_auto_masks_for_image_async(image_path)


if __name__ == "__main__":
    # Demo: Full pipeline (extract -> save -> load -> visualize) for CLIPSAM
    data_root = Path("datasets/f3rm/custom/betabook/small")
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
    extractor2 = CLIPSAMExtractor(device=device, data_dir=data_root, text_prompts=["book"], verbose=True)
    batch_masks2 = run_async_in_any_context(lambda: extractor2.extract_batch_async(image_paths))

    # Use the first mode for visualization
    batch_masks = batch_masks

    print(f"Extracted CLIPSAM auto-masks for {len(batch_masks)} images. Visualizing results...")

    # Visualize a few results
    vis_count = min(4, len(batch_masks))
    fig, axes = plt.subplots(2, vis_count, figsize=(4 * vis_count, 8))
    if vis_count == 1:
        axes = axes.reshape(2, 1)
    for i, (auto_masks, img_path) in enumerate(zip(batch_masks[:vis_count], image_paths[:vis_count])):
        rgb_img = Image.open(img_path).convert("RGB")
        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f"RGB {i+1}")
        axes[0, i].axis('off')

        # Generate and display mask
        inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
            auto_masks,
            min_iou=float(SAM2Args.pred_iou_thresh),
            min_area=float(SAM2Args.min_mask_region_area),
            assign_by="area",
            start_from="low",
        )
        if inst_mask is None:
            # No valid masks found, create empty instance mask
            if auto_masks:
                h, w = auto_masks[0]['segmentation'].shape
            else:
                # Use actual image dimensions as fallback
                img = Image.open(img_path)
                h, w = img.height, img.width
            inst_mask = np.zeros((h, w), dtype=np.uint16)
        viz_mask, cmap, norm = SAM2utils.make_viz_mask_and_cmap(inst_mask)
        axes[1, i].imshow(viz_mask, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1, i].set_title(f"CLIPSAM Mask {i+1} ({len(np.unique(inst_mask)) - 1} inst)")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()

    # Demo 2: Test the full pipeline (extract -> save -> load -> visualize)
    print("\n" + "=" * 60)
    print("DEMO 2: Full pipeline test (extract -> save -> load -> visualize)")
    print("=" * 60)

    # Save shards
    test_dir = Path("test_clipsam_pipeline")
    test_dir.mkdir(exist_ok=True)
    shard_size = 4
    n_imgs = len(batch_masks)
    n_shards = math.ceil(n_imgs / shard_size)

    print("Saving CLIPSAM shards...")
    for i in range(n_shards):
        s, e = i * shard_size, min((i + 1) * shard_size, n_imgs)
        packed_batch = pack_batch_auto_masks(batch_masks[s:e])
        np.savez_compressed(test_dir / f"chunk_{i:04d}.npz", **packed_batch)

    # Load shards with SAM2LazyAutoMasks and visualize a few
    print("Loading shards with SAM2LazyAutoMasks...")
    shard_paths = sorted(test_dir.glob("chunk_*.npz"))
    lazy_shards = SAM2LazyAutoMasks(shard_paths)
    print(f"Loaded {len(lazy_shards)} images from {len(shard_paths)} shards")

    print("Visualizing loaded data...")
    vis_count = min(3, len(lazy_shards))
    fig, axes = plt.subplots(2, vis_count, figsize=(4 * vis_count, 8))
    if vis_count == 1:
        axes = axes.reshape(2, 1)
    for i in range(vis_count):
        rgb_img = Image.open(image_paths[i]).convert("RGB")
        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f"RGB {i+1}")
        axes[0, i].axis('off')

        # Loaded masks
        loaded_masks = lazy_shards[i]
        inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
            loaded_masks,
            min_iou=float(SAM2Args.pred_iou_thresh),
            min_area=float(SAM2Args.min_mask_region_area),
            assign_by="area",
            start_from="low",
        )
        if inst_mask is None:
            # No valid masks found, create empty instance mask
            if loaded_masks:
                h, w = loaded_masks[0]['segmentation'].shape
            else:
                # Use actual image dimensions as fallback
                img = Image.open(image_paths[i])
                h, w = img.height, img.width
            inst_mask = np.zeros((h, w), dtype=np.uint16)
        viz_mask, cmap, norm = SAM2utils.make_viz_mask_and_cmap(inst_mask)
        axes[1, i].imshow(viz_mask, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1, i].set_title(f"Loaded CLIPSAM {i+1} ({len(loaded_masks)} masks, {len(np.unique(inst_mask)) - 1} inst)")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()

    # Print comparison
    print("\nComparison:")
    for i in range(min(3, len(batch_masks))):
        original_count = len(batch_masks[i])
        loaded_count = len(lazy_shards[i])
        print(f"Image {i}: Original={original_count} masks, Loaded={loaded_count} masks")

    # Cleanup
    plt.close('all')
    del lazy_shards
    gc.collect()
    shutil.rmtree(test_dir)
    print(f"Cleaned up test directory: {test_dir}")
