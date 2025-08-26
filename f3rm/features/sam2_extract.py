import gc
import asyncio
import glob
import math
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from sam2.features.client.sam2_client import SAM2FeaturesUnified
from sam2.features.utils import SAM2utils, AsyncMultiWrapper
from f3rm.features.utils import resolve_devices_and_workers, run_async_in_any_context, SAM2LazyAutoMasks, pack_auto_masks, pack_batch_auto_masks


class SAM2Args:
    points_per_side: int = 64
    points_per_batch: int = 128
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.9
    min_mask_region_area: int = 0
    preset: Optional[str] = "coarse"
    load_size: int = 2048
    model_cfg: str = "/robodata/smodak/repos/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint_path: str = "/robodata/smodak/repos/sam2/checkpoints/sam2.1_hiera_large.pt"
    batch_size_per_gpu: int = 4

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the SAM2 model parameters."""
        return {
            "points_per_side": cls.points_per_side,
            "points_per_batch": cls.points_per_batch,
            "pred_iou_thresh": cls.pred_iou_thresh,
            "stability_score_thresh": cls.stability_score_thresh,
            "min_mask_region_area": cls.min_mask_region_area,
            "preset": cls.preset,
            "load_size": cls.load_size,
            "model_cfg": cls.model_cfg,
            "checkpoint_path": cls.checkpoint_path,
        }


async def process_single_image_async(image_path: str, sam2: AsyncMultiWrapper) -> List[dict]:
    pil_img = Image.open(image_path).convert("RGB")
    original_width, original_height = pil_img.width, pil_img.height
    downscaled_img = SAM2utils.prevent_oom_resizing(pil_img, target=SAM2Args.load_size)

    auto_masks = await sam2.auto_mask_async(
        image=downscaled_img,
        model_cfg=SAM2Args.model_cfg,
        checkpoint_path=SAM2Args.checkpoint_path,
        preset=SAM2Args.preset,
        points_per_side=SAM2Args.points_per_side,
        points_per_batch=SAM2Args.points_per_batch,
        pred_iou_thresh=SAM2Args.pred_iou_thresh,
        stability_score_thresh=SAM2Args.stability_score_thresh,
        min_mask_region_area=SAM2Args.min_mask_region_area,
        output_mode="binary_mask",
    )
    # Upscale to original size if needed using utility
    if (downscaled_img.width, downscaled_img.height) != (original_width, original_height):
        target = max(original_width, original_height)
        auto_masks = SAM2utils.resize_auto_masks(auto_masks, target=target)

    return auto_masks


class SAM2Extractor:
    def __init__(self, device: torch.device, verbose: bool = False) -> None:
        devices_param, num_workers = resolve_devices_and_workers(device, SAM2Args.batch_size_per_gpu)
        if verbose:
            print("Initializing SAM2 client")
        self.client = AsyncMultiWrapper(SAM2FeaturesUnified, num_objects=num_workers, devices=devices_param)
        self.num_workers = num_workers
        # Sequential warm-up to avoid TorchScript race in torchvision Resize when constructing generators in parallel
        if verbose:
            print("Warming up SAM2 workers...")
        tiny = Image.new("RGB", (8, 8), color=0)
        for _ in range(self.num_workers):
            _ = self.client.auto_mask(
                image=tiny,
                model_cfg=SAM2Args.model_cfg,
                checkpoint_path=SAM2Args.checkpoint_path,
                preset=SAM2Args.preset,
                points_per_side=4,
                points_per_batch=8,
                pred_iou_thresh=SAM2Args.pred_iou_thresh,
                stability_score_thresh=SAM2Args.stability_score_thresh,
                min_mask_region_area=0,
                output_mode="binary_mask",
            )

    async def extract_batch_async(self, image_paths: List[str]):
        results: List[List[dict]] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Processing & extracting SAM2 auto-masks", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_async(path, self.client) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="SAM2 auto_mask", leave=False)
            results.extend(batch_results)
            gc.collect()
        return results


def make_sam2_extractor(device: torch.device, verbose: bool = False) -> "SAM2Extractor":
    return SAM2Extractor(device=device, verbose=verbose)


def extract_sam2_features(image_paths: List[str], device: torch.device, verbose: bool = False):
    extractor = make_sam2_extractor(device, verbose=verbose)
    return run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))


if __name__ == "__main__":
    # Get all images in the directory
    image_dir = "datasets/f3rm/panda/scene_001/images"
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png"))
    image_paths = image_paths[:10]
    print(f"Found {len(image_paths)} images in {image_dir}")
    print("Extraction:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auto_masks_per_image = extract_sam2_features(image_paths, device=device, verbose=True)
    print(f"Extracted auto-masks for {len(auto_masks_per_image)} images. Visualizing instance conversion for first few...")
    vis_count = min(4, len(auto_masks_per_image))
    fig, axes = plt.subplots(2, vis_count, figsize=(4 * vis_count, 8))
    if vis_count == 1:
        axes = axes.reshape(2, 1)

    for i, (auto_masks, image_path) in enumerate(zip(auto_masks_per_image[:vis_count], image_paths[:vis_count])):
        # Load and display RGB image
        rgb_img = Image.open(image_path).convert("RGB")
        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f"RGB Image {i+1}")
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
                img = Image.open(image_paths[i])
                h, w = img.height, img.width
            inst_mask = np.zeros((h, w), dtype=np.uint16)
        viz_mask, cmap, norm = SAM2utils.make_viz_mask_and_cmap(inst_mask)
        axes[1, i].imshow(viz_mask, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1, i].set_title(f"Instance Mask {i+1} ({len(np.unique(inst_mask)) - 1} inst)")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Demo 2: Test the full pipeline (extract -> save -> load -> visualize)
    print("\n" + "=" * 60)
    print("DEMO 2: Full pipeline test (extract -> save -> load -> visualize)")
    print("=" * 60)

    # Setup paths
    test_dir = Path("test_sam2_pipeline")
    test_dir.mkdir(exist_ok=True)
    shard_size = 4

    # Extract features
    print("Extracting SAM2 features...")
    extractor = make_sam2_extractor(device, verbose=True)
    auto_masks_per_image = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))

    # Save shards (following extract_features_standalone.py logic)
    print("Saving shards...")
    n_imgs = len(auto_masks_per_image)
    n_shards = math.ceil(n_imgs / shard_size)

    for i in range(n_shards):
        s, e = i * shard_size, min((i + 1) * shard_size, n_imgs)
        batch_masks = auto_masks_per_image[s:e]

        # Pack batch data using consolidated function
        packed_batch = pack_batch_auto_masks(batch_masks)

        # Save shard
        np.savez_compressed(
            test_dir / f"chunk_{i:04d}.npz",
            **packed_batch
        )

    # Load shards using SAM2LazyAutoMasks
    print("Loading shards with SAM2LazyAutoMasks...")
    shard_paths = sorted(test_dir.glob("chunk_*.npz"))
    lazy_shards = SAM2LazyAutoMasks(shard_paths)
    print(f"Loaded {len(lazy_shards)} images from {len(shard_paths)} shards")

    # Visualize loaded data
    print("Visualizing loaded data...")
    vis_count = min(3, len(lazy_shards))
    fig, axes = plt.subplots(2, vis_count, figsize=(4 * vis_count, 8))
    if vis_count == 1:
        axes = axes.reshape(2, 1)

    for i in range(vis_count):
        # Original RGB
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
        axes[1, i].set_title(f"Loaded Mask {i+1} ({len(loaded_masks)} masks, {len(np.unique(inst_mask)) - 1} inst)")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Print comparison
    print("\nComparison:")
    for i in range(min(3, len(auto_masks_per_image))):
        original_count = len(auto_masks_per_image[i])
        loaded_count = len(lazy_shards[i])
        print(f"Image {i}: Original={original_count} masks, Loaded={loaded_count} masks")

    # Cleanup - close figures and lazy shards before removing directory
    plt.close('all')
    del lazy_shards
    import gc
    gc.collect()

    # Cleanup
    shutil.rmtree(test_dir)
    print(f"Cleaned up test directory: {test_dir}")
