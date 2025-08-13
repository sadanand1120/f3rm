import gc
import asyncio
import glob
from typing import List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from sam2.features.client.sam2_client import SAM2FeaturesUnified
from sam2.features.utils import SAM2utils, AsyncMultiWrapper
from f3rm.features.utils import resolve_devices_and_workers, run_async_in_any_context


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


def pack_auto_masks(auto_masks: List[dict]) -> List[dict]:
    optimized: List[dict] = []
    for m in auto_masks:
        seg = m.get("segmentation", None)
        if isinstance(seg, np.ndarray):
            h, w = seg.shape
            packed = np.packbits(seg.astype(np.uint8).reshape(-1))
            m = m.copy()
            m["segmentation"] = {
                "format": "packbits",
                "shape": (h, w),
                "data": packed.tobytes(),
            }
        optimized.append(m)
    return optimized


def unpack_auto_masks(auto_masks: List[dict]) -> List[dict]:
    decoded: List[dict] = []
    for m in auto_masks:
        seg = m.get("segmentation")
        if isinstance(seg, dict) and seg.get("format") == "packbits":
            h, w = seg["shape"]
            packed = np.frombuffer(seg["data"], dtype=np.uint8)
            flat = np.unpackbits(packed)[: h * w]
            seg_arr = flat.reshape(h, w).astype(bool)
            md = m.copy()
            md["segmentation"] = seg_arr
            decoded.append(md)
        else:
            decoded.append(m)
    return decoded


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
        results: List[dict] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Processing & extracting SAM2 auto-masks", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_async(path, self.client) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="SAM2 auto_mask", leave=False)
            # pack for storage efficiency
            results.extend([pack_auto_masks(m) for m in batch_results])
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
    fig, axes = plt.subplots(1, vis_count, figsize=(4 * vis_count, 4))
    if vis_count == 1:
        axes = [axes]
    for ax, auto_masks in zip(axes, auto_masks_per_image[:vis_count]):
        decoded = unpack_auto_masks(auto_masks)
        inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
            decoded,
            min_iou=float(SAM2Args.pred_iou_thresh),
            min_area=float(SAM2Args.min_mask_region_area),
            assign_by="area",
            start_from="low",
        )
        viz_mask, cmap, norm = SAM2utils.make_viz_mask_and_cmap(inst_mask)
        ax.imshow(viz_mask, cmap=cmap, norm=norm, interpolation='nearest')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
