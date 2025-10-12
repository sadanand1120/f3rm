import gc
import asyncio
import glob
import math
import os
import shutil
import cv2
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from sam2.features.client.sam2_client import SAM2FeaturesUnified
from sam2.features.utils import SAM2utils, AsyncMultiWrapper
from f3rm.features.utils import resolve_devices_and_workers, run_async_in_any_context, pack_auto_masks, unpack_auto_masks, visualize_auto_masks_demo


class SAM2Args:
    points_per_side: int = 64
    points_per_batch: int = 128
    pred_iou_thresh: float = None
    stability_score_thresh: float = None
    stability_score_offset: float = None
    box_nms_thresh: float = None
    min_mask_region_area: int = None
    use_m2m: bool = True
    preset: Optional[str] = "coarse"
    load_size: int = 2048   # final save is still at image size
    model_cfg: str = "/robodata/smodak/repos/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint_path: str = "/robodata/smodak/repos/sam2/checkpoints/sam2.1_hiera_large.pt"
    batch_size_per_gpu: int = 4
    use_object_masks: bool = False

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the SAM2 model parameters."""
        return {
            "points_per_side": cls.points_per_side,
            "points_per_batch": cls.points_per_batch,
            "pred_iou_thresh": cls.pred_iou_thresh,
            "stability_score_thresh": cls.stability_score_thresh,
            "stability_score_offset": cls.stability_score_offset,
            "box_nms_thresh": cls.box_nms_thresh,
            "min_mask_region_area": cls.min_mask_region_area,
            "use_m2m": cls.use_m2m,
            "preset": cls.preset,
            "load_size": cls.load_size,
            "model_cfg": cls.model_cfg,
            "checkpoint_path": cls.checkpoint_path,
            "use_object_masks": cls.use_object_masks,
        }


def load_object_mask(image_path: str, data_dir: Path) -> Optional[np.ndarray]:
    """Load object mask from object_masks directory corresponding to the image."""
    image_name = Path(image_path).name
    mask_path = data_dir / "object_masks" / image_name

    if not mask_path.exists():
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Convert to boolean mask (255 -> True, 0 -> False)
    return mask > 0


def object_mask_to_auto_masks(object_mask: np.ndarray) -> List[dict]:
    """Convert object mask to SAM2-style auto masks format."""
    if object_mask is None or not np.any(object_mask):
        return []

    # Create a single auto mask from the object mask
    h, w = object_mask.shape
    ys, xs = np.where(object_mask)

    if len(ys) == 0:
        return []

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]

    return [{
        "segmentation": object_mask,
        "bbox": bbox,
        "predicted_iou": np.float16(1.0),  # Perfect IoU for ground truth
        "area": np.float16(np.sum(object_mask)),  # Actual area
    }]


async def process_single_image_async(image_path: str, sam2: AsyncMultiWrapper, data_dir: Optional[Path] = None) -> List[dict]:
    if SAM2Args.use_object_masks and data_dir is not None:
        # Use object masks instead of SAM2 processing
        object_mask = load_object_mask(image_path, data_dir)
        assert object_mask is not None, f"Object mask not found for image: {image_path}"
        auto_masks = object_mask_to_auto_masks(object_mask)
        return auto_masks

    # Original SAM2 processing
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
        stability_score_offset=SAM2Args.stability_score_offset,
        min_mask_region_area=SAM2Args.min_mask_region_area,
        box_nms_thresh=SAM2Args.box_nms_thresh,
        use_m2m=SAM2Args.use_m2m,
        output_mode="binary_mask",
    )
    # Upscale to original size if needed using utility
    if (downscaled_img.width, downscaled_img.height) != (original_width, original_height):
        target = max(original_width, original_height)
        auto_masks = SAM2utils.resize_auto_masks(auto_masks, target=target)

    return auto_masks


class SAM2Extractor:
    def __init__(self, device: torch.device, data_dir: Optional[Path] = None, verbose: bool = False) -> None:
        self.data_dir = data_dir
        devices_param, num_workers = resolve_devices_and_workers(device, SAM2Args.batch_size_per_gpu)
        if verbose:
            print("Initializing SAM2 client")
        self.client = AsyncMultiWrapper(SAM2FeaturesUnified, num_objects=num_workers, devices=devices_param)
        self.num_workers = num_workers

        # Only warm up SAM2 workers if not using object masks
        if not SAM2Args.use_object_masks:
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
                    stability_score_offset=SAM2Args.stability_score_offset,
                    box_nms_thresh=SAM2Args.box_nms_thresh,
                    min_mask_region_area=0,
                    use_m2m=SAM2Args.use_m2m,
                    output_mode="binary_mask",
                )

    async def extract_batch_async(self, image_paths: List[str]):
        results: List[List[dict]] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Processing & extracting SAM2 auto-masks", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_async(path, self.client, self.data_dir) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="SAM2 auto_mask", leave=False)
            results.extend(batch_results)
            gc.collect()
        return results


def extract_sam2_features(image_paths: List[str], device: torch.device, data_dir: Optional[Path] = None, verbose: bool = False):
    extractor = SAM2Extractor(device=device, data_dir=data_dir, verbose=verbose)
    return run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))


def examine_saved(sam2_feat_dir: str):
    """Create .mp4 video of saved SAM2 features with mask visualization."""
    meta_path = os.path.join(sam2_feat_dir, "meta.pt")
    assert os.path.exists(meta_path), f"SAM2 meta not found at {meta_path}"

    meta = torch.load(meta_path)
    image_fnames = meta["image_fnames"]
    n_images = len(image_fnames)

    # Load first image to get dimensions
    first_data = np.load(os.path.join(sam2_feat_dir, "image_000000.npz"))
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

    video_path = os.path.join(sam2_feat_dir, "features_viz.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 2.0, (W, H))

    for i in tqdm(range(n_images), desc="Creating SAM2 features video"):
        feat_path = os.path.join(sam2_feat_dir, f"image_{i:06d}.npz")
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
        colors = plt.cm.Set3(np.linspace(0, 1, max(1, len(auto_masks))))
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
    # examine_saved("datasets/f3rm/opt/objaverse/car2/features/sam2")
    # examine_saved("datasets/f3rm/opt/objaverse/car2/features/sam2.bkp")

    # Get all images in the directory
    # image_dir = "datasets/f3rm/panda/scene_001/images"
    image_dir = "datasets/f3rm/opt/objaverse/car2/images"
    data_dir = Path("datasets/f3rm/opt/objaverse/car2")
    SAM2Args.use_object_masks = False
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png"))
    image_paths = image_paths[32:36]
    print(f"Found {len(image_paths)} images in {image_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test with object masks
    print("Testing...")
    auto_masks_per_image = extract_sam2_features(image_paths, device=device, data_dir=data_dir, verbose=True)
    print(f"Extracted auto-masks for {len(auto_masks_per_image)} images. Visualizing...")
    visualize_auto_masks_demo(auto_masks_per_image, image_paths, "SAM2", max_vis=4, pred_iou_thresh=SAM2Args.pred_iou_thresh, min_mask_region_area=SAM2Args.min_mask_region_area)

    # Demo 2: Test the full pipeline (extract -> save -> load -> visualize)
    print("\n" + "=" * 60)
    print("DEMO 2: Full pipeline test (extract -> save -> load -> visualize)")
    print("=" * 60)

    # Setup paths
    test_dir = Path("test_sam2_pipeline")
    test_dir.mkdir(exist_ok=True)

    # Extract features
    print("Extracting SAM2 features...")
    extractor = SAM2Extractor(device=device, data_dir=data_dir, verbose=True)
    auto_masks_per_image = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))

    # Save per-image files (following new per-image system)
    print("Saving per-image files...")
    for i, auto_masks in enumerate(auto_masks_per_image):
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
    for i in range(len(auto_masks_per_image)):
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
    visualize_auto_masks_demo(loaded_masks, image_paths, "Loaded SAM2", max_vis=3, pred_iou_thresh=SAM2Args.pred_iou_thresh, min_mask_region_area=SAM2Args.min_mask_region_area)

    # Print summary
    print("\nExtracted auto-masks for demo:")
    for i in range(min(3, len(auto_masks_per_image))):
        original_count = len(auto_masks_per_image[i])
        loaded_count = len(loaded_masks[i]) if i < len(loaded_masks) else 0
        print(f"Image {i}: {original_count} original masks, {loaded_count} loaded masks")

    # Cleanup
    plt.close('all')
    gc.collect()
    shutil.rmtree(test_dir, ignore_errors=True)  # nfs sometimes will still leave a empty directory behind
    print(f"Cleaned up test directory: {test_dir}")
