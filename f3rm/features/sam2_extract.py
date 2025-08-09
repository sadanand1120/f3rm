import gc
import asyncio
import inspect
import os
import glob
from typing import List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
import aiohttp
import torch
import matplotlib.pyplot as plt

from sam2.server.client.sam2_client import (
    encode_image,
    decode_auto_masks,
    auto_masks_to_instance_mask,
    _prevent_oom_downscale_img,
    make_viz_mask_and_cmap,
    generate_sam2_auto_masks,
)


class SAM2Args:
    points_per_side: int = 36
    points_per_batch: int = 128
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.9
    min_mask_region_area: int = 0
    batch_size: int = 2  # Number of concurrent requests to server
    server_config_path: str = "/robodata/smodak/repos/orienter/clients/servers.yaml"
    server_name: str = "sam2"
    preset: Optional[str] = "coarse"  # e.g., "coarse", "fine_grained"
    assign_by: str = "area"
    start_from: str = "low"

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the SAM2 model parameters."""
        return {
            "points_per_side": cls.points_per_side,
            "points_per_batch": cls.points_per_batch,
            "pred_iou_thresh": cls.pred_iou_thresh,
            "stability_score_thresh": cls.stability_score_thresh,
            "min_mask_region_area": cls.min_mask_region_area,
            "batch_size": cls.batch_size,
            "server_config_path": cls.server_config_path,
            "server_name": cls.server_name,
            "preset": cls.preset,
            "assign_by": cls.assign_by,
            "start_from": cls.start_from,
        }


async def process_single_image_async(image_path: str, selected_server: dict, session: aiohttp.ClientSession) -> np.ndarray:
    """Process a single image asynchronously."""
    # Load and preprocess image
    pil_img = Image.open(image_path).convert("RGB")
    original_width, original_height = pil_img.width, pil_img.height
    downscaled_img = _prevent_oom_downscale_img(pil_img, target=2048)

    # Prepare payload
    payload = {'image': encode_image(downscaled_img)}
    # Include only supported, non-None args by inspecting client function signature
    sig = inspect.signature(generate_sam2_auto_masks)
    allowed = set(sig.parameters.keys()) - {"image", "image_url", "base_url", "api_key"}
    for key, value in SAM2Args.id_dict().items():
        if key in allowed and value is not None:
            payload[key] = value

    # Prepare headers
    headers = {'Content-Type': 'application/json'}
    if selected_server.get("api_key"):
        headers['Authorization'] = f'Bearer {selected_server["api_key"]}'

    # Make async request with retry (errors still propagate after final attempt)
    max_retries = 100
    retry_delay_s = 10
    result = None
    retry_bar = None
    for attempt in range(max_retries):
        try:
            async with session.post(f"{selected_server['base_url']}/auto_mask", json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                break
        except Exception as e:
            # Lazy-create a temporary retry bar that doesn't interfere with the main bar
            if retry_bar is None:
                retry_bar = tqdm(total=max_retries, desc="SAM2 retry", position=2, leave=False, colour="red")
                retry_bar.set_postfix_str(os.path.basename(image_path))
            retry_bar.update(1)
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_delay_s)
    if retry_bar is not None:
        retry_bar.close()

    # Decode masks
    auto_masks = decode_auto_masks(result['masks'])

    # Convert to instance mask using shared utility and our thresholds
    instance_mask = auto_masks_to_instance_mask(
        auto_masks,
        min_iou=float(SAM2Args.pred_iou_thresh),
        min_area=float(SAM2Args.min_mask_region_area),
        assign_by=SAM2Args.assign_by,
        start_from=SAM2Args.start_from,
    )

    # Upscale back to original size
    if (downscaled_img.width, downscaled_img.height) != (original_width, original_height):
        # Convert numpy array to PIL Image for resizing
        pil_mask = Image.fromarray(instance_mask.astype(np.uint8))
        upscaled_mask = pil_mask.resize((original_width, original_height), Image.NEAREST)
        instance_mask = np.array(upscaled_mask)

    del auto_masks
    return instance_mask


async def extract_sam2_masks_async(image_paths: List[str], verbose=False) -> np.ndarray:
    # Load server config
    with open(SAM2Args.server_config_path, "r") as f:
        servers = yaml.safe_load(f)
    selected_server = servers[SAM2Args.server_name]
    if verbose:
        print(f"Loaded SAM2 server config from {SAM2Args.server_config_path}")
    timeout = aiohttp.ClientTimeout(total=1e4, connect=1e4)
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=10, enable_cleanup_closed=True, force_close=True)
    instance_masks = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for i in tqdm(range(0, len(image_paths), SAM2Args.batch_size), desc="Processing & extracting SAM2 masks", position=1, leave=verbose):
            batch_paths = image_paths[i:i + SAM2Args.batch_size]
            tasks = [process_single_image_async(path, selected_server, session) for path in batch_paths]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            instance_masks.extend(batch_results)
            gc.collect()

    instance_masks = np.array(instance_masks)
    instance_masks = instance_masks[..., np.newaxis]
    if verbose:
        print(f"Extracted SAM2 instance masks of shape {instance_masks.shape}")
    return instance_masks


def extract_sam2_masks(image_paths: List[str], verbose=False) -> np.ndarray:
    """Synchronous wrapper for async extraction."""
    return asyncio.run(extract_sam2_masks_async(image_paths, verbose))


def extract_sam2_features(image_paths: List[str], device: torch.device, verbose: bool = False) -> torch.Tensor:
    """Adapter to match CLIP/DINO extractor signature. Returns a torch.Tensor of shape (B, H, W, 1) with integer instance ids."""
    masks = extract_sam2_masks(image_paths, verbose=verbose)
    if masks.dtype != np.int32:
        masks = masks.astype(np.int32, copy=False)
    return torch.from_numpy(masks)


if __name__ == "__main__":
    # Get all images in the directory
    image_dir = "datasets/f3rm/panda/scene_001/images"
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png"))
    image_paths = image_paths[:10]
    print(f"Found {len(image_paths)} images in {image_dir}")
    print("Async extraction:")
    instance_masks = extract_sam2_masks(image_paths, verbose=True)
    print(f"Instance mask shape: {instance_masks.shape}")

    # Visualize in subsets of 4 with 2x2 grid
    for i in range(0, len(instance_masks), 4):
        subset_masks = instance_masks[i:i + 4]
        subset_paths = image_paths[i:i + 4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        for j, (mask, img_path) in enumerate(zip(subset_masks, subset_paths)):
            if j >= 4:  # Safety check
                break
            # Remove channel dimension for visualization (mask is now 4D: batch, H, W, 1)
            mask_2d = mask.squeeze()
            viz_mask, cmap, norm = make_viz_mask_and_cmap(mask_2d.astype(np.int32))
            axes[j].imshow(viz_mask, cmap=cmap, norm=norm, interpolation='nearest')
            axes[j].axis('off')
            num_instances = int(mask_2d.max())
            axes[j].set_title(f"{img_path.split('/')[-1]}\nInstances: {num_instances}")
        # Hide unused subplots
        for j in range(len(subset_masks), 4):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()
