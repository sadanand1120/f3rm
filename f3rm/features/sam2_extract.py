import gc
import asyncio
import base64
import io
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import yaml
import os
import aiohttp

from sam2.server.client.sam2_client import masks_to_instance_mask, decode_auto_masks, encode_image
from sam2.features.utils import SAM2utils


class SAM2Args:
    downscale_factor: int = -1
    points_per_side: int = 48
    pred_iou_thresh: float = 0.85
    stability_score_thresh: float = 0.95
    min_mask_region_area: int = 100
    batch_size: int = 2  # Number of concurrent requests to server
    server_config_path: str = "/robodata/smodak/repos/orienter/clients/servers.yaml"
    server_name: str = "sam2"

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the SAM2 model parameters."""
        return {
            "downscale_factor": cls.downscale_factor,
            "points_per_side": cls.points_per_side,
            "pred_iou_thresh": cls.pred_iou_thresh,
            "stability_score_thresh": cls.stability_score_thresh,
            "min_mask_region_area": cls.min_mask_region_area,
            "batch_size": cls.batch_size,
        }


async def process_single_image_async(image_path: str, selected_server: dict, session: aiohttp.ClientSession) -> np.ndarray:
    """Process a single image asynchronously."""
    # Load and preprocess image
    pil_img = Image.open(image_path).convert("RGB")
    original_width, original_height = pil_img.width, pil_img.height

    # Auto-infer downscale factor if set to -1
    if SAM2Args.downscale_factor == -1:
        max_side = max(original_width, original_height)
        if max_side > 2048:
            downscale_factor = max_side / 2048
        else:
            downscale_factor = 1
    else:
        downscale_factor = SAM2Args.downscale_factor

    new_width = int(original_width // downscale_factor)
    new_height = int(original_height // downscale_factor)
    downscaled_img = pil_img.resize((new_width, new_height))

    # Prepare payload
    payload = {
        'image': encode_image(downscaled_img),
        'points_per_side': SAM2Args.points_per_side,
        'pred_iou_thresh': SAM2Args.pred_iou_thresh,
        'stability_score_thresh': SAM2Args.stability_score_thresh,
        'min_mask_region_area': SAM2Args.min_mask_region_area
    }

    # Prepare headers
    headers = {'Content-Type': 'application/json'}
    if selected_server.get("api_key"):
        headers['Authorization'] = f'Bearer {selected_server["api_key"]}'

    # Make async request with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with session.post(
                f"{selected_server['base_url']}/auto_mask",
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                result = await response.json()
                break  # Success, exit retry loop
        except (aiohttp.ClientOSError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            await asyncio.sleep(1)  # Wait before retry
            continue

    # Decode masks
    auto_masks = decode_auto_masks(result['masks'])

    # Convert to instance mask
    instance_mask = masks_to_instance_mask(auto_masks)

    # Upscale back to original size
    if downscale_factor > 1:
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

    # Configure timeout and connection settings
    timeout = aiohttp.ClientTimeout(total=300, connect=30)  # 5 min total, 30s connect
    connector = aiohttp.TCPConnector(
        limit=100,  # Max connections
        limit_per_host=10,  # Max connections per host
        keepalive_timeout=30,  # Keep connections alive for 30s
        enable_cleanup_closed=True,  # Clean up closed connections
    )

    instance_masks = []
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers={'Connection': 'keep-alive'}
    ) as session:
        for i in tqdm(range(0, len(image_paths), SAM2Args.batch_size),
                      desc="Processing & extracting SAM2 masks", leave=verbose):
            batch_paths = image_paths[i:i + SAM2Args.batch_size]

            # Process batch concurrently with retry logic
            tasks = [process_single_image_async(path, selected_server, session) for path in batch_paths]

            # Use gather with return_exceptions=True to handle individual failures
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for exceptions and retry failed requests
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    if verbose:
                        print(f"Retrying failed request for {batch_paths[j]}: {result}")
                    # Retry once
                    try:
                        batch_results[j] = await process_single_image_async(batch_paths[j], selected_server, session)
                    except Exception as e:
                        print(f"Failed to process {batch_paths[j]} after retry: {e}")
                        # Return empty mask as fallback
                        batch_results[j] = np.zeros((1, 1), dtype=np.int32)

            instance_masks.extend(batch_results)

            gc.collect()

    instance_masks = np.array(instance_masks)
    # Add channel dimension to match CLIP/DINO format (batch, height, width, channels)
    instance_masks = instance_masks[..., np.newaxis]
    if verbose:
        print(f"Extracted SAM2 instance masks of shape {instance_masks.shape}")

    return instance_masks


def extract_sam2_masks(image_paths: List[str], verbose=False) -> np.ndarray:
    """Synchronous wrapper for async extraction."""
    return asyncio.run(extract_sam2_masks_async(image_paths, verbose))


if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Get all images in the directory
    image_dir = "datasets/f3rm/panda/scene_001/images"
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png"))

    print(f"Found {len(image_paths)} images in {image_dir}")

    # Process all images
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
            mask_2d = mask.squeeze()  # Remove singleton dimensions

            # Create colormap
            n = mask_2d.max() + 1
            base_cmap = plt.get_cmap('tab20', n)
            colors = base_cmap(np.arange(n))
            colors[0] = [0, 0, 0, 1]  # RGBA for black
            cmap = ListedColormap(colors)

            # Display mask
            axes[j].imshow(mask_2d, cmap=cmap, interpolation='nearest')
            axes[j].axis('off')
            axes[j].set_title(f"{img_path.split('/')[-1]}\nInstances: {n-1}")

        # Hide unused subplots
        for j in range(len(subset_masks), 4):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
