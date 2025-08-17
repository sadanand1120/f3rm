from typing import Optional, Tuple, Callable, Any, List, Dict

import asyncio
import concurrent.futures
import numpy as np
import torch
from pathlib import Path


def resolve_devices_and_workers(device: torch.device, batch_size_per_gpu: int) -> Tuple[Optional[torch.device], int]:
    """Return (devices_param, num_workers) for AsyncMultiWrapper using per-GPU worker count.

    - device == cuda with no index → round-robin across all GPUs, num_workers = num_gpus * batch_size_per_gpu
    - device == cuda:X → pin to that GPU, num_workers = batch_size_per_gpu
    - device == cpu → single worker
    """
    if device.type == "cuda":
        if device.index is None:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            num_workers = max(1, (n_gpus or 1) * max(1, batch_size_per_gpu))
            return None, num_workers
        else:
            num_workers = max(1, batch_size_per_gpu)
            return torch.device(f"cuda:{device.index}"), num_workers
    return torch.device("cpu"), 1


def run_async_in_any_context(coro_fn: Callable[[], Any]) -> Any:
    """Run an async coroutine function regardless of existing event loop.

    Expects a no-arg function that returns an awaitable when called.
    """
    try:
        asyncio.get_running_loop()

        def _thread_run():
            return asyncio.run(coro_fn())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_thread_run)
            return fut.result()
    except RuntimeError:
        return asyncio.run(coro_fn())


class BaseLazyShards:
    """Base class for lazy shard management."""

    def __init__(self, shard_paths: List[Path]):
        self.paths = shard_paths
        self.lengths = []
        self._setup_lengths()
        self.cum = np.cumsum([0] + self.lengths)

    def _setup_lengths(self):
        """Override in subclasses to set up lengths based on data type."""
        raise NotImplementedError

    def _loc(self, idx_img: int) -> Tuple[int, int]:
        sid = int(np.searchsorted(self.cum, idx_img, side="right") - 1)
        return sid, idx_img - self.cum[sid]

    def _get_shard(self, sid: int):
        """Override in subclasses to implement shard loading."""
        raise NotImplementedError


class LazyFeatures(BaseLazyShards):
    """Memory-mapped shards with O(1) random access.

    Exposes `feat[idx_img, y, x] → torch.Tensor(C)` and keeps each shard
    mapped only once (OS handles paging).  Nothing is ever `torch.cat`-ed.
    """

    def __init__(self, shard_paths: List[Path]):
        self.mmaps = [None] * len(shard_paths)          # lazy mmap
        super().__init__(shard_paths)

    def _setup_lengths(self):
        for p in self.paths:
            # Load just shape info without loading full data
            arr = np.load(p, mmap_mode="r")
            self.lengths.append(arr.shape[0])
            if self.mmaps[0] is None:                     # keep dims
                self.H, self.W, self.C = arr.shape[1:]
            del arr  # Explicitly free memory after getting shape info

    def _get_shard(self, sid: int):
        if self.mmaps[sid] is None:
            self.mmaps[sid] = np.load(self.paths[sid], mmap_mode="r", allow_pickle=False)
        return self.mmaps[sid]

    # single triple access
    def __getitem__(self, triple):
        if isinstance(triple, int):
            # Full image access: feat[idx_img] → torch.Tensor(H, W, C)
            sid, loc = self._loc(int(triple))
            shard_data = self._get_shard(sid)
            return torch.from_numpy(shard_data[loc])
        else:
            # Pixel access: feat[idx_img, y, x] → torch.Tensor(C)
            idx_img, y, x = triple
            sid, loc = self._loc(int(idx_img))
            feat = self._get_shard(sid)[loc, int(y), int(x)]
            return torch.from_numpy(feat)


class SAM2LazyAutoMasks(BaseLazyShards):
    """Sharded loader for SAM2 auto-masks stored in memory-mappable .npz format.

    - Mirrors the interface of `LazyFeatures` for consistency
    - Each shard: .npz file with concatenated arrays and offsets for memory mapping
    - Each element: List[Dict] of auto masks for an image
    - Provides O(1) random access and on-demand shard loading with memory mapping
    """

    def __init__(self, shard_paths: List[Path]):
        self._loaded = [None] * len(shard_paths)
        super().__init__(shard_paths)

    def _setup_lengths(self):
        for p in self.paths:
            # Load just shape info without loading full data
            with np.load(p, mmap_mode='r') as data:
                self.lengths.append(int(len(data['num_masks'])))

    def _get_shard(self, sid: int):
        if self._loaded[sid] is None:
            # Load with memory mapping for concatenated arrays
            self._loaded[sid] = np.load(self.paths[sid], mmap_mode='r')
        return self._loaded[sid]

    def __len__(self) -> int:
        return int(self.cum[-1])

    def __getitem__(self, idx_img: int) -> List[dict]:
        sid, loc = self._loc(int(idx_img))
        shard_data = self._get_shard(sid)

        # Get per-image counts and offsets
        num_masks = int(shard_data['num_masks'][loc])
        image_start = int(shard_data['image_offsets'][loc])            # byte offset into mask_data
        image_end = int(shard_data['image_offsets'][loc + 1])

        # Compute per-mask index range for metadata arrays (shapes/bboxes/scores)
        # Use cumulative sum of num_masks up to this image index
        mask_start = int(np.sum(shard_data['num_masks'][:loc]))
        mask_end = mask_start + num_masks

        if num_masks == 0:
            return []

        # Extract data for this image using offsets
        mask_data = shard_data['mask_data'][image_start:image_end]
        mask_shapes = shard_data['mask_shapes'][mask_start:mask_end]
        bbox_data = shard_data['bbox_data'][mask_start:mask_end]
        pred_iou_data = shard_data['pred_iou_data'][mask_start:mask_end]
        area_data = shard_data['area_data'][mask_start:mask_end]

        packed_entry = {
            "num_masks": num_masks,
            "mask_data": mask_data,
            "mask_shapes": mask_shapes,
            "bbox_data": bbox_data,
            "pred_iou_data": pred_iou_data,
            "area_data": area_data
        }

        return self._unpack_auto_masks(packed_entry)

    def _unpack_auto_masks(self, packed_data: dict) -> List[dict]:
        """Unpack auto masks from memory-mappable format back to original format."""
        if packed_data["num_masks"] == 0:
            return []

        decoded = []
        mask_data = packed_data["mask_data"]
        mask_shapes = packed_data["mask_shapes"]
        bbox_data = packed_data["bbox_data"]
        pred_iou_data = packed_data["pred_iou_data"]
        area_data = packed_data["area_data"]

        start_idx = 0
        for i in range(packed_data["num_masks"]):
            h, w = mask_shapes[i]
            bbox = bbox_data[i]
            pred_iou = pred_iou_data[i]
            area = area_data[i]

            # Calculate packed size for this mask
            packed_size = (h * w + 7) // 8  # Round up for packbits
            end_idx = start_idx + packed_size

            # Extract and unpack this mask's data
            packed = mask_data[start_idx:end_idx]
            flat = np.unpackbits(packed)[:h * w]
            seg_arr = flat.reshape(h, w).astype(bool)

            # Reconstruct original mask dict
            mask_dict = {
                "segmentation": seg_arr,
                "bbox": bbox.tolist(),
                "predicted_iou": float(pred_iou),
                "area": float(area)
            }
            decoded.append(mask_dict)

            start_idx = end_idx

        return decoded


def pack_auto_masks(auto_masks: List[dict]) -> dict:
    """Pack auto masks into a memory-mappable format.

    Returns a dict with:
    - num_masks: number of masks per image
    - mask_data: packed binary data for all masks
    - mask_shapes: shapes of each mask
    - bbox_data: bbox coordinates as float32 array
    - pred_iou_data: predicted IoU scores as float32 array
    - area_data: mask areas as float32 array
    """
    if not auto_masks:
        return {
            "num_masks": 0,
            "mask_data": np.array([], dtype=np.uint8),
            "mask_shapes": np.array([], dtype=np.int32),
            "bbox_data": np.array([], dtype=np.float32),
            "pred_iou_data": np.array([], dtype=np.float32),
            "area_data": np.array([], dtype=np.float32)
        }

    # Collect all mask data
    all_packed_data = []
    all_shapes = []
    all_bboxes = []
    all_scores = []
    all_areas = []

    for m in auto_masks:
        seg = m.get("segmentation", None)
        if isinstance(seg, np.ndarray):
            h, w = seg.shape
            packed = np.packbits(seg.astype(np.uint8).reshape(-1))
            all_packed_data.append(packed)
            all_shapes.append([h, w])

            # Extract bbox, score, and area as regular arrays
            bbox = m.get("bbox", [0, 0, 0, 0])
            score = m.get("predicted_iou", m.get("score", 0.0))
            area = m.get("area", h * w)  # Default to full mask area if not provided
            all_bboxes.append(bbox)
            all_scores.append(score)
            all_areas.append(area)

    if not all_packed_data:
        return {
            "num_masks": 0,
            "mask_data": np.array([], dtype=np.uint8),
            "mask_shapes": np.array([], dtype=np.int32),
            "bbox_data": np.array([], dtype=np.float32),
            "pred_iou_data": np.array([], dtype=np.float32),
            "area_data": np.array([], dtype=np.float32)
        }

    # Concatenate all packed data
    total_packed_size = sum(len(data) for data in all_packed_data)
    combined_data = np.empty(total_packed_size, dtype=np.uint8)

    start_idx = 0
    for data in all_packed_data:
        end_idx = start_idx + len(data)
        combined_data[start_idx:end_idx] = data
        start_idx = end_idx

    return {
        "num_masks": len(all_packed_data),  # Actual number of packed masks
        "mask_data": combined_data,
        "mask_shapes": np.array(all_shapes, dtype=np.int32),
        "bbox_data": np.array(all_bboxes, dtype=np.float32),
        "pred_iou_data": np.array(all_scores, dtype=np.float32),
        "area_data": np.array(all_areas, dtype=np.float32)
    }


def pack_batch_auto_masks(batch_masks: List[List[dict]]) -> dict:
    """Pack a batch of auto masks into shard format.

    Args:
        batch_masks: List of auto mask lists, one per image

    Returns:
        dict with concatenated arrays ready for np.savez_compressed:
        - num_masks: per-image mask counts
        - mask_data: concatenated packed bytes
        - mask_shapes: concatenated (H,W) per mask
        - bbox_data: concatenated bbox coordinates
        - pred_iou_data: concatenated predicted IoU scores
        - area_data: concatenated mask areas
        - image_offsets: byte offsets into mask_data per image
    """
    all_num_masks = []
    all_mask_data = []
    all_mask_shapes = []
    all_bbox_data = []
    all_pred_iou_data = []
    all_area_data = []
    image_offsets = [0]

    for image_masks in batch_masks:
        packed = pack_auto_masks(image_masks)
        all_num_masks.append(packed["num_masks"])
        all_mask_data.append(packed["mask_data"])
        all_mask_shapes.append(packed["mask_shapes"])
        all_bbox_data.append(packed["bbox_data"])
        all_pred_iou_data.append(packed["pred_iou_data"])
        all_area_data.append(packed["area_data"])
        image_offsets.append(image_offsets[-1] + len(packed["mask_data"]))

    return {
        "num_masks": np.array(all_num_masks, dtype=np.int32),
        "mask_data": np.concatenate(all_mask_data) if all_mask_data else np.array([], dtype=np.uint8),
        "mask_shapes": np.concatenate(all_mask_shapes) if all_mask_shapes else np.array([], dtype=np.int32),
        "bbox_data": np.concatenate(all_bbox_data) if all_bbox_data else np.array([], dtype=np.float32),
        "pred_iou_data": np.concatenate(all_pred_iou_data) if all_pred_iou_data else np.array([], dtype=np.float32),
        "area_data": np.concatenate(all_area_data) if all_area_data else np.array([], dtype=np.float32),
        "image_offsets": np.array(image_offsets, dtype=np.int32)
    }
