import asyncio
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from f3rm.features.utils import AsyncMultiWrapper, resolve_devices_and_workers, run_async_in_any_context


class SAMArgs:
    checkpoint_path: str = "/robodata/smodak/repos/f3rm/checkpoints/sam_vit_h_4b8939.pth"
    model_type: str = "vit_h"
    sort_mode: str = "area"
    min_mask_area_perc: float = 0.01
    points_per_side: Optional[int] = 8
    pred_iou_thresh: Optional[float] = None
    stability_score_thresh: Optional[float] = None
    box_nms_thresh: Optional[float] = None
    crop_n_layers: Optional[int] = None
    crop_nms_thresh: Optional[float] = None
    workers_per_gpu: int = 2

    @classmethod
    def resolved_checkpoint_path(cls) -> str:
        return str(Path(cls.checkpoint_path).expanduser().resolve())

    @classmethod
    def id_dict(cls) -> dict[str, Any]:
        return {
            "checkpoint_path": cls.resolved_checkpoint_path(),
            "model_type": cls.model_type,
            "sort_mode": cls.sort_mode,
            "min_mask_area_perc": cls.min_mask_area_perc,
            "points_per_side": cls.points_per_side,
            "pred_iou_thresh": cls.pred_iou_thresh,
            "stability_score_thresh": cls.stability_score_thresh,
            "box_nms_thresh": cls.box_nms_thresh,
            "crop_n_layers": cls.crop_n_layers,
            "crop_nms_thresh": cls.crop_nms_thresh,
        }


def _load_image_rgb(image: Union[str, Path, Image.Image]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil_image:
            return np.array(pil_image.convert("RGB"), dtype=np.uint8, copy=True)
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    raise TypeError("image must be a path or PIL image")


def _mask_score(mask: dict[str, Any], sort_mode: str) -> float:
    predicted_iou = float(mask.get("predicted_iou", 0.0))
    stability = float(mask.get("stability_score", 0.0))
    area = float(mask.get("area", 0.0))

    if sort_mode == "score":
        return predicted_iou * stability
    if sort_mode == "predicted_iou":
        return predicted_iou
    if sort_mode == "stability":
        return stability
    if sort_mode == "area":
        return area
    raise ValueError(f"Unknown sort_mode={sort_mode!r}")


def _flatten_masks(
    masks: List[dict[str, Any]],
    image_shape: tuple[int, int, int],
    sort_mode: str,
    min_mask_area_perc: float,
) -> np.ndarray:
    height, width = image_shape[:2]
    min_mask_area_pixels = float(min_mask_area_perc) * height * width

    if min_mask_area_pixels > 0.0:
        masks = [mask for mask in masks if float(mask.get("area", 0.0)) >= min_mask_area_pixels]
    if not masks:
        return np.full((height, width), -1, dtype=np.int32)

    scores = np.asarray([_mask_score(mask, sort_mode) for mask in masks], dtype=np.float32)
    winning_raw_idx = np.full((height, width), -1, dtype=np.int32)
    winning_score = np.full((height, width), -np.inf, dtype=np.float32)
    order = np.argsort(scores)[::-1]

    for raw_idx in order:
        segmentation = np.asarray(masks[raw_idx]["segmentation"], dtype=bool)
        better = segmentation & (scores[raw_idx] > winning_score)
        winning_raw_idx[better] = int(raw_idx)
        winning_score[better] = float(scores[raw_idx])

    flat_labels = np.full((height, width), -1, dtype=np.int32)
    compact_id = 0
    for raw_idx in order:
        claimed = winning_raw_idx == int(raw_idx)
        area_claimed = int(claimed.sum())
        if area_claimed == 0:
            continue
        if min_mask_area_pixels > 0.0 and area_claimed < min_mask_area_pixels:
            continue
        flat_labels[claimed] = compact_id
        compact_id += 1
    return flat_labels


class _SAMWorker(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_l",
        sort_mode: str = "area",
        min_mask_area_perc: float = 0.01,
        points_per_side: Optional[int] = 8,
        pred_iou_thresh: Optional[float] = None,
        stability_score_thresh: Optional[float] = None,
        box_nms_thresh: Optional[float] = None,
        crop_n_layers: Optional[int] = None,
        crop_nms_thresh: Optional[float] = None,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__()
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        requested_device = torch.device(device)
        self.device = torch.device(requested_device if requested_device.type != "cuda" or torch.cuda.is_available() else "cpu")
        self.sort_mode = sort_mode
        self.min_mask_area_perc = float(min_mask_area_perc)

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)

        amg_kwargs: dict[str, Any] = {}
        if points_per_side is not None:
            amg_kwargs["points_per_side"] = points_per_side
        if pred_iou_thresh is not None:
            amg_kwargs["pred_iou_thresh"] = pred_iou_thresh
        if stability_score_thresh is not None:
            amg_kwargs["stability_score_thresh"] = stability_score_thresh
        if box_nms_thresh is not None:
            amg_kwargs["box_nms_thresh"] = box_nms_thresh
        if crop_n_layers is not None:
            amg_kwargs["crop_n_layers"] = crop_n_layers
        if crop_nms_thresh is not None:
            amg_kwargs["crop_nms_thresh"] = crop_nms_thresh

        self.mask_generator = SamAutomaticMaskGenerator(sam, **amg_kwargs)

    @torch.inference_mode()
    def extract_labels(self, image: Union[str, Path, Image.Image], output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        image_rgb = _load_image_rgb(image)
        masks = self.mask_generator.generate(image_rgb)
        flat_labels = _flatten_masks(
            masks=masks,
            image_shape=image_rgb.shape,
            sort_mode=self.sort_mode,
            min_mask_area_perc=self.min_mask_area_perc,
        )
        labels = torch.from_numpy(flat_labels)
        if output_dtype is not None:
            labels = labels.to(dtype=output_dtype)
        return labels


class SAMExtractor:
    def __init__(self, device: torch.device, verbose: bool = False) -> None:
        devices_param, num_workers = resolve_devices_and_workers(device, SAMArgs.workers_per_gpu)
        if verbose:
            print("Initializing SAM workers")
        self.client_workers = AsyncMultiWrapper(
            _SAMWorker,
            num_objects=num_workers,
            devices=devices_param,
            checkpoint_path=SAMArgs.resolved_checkpoint_path(),
            model_type=SAMArgs.model_type,
            sort_mode=SAMArgs.sort_mode,
            min_mask_area_perc=SAMArgs.min_mask_area_perc,
            points_per_side=SAMArgs.points_per_side,
            pred_iou_thresh=SAMArgs.pred_iou_thresh,
            stability_score_thresh=SAMArgs.stability_score_thresh,
            box_nms_thresh=SAMArgs.box_nms_thresh,
            crop_n_layers=SAMArgs.crop_n_layers,
            crop_nms_thresh=SAMArgs.crop_nms_thresh,
        ).workers
        if verbose:
            print("Warming up SAM workers...")
        tiny = Image.new("RGB", (8, 8), color=0)
        for worker in self.client_workers:
            _ = worker.extract_labels(image=tiny)

    async def _stream_batch_async(
        self,
        image_paths: List[str],
        on_result: Callable[[int, torch.Tensor], None],
        output_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if not image_paths:
            return

        loop = asyncio.get_running_loop()
        queue = iter(enumerate(image_paths))

        with tqdm(total=len(image_paths), desc="SAM tasks", leave=False) as pbar:
            async def _worker_loop(worker: _SAMWorker) -> None:
                while True:
                    try:
                        image_idx, image_path = next(queue)
                    except StopIteration:
                        return
                    result = await loop.run_in_executor(
                        None,
                        lambda path=image_path, sam_worker=worker: sam_worker.extract_labels(
                            image=path,
                            output_dtype=output_dtype,
                        ),
                    )
                    on_result(image_idx, result)
                    pbar.update(1)

            await asyncio.gather(*(_worker_loop(worker) for worker in self.client_workers))

    def stream_batch(
        self,
        image_paths: List[str],
        on_result: Callable[[int, torch.Tensor], None],
        output_dtype: Optional[torch.dtype] = None,
    ) -> None:
        run_async_in_any_context(lambda: self._stream_batch_async(image_paths, on_result, output_dtype=output_dtype))


def colorize_labels(label_map: np.ndarray, seed: int = 0) -> np.ndarray:
    height, width = label_map.shape
    out = np.zeros((height, width, 3), dtype=np.uint8)
    labels = np.unique(label_map)
    labels = labels[labels >= 0]
    if len(labels) == 0:
        return out
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 255, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    for label in labels:
        out[label_map == int(label)] = colors[int(label)]
    return out


def examine_saved(sam_feat_dir: str) -> None:
    """Create an MP4 video of saved SAM labels using categorical colors."""
    import cv2

    meta_path = os.path.join(sam_feat_dir, "meta.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"SAM meta not found at {meta_path}")

    meta = torch.load(meta_path, map_location="cpu")
    image_fnames = meta["image_fnames"]
    n_images = len(image_fnames)

    first_labels = np.load(os.path.join(sam_feat_dir, "image_000000.npy"))
    height, width = first_labels.shape
    video_path = os.path.join(sam_feat_dir, "features_viz.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 2.0, (width, height))

    try:
        for image_idx in tqdm(range(n_images), desc="Creating SAM features video"):
            feat_path = os.path.join(sam_feat_dir, f"image_{image_idx:06d}.npy")
            labels = np.load(feat_path)
            frame = colorize_labels(labels)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not created at {video_path}")
