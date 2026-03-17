import asyncio
import concurrent.futures
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


def parse_comma_separated_labels(raw_text: str) -> List[str]:
    """Parse comma-separated labels and drop empty tokens."""
    return [x.strip() for x in raw_text.split(",") if x.strip()]


def l2_normalize_embeddings(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize embedding tensors on the last dimension."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def compute_similarity_scores(
    clip_features: torch.Tensor,
    pos_embed: torch.Tensor,
    neg_embed: Optional[torch.Tensor] = None,
    softmax_temp: float = 0.1,
) -> torch.Tensor:
    """Compute similarity exactly following F3RM model semantics."""
    if pos_embed.ndim == 1:
        pos_embed = pos_embed.unsqueeze(0)

    clip_features = l2_normalize_embeddings(clip_features.float())
    pos_embed = l2_normalize_embeddings(pos_embed.float())
    clip_features = clip_features.to(dtype=pos_embed.dtype)

    if neg_embed is None:
        return clip_features @ pos_embed.T

    neg_embed = l2_normalize_embeddings(neg_embed.float())
    text_embs = torch.cat([pos_embed, neg_embed], dim=0)
    raw_sims = clip_features @ text_embs.T
    pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
    pos_sims = pos_sims.broadcast_to(neg_sims.shape)
    paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)
    probs = (paired_sims / max(float(softmax_temp), 1e-6)).softmax(dim=-1)[..., :1]
    torch.nan_to_num_(probs, nan=0.0)
    sims, _ = probs.min(dim=-1, keepdim=True)
    return sims


def resolve_devices_and_workers(device: torch.device, batch_size_per_gpu: int) -> Tuple[Optional[torch.device], int]:
    """Return (devices_param, num_workers) for AsyncMultiWrapper using per-GPU worker count."""
    if device.type == "cuda":
        if device.index is None:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            num_workers = max(1, (n_gpus or 1) * max(1, batch_size_per_gpu))
            return None, num_workers
        num_workers = max(1, batch_size_per_gpu)
        return torch.device(f"cuda:{device.index}"), num_workers
    return torch.device("cpu"), 1


def run_async_in_any_context(coro_fn: Callable[[], Any]) -> Any:
    """Run an async coroutine function regardless of existing event loop."""
    try:
        asyncio.get_running_loop()

        def _thread_run():
            return asyncio.run(coro_fn())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_thread_run)
            return fut.result()
    except RuntimeError:
        return asyncio.run(coro_fn())


class BatchFeatureLoader:
    """Batch feature loader for per-image CLIP features."""

    def __init__(
        self,
        data_dir: Path,
        feature_type: str,
        image_fnames: List[str],
        device: torch.device,
        max_cpu_images: int = 128,
        max_gpu_images: int = 16,
        pin_cpu_tensors: bool = True,
    ):
        if feature_type != "CLIP":
            raise ValueError(f"Unsupported feature type: {feature_type}")

        self.data_dir = data_dir
        self.feature_type = feature_type
        self.image_fnames = image_fnames
        self.device = device
        self.root, _ = get_cache_paths(data_dir, feature_type)
        self.max_cpu_images = int(max_cpu_images)
        self.max_gpu_images = int(max_gpu_images)
        self._cpu_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._gpu_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._use_pinned = torch.cuda.is_available() and bool(pin_cpu_tensors)
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        sample_features = self._load_single_image_cpu(0)
        self.H, self.W, self.C = sample_features.shape
        self.dtype = sample_features.dtype

    def _load_single_image_cpu(self, img_idx: int) -> torch.Tensor:
        data = np.load(self.root / f"image_{img_idx:06d}.npy", mmap_mode="r")
        tensor = torch.from_numpy(data)
        return tensor.pin_memory() if self._use_pinned else tensor

    def _get_cpu_tensor(self, img_idx: int) -> torch.Tensor:
        if img_idx in self._cpu_cache:
            tensor = self._cpu_cache.pop(img_idx)
            self._cpu_cache[img_idx] = tensor
            return tensor
        tensor = self._load_single_image_cpu(img_idx)
        self._cpu_cache[img_idx] = tensor
        if len(self._cpu_cache) > self.max_cpu_images:
            self._cpu_cache.popitem(last=False)
        return tensor

    def _get_gpu_tensor(self, img_idx: int) -> torch.Tensor:
        if img_idx in self._gpu_cache:
            tensor = self._gpu_cache.pop(img_idx)
            self._gpu_cache[img_idx] = tensor
            return tensor
        cpu_tensor = self._get_cpu_tensor(img_idx)
        if self._stream:
            with torch.cuda.stream(self._stream):
                gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self._stream)
        else:
            gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
        self._gpu_cache[img_idx] = gpu_tensor
        if len(self._gpu_cache) > self.max_gpu_images:
            self._gpu_cache.popitem(last=False)
        return gpu_tensor

    def load_batch_images(self, camera_indices: torch.Tensor) -> Dict[int, torch.Tensor]:
        batch_features: Dict[int, torch.Tensor] = {}
        for cam_idx in camera_indices.unique():
            cam_idx_int = int(cam_idx.item())
            batch_features[cam_idx_int] = self._get_gpu_tensor(cam_idx_int)
        return batch_features

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._get_gpu_tensor(index)


def get_cache_paths(data_dir: Path, feature_type: str) -> Tuple[Path, Path]:
    root = data_dir / "features" / feature_type.lower()
    return root, root / "meta.pt"
