from typing import Optional, Tuple, Callable, Any, List, Dict
from collections import OrderedDict
import json

import asyncio
import concurrent.futures
import numpy as np
import torch
from pathlib import Path
from PIL import Image

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
    """Compute similarity exactly following F3RM model semantics.

    Returns:
        If `neg_embed` is None: cosine similarities to positives in [-1, 1].
        Else: paired-softmax min probability in [0, 1].
    """
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


def parse_prefixed_feature_type(feature_type: str, prefix: str) -> List[str]:
    if not feature_type.startswith(prefix):
        raise ValueError(f"Invalid feature type: {feature_type}. Must start with '{prefix}'")
    prompts_part = feature_type[len(prefix):]
    if prompts_part == "":
        return []
    return [w.lower() for w in prompts_part.split("_") if w.strip()]


def compose_prefixed_feature_type(prefix: str, text_prompts: Optional[List[str]]) -> str:
    if text_prompts is None or len(text_prompts) == 0:
        return prefix
    joined = "_".join([p.lower() for p in text_prompts])
    return f"{prefix}{joined}"


def infer_sam3_feature_type(text_prompts: Optional[List[str]]) -> str:
    return compose_prefixed_feature_type("SAM3_", text_prompts)


def get_sam3_feature_root(data_dir: Path, sam3_feature_type: str) -> Path:
    return Path(data_dir) / "features" / sam3_feature_type.lower()


def ensure_sam3_feature_cache(
    data_dir: Path,
    sam3_feature_type: str,
    *,
    consumer: Optional[str] = None,
    require_npz: bool = True,
) -> Path:
    sam3_root = get_sam3_feature_root(data_dir, sam3_feature_type)
    meta_path = sam3_root / "meta.pt"
    if not meta_path.exists():
        if consumer is None:
            raise FileNotFoundError(f"Missing SAM3 meta: {meta_path}")
        raise FileNotFoundError(
            f"Missing SAM3 meta: {meta_path} (expected for {consumer}). "
            f"Run SAM3 extraction for feature type '{sam3_feature_type}' first."
        )
    if require_npz and not list(sam3_root.glob("image_*.npz")):
        raise FileNotFoundError(f"Missing SAM3 per-image features under {sam3_root}")
    return sam3_root


def load_sam3_image_fnames(data_dir: Path, sam3_feature_type: str) -> List[str]:
    sam3_root = ensure_sam3_feature_cache(data_dir, sam3_feature_type, require_npz=False)
    sam3_meta = torch.load(sam3_root / "meta.pt")
    return [str(p) for p in sam3_meta["image_fnames"]]


def normalize_sam3_masks(raw_masks: Any, image_path: Optional[str] = None) -> np.ndarray:
    masks = np.asarray(raw_masks)
    if masks.ndim == 4:
        masks = masks[:, 0, ...]
    elif masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim != 3:
        if image_path is None:
            raise ValueError(f"Invalid SAM3 mask shape: {masks.shape}")
        with Image.open(image_path) as img:
            return np.zeros((0, img.height, img.width), dtype=bool)
    return masks.astype(bool)


class BatchFeatureLoader:
    """Simple batch feature loader for per-image stored features."""

    def __init__(self, data_dir: Path, feature_type: str, image_fnames: List[str], device: torch.device,
                 max_cpu_images: int = 128, max_gpu_images: int = 16):
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.image_fnames = image_fnames
        self.device = device
        self.root, _ = get_cache_paths(data_dir, feature_type)
        # CPU/GPU LRU caches
        self.max_cpu_images = int(max_cpu_images)
        self.max_gpu_images = int(max_gpu_images)
        self._cpu_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._gpu_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._use_pinned = torch.cuda.is_available()
        # Prefetch stream for async H2D
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        sample_features = self._load_single_image_cpu(0)
        if feature_type == "CLIP":
            self.H, self.W, self.C = sample_features.shape
            self.dtype = sample_features.dtype
        elif feature_type.startswith("FOREGROUND_"):
            self.H, self.W, self.C = sample_features.shape
            self.dtype = sample_features.dtype
        elif feature_type.startswith("SAM3_"):
            self.K, _, self.H, self.W = sample_features.shape
            self.dtype = sample_features.dtype
        del sample_features

    def _load_single_image_cpu(self, img_idx: int):
        """Load features for a single image onto CPU (optionally pinned)."""
        if self.feature_type == "CLIP":
            data = np.load(self.root / f"image_{img_idx:06d}.npy", mmap_mode="r")
            t = torch.from_numpy(data)
            return t.pin_memory() if self._use_pinned else t
        elif self.feature_type.startswith("FOREGROUND_"):
            data = np.load(self.root / f"image_{img_idx:06d}.npy", mmap_mode="r")
            t = torch.from_numpy(data)
            return t.pin_memory() if self._use_pinned else t
        elif self.feature_type.startswith("SAM3_"):
            data = np.load(self.root / f"image_{img_idx:06d}.npz")
            masks = data["masks"]
            return masks
        elif self.feature_type == "TEXT":
            with open(self.root / f"image_{img_idx:06d}.json", 'r') as f:
                return json.load(f)

        raise ValueError(f"Unknown feature type: {self.feature_type}")

    def _get_cpu_tensor(self, img_idx: int) -> torch.Tensor:
        if img_idx in self._cpu_cache:
            t = self._cpu_cache.pop(img_idx)
            self._cpu_cache[img_idx] = t
            return t
        t = self._load_single_image_cpu(img_idx)
        self._cpu_cache[img_idx] = t
        if len(self._cpu_cache) > self.max_cpu_images:
            self._cpu_cache.popitem(last=False)
        return t

    def _get_gpu_tensor(self, img_idx: int) -> torch.Tensor:
        if img_idx in self._gpu_cache:
            t = self._gpu_cache.pop(img_idx)
            self._gpu_cache[img_idx] = t
            return t
        cpu_t = self._get_cpu_tensor(img_idx)
        if self._stream:
            with torch.cuda.stream(self._stream):
                gpu_t = cpu_t.to(self.device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self._stream)
        else:
            gpu_t = cpu_t.to(self.device, non_blocking=True)
        self._gpu_cache[img_idx] = gpu_t
        if len(self._gpu_cache) > self.max_gpu_images:
            # Evict least-recently-used
            old_idx, old_t = self._gpu_cache.popitem(last=False)
            del old_t
        return gpu_t

    def load_batch_images(self, camera_indices: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Load features for a batch of camera indices using CPU/GPU LRU caches and async H2D."""
        batch_features: Dict[int, torch.Tensor] = {}
        unique_indices = camera_indices.unique()
        for cam_idx in unique_indices:
            cam_idx_int = int(cam_idx.item())
            # Only tensor-backed features are cached on GPU. For list/JSON types we fall back to CPU read.
            if self.feature_type == "CLIP" or self.feature_type.startswith("FOREGROUND_"):
                batch_features[cam_idx_int] = self._get_gpu_tensor(cam_idx_int)
            else:
                # CPU-backed feature types (e.g. SAM3/TEXT)
                batch_features[cam_idx_int] = self._load_single_image_cpu(cam_idx_int)
        return batch_features

    def __getitem__(self, index: int):
        """Direct access by image index for pipeline compatibility."""
        if self.feature_type == "CLIP" or self.feature_type.startswith("FOREGROUND_"):
            return self._get_gpu_tensor(index)
        else:
            return self._load_single_image_cpu(index)


def get_cache_paths(data_dir: Path, feature_type: str) -> Tuple[Path, Path]:
    """Get cache directory and metadata paths for a feature type."""
    root = data_dir / "features" / feature_type.lower()
    return root, root / "meta.pt"
