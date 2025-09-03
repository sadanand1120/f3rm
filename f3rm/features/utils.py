from typing import Optional, Tuple, Callable, Any, List, Dict
import json

import asyncio
import concurrent.futures
import numpy as np
import os
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path

from nerfstudio.cameras import camera_utils


def get_conf_temp_scaled_logits(logits, confidence, drop_exp_factor=6, eps=1e-8):
    """
    Scale logits by confidence.
    T = 1/max(confidence^drop_exp_factor, eps). Lower confidence -> higher T -> flatter probs.
    """
    x = torch.as_tensor(logits, dtype=torch.float32)
    T = 1.0 / max(float(confidence)**drop_exp_factor, eps)
    return x / T


def probs_to_normal(
    probs: torch.Tensor,
    *,
    n_bins: int,
    angle_min_deg: float,
    period_deg: float,                 # here: total span (e.g., 180.0)
    bin_width_deg: Optional[float] = None,
    min_std_deg: float = 1e-3,
    window_deg: float = 30.0,          # fit window half-width
    peak_frac: float = 0.2,            # keep bins >= peak_frac * p_max
    gamma: float = 2.0,                # weights: w = p**gamma
    ridge: float = 0.0,                # tiny L2 on LS (e.g., 1e-4) stabilizes
    shrink_to_moment: float = 0.5,     # 0..1; blend LS σ toward moment σ
    sigma_floor_deg: Optional[float] = 1.0,
    sigma_ceil_deg: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a single Gaussian N(mean_deg, std_deg) on a bounded linear range [angle_min, angle_min+period).
    Mean: sub-bin peak via quadratic fit on log-probs (no wrap).
    Std: weighted LS on log p ≈ -(x-μ)^2/(2σ^2), shrunk toward local moment.
    """
    p = probs.float()
    assert p.ndim == 1 and p.numel() == n_bins, "probs must be 1D with length n_bins"
    if bin_width_deg is None:
        bin_width_deg = period_deg / n_bins
    else:
        if not float(abs(n_bins * bin_width_deg - period_deg)) < 1e-6:
            raise ValueError("n_bins*bin_width_deg must equal period_deg")

    eps = 1e-12
    p = p + eps
    n = n_bins

    # ---- mode & sub-bin refinement (no wrapping at edges)
    k = int(torch.argmax(p).item())
    km = k - 1 if k - 1 >= 0 else k
    kp = k + 1 if k + 1 < n else k
    Lm, L0, Lp = torch.log(p[km]), torch.log(p[k]), torch.log(p[kp])
    denom = (Lm - 2.0 * L0 + Lp)
    concave = bool((denom < -1e-9).item() and km != k and kp != k)

    if concave:
        delta = 0.5 * (Lm - Lp) / denom
        delta = torch.clamp(delta, -0.5, 0.5)
        mean_bins = torch.tensor(float(k), dtype=p.dtype, device=p.device) + delta
    else:
        mean_bins = torch.tensor(float(k), dtype=p.dtype, device=p.device)

    mean_deg = angle_min_deg + mean_bins * bin_width_deg

    # ---- select local window (linear, clipped to range)
    idx = torch.arange(n, device=p.device, dtype=p.dtype)
    centers = angle_min_deg + idx * bin_width_deg
    diff = centers - mean_deg
    mask = (diff.abs() <= window_deg) & (p >= (peak_frac * p[k]))
    if int(mask.sum().item()) < 3:
        # fallback: curvature or minimal std
        if concave:
            # from denom ≈ -1/σ_bins^2  -> σ ≈ 1/sqrt(-denom) bins
            sigma_bins = 1.0 / torch.sqrt(-denom)
            std_deg = sigma_bins * bin_width_deg
        else:
            std_deg = torch.tensor(sigma_floor_deg or min_std_deg, device=p.device, dtype=p.dtype)
        std_deg = torch.clamp(std_deg, min=min_std_deg)
        if sigma_floor_deg is not None:
            std_deg = torch.clamp(std_deg, min=sigma_floor_deg)
        if sigma_ceil_deg is not None:
            std_deg = torch.clamp(std_deg, max=sigma_ceil_deg)
        return mean_deg, std_deg

    # ---- weighted LS for σ using y = log p - log p(μ) ≈ -(x-μ)^2/(2σ^2)
    x = diff[mask]                          # degrees
    y = torch.log(p[mask]) - torch.log(p[k])
    X = x * x
    w = (p[mask] ** gamma)
    denom_ls = (w * X * X).sum() + ridge + 1e-12
    s = (w * X * y).sum() / denom_ls        # slope
    sigma_ls = torch.sqrt(torch.clamp(-1.0 / (2.0 * s + 1e-12), min=min_std_deg**2))

    # ---- moment-based σ (conservative)
    mu_local = (w * x).sum() / w.sum()
    var_local = torch.clamp((w * (x - mu_local) ** 2).sum() / w.sum(), min=min_std_deg**2)
    sigma_mom = torch.sqrt(var_local)

    # ---- shrink & clamp
    alpha = float(torch.clamp(torch.tensor(shrink_to_moment), 0.0, 1.0))
    std_deg = (1 - alpha) * sigma_ls + alpha * sigma_mom
    if sigma_floor_deg is not None:
        std_deg = torch.clamp(std_deg, min=sigma_floor_deg)
    if sigma_ceil_deg is not None:
        std_deg = torch.clamp(std_deg, max=sigma_ceil_deg)
    std_deg = torch.clamp(std_deg, min=min_std_deg)

    return mean_deg, std_deg


def normal_to_probs(
    mean_deg: torch.Tensor,
    std_deg: torch.Tensor,
    *,
    n_bins: int,
    angle_min_deg: float,
    period_deg: float,
    bin_width_deg: Optional[float] = None,
    integrate_bins: bool = True,
    subsamples_per_bin: int = 7,   # odd recommended (includes center)
) -> torch.Tensor:
    """
    Expand (mean_deg, std_deg) to discrete probs over [angle_min, angle_min+period).
    If integrate_bins=True, averages inside each bin to approximate bin mass.
    """
    if bin_width_deg is None:
        bin_width_deg = period_deg / n_bins
    else:
        if not float(abs(n_bins * bin_width_deg - period_deg)) < 1e-6:
            raise ValueError("n_bins*bin_width_deg must equal period_deg")

    device, dtype = mean_deg.device, mean_deg.dtype
    idx = torch.arange(n_bins, device=device, dtype=dtype)
    centers = angle_min_deg + idx * bin_width_deg

    mean = mean_deg[..., None]
    std = torch.clamp(std_deg, min=1e-6)[..., None]

    if (not integrate_bins) or subsamples_per_bin <= 1:
        diff = centers - mean
        logits = -0.5 * (diff / std) ** 2
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.exp(logits)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    J = int(subsamples_per_bin)
    if J % 2 == 0:
        J += 1
    offs = (torch.linspace(-0.5, 0.5, J, device=device, dtype=dtype) * bin_width_deg)  # (J,)

    c = centers.view(1, 1, n_bins)     # (1,J,n)
    o = offs.view(1, J, 1)             # (1,J,1)
    m = mean[..., None]                # (...,1,1)
    s = std[..., None]                 # (...,1,1)

    samp = c + o                       # (...,J,n)
    diff = samp - m
    logits = -0.5 * (diff / s) ** 2
    vals = torch.exp(logits)           # (...,J,n)
    probs = vals.mean(dim=-2)          # (...,n)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def _kappa_from_resultant(R: torch.Tensor) -> torch.Tensor:
    # Approx inverse of A(kappa)=I1/I0 (Mardia & Jupp style piecewise)
    eps = 1e-12
    R = torch.clamp(R, 0.0, 0.999999)
    k1 = 2 * R + R**3 + 5 * R**5 / 6
    k2 = -0.4 + 1.39 * R + 0.43 / (1 - R + eps)
    k3 = 1.0 / (R**3 - 4 * R**2 + 3 * R + eps)
    return torch.where(R < 0.53, k1, torch.where(R < 0.85, k2, k3))


def probs_to_von_mises(
    probs: torch.Tensor,
    *,
    n_bins: int,
    angle_min_deg: float,
    period_deg: float,
    bin_width_deg: Optional[float] = None,
    min_kappa: float = 1e-3,
    window_deg: float = 30.0,
    peak_frac: float = 0.2,
    gamma: float = 2.0,
    ridge: float = 0.0,              # small L2 on LS fit (e.g., 1e-4) to tame κ
    shrink_to_resultant: float = 0.5,  # 0..1; blend LS κ toward resultant-based κ
    sigma_floor_deg: float = 2.0,    # don't fit narrower than this (deg)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit (mean_deg, kappa) from a discrete circular distribution.
    - sub-bin peak via quadratic on log p around argmax
    - κ via weighted LS on log p ≈ -(κ/2) δ^2, shrunk toward resultant-based κ
    - κ capped so std >= sigma_floor_deg
    """
    p = probs.float()
    assert p.ndim == 1 and p.numel() == n_bins, "probs must be 1D with length n_bins"

    if bin_width_deg is None:
        bin_width_deg = period_deg / n_bins
    else:
        if not float(abs(n_bins * bin_width_deg - period_deg)) < 1e-6:
            raise ValueError("n_bins*bin_width_deg must equal period_deg")

    eps = 1e-12
    p = p + eps
    n = n_bins

    # ---- sub-bin peak
    k = int(torch.argmax(p).item())
    km, kp = (k - 1) % n, (k + 1) % n
    Lm, L0, Lp = torch.log(p[km]), torch.log(p[k]), torch.log(p[kp])
    denom = (Lm - 2.0 * L0 + Lp)
    concave = bool((denom < -1e-9).item())

    if concave:
        delta = 0.5 * (Lm - Lp) / denom
        delta = torch.clamp(delta, -0.5, 0.5)
        mean_bins = (torch.tensor(float(k), dtype=p.dtype, device=p.device) + delta) % n
    else:
        mean_bins = torch.tensor(float(k), dtype=p.dtype, device=p.device)

    mean_deg = (angle_min_deg + mean_bins * bin_width_deg - angle_min_deg) % period_deg + angle_min_deg

    # ---- local window
    idx = torch.arange(n, device=p.device, dtype=p.dtype)
    centers = angle_min_deg + idx * bin_width_deg
    half = period_deg / 2.0
    diff_deg = ((centers - mean_deg + half) % period_deg) - half

    peak = p[k]
    mask = (diff_deg.abs() <= window_deg) & (p >= peak_frac * peak)
    if int(mask.sum().item()) < 3:
        # fallback to curvature or minimum
        if concave:
            dtheta = torch.deg2rad(torch.tensor(bin_width_deg, device=p.device, dtype=p.dtype))
            kappa_ls = torch.clamp((-denom) / (dtheta * dtheta), min=min_kappa)
        else:
            kappa_ls = torch.tensor(min_kappa, device=p.device, dtype=p.dtype)
    else:
        x = torch.deg2rad(diff_deg[mask])                        # δ (rad)
        y = torch.log(p[mask]) - torch.log(peak)                 # log ratio
        X = x * x
        w = (p[mask] ** gamma)
        # weighted LS through origin with optional ridge
        denom_ls = (w * X * X).sum() + ridge + 1e-12
        s = (w * X * y).sum() / denom_ls                         # slope
        kappa_ls = torch.clamp(-2.0 * s, min=min_kappa)

    # ---- shrink κ toward resultant-based estimate (more conservative)
    if int(mask.sum().item()) >= 3:
        x = torch.deg2rad(diff_deg[mask])
        w = (p[mask] ** gamma)
        C = (w * torch.cos(x)).sum()
        S = (w * torch.sin(x)).sum()
        R = torch.sqrt(C * C + S * S) / (w.sum() + 1e-12)
        kappa_R = torch.clamp(_kappa_from_resultant(R), min=min_kappa)
        alpha = float(torch.clamp(torch.tensor(shrink_to_resultant), 0.0, 1.0))
        kappa = (1 - alpha) * kappa_ls + alpha * kappa_R
    else:
        kappa = kappa_ls

    # ---- cap κ so fitted std doesn't go below sigma_floor_deg
    sigma_floor_rad = torch.deg2rad(torch.tensor(sigma_floor_deg, device=p.device, dtype=p.dtype))
    kappa_cap = 1.0 / (sigma_floor_rad * sigma_floor_rad + 1e-12)
    kappa = torch.clamp(kappa, min=min_kappa, max=kappa_cap)

    return mean_deg, kappa


def von_mises_to_probs(
    mean_deg: torch.Tensor,
    kappa: torch.Tensor,
    *,
    n_bins: int,
    angle_min_deg: float,
    period_deg: float,
    bin_width_deg: Optional[float] = None,
    integrate_bins: bool = True,
    subsamples_per_bin: int = 7,
) -> torch.Tensor:
    """
    Expand (mean_deg, kappa) to discrete probs. With integrate_bins, approximate bin mass
    by averaging inside each bin (reduces peak overshoot for large κ).
    """
    if bin_width_deg is None:
        bin_width_deg = period_deg / n_bins
    else:
        if not float(abs(n_bins * bin_width_deg - period_deg)) < 1e-6:
            raise ValueError("n_bins*bin_width_deg must equal period_deg")

    device, dtype = mean_deg.device, mean_deg.dtype
    idx = torch.arange(n_bins, device=device, dtype=dtype)
    centers = angle_min_deg + idx * bin_width_deg
    half = period_deg / 2.0

    mean = mean_deg[..., None]
    kap = torch.clamp(kappa, min=1e-6)[..., None]

    if (not integrate_bins) or subsamples_per_bin <= 1:
        diff_deg = ((centers - mean + half) % period_deg) - half
        diff_rad = torch.deg2rad(diff_deg)
        logits = kap * torch.cos(diff_rad)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.exp(logits)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    J = int(subsamples_per_bin)
    if J % 2 == 0:
        J += 1
    offs = (torch.linspace(-0.5, 0.5, J, device=device, dtype=dtype) * bin_width_deg)

    c = centers.view(1, 1, n_bins)
    o = offs.view(1, J, 1)
    m = mean[..., None]

    samp = c + o
    diff_deg = ((samp - m + half) % period_deg) - half
    diff_rad = torch.deg2rad(diff_deg)

    vals = torch.exp(kap[..., None] * torch.cos(diff_rad))      # (..., J, n_bins)
    probs = vals.mean(dim=-2)                                   # (..., n_bins)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def build_transform_lookup(frames):
    return {frame["file_path"]: np.asarray(frame["transform_matrix"]) for frame in frames}


def get_nerf_ccs_to_normal_ccs_T():
    """Get the transformation matrix from NeRF CCS to normal CCS."""
    T = np.asarray([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    return T


def get_nerf_ccs_to_orig_nerf_world(filename, transforms_lookup):
    for k, v in transforms_lookup.items():
        if os.path.basename(k) == filename:
            return v
    raise KeyError(f"No transform found for {filename}")


def get_orig_to_final_nerf_world_transform_scale(dataset_transforms_path: str, model_outputdir_path: str = None):
    """
    if model output dir path not provided, assumes defaults and computes
    """
    dataset_transforms_path = Path(dataset_transforms_path)
    dataset_transforms_data = json.load(open(dataset_transforms_path, "r"))
    if model_outputdir_path is not None:
        model_dp_transforms_path = Path(model_outputdir_path) / "dataparser_transforms.json"
        model_config_path = Path(model_outputdir_path) / "config.yml"
        model_dp_transforms_data = json.load(open(model_dp_transforms_path, "r"))
        model_config = yaml.load(model_config_path.read_text(), Loader=yaml.Loader)
        loaded_global_transform = np.array(model_dp_transforms_data["transform"])
        loaded_global_transform_44 = np.vstack([loaded_global_transform, np.array([[0, 0, 0, 1]])])
        loaded_global_scale = model_dp_transforms_data["scale"]
        if "applied_transform" in dataset_transforms_data:
            applied_transform = np.asarray(dataset_transforms_data["applied_transform"])
            applied_transform_44 = np.vstack([applied_transform, np.array([[0, 0, 0, 1]])])
        else:
            applied_transform_44 = np.eye(4)
        if "applied_scale" in dataset_transforms_data:
            applied_scale = float(dataset_transforms_data["applied_scale"])
        else:
            applied_scale = 1.0
        T_orig_to_final_nerf_world = loaded_global_transform_44 @ np.linalg.inv(applied_transform_44)
        orig_to_final_nerf_world_scale = loaded_global_scale / applied_scale
        return T_orig_to_final_nerf_world, orig_to_final_nerf_world_scale
    else:
        # TODO: add asserts to pipeline, model etc to NOT change these defaults
        _model_default_orientation_method = "up"
        _model_default_center_method = "poses"
        _model_default_auto_scale_poses = True
        _model_default_scale_factor = 1.0
        # orient and center
        all_poses = []
        for frame in dataset_transforms_data["frames"]:
            pose = np.array(frame["transform_matrix"], dtype=np.float32)
            all_poses.append(pose)
        all_poses = torch.from_numpy(np.array(all_poses))
        if "orientation_override" in dataset_transforms_data:
            orientation_method = dataset_transforms_data["orientation_override"]
        else:
            orientation_method = _model_default_orientation_method
        center_method = _model_default_center_method
        oriented_poses, transform_matrix = camera_utils.auto_orient_and_center_poses(all_poses, method=orientation_method, center_method=center_method)
        transform_matrix_44 = torch.cat([transform_matrix, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], dim=0)
        # scale poses
        scale_factor = 1.0
        if _model_default_auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(oriented_poses[:, :3, 3])))
        scale_factor *= _model_default_scale_factor
        return transform_matrix_44.cpu().numpy(), float(scale_factor)


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

        # Validate that we have data to unpack
        if len(mask_data) == 0 or len(mask_shapes) == 0:
            return []

        start_idx = 0
        for i in range(packed_data["num_masks"]):
            h, w = mask_shapes[i]
            bbox = bbox_data[i]
            pred_iou = pred_iou_data[i]
            area = area_data[i]

            # Calculate packed size for this mask
            packed_size = (h * w + 7) // 8  # Round up for packbits
            end_idx = start_idx + packed_size

            # Validate we have enough data
            if end_idx > len(mask_data):
                break

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


class TextLazyFeatures(BaseLazyShards):
    """Lazy loading for TEXT features from sharded .json files."""

    def __init__(self, shard_paths: List[Path]):
        super().__init__(shard_paths)

    def _setup_lengths(self):
        """Count total number of images across all shards."""
        for p in self.paths:
            with open(p, 'r') as f:
                data = json.load(f)
            self.lengths.append(len(data))

    def _get_shard(self, sid: int):
        """Load a specific shard."""
        with open(self.paths[sid], 'r') as f:
            return json.load(f)

    def __len__(self) -> int:
        return int(self.cum[-1])

    def __getitem__(self, idx_img: int) -> List[str]:
        """Get text objects for a specific image index."""
        sid, loc = self._loc(int(idx_img))
        shard_data = self._get_shard(sid)

        # Get image path at this index and return its objects
        image_paths = list(shard_data.keys())
        return shard_data[image_paths[loc]]


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
            "mask_shapes": np.empty((0, 2), dtype=np.int32),
            "bbox_data": np.empty((0, 4), dtype=np.float32),
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
            "mask_shapes": np.empty((0, 2), dtype=np.int32),
            "bbox_data": np.empty((0, 4), dtype=np.float32),
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
        # Ensure 2D (N,2) for shapes, 2D (N,4) for bboxes
        ms = packed["mask_shapes"].reshape(-1, 2)
        bb = packed["bbox_data"].reshape(-1, 4)
        all_mask_shapes.append(ms)
        all_bbox_data.append(bb)
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


class ORIENTANYLazyFeatures(BaseLazyShards):
    """Optimized lazy loading for ORIENTANY features with aggressive caching and memory mapping."""

    def __init__(self, pixel_shard_paths: List[Path], instance_shard_paths: List[Path]):
        self.instance_shard_paths = instance_shard_paths
        assert len(pixel_shard_paths) == len(instance_shard_paths), "Pixel and instance shards must match"

        # Initialize caching attributes BEFORE calling super().__init__()
        self.pixel_mmaps = [None] * len(pixel_shard_paths)  # Memory mapped pixel data
        self.instance_cache = [None] * len(instance_shard_paths)  # Cached JSON data
        self.full_image_cache = {}  # LRU-style cache for full reconstructed images
        self.max_image_cache = 50  # Cache up to 50 full images in memory

        super().__init__(pixel_shard_paths)  # Use pixel shards for indexing

    def _setup_lengths(self):
        """Count total number of images across all pixel shards."""
        for p in self.paths:
            # Use memory mapping for instant access to shape info
            arr = np.load(p, mmap_mode="r")
            self.lengths.append(arr.shape[0])
            # Keep first mmap for dimensions
            if self.pixel_mmaps[0] is None:
                self.pixel_mmaps[0] = arr
                self.H, self.W = arr.shape[1:3]
            elif arr is not self.pixel_mmaps[0]:
                del arr

    def _get_pixel_shard(self, shard_idx: int):
        """Get memory-mapped pixel shard."""
        if self.pixel_mmaps[shard_idx] is None:
            self.pixel_mmaps[shard_idx] = np.load(self.paths[shard_idx], mmap_mode="r")
        return self.pixel_mmaps[shard_idx]

    def _get_instance_shard(self, shard_idx: int):
        """Get cached instance features shard."""
        if self.instance_cache[shard_idx] is None:
            with open(self.instance_shard_paths[shard_idx], 'r') as f:
                self.instance_cache[shard_idx] = json.load(f)
        return self.instance_cache[shard_idx]

    def __len__(self) -> int:
        return int(self.cum[-1])

    def __getitem__(self, idx) -> np.ndarray:
        """Optimized ORIENTANY feature access with aggressive caching.

        For single image access: feat[idx_img] → returns (H, W, 10) array
        """
        if not isinstance(idx, int):
            raise ValueError("Only single image access supported: feat[idx_img]")

        # Check cache first
        if idx in self.full_image_cache:
            return self.full_image_cache[idx]

        shard_idx, local_idx = self._loc(idx)

        # Use memory-mapped pixel data
        pixel_data = self._get_pixel_shard(shard_idx)[local_idx]  # (H, W, 3)

        # Use cached instance features
        instance_features = self._get_instance_shard(shard_idx)[local_idx]  # {instance_id: 8D_mixed_distribution_params}

        # Reconstruct efficiently using vectorized operations
        h, w, _ = pixel_data.shape
        full_features = np.zeros((h, w, 10), dtype=np.float32)

        # Set foreground one-hot (vectorized)
        full_features[..., 8:10] = pixel_data[..., :2]

        # Vectorized instance feature assignment
        instance_ids = pixel_data[..., 2]
        unique_ids = np.unique(instance_ids)

        for instance_id in unique_ids:
            if instance_id == 0:  # Skip background
                continue
            instance_id_str = str(int(instance_id))
            if instance_id_str in instance_features:
                mask = (instance_ids == instance_id)
                instance_feat = instance_features[instance_id_str]
                if isinstance(instance_feat, list):
                    instance_feat = np.array(instance_feat, dtype=np.float32)
                full_features[mask, :8] = instance_feat

        # Cache the result (with simple LRU eviction)
        if len(self.full_image_cache) >= self.max_image_cache:
            # Remove oldest entry (simple FIFO for speed)
            oldest_key = next(iter(self.full_image_cache))
            del self.full_image_cache[oldest_key]

        self.full_image_cache[idx] = full_features
        return full_features

    def clear_cache(self):
        """Clear only the full image cache to manage memory, keep shard caches for speed."""
        self.full_image_cache.clear()
