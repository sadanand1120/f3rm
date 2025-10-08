from typing import Optional, Tuple, Callable, Any, List, Dict
from collections import OrderedDict
import json

import asyncio
import concurrent.futures
import numpy as np
import os
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from sam2.features.utils import SAM2utils

from nerfstudio.cameras import camera_utils


def get_conf_temp_scaled_logits(logits, confidence, drop_exp_factor=6, eps=1e-8):
    """Scale logits by confidence. T = 1/max(confidence^drop_exp_factor, eps). Lower confidence -> higher T -> flatter probs."""
    x = torch.as_tensor(logits, dtype=torch.float32)
    T = 1.0 / max(float(confidence)**drop_exp_factor, eps)
    return x / T


def probs_to_normal(
    probs: torch.Tensor,
    *,
    n_bins: int,
    angle_min_deg: float,
    period_deg: float,
    bin_width_deg: Optional[float] = None,
    min_std_deg: float = 1e-3,
    window_deg: float = 60.0,  # Increased from 45.0 to capture even more of the distribution
    peak_frac: float = 0.05,   # Reduced from 0.1 to include even more tail
    gamma: float = 1.2,        # Reduced from 1.5 to be even less aggressive
    ridge: float = 0.0,
    shrink_to_moment: float = 0.2,  # Reduced from 0.3 to be even less conservative
    sigma_floor_deg: Optional[float] = 0.3,  # Reduced from 0.5 to allow even narrower fits
    sigma_ceil_deg: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit a single Gaussian N(mean_deg, std_deg) on a bounded linear range [angle_min, angle_min+period)."""
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
    window_deg: float = 60.0,        # Increased from 45.0 to capture even more of the distribution
    peak_frac: float = 0.05,         # Reduced from 0.1 to include even more tail
    gamma: float = 1.2,              # Reduced from 1.5 to be even less aggressive
    ridge: float = 0.0,              # small L2 on LS fit (e.g., 1e-4) to tame κ
    shrink_to_resultant: float = 0.2,  # Reduced from 0.3 to be even less conservative
    sigma_floor_deg: float = 0.5,    # Reduced from 1.0 to allow even narrower fits
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


def pack_auto_masks(auto_masks: List[dict]) -> dict:
    """Pack auto masks into a memory-mappable format.

    Returns a dict with:
    - num_masks: number of masks per image
    - mask_data: packed binary data for all masks
    - mask_shapes: shapes of each mask
    - bbox_data: bbox coordinates as float16 array
    - pred_iou_data: predicted IoU scores as float16 array
    - area_data: mask areas as float16 array
    """
    if not auto_masks:
        return {
            "num_masks": 0,
            "mask_data": np.array([], dtype=np.uint8),
            "mask_shapes": np.empty((0, 2), dtype=np.int32),
            "bbox_data": np.empty((0, 4), dtype=np.float16),      # Use fp16 for VRAM efficiency
            "pred_iou_data": np.array([], dtype=np.float16),      # Use fp16 for VRAM efficiency
            "area_data": np.array([], dtype=np.float16)           # Use fp16 for VRAM efficiency
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
            "bbox_data": np.empty((0, 4), dtype=np.float16),      # Use fp16 for VRAM efficiency
            "pred_iou_data": np.array([], dtype=np.float16),      # Use fp16 for VRAM efficiency
            "area_data": np.array([], dtype=np.float16)           # Use fp16 for VRAM efficiency
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
        "bbox_data": np.array(all_bboxes, dtype=np.float16),      # Use fp16 for VRAM efficiency
        "pred_iou_data": np.array(all_scores, dtype=np.float16),  # Use fp16 for VRAM efficiency
        "area_data": np.array(all_areas, dtype=np.float16)        # Use fp16 for VRAM efficiency
    }


def unpack_auto_masks(packed_data: dict) -> List[dict]:
    """Unpack auto masks from memory-mappable format back to original format.

    This is the standalone version of the unpacking logic used in BatchFeatureLoader.
    """
    if packed_data["num_masks"] == 0:
        return []

    decoded = []
    mask_data = packed_data["mask_data"]
    mask_shapes = packed_data["mask_shapes"]
    bbox_data = packed_data["bbox_data"]
    pred_iou_data = packed_data["pred_iou_data"]
    area_data = packed_data["area_data"]

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

        if end_idx > len(mask_data):
            break

        # Extract and unpack this mask's data
        packed = mask_data[start_idx:end_idx]
        flat = np.unpackbits(packed)[:h * w]
        seg_arr = flat.reshape(h, w).astype(bool)

        # Reconstruct original mask dict with fp16 optimization
        mask_dict = {
            "segmentation": seg_arr,
            "bbox": bbox.tolist(),
            "predicted_iou": np.float16(pred_iou),  # Use fp16 for VRAM efficiency
            "area": np.float16(area)                # Use fp16 for VRAM efficiency
        }
        decoded.append(mask_dict)
        start_idx = end_idx

    return decoded


def visualize_auto_masks_demo(auto_masks_list: List[List[dict]], image_paths: List[str], title_prefix: str = "Mask", max_vis: int = 4, pred_iou_thresh: float = 0.8, min_mask_region_area: int = 0):
    """Common visualization function for auto masks demos.

    Args:
        auto_masks_list: List of auto mask lists, one per image
        image_paths: List of image file paths
        title_prefix: Prefix for the mask titles (e.g., "SAM2", "CLIPSAM")
        max_vis: Maximum number of images to visualize
        pred_iou_thresh: IoU threshold for mask filtering
        min_mask_region_area: Minimum mask area threshold
    """
    vis_count = min(max_vis, len(auto_masks_list))
    fig, axes = plt.subplots(2, vis_count, figsize=(4 * vis_count, 8))
    if vis_count == 1:
        axes = axes.reshape(2, 1)

    for i, (auto_masks, image_path) in enumerate(zip(auto_masks_list[:vis_count], image_paths[:vis_count])):
        # Load and display RGB image
        rgb_img = Image.open(image_path).convert("RGB")
        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f"RGB {i+1}")
        axes[0, i].axis('off')

        # Generate and display mask
        inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
            auto_masks,
            min_iou=float(pred_iou_thresh),
            min_area=float(min_mask_region_area),
            assign_by="area",
            start_from="low",
        )
        if inst_mask is None:
            # No valid masks found, create empty instance mask
            if auto_masks:
                h, w = auto_masks[0]['segmentation'].shape
            else:
                # Use actual image dimensions as fallback
                img = Image.open(image_path)
                h, w = img.height, img.width
            inst_mask = np.zeros((h, w), dtype=np.uint16)
        viz_mask, cmap, norm = SAM2utils.make_viz_mask_and_cmap(inst_mask)
        axes[1, i].imshow(viz_mask, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1, i].set_title(f"{title_prefix} {i+1} ({len(np.unique(inst_mask)) - 1} inst)")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


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
        if feature_type in ("CLIP", "DINO"):
            self.H, self.W, self.C = sample_features.shape
            self.dtype = sample_features.dtype
        elif feature_type.startswith("FOREGROUND_"):
            self.H, self.W, self.C = sample_features.shape
            self.dtype = sample_features.dtype
        elif feature_type.startswith("ORIENTANY_"):
            self.H, self.W = sample_features.shape[:2]
            self.dtype = sample_features.dtype
        del sample_features

    def _load_single_image_cpu(self, img_idx: int):
        """Load features for a single image onto CPU (optionally pinned)."""
        if self.feature_type in ("CLIP", "DINO"):
            data = np.load(self.root / f"image_{img_idx:06d}.npy", mmap_mode="r")
            t = torch.from_numpy(data)
            return t.pin_memory() if self._use_pinned else t
        elif self.feature_type.startswith("FOREGROUND_"):
            data = np.load(self.root / f"image_{img_idx:06d}.npy", mmap_mode="r")
            t = torch.from_numpy(data)
            return t.pin_memory() if self._use_pinned else t
        elif self.feature_type.startswith("ORIENTANY_"):
            pixel_data = np.load(self.root / f"image_{img_idx:06d}_pixel.npy", mmap_mode="r")
            with open(self.root / f"image_{img_idx:06d}_instances.json", 'r') as f:
                instance_features = json.load(f)

            h, w, _ = pixel_data.shape
            full_features = np.zeros((h, w, 9), dtype=np.float16)  # 7D features + 2D foreground
            full_features[..., 7:9] = pixel_data[..., :2]  # foreground one-hot
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
                        instance_feat = np.array(instance_feat, dtype=np.float16)
                    full_features[mask, :7] = instance_feat  # 7D features: R_x, R_z, confidence
            t = torch.from_numpy(full_features)
            return t.pin_memory() if self._use_pinned else t

        elif self.feature_type.startswith("CLIPSAM_") or self.feature_type == "SAM2":
            data = np.load(self.root / f"image_{img_idx:06d}.npz")
            packed_data = {
                "num_masks": int(data['num_masks']),
                "mask_data": data['mask_data'],
                "mask_shapes": data['mask_shapes'],
                "bbox_data": data['bbox_data'],
                "pred_iou_data": data['pred_iou_data'],
                "area_data": data['area_data']
            }
            return unpack_auto_masks(packed_data)
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
            if self.feature_type in ("CLIP", "DINO") or self.feature_type.startswith("FOREGROUND_") or self.feature_type.startswith("ORIENTANY_"):
                batch_features[cam_idx_int] = self._get_gpu_tensor(cam_idx_int)
            else:
                # SAM2/TEXT types
                batch_features[cam_idx_int] = self._load_single_image_cpu(cam_idx_int)
        return batch_features

    def __getitem__(self, index: int):
        """Direct access by image index for pipeline compatibility."""
        if self.feature_type in ("CLIP", "DINO") or self.feature_type.startswith("FOREGROUND_") or self.feature_type.startswith("ORIENTANY_"):
            return self._get_gpu_tensor(index)
        else:
            return self._load_single_image_cpu(index)


def get_cache_paths(data_dir: Path, feature_type: str) -> Tuple[Path, Path]:
    """Get cache directory and metadata paths for a feature type."""
    if feature_type.startswith("CLIPSAM_") or feature_type.startswith("FOREGROUND_") or feature_type.startswith("ORIENTANY_"):
        root = data_dir / "features" / feature_type.lower()
    else:
        root = data_dir / "features" / feature_type.lower()
    return root, root / "meta.pt"
