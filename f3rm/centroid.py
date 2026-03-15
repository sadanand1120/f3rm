#!/usr/bin/env python3
import concurrent.futures
import gc
import multiprocessing as mp
from pathlib import Path
import time
import json
from dataclasses import dataclass, asdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from plyfile import PlyData
from tqdm.auto import tqdm
import viser
from nerfstudio.configs.method_configs import all_descriptions, all_methods
from nerfstudio.utils.eval_utils import eval_setup
from pytorch3d.transforms import quaternion_to_matrix

FACTOR_NERF_GL_TO_OPENCV = [1.0, -1.0, -1.0]
FACTOR_SAM3D_P3D_TO_OPENCV = [-1.0, -1.0, 1.0]


@dataclass
class ScaleCalibrationResult:
    n_pixels: int
    sam3d_to_nerf_scale: float
    mse_before: float
    mse_after: float
    median_before: float
    median_after: float


def apply_axis_multipliers(
    points: np.ndarray | torch.Tensor,
    factors_xyz: list[float],
) -> np.ndarray | torch.Tensor:
    if isinstance(points, torch.Tensor):
        scale = torch.tensor(factors_xyz, dtype=points.dtype, device=points.device)
        return points * scale
    return points * np.asarray(factors_xyz, dtype=points.dtype)


def sam3d_p3d_to_opencv_camera(points: np.ndarray) -> np.ndarray:
    return apply_axis_multipliers(points, FACTOR_SAM3D_P3D_TO_OPENCV)


def iter_manifest_paths(data_dir: Path, sam3d_feature_name: str) -> list[Path]:
    return sorted((data_dir / "features" / sam3d_feature_name).glob("frame_*/manifest.json"))


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text())


def load_masks_from_manifest(manifest: dict[str, Any]) -> np.ndarray:
    masks = np.load(Path(manifest["sam3_npz_path"]).resolve())["masks"]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    return masks.astype(bool)


def load_sam3d_pointmap_from_manifest(manifest: dict[str, Any]) -> np.ndarray:
    pointmap = np.load(Path(manifest["pointmap_file"]).resolve())["pointmap_hwc"]
    return pointmap.astype(np.float32)


def load_sam3d_centroid_p3d(instance_record: dict[str, Any]) -> np.ndarray:
    centroid_path = Path(instance_record["centroid_path"]).resolve()
    centroid_meta = json.loads(centroid_path.read_text())
    return np.asarray(centroid_meta["centroid_sam3d_p3d"], dtype=np.float32).reshape(3)


def build_nerf_index_map(pipeline: Any) -> dict[Path, tuple[str, Any, int]]:
    index_map: dict[Path, tuple[str, Any, int]] = {}
    for split in ("train", "eval"):
        dataset = getattr(pipeline.datamanager, f"{split}_dataset")
        for image_idx, image_path in enumerate(dataset.image_filenames):
            index_map[Path(image_path).resolve()] = (split, dataset, image_idx)
    return index_map


def render_nerf_pointmap_camera(pipeline: Any, dataset: Any, image_idx: int, device: torch.device) -> np.ndarray:
    pointmap_cam_cv, _ = render_nerf_pointmap_and_centroid_camera(pipeline, dataset, image_idx, device)
    return pointmap_cam_cv


def render_nerf_pointmap_and_centroid_camera(
    pipeline: Any,
    dataset: Any,
    image_idx: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    camera = dataset.cameras[image_idx:image_idx + 1].to(device)
    ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True).to(device)
    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle, render_features=True)
    depth_ray = outputs["depth"].squeeze(-1)
    pointmap_world = ray_bundle.origins + ray_bundle.directions * depth_ray.unsqueeze(-1)

    c2w = camera.camera_to_worlds[0]
    rotation = c2w[:, :3]
    translation = c2w[:, 3]
    pointmap_cam_gl = (pointmap_world - translation) @ rotation
    pointmap_cam_cv = apply_axis_multipliers(pointmap_cam_gl, FACTOR_NERF_GL_TO_OPENCV)
    centroid_cam_gl = outputs["centroid"]
    centroid_cam_cv = apply_axis_multipliers(centroid_cam_gl, FACTOR_NERF_GL_TO_OPENCV)
    return (
        pointmap_cam_cv.detach().cpu().numpy().astype(np.float32),
        centroid_cam_cv.detach().cpu().numpy().astype(np.float32),
    )


def render_nerf_centroid_camera(pipeline: Any, dataset: Any, image_idx: int, device: torch.device) -> np.ndarray:
    _, centroid_cam_cv = render_nerf_pointmap_and_centroid_camera(pipeline, dataset, image_idx, device)
    return centroid_cam_cv


def load_optimized_pose(instance_record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_path = Path(instance_record["optimized_pose_path"]).resolve()
    pose = json.loads(pose_path.read_text())
    quat_wxyz = np.asarray(pose["rotation_wxyz_l2c"], dtype=np.float32).reshape(4)
    translation = np.asarray(pose["translation_l2c"], dtype=np.float32).reshape(3)
    scale = np.asarray(pose["scale_l2c"], dtype=np.float32).reshape(3)
    rotation = quaternion_to_matrix(torch.from_numpy(quat_wxyz).float().unsqueeze(0))[0].cpu().numpy().astype(np.float32)
    return rotation, translation, scale


def load_gaussian_local_stats(gs_local_path: Path) -> dict[str, np.ndarray]:
    ply = PlyData.read(str(gs_local_path))["vertex"]
    points_local = np.stack(
        [
            np.asarray(ply["x"]),
            np.asarray(ply["y"]),
            np.asarray(ply["z"]),
        ],
        axis=1,
    ).astype(np.float32)
    opacity_logits = np.asarray(ply["opacity"], dtype=np.float32)
    log_scales = np.stack(
        [
            np.asarray(ply["scale_0"], dtype=np.float32),
            np.asarray(ply["scale_1"], dtype=np.float32),
            np.asarray(ply["scale_2"], dtype=np.float32),
        ],
        axis=1,
    )
    return {
        "points_local": points_local,
        "opacity_logits": opacity_logits,
        "log_scales": log_scales,
    }


def transform_points_to_sam3d_p3d(points_local: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (points_local * scale[None, :]) @ rotation + translation[None, :]


def fit_scale_mse_optimal(sam3d_points: np.ndarray, nerf_points: np.ndarray) -> tuple[float, float, float, float, float]:
    valid = np.isfinite(sam3d_points).all(axis=1) & np.isfinite(nerf_points).all(axis=1)
    sam_valid = sam3d_points[valid]
    nerf_valid = nerf_points[valid]
    if sam_valid.shape[0] == 0:
        raise ValueError("No valid correspondences")
    denom = float(np.sum(sam_valid * sam_valid))
    if denom <= 1e-12:
        raise ValueError("Degenerate SAM3D points")
    k_i = float(np.sum(sam_valid * nerf_valid) / denom)
    diff_before = sam_valid - nerf_valid
    diff_after = (k_i * sam_valid) - nerf_valid
    mse_before = float(np.mean(np.sum(diff_before ** 2, axis=1)))
    mse_after = float(np.mean(np.sum(diff_after ** 2, axis=1)))
    median_before = float(np.median(np.linalg.norm(diff_before, axis=1)))
    median_after = float(np.median(np.linalg.norm(diff_after, axis=1)))
    return k_i, mse_before, mse_after, median_before, median_after


def fit_scale_median_ratios(sam3d_points: np.ndarray, nerf_points: np.ndarray) -> tuple[float, float, float, float, float]:
    valid = np.isfinite(sam3d_points).all(axis=1) & np.isfinite(nerf_points).all(axis=1)
    sam_valid = sam3d_points[valid]
    nerf_valid = nerf_points[valid]
    if sam_valid.shape[0] == 0:
        raise ValueError("No valid correspondences")
    sam_norms = np.linalg.norm(sam_valid, axis=1)
    nerf_norms = np.linalg.norm(nerf_valid, axis=1)
    valid_norms = sam_norms > 1e-12
    if not np.any(valid_norms):
        raise ValueError("Degenerate SAM3D points")
    k_i = float(np.median(nerf_norms[valid_norms] / sam_norms[valid_norms]))
    diff_before = sam_valid - nerf_valid
    diff_after = (k_i * sam_valid) - nerf_valid
    mse_before = float(np.mean(np.sum(diff_before ** 2, axis=1)))
    mse_after = float(np.mean(np.sum(diff_after ** 2, axis=1)))
    median_before = float(np.median(np.linalg.norm(diff_before, axis=1)))
    median_after = float(np.median(np.linalg.norm(diff_after, axis=1)))
    return k_i, mse_before, mse_after, median_before, median_after


def downsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[keep]


def save_scale_distribution_plot(
    scale_results: dict[str, dict[int, dict[str, Any]]],
    data_dir: Path,
) -> Path:
    scales = [
        float(instance_meta["sam3d_to_nerf_scale"])
        for image_meta in scale_results.values()
        for instance_meta in image_meta.values()
    ]
    out_path = Path.cwd() / f"scales_distn_{Path(data_dir).name}.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scales, bins=40, color="#2c7fb8", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Estimated scale k_i")
    ax.set_ylabel("Count")
    ax.set_title(f"Scale distribution: {Path(data_dir).name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def resolve_worker_devices(device: torch.device, num_workers_per_gpu: int) -> list[torch.device]:
    workers_per_gpu = max(1, int(num_workers_per_gpu))
    if device.type == "cuda":
        if device.index is None:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_gpus == 0:
                return [torch.device("cpu")]
            return [
                torch.device(f"cuda:{gpu_idx}")
                for gpu_idx in range(n_gpus)
                for _ in range(workers_per_gpu)
            ]
        return [torch.device(f"cuda:{device.index}") for _ in range(workers_per_gpu)]
    return [torch.device("cpu")]


def shutdown_executors(
    executors: list[concurrent.futures.ProcessPoolExecutor], force: bool
) -> None:
    for executor in executors:
        if force:
            processes = getattr(executor, "_processes", None)
            if processes:
                for proc in list(processes.values()):
                    if proc.is_alive():
                        proc.terminate()
            executor.shutdown(wait=False, cancel_futures=True)
            if processes:
                for proc in list(processes.values()):
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        proc.kill()
        else:
            executor.shutdown(wait=True, cancel_futures=False)


def register_f3rm_method() -> None:
    if "f3rm" in all_methods:
        return
    from f3rm.f3rm_config import f3rm_method

    all_methods["f3rm"] = f3rm_method.config
    all_descriptions["f3rm"] = f3rm_method.description


class NerfToSAM3DScaleCalibrator:
    def __init__(
        self,
        data_dir: Path,
        config_path: Path,
        sam3d_feature_name: str = "sam3d_",
        device: torch.device | None = None,
        do_simpler_median_ratios: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.sam3d_feature_name = sam3d_feature_name
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.do_simpler_median_ratios = do_simpler_median_ratios

    def _setup_pipeline(self) -> None:
        register_f3rm_method()
        _, self.pipeline, _, _ = eval_setup(config_path=self.config_path, test_mode="test")
        self.nerf_index_map = build_nerf_index_map(self.pipeline)

    def _process_instance(
        self,
        image_path: Path,
        sam3d_instance_idx: int,
        instance_record: dict[str, Any],
        masks: np.ndarray,
        nerf_pointmap_cam: np.ndarray,
        sam3d_pointmap_cv: np.ndarray,
    ) -> dict[str, Any] | None:
        sam3_mask_index = int(instance_record["mask_index"])
        if sam3_mask_index >= masks.shape[0]:
            return None
        valid_mask = masks[sam3_mask_index] & np.isfinite(sam3d_pointmap_cv).all(axis=-1) & np.isfinite(nerf_pointmap_cam).all(axis=-1)
        sam3d_points = sam3d_pointmap_cv[valid_mask]
        nerf_points = nerf_pointmap_cam[valid_mask]
        if sam3d_points.shape[0] == 0:
            return None
        fit_fn = fit_scale_median_ratios if self.do_simpler_median_ratios else fit_scale_mse_optimal
        k_i, mse_before, mse_after, median_before, median_after = fit_fn(sam3d_points, nerf_points)
        result = ScaleCalibrationResult(
            n_pixels=int(sam3d_points.shape[0]),
            sam3d_to_nerf_scale=float(k_i),
            mse_before=float(mse_before),
            mse_after=float(mse_after),
            median_before=float(median_before),
            median_after=float(median_after),
        )
        result_dict = asdict(result)
        result_dict["sam3d_instance_idx"] = int(sam3d_instance_idx)
        return result_dict

    def _process_manifest(self, manifest_path: Path) -> tuple[str, dict[int, dict[str, Any]]]:
        manifest = load_manifest(manifest_path)
        image_path = Path(manifest["image_path"]).resolve()
        _, dataset, image_idx = self.nerf_index_map[image_path]
        nerf_pointmap_cam = render_nerf_pointmap_camera(self.pipeline, dataset, image_idx, self.device)
        sam3d_pointmap_cv = sam3d_p3d_to_opencv_camera(load_sam3d_pointmap_from_manifest(manifest))
        masks = load_masks_from_manifest(manifest)
        instance_results: dict[int, dict[str, Any]] = {}
        for sam3d_instance_idx, instance_record in enumerate(manifest["instances"]):
            result = self._process_instance(image_path, sam3d_instance_idx, instance_record, masks, nerf_pointmap_cam, sam3d_pointmap_cv)
            if result is not None:
                instance_results[int(instance_record["mask_index"])] = result
        return str(image_path), instance_results

    def discover_manifest_paths(self) -> list[Path]:
        return iter_manifest_paths(self.data_dir, self.sam3d_feature_name)

    def process_manifest_path(self, manifest_path: Path | str) -> tuple[str, dict[int, dict[str, Any]]]:
        if not hasattr(self, "pipeline"):
            self._setup_pipeline()
        return self._process_manifest(Path(manifest_path))


_MP_SCALE_CALIBRATOR: NerfToSAM3DScaleCalibrator | None = None


def _mp_scale_worker_init(
    device_str: str,
    data_dir: str,
    config_path: str,
    sam3d_feature_name: str,
    do_simpler_median_ratios: bool,
) -> None:
    global _MP_SCALE_CALIBRATOR
    device = torch.device(device_str)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    _MP_SCALE_CALIBRATOR = NerfToSAM3DScaleCalibrator(
        data_dir=Path(data_dir),
        config_path=Path(config_path),
        sam3d_feature_name=sam3d_feature_name,
        device=device,
        do_simpler_median_ratios=do_simpler_median_ratios,
    )
    _MP_SCALE_CALIBRATOR._setup_pipeline()


def _mp_scale_worker_run(task_idx: int, manifest_path: str) -> tuple[int, str, dict[int, dict[str, Any]]]:
    global _MP_SCALE_CALIBRATOR
    if _MP_SCALE_CALIBRATOR is None:
        raise RuntimeError("Scale calibration worker is not initialized.")
    image_path, instance_results = _MP_SCALE_CALIBRATOR.process_manifest_path(manifest_path)
    return task_idx, image_path, instance_results


def run_parallel_scale_calibration(
    data_dir: Path,
    config_path: Path,
    sam3d_feature_name: str = "sam3d_",
    device: torch.device | None = None,
    num_workers_per_gpu: int = 2,
    do_simpler_median_ratios: bool = True,
) -> dict[str, dict[int, dict[str, Any]]]:
    cache_path = Path(config_path).parent / "nerf_to_sam3d_scales.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return {
            image_path: {int(mask_idx): scale_meta for mask_idx, scale_meta in image_meta.items()}
            for image_path, image_meta in cached.items()
        }

    scale_calibrator = NerfToSAM3DScaleCalibrator(
        data_dir,
        config_path,
        sam3d_feature_name,
        device=device,
        do_simpler_median_ratios=do_simpler_median_ratios,
    )
    manifest_paths = scale_calibrator.discover_manifest_paths()
    worker_devices = resolve_worker_devices(scale_calibrator.device, num_workers_per_gpu)
    executors: list[concurrent.futures.ProcessPoolExecutor] = []
    futures: list[concurrent.futures.Future] = []
    scale_results: dict[str, dict[int, dict[str, Any]]] = {}
    interrupted = False
    try:
        ctx = mp.get_context("spawn")
        for worker_device in worker_devices:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=_mp_scale_worker_init,
                initargs=(
                    str(worker_device),
                    str(data_dir),
                    str(config_path),
                    sam3d_feature_name,
                    do_simpler_median_ratios,
                ),
            )
            executors.append(executor)

        for task_idx, manifest_path in enumerate(manifest_paths):
            executor = executors[task_idx % len(executors)]
            futures.append(executor.submit(_mp_scale_worker_run, task_idx, str(manifest_path)))

        ordered_results: list[tuple[str, dict[int, dict[str, Any]]] | None] = [None] * len(manifest_paths)
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Calibrating scales",
            leave=False,
        ):
            task_idx, image_path, instance_results = future.result()
            ordered_results[task_idx] = (image_path, instance_results)

        for item in ordered_results:
            if item is None:
                continue
            image_path, instance_results = item
            scale_results[image_path] = instance_results
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        shutdown_executors(executors, force=True)
        raise
    except Exception:
        shutdown_executors(executors, force=True)
        raise
    finally:
        if not interrupted:
            shutdown_executors(executors, force=False)
        gc.collect()

    cache_path.write_text(json.dumps(scale_results))
    return scale_results


def launch_viser(
    scale_results: dict[str, dict[int, dict[str, Any]]],
    data_dir: Path,
    config_path: Path,
    image_path: Path,
    sam3d_instance_idx_to_vis: int,
    sam3d_feature_name: str = "sam3d_",
    max_vis_points: int = 120_000,
    viser_port: int = 8890,
    nerf_point_size: float = 0.0035,
    nerf_centroid_point_size: float = 0.0008,
    sam3d_model_point_size: float = 0.0012,
) -> viser.ViserServer:
    data_dir = Path(data_dir)
    image_path = Path(image_path).resolve()
    _, pipeline, _, _ = eval_setup(config_path=Path(config_path), test_mode="test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, dataset, image_idx = build_nerf_index_map(pipeline)[image_path]
    nerf_pointmap_cam = render_nerf_pointmap_camera(pipeline, dataset, image_idx, device)
    nerf_centroid_cam = render_nerf_centroid_camera(pipeline, dataset, image_idx, device)

    manifest_path = data_dir / "features" / sam3d_feature_name / image_path.stem / "manifest.json"
    manifest = load_manifest(manifest_path)
    print(f"SAM3D instances in {image_path.name}: {len(manifest['instances'])}")
    instance_record = manifest["instances"][sam3d_instance_idx_to_vis]
    masks = load_masks_from_manifest(manifest)
    sam3d_pointmap_cv = sam3d_p3d_to_opencv_camera(load_sam3d_pointmap_from_manifest(manifest))
    scale_meta = scale_results[str(image_path)][int(instance_record["mask_index"])]
    k_i = float(scale_meta["sam3d_to_nerf_scale"])

    sam3_mask_index = int(instance_record["mask_index"])
    valid_mask = masks[sam3_mask_index] & np.isfinite(sam3d_pointmap_cv).all(axis=-1) & np.isfinite(nerf_pointmap_cam).all(axis=-1)
    nerf_instance_points = nerf_pointmap_cam[valid_mask].astype(np.float32)
    nerf_centroid_points = nerf_centroid_cam[masks[sam3_mask_index] & np.isfinite(nerf_centroid_cam).all(axis=-1)].astype(np.float32)
    sam3d_pointmap_scaled = (k_i * sam3d_pointmap_cv[valid_mask]).astype(np.float32)
    sam3d_centroid_scaled = (
        k_i * sam3d_p3d_to_opencv_camera(load_sam3d_centroid_p3d(instance_record))
    ).astype(np.float32)
    sam3d_model_raw = sam3d_p3d_to_opencv_camera(
        transform_points_to_sam3d_p3d(
            load_gaussian_local_stats(Path(instance_record["gs_local_path"]).resolve())["points_local"],
            *load_optimized_pose(instance_record),
        )
    ).astype(np.float32)
    sam3d_model_scaled = (k_i * sam3d_model_raw).astype(np.float32)

    server = viser.ViserServer(port=viser_port)
    server.scene.add_point_cloud(
        "/nerf_instance",
        points=downsample_points(nerf_instance_points, max_vis_points, seed=1),
        colors=(0.1, 0.35, 0.95),
        point_size=nerf_point_size,
    )
    server.scene.add_point_cloud(
        "/sam3d_pointmap_scaled",
        points=downsample_points(sam3d_pointmap_scaled, max_vis_points, seed=11),
        colors=(0.0, 0.0, 0.0),
        point_size=nerf_point_size,
    )
    server.scene.add_point_cloud(
        "/nerf_centroid",
        points=downsample_points(nerf_centroid_points, max_vis_points, seed=7),
        colors=(1.0, 0.0, 1.0),
        point_size=nerf_centroid_point_size,
    )
    server.scene.add_point_cloud(
        "/sam3d_raw",
        points=downsample_points(sam3d_model_raw, max_vis_points, seed=2),
        colors=(0.95, 0.2, 0.2),
        point_size=sam3d_model_point_size,
    )
    server.scene.add_point_cloud(
        "/sam3d_scaled",
        points=downsample_points(sam3d_model_scaled, max_vis_points, seed=3),
        colors=(0.1, 0.9, 0.2),
        point_size=sam3d_model_point_size,
    )
    server.scene.add_point_cloud(
        "/sam3d_centroid_scaled",
        points=sam3d_centroid_scaled[None, :],
        colors=(1.0, 0.55, 0.0),
        point_size=0.03,
    )

    print(f"Viser running: http://localhost:{viser_port}")
    print(f"Public URL: {server.request_share_url(verbose=True)}")
    return server


if __name__ == "__main__":
    DATA_DIR = Path("datasets/f3rm/fresh/objaverse/car2new")
    CONFIG_PATH = Path("mar10_outputs/car2new_cent/f3rm/2026-03-11_085644/config.yml")
    SAM3D_FEATURE_NAME = "sam3d_"
    IMGPATH = DATA_DIR / "images/frame_00059.png"
    SAM3D_INSTANCE_IDX_TO_VIS = 0
    NUM_WORKERS_PER_GPU = 2

    scale_results = run_parallel_scale_calibration(
        data_dir=DATA_DIR,
        config_path=CONFIG_PATH,
        sam3d_feature_name=SAM3D_FEATURE_NAME,
        num_workers_per_gpu=NUM_WORKERS_PER_GPU,
    )

    print(f"processed images: {len(scale_results)}")
    distn_path = save_scale_distribution_plot(scale_results, DATA_DIR)
    print(f"saved scale distribution: {distn_path}")

    server = launch_viser(
        scale_results,
        DATA_DIR,
        CONFIG_PATH,
        IMGPATH,
        SAM3D_INSTANCE_IDX_TO_VIS,
        SAM3D_FEATURE_NAME,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
