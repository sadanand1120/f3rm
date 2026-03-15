#!/usr/bin/env python3
import argparse
import concurrent.futures
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
from nerfstudio.utils.eval_utils import eval_setup

from f3rm.centroid import (
    build_nerf_index_map,
    downsample_points,
    iter_manifest_paths,
    load_manifest,
    load_masks_from_manifest,
    load_sam3d_centroid_p3d,
    load_sam3d_pointmap_from_manifest,
    register_f3rm_method,
    render_nerf_pointmap_and_centroid_camera,
    resolve_worker_devices,
    run_parallel_scale_calibration,
    sam3d_p3d_to_opencv_camera,
    shutdown_executors,
)


def _empty_points() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


def _cache_dir(config_path: Path) -> Path:
    return Path(config_path).parent / "centroid_examiner_cache"


def _cache_index_path(config_path: Path) -> Path:
    return _cache_dir(config_path) / "index.json"


def _render_cache_dir(config_path: Path) -> Path:
    return _cache_dir(config_path) / "_tmp_renders"


def _latest_available_ckpt_step(config_path: Path) -> int | None:
    model_dir = Path(config_path).parent / "nerfstudio_models"
    ckpts = sorted(model_dir.glob("step-*.ckpt"))
    if not ckpts:
        return None
    return max(int(ckpt.stem.split("-")[1]) for ckpt in ckpts)


def wait_for_ckpt_step(config_path: Path, wait_for_ckpt_step: int) -> None:
    while True:
        latest_step = _latest_available_ckpt_step(config_path)
        if latest_step is not None and latest_step >= wait_for_ckpt_step:
            print(
                f"wait_for_ckpt_step={wait_for_ckpt_step}: found checkpoint step {latest_step}, "
                "sleeping 30s before continuing"
            )
            time.sleep(30)
            return
        print(
            f"wait_for_ckpt_step={wait_for_ckpt_step}: latest available is "
            f"{latest_step if latest_step is not None else 'none'}, sleeping 120s"
        )
        time.sleep(120)


class CentroidExaminerRenderWorker:
    def __init__(
        self,
        data_dir: Path,
        config_path: Path,
        sam3d_feature_name: str,
        render_cache_dir: Path,
        device: torch.device | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.sam3d_feature_name = sam3d_feature_name
        self.render_cache_dir = Path(render_cache_dir)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_pipeline(self) -> None:
        register_f3rm_method()
        _, self.pipeline, _, _ = eval_setup(config_path=self.config_path, test_mode="test")
        self.nerf_index_map = build_nerf_index_map(self.pipeline)

    def process_manifest_path(
        self,
        task_idx: int,
        manifest_path: Path,
    ) -> dict[str, Any]:
        if not hasattr(self, "pipeline"):
            self._setup_pipeline()

        manifest = load_manifest(manifest_path)
        image_path = Path(manifest["image_path"]).resolve()
        _, dataset, image_idx = self.nerf_index_map[image_path]
        nerf_pointmap_cam, nerf_centroid_cam = render_nerf_pointmap_and_centroid_camera(
            self.pipeline,
            dataset,
            image_idx,
            self.device,
        )
        render_name = f"render_{task_idx:06d}_{image_path.stem}.npz"
        np.savez(
            self.render_cache_dir / render_name,
            nerf_pointmap_cam=nerf_pointmap_cam.astype(np.float32),
            nerf_centroid_cam=nerf_centroid_cam.astype(np.float32),
        )
        return {
            "image_path": str(image_path),
            "render_path": render_name,
        }


def _build_examiner_payload(
    task_idx: int,
    manifest_path: Path,
    render_path: Path,
    cache_dir: Path,
    scale_results: dict[str, dict[int, dict[str, Any]]],
    max_vis_points: int,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    image_path = Path(manifest["image_path"]).resolve()
    with np.load(render_path) as render_data:
        nerf_pointmap_cam = render_data["nerf_pointmap_cam"]
        nerf_centroid_cam = render_data["nerf_centroid_cam"]
    sam3d_pointmap_cv = sam3d_p3d_to_opencv_camera(load_sam3d_pointmap_from_manifest(manifest))
    masks = load_masks_from_manifest(manifest)
    image_scales = scale_results.get(str(image_path), {})

    npz_arrays: dict[str, np.ndarray] = {}
    instances_meta: list[dict[str, Any]] = []
    for sam3d_instance_idx, instance_record in enumerate(manifest["instances"]):
        sam3_mask_index = int(instance_record["mask_index"])
        if sam3_mask_index >= masks.shape[0]:
            continue

        scale_meta = image_scales.get(sam3_mask_index)
        scale = None if scale_meta is None else float(scale_meta["sam3d_to_nerf_scale"])
        mask = masks[sam3_mask_index]
        valid_mask = mask & np.isfinite(sam3d_pointmap_cv).all(axis=-1) & np.isfinite(nerf_pointmap_cam).all(axis=-1)
        nerf_centroid_mask = mask & np.isfinite(nerf_centroid_cam).all(axis=-1)

        arr_idx = len(instances_meta)
        nerf_instance_points = downsample_points(
            nerf_pointmap_cam[valid_mask].astype(np.float32),
            max_vis_points,
            seed=task_idx * 1000 + arr_idx * 10 + 1,
        )
        nerf_centroid_points = downsample_points(
            nerf_centroid_cam[nerf_centroid_mask].astype(np.float32),
            max_vis_points,
            seed=task_idx * 1000 + arr_idx * 10 + 2,
        )

        if scale is None:
            sam3d_pointmap_scaled = _empty_points()
            sam3d_centroid_scaled = _empty_points()
        else:
            sam3d_pointmap_scaled = downsample_points(
                (scale * sam3d_pointmap_cv[valid_mask]).astype(np.float32),
                max_vis_points,
                seed=task_idx * 1000 + arr_idx * 10 + 3,
            )
            sam3d_centroid_scaled = (
                scale * sam3d_p3d_to_opencv_camera(load_sam3d_centroid_p3d(instance_record))
            )[None, :].astype(np.float32)

        npz_arrays[f"nerf_instance_points_{arr_idx}"] = nerf_instance_points
        npz_arrays[f"nerf_centroid_points_{arr_idx}"] = nerf_centroid_points
        npz_arrays[f"sam3d_pointmap_scaled_{arr_idx}"] = sam3d_pointmap_scaled
        npz_arrays[f"sam3d_centroid_scaled_{arr_idx}"] = sam3d_centroid_scaled
        instances_meta.append(
            {
                "sam3d_instance_idx": sam3d_instance_idx,
                "mask_index": sam3_mask_index,
                "scale": scale,
                "array_index": arr_idx,
            }
        )

    payload_name = f"image_{task_idx:06d}_{image_path.stem}.npz"
    np.savez(cache_dir / payload_name, **npz_arrays)
    return {
        "image_path": str(image_path),
        "payload_path": payload_name,
        "num_instances": len(instances_meta),
        "instances": instances_meta,
    }


_MP_EXAMINER_WORKER: CentroidExaminerRenderWorker | None = None
_MP_PAYLOAD_CACHE_DIR: Path | None = None
_MP_PAYLOAD_MAX_VIS_POINTS: int | None = None
_MP_PAYLOAD_SCALE_RESULTS: dict[str, dict[int, dict[str, Any]]] | None = None


def _mp_examiner_worker_init(
    device_str: str,
    data_dir: str,
    config_path: str,
    sam3d_feature_name: str,
    render_cache_dir: str,
) -> None:
    global _MP_EXAMINER_WORKER
    device = torch.device(device_str)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    _MP_EXAMINER_WORKER = CentroidExaminerRenderWorker(
        data_dir=Path(data_dir),
        config_path=Path(config_path),
        sam3d_feature_name=sam3d_feature_name,
        render_cache_dir=Path(render_cache_dir),
        device=device,
    )
    _MP_EXAMINER_WORKER._setup_pipeline()


def _mp_examiner_worker_run(task_idx: int, manifest_path: str) -> tuple[int, dict[str, Any]]:
    global _MP_EXAMINER_WORKER
    if _MP_EXAMINER_WORKER is None:
        raise RuntimeError("Centroid examiner worker is not initialized.")
    meta = _MP_EXAMINER_WORKER.process_manifest_path(task_idx, Path(manifest_path))
    return task_idx, meta


def _mp_payload_worker_init(
    cache_dir: str,
    max_vis_points: int,
    scale_results_json: str,
) -> None:
    global _MP_PAYLOAD_CACHE_DIR, _MP_PAYLOAD_MAX_VIS_POINTS, _MP_PAYLOAD_SCALE_RESULTS
    _MP_PAYLOAD_CACHE_DIR = Path(cache_dir)
    _MP_PAYLOAD_MAX_VIS_POINTS = max_vis_points
    _MP_PAYLOAD_SCALE_RESULTS = {
        image_path: {int(mask_idx): scale_meta for mask_idx, scale_meta in image_meta.items()}
        for image_path, image_meta in json.loads(scale_results_json).items()
    }


def _mp_payload_worker_run(task_idx: int, manifest_path: str, render_path: str) -> tuple[int, dict[str, Any]]:
    global _MP_PAYLOAD_CACHE_DIR, _MP_PAYLOAD_MAX_VIS_POINTS, _MP_PAYLOAD_SCALE_RESULTS
    if _MP_PAYLOAD_CACHE_DIR is None or _MP_PAYLOAD_MAX_VIS_POINTS is None or _MP_PAYLOAD_SCALE_RESULTS is None:
        raise RuntimeError("Centroid examiner payload worker is not initialized.")
    meta = _build_examiner_payload(
        task_idx=task_idx,
        manifest_path=Path(manifest_path),
        render_path=Path(render_path),
        cache_dir=_MP_PAYLOAD_CACHE_DIR,
        scale_results=_MP_PAYLOAD_SCALE_RESULTS,
        max_vis_points=_MP_PAYLOAD_MAX_VIS_POINTS,
    )
    return task_idx, meta


def precompute_examiner_cache(
    data_dir: Path,
    config_path: Path,
    sam3d_feature_name: str,
    scale_results: dict[str, dict[int, dict[str, Any]]],
    max_vis_points: int,
    num_workers_per_gpu: int,
    num_cpu_workers: int,
) -> list[dict[str, Any]]:
    cache_dir = _cache_dir(config_path)
    cache_dir.mkdir(exist_ok=True)
    index_path = _cache_index_path(config_path)
    if index_path.exists():
        return json.loads(index_path.read_text())["images"]
    render_cache_dir = _render_cache_dir(config_path)
    shutil.rmtree(render_cache_dir, ignore_errors=True)
    render_cache_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worker_devices = resolve_worker_devices(device, num_workers_per_gpu)
    manifest_paths = iter_manifest_paths(Path(data_dir), sam3d_feature_name)
    render_executors: list[concurrent.futures.ProcessPoolExecutor] = []
    render_futures: list[concurrent.futures.Future] = []
    payload_executor: concurrent.futures.ProcessPoolExecutor | None = None
    payload_futures: list[concurrent.futures.Future] = []
    interrupted = False
    try:
        ctx = mp.get_context("spawn")
        for worker_device in worker_devices:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=_mp_examiner_worker_init,
                initargs=(
                    str(worker_device),
                    str(data_dir),
                    str(config_path),
                    sam3d_feature_name,
                    str(render_cache_dir),
                ),
            )
            render_executors.append(executor)

        for task_idx, manifest_path in enumerate(manifest_paths):
            executor = render_executors[task_idx % len(render_executors)]
            render_futures.append(executor.submit(_mp_examiner_worker_run, task_idx, str(manifest_path)))

        ordered_meta: list[dict[str, Any] | None] = [None] * len(manifest_paths)
        for future in tqdm(
            concurrent.futures.as_completed(render_futures),
            total=len(render_futures),
            desc="Rendering examiner views",
            leave=False,
        ):
            task_idx, meta = future.result()
            ordered_meta[task_idx] = meta
        shutdown_executors(render_executors, force=False)
        render_executors = []
        payload_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, min(num_cpu_workers, len(manifest_paths))),
            mp_context=ctx,
            initializer=_mp_payload_worker_init,
            initargs=(str(cache_dir), max_vis_points, json.dumps(scale_results)),
        )
        for task_idx, manifest_path in enumerate(manifest_paths):
            if ordered_meta[task_idx] is None:
                continue
            payload_futures.append(
                payload_executor.submit(
                    _mp_payload_worker_run,
                    task_idx,
                    str(manifest_path),
                    str(render_cache_dir / ordered_meta[task_idx]["render_path"]),
                )
            )

        ordered_payloads: list[dict[str, Any] | None] = [None] * len(manifest_paths)
        for future in tqdm(
            concurrent.futures.as_completed(payload_futures),
            total=len(payload_futures),
            desc="Building examiner payloads",
            leave=False,
        ):
            task_idx, meta = future.result()
            ordered_payloads[task_idx] = meta
        images = [item for item in ordered_payloads if item is not None]
    except KeyboardInterrupt:
        interrupted = True
        for future in render_futures + payload_futures:
            future.cancel()
        if payload_executor is not None:
            shutdown_executors([payload_executor], force=True)
        shutdown_executors(render_executors, force=True)
        raise
    except Exception:
        if payload_executor is not None:
            shutdown_executors([payload_executor], force=True)
        shutdown_executors(render_executors, force=True)
        raise
    finally:
        if not interrupted:
            if payload_executor is not None:
                shutdown_executors([payload_executor], force=False)
            shutdown_executors(render_executors, force=False)
        shutil.rmtree(render_cache_dir, ignore_errors=True)
        gc.collect()

    index_path.write_text(json.dumps({"images": images}))
    return images


def launch_examiner(
    image_meta: list[dict[str, Any]],
    config_path: Path,
    viser_port: int,
    nerf_point_size: float,
    nerf_centroid_point_size: float,
) -> viser.ViserServer:
    if not image_meta:
        raise ValueError("No payloads to visualize.")

    cache_dir = _cache_dir(config_path)
    max_instance_count = max(item["num_instances"] for item in image_meta)
    server = viser.ViserServer(port=viser_port)
    handles = {
        "nerf_instance": None,
        "sam3d_pointmap_scaled": None,
        "nerf_centroid": None,
        "sam3d_centroid_scaled": None,
    }
    state = {
        "sync": False,
        "loaded_image_idx": None,
        "loaded_instances": {},
    }

    with server.gui.add_folder("Centroid Examiner"):
        prev_image_btn = server.gui.add_button("Previous Image")
        next_image_btn = server.gui.add_button("Next Image")
        image_slider = server.gui.add_slider("Image Index", min=0, max=len(image_meta) - 1, step=1, initial_value=0)
        instance_slider = server.gui.add_slider("Instance Index", min=0, max=max(0, max_instance_count - 1), step=1, initial_value=0)
        status = server.gui.add_markdown("Preparing scene")

    def load_current_image(image_idx: int) -> None:
        if state["loaded_image_idx"] == image_idx:
            return
        state["loaded_image_idx"] = image_idx
        state["loaded_instances"] = {}
        image_item = image_meta[image_idx]
        payload_path = cache_dir / image_item["payload_path"]
        with np.load(payload_path) as data:
            for instance_meta in image_item["instances"]:
                array_index = instance_meta["array_index"]
                state["loaded_instances"][array_index] = {
                    "nerf_instance_points": data[f"nerf_instance_points_{array_index}"],
                    "nerf_centroid_points": data[f"nerf_centroid_points_{array_index}"],
                    "sam3d_pointmap_scaled": data[f"sam3d_pointmap_scaled_{array_index}"],
                    "sam3d_centroid_scaled": data[f"sam3d_centroid_scaled_{array_index}"],
                }

    def render_current() -> None:
        image_idx = int(image_slider.value)
        image_item = image_meta[image_idx]
        load_current_image(image_idx)
        instance_count = image_item["num_instances"]
        if instance_count == 0:
            instance_idx = 0
            instance_meta = {"sam3d_instance_idx": 0, "mask_index": -1, "scale": None, "array_index": -1}
            instance_arrays = {
                "nerf_instance_points": _empty_points(),
                "nerf_centroid_points": _empty_points(),
                "sam3d_pointmap_scaled": _empty_points(),
                "sam3d_centroid_scaled": _empty_points(),
            }
        else:
            instance_idx = min(int(instance_slider.value), instance_count - 1)
            if instance_idx != int(instance_slider.value):
                state["sync"] = True
                instance_slider.value = instance_idx
                state["sync"] = False
            instance_meta = image_item["instances"][instance_idx]
            instance_arrays = state["loaded_instances"][instance_meta["array_index"]]

        for name, handle in handles.items():
            if handle is not None:
                handle.remove()
            handles[name] = None

        handles["nerf_instance"] = server.scene.add_point_cloud(
            "/nerf_instance",
            points=instance_arrays["nerf_instance_points"],
            colors=(0.1, 0.35, 0.95),
            point_size=nerf_point_size,
        )
        handles["sam3d_pointmap_scaled"] = server.scene.add_point_cloud(
            "/sam3d_pointmap_scaled",
            points=instance_arrays["sam3d_pointmap_scaled"],
            colors=(0.0, 0.0, 0.0),
            point_size=nerf_point_size,
        )
        handles["nerf_centroid"] = server.scene.add_point_cloud(
            "/nerf_centroid",
            points=instance_arrays["nerf_centroid_points"],
            colors=(1.0, 0.0, 1.0),
            point_size=nerf_centroid_point_size,
        )
        handles["sam3d_centroid_scaled"] = server.scene.add_point_cloud(
            "/sam3d_centroid_scaled",
            points=instance_arrays["sam3d_centroid_scaled"],
            colors=(1.0, 0.55, 0.0),
            point_size=0.03,
        )

        scale = instance_meta["scale"]
        scale_str = "missing" if scale is None else f"{float(scale):.6f}"
        status.content = (
            f"image {image_idx + 1}/{len(image_meta)}\n"
            f"path: {Path(image_item['image_path']).name}\n"
            f"instances: {instance_count}\n"
            f"instance idx: {instance_idx}\n"
            f"mask idx: {instance_meta['mask_index']}\n"
            f"scale: {scale_str}"
        )

    def set_image(image_idx: int) -> None:
        state["sync"] = True
        image_slider.value = image_idx % len(image_meta)
        instance_slider.value = 0
        state["sync"] = False
        render_current()

    @image_slider.on_update
    def _(_event) -> None:
        if state["sync"]:
            return
        render_current()

    @instance_slider.on_update
    def _(_event) -> None:
        if state["sync"]:
            return
        render_current()

    @prev_image_btn.on_click
    def _(_event) -> None:
        set_image(int(image_slider.value) - 1)

    @next_image_btn.on_click
    def _(_event) -> None:
        set_image(int(image_slider.value) + 1)

    render_current()
    print(f"Viser running: http://localhost:{viser_port}")
    print(f"Public URL: {server.request_share_url(verbose=True)}")
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/f3rm/fresh/objaverse/car2new"))
    parser.add_argument("--config-path", type=Path, default=Path("mar10_outputs/car2new_cent/f3rm/2026-03-11_085644/config.yml"))
    parser.add_argument("--sam3d-feature-name", type=str, default="sam3d_")
    parser.add_argument("--max-vis-points", type=int, default=120_000)
    parser.add_argument("--viser-port", type=int, default=8891)
    parser.add_argument("--nerf-point-size", type=float, default=0.0035)
    parser.add_argument("--nerf-centroid-point-size", type=float, default=0.0008)
    parser.add_argument("--num-workers-per-gpu", type=int, default=2)
    parser.add_argument("--num-cpu-workers", type=int, default=8)
    parser.add_argument("--wait-for-ckpt-step", type=int, default=79999)
    args = parser.parse_args()

    if args.wait_for_ckpt_step is not None:
        wait_for_ckpt_step(args.config_path, args.wait_for_ckpt_step)

    scale_results = run_parallel_scale_calibration(
        data_dir=args.data_dir,
        config_path=args.config_path,
        sam3d_feature_name=args.sam3d_feature_name,
        num_workers_per_gpu=args.num_workers_per_gpu,
    )
    image_meta = precompute_examiner_cache(
        data_dir=args.data_dir,
        config_path=args.config_path,
        sam3d_feature_name=args.sam3d_feature_name,
        scale_results=scale_results,
        max_vis_points=args.max_vis_points,
        num_workers_per_gpu=args.num_workers_per_gpu,
        num_cpu_workers=args.num_cpu_workers,
    )
    print(f"Prepared examiner cache for {len(image_meta)} images")
    print(f"Cache dir: {_cache_dir(args.config_path)}")
    print("Starting Viser")
    server = launch_examiner(
        image_meta=image_meta,
        config_path=args.config_path,
        viser_port=args.viser_port,
        nerf_point_size=args.nerf_point_size,
        nerf_centroid_point_size=args.nerf_centroid_point_size,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
