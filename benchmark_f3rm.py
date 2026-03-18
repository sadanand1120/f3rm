#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, strftime
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
CONTAINER_NAME = "fresh"
CONDA_SH = "/opt/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV = "f3rm"
SUMMARY_START = "=== F3RM_BENCHMARK_SUMMARY_BEGIN ==="
SUMMARY_END = "=== F3RM_BENCHMARK_SUMMARY_END ==="
GPU_ACTIVE_UTIL_THRESHOLD = 10


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    dataset: Path
    extract_gpu_ids: tuple[int, ...]
    train_gpu_ids: tuple[int, ...]
    skip_visualization: bool = True


PROFILES = {
    "smoke": BenchmarkProfile(
        name="smoke",
        dataset=REPO_ROOT / "datasets/f3rm/test/poster_smoke",
        extract_gpu_ids=(1, 2),
        train_gpu_ids=(1,),
    ),
    "measure": BenchmarkProfile(
        name="measure",
        dataset=REPO_ROOT / "datasets/f3rm/test/poster2",
        extract_gpu_ids=(1, 2),
        train_gpu_ids=(1,),
    ),
}


class BenchmarkError(RuntimeError):
    pass


class GpuPoller:
    def __init__(self, gpu_ids: tuple[int, ...], sample_interval_s: float = 0.5) -> None:
        self.gpu_ids = gpu_ids
        self.sample_interval_s = sample_interval_s
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self, t_s: float) -> None:
        cmd = [
            "nvidia-smi",
            "-i",
            ",".join(str(gpu_id) for gpu_id in self.gpu_ids),
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        tick_samples = []
        for raw_line in output.strip().splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) != 4:
                continue
            tick_samples.append(
                {
                    "time_s": round(t_s, 3),
                    "gpu": int(parts[0]),
                    "utilization_gpu": int(parts[1]),
                    "utilization_memory": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                }
            )
        with self._lock:
            self._samples.extend(tick_samples)

    def _run(self) -> None:
        start = perf_counter()
        while not self._stop.is_set():
            try:
                self._sample_once(perf_counter() - start)
            except Exception:
                pass
            if self._stop.wait(self.sample_interval_s):
                break

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        with self._lock:
            return list(self._samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed benchmark harness for F3RM extraction + training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--run-tag", type=str, default=None, help="Optional stable suffix for the benchmark run directory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep-ns-output",
        action="store_true",
        help="Keep Nerfstudio output directories after parsing. Logs and summary are always kept.",
    )
    return parser.parse_args()


def run_in_container(args: argparse.Namespace) -> int:
    child_args = ["python", "benchmark_f3rm.py", "--profile", args.profile, "--seed", str(args.seed)]
    if args.run_tag is not None:
        child_args.extend(["--run-tag", args.run_tag])
    if args.keep_ns_output:
        child_args.append("--keep-ns-output")
    command = " && ".join(
        [
            f"source {shlex.quote(CONDA_SH)}",
            f"conda activate {shlex.quote(CONDA_ENV)}",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            "export F3RM_BENCHMARK_INSIDE=1",
            shlex.join(child_args),
        ]
    )
    return subprocess.run(["docker", "exec", CONTAINER_NAME, "bash", "-lc", command], cwd=REPO_ROOT).returncode


def ensure_dataset(dataset: Path) -> None:
    if not dataset.exists():
        raise BenchmarkError(f"Dataset directory not found: {dataset}")
    transforms = dataset / "transforms.json"
    if not transforms.exists():
        raise BenchmarkError(f"transforms.json not found: {transforms}")


def scan_dataset(dataset: Path) -> dict[str, Any]:
    from PIL import Image

    images = sorted((dataset / "images").glob("*.png"))
    if not images:
        raise BenchmarkError(f"No PNG images found in {dataset / 'images'}")
    with Image.open(images[0]) as image:
        width, height = image.size
    return {
        "num_images_total": len(images),
        "train_image_wh": [width, height],
        "train_split_fraction": 0.95,
    }


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def select_visual_check_indices(num_images: int, num_checks: int = 4) -> list[int]:
    if num_images < num_checks:
        raise BenchmarkError(f"Need at least {num_checks} images for visual checks, found {num_images}")
    if num_checks == 1:
        return [0]
    indices = []
    for i in range(num_checks):
        idx = round(i * (num_images - 1) / (num_checks - 1))
        if not indices or idx != indices[-1]:
            indices.append(idx)
    while len(indices) < num_checks:
        candidate = indices[-1] + 1
        if candidate >= num_images:
            break
        indices.append(candidate)
    if len(indices) != num_checks:
        raise BenchmarkError(f"Failed to select {num_checks} unique visual-check indices from {num_images} images")
    return indices


def create_extract_visual_checks(feature_cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    from f3rm.features.utils import apply_pca_colormap

    meta_path = feature_cache_dir / "meta.pt"
    if not meta_path.exists():
        raise BenchmarkError(f"Feature metadata not found: {meta_path}")
    meta = torch.load(meta_path, map_location="cpu")
    image_fnames = [Path(path) for path in meta.get("image_fnames", [])]
    if not image_fnames:
        raise BenchmarkError(f"No image_fnames recorded in {meta_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    indices = select_visual_check_indices(len(image_fnames))
    paths: list[str] = []

    for idx in indices:
        image_path = image_fnames[idx]
        feature_path = feature_cache_dir / f"image_{idx:06d}.npy"
        if not image_path.exists():
            raise BenchmarkError(f"Original image for visual check not found: {image_path}")
        if not feature_path.exists():
            raise BenchmarkError(f"Feature file for visual check not found: {feature_path}")

        with Image.open(image_path) as image:
            original = image.convert("RGB")
            original_np = np.array(original)

        feature = torch.from_numpy(np.load(feature_path)).float()
        pca_image = apply_pca_colormap(feature, niter=5, q_min=0.01, q_max=0.99).cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(original_np)
        axes[0].set_title(f"Original\nidx={idx} {image_path.name}")
        axes[1].imshow(pca_image)
        axes[1].set_title(f"Feature PCA\nimage_{idx:06d}.npy")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()

        out_path = output_dir / f"visual_check_{idx:06d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(out_path))

    return {
        "count": len(paths),
        "indices": indices,
        "paths": paths,
    }


def tail_lines(path: Path, n: int = 40) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-n:]


def parse_marker_scalars(log_path: Path, prefix: str) -> dict[str, float]:
    results: dict[str, float] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):].strip()
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)
        try:
            results[key] = float(value)
        except ValueError:
            continue
    return results


def parse_marker_values(log_path: Path, prefix: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):].strip()
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)
        results[key] = value
    return results


def summarize_gpu_samples(samples: list[dict[str, Any]], gpu_ids: tuple[int, ...]) -> dict[str, Any]:
    per_gpu: dict[str, Any] = {}
    by_tick: dict[float, dict[int, dict[str, Any]]] = {}
    for sample in samples:
        gpu = sample["gpu"]
        key = str(gpu)
        summary = per_gpu.setdefault(
            key,
            {"samples": 0, "max_utilization_gpu": 0, "max_memory_mb": 0, "active_samples": 0},
        )
        summary["samples"] += 1
        summary["max_utilization_gpu"] = max(summary["max_utilization_gpu"], sample["utilization_gpu"])
        summary["max_memory_mb"] = max(summary["max_memory_mb"], sample["memory_used_mb"])
        if sample["utilization_gpu"] > GPU_ACTIVE_UTIL_THRESHOLD:
            summary["active_samples"] += 1
        by_tick.setdefault(sample["time_s"], {})[gpu] = sample

    overlap_samples = 0
    for tick in by_tick.values():
        if all(tick.get(gpu, {}).get("utilization_gpu", 0) > GPU_ACTIVE_UTIL_THRESHOLD for gpu in gpu_ids):
            overlap_samples += 1

    return {
        "gpu_ids": list(gpu_ids),
        "sample_count": len(samples),
        "overlap_active_samples": overlap_samples,
        "parallel_ok": overlap_samples > 0 and all(per_gpu.get(str(gpu), {}).get("active_samples", 0) > 0 for gpu in gpu_ids),
        "per_gpu": per_gpu,
    }


def peak_memory_mb(summary: dict[str, Any]) -> int:
    per_gpu = summary.get("per_gpu", {})
    return max((int(stats.get("max_memory_mb", 0)) for stats in per_gpu.values()), default=0)


def stream_process_output(process: subprocess.Popen[bytes], log_file) -> None:
    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(4096)
        if not chunk:
            break
        log_file.write(chunk)
        log_file.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()


def run_logged_subprocess(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    gpu_ids: tuple[int, ...],
) -> tuple[float, list[dict[str, Any]]]:
    poller = GpuPoller(gpu_ids)
    start = perf_counter()
    with log_path.open("wb") as log_file:
        print(f"[benchmark] running: {shlex.join(command)}", flush=True)
        print(f"[benchmark] logging to: {log_path}", flush=True)
        poller.start()
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            try:
                stream_process_output(process, log_file)
            finally:
                returncode = process.wait()
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, command)
        finally:
            samples = poller.stop()
    return perf_counter() - start, samples


def locate_single_run_dir(ns_output_dir: Path, experiment_name: str) -> Path:
    candidate_root = ns_output_dir / experiment_name / "f3rm"
    if not candidate_root.exists():
        raise BenchmarkError(f"Nerfstudio output root not found: {candidate_root}")
    candidates = [path for path in candidate_root.iterdir() if path.is_dir()]
    if not candidates:
        raise BenchmarkError(f"No timestamped Nerfstudio run directories found in {candidate_root}")
    if len(candidates) > 1:
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def load_event_scalars(run_dir: Path) -> dict[str, float]:
    from tensorboard.backend.event_processing import event_accumulator

    accumulator = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars: dict[str, float] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if events:
            scalars[tag] = float(events[-1].value)
    return scalars


def run_benchmark(args: argparse.Namespace) -> int:
    profile = PROFILES[args.profile]
    ensure_dataset(profile.dataset)
    dataset_stats = scan_dataset(profile.dataset)

    run_id = args.run_tag or strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "benchmark_runs" / profile.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ns_output_dir = run_dir / "ns_outputs"
    summary_path = run_dir / "summary.json"
    extract_log = run_dir / "extract.log"
    train_log = run_dir / "train.log"
    extract_gpu_csv = run_dir / "extract_gpu.csv"
    train_gpu_csv = run_dir / "train_gpu.csv"
    extract_visual_dir = run_dir / "extract_visual_check"
    feature_cache_dir = profile.dataset / "features" / "clip"

    experiment_name = f"{profile.name}_{run_id}"
    train_image_wh = dataset_stats["train_image_wh"]
    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHONHASHSEED": str(args.seed),
            "F3RM_NUM_IMAGES_TOTAL": str(dataset_stats["num_images_total"]),
            "F3RM_TRAIN_SPLIT_FRACTION": str(dataset_stats["train_split_fraction"]),
            "F3RM_TRAIN_IMAGE_WH": f"{train_image_wh[0]},{train_image_wh[1]}",
        }
    )

    extract_env = base_env | {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in profile.extract_gpu_ids)}
    train_env = base_env | {
        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in profile.train_gpu_ids),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    extract_command = [
        sys.executable,
        "f3rm/features/extract_features_standalone.py",
        "--data",
        str(profile.dataset),
        "--feature-type",
        "CLIP",
    ]
    if profile.skip_visualization:
        extract_command.append("--skip-visualization")

    train_command = [
        "ns-train",
        "f3rm",
        "--machine.num-devices",
        "1",
        "--machine.seed",
        str(args.seed),
        "--vis",
        "tensorboard",
        "--data",
        str(profile.dataset),
        "--output-dir",
        str(ns_output_dir),
        "--experiment-name",
        experiment_name,
    ]

    summary: dict[str, Any] = {
        "status": "ok",
        "profile": profile.name,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "dataset": str(profile.dataset),
        "dataset_num_images_total": dataset_stats["num_images_total"],
        "train_image_wh": train_image_wh,
        "feature_cache_dir": str(feature_cache_dir),
        "extract_command": extract_command,
        "train_command": train_command,
        "extract_log": str(extract_log),
        "train_log": str(train_log),
    }

    try:
        remove_dir(feature_cache_dir)

        extract_wall_s, extract_gpu_samples = run_logged_subprocess(
            command=extract_command,
            env=extract_env,
            log_path=extract_log,
            gpu_ids=profile.extract_gpu_ids,
        )
        extract_visual_check = create_extract_visual_checks(feature_cache_dir=feature_cache_dir, output_dir=extract_visual_dir)
        with extract_gpu_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["time_s", "gpu", "utilization_gpu", "utilization_memory", "memory_used_mb"],
            )
            writer.writeheader()
            writer.writerows(extract_gpu_samples)

        train_wall_s, train_gpu_samples = run_logged_subprocess(
            command=train_command,
            env=train_env,
            log_path=train_log,
            gpu_ids=profile.train_gpu_ids,
        )
        with train_gpu_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["time_s", "gpu", "utilization_gpu", "utilization_memory", "memory_used_mb"],
            )
            writer.writeheader()
            writer.writerows(train_gpu_samples)

        ns_run_dir = locate_single_run_dir(ns_output_dir, experiment_name)
        event_scalars = load_event_scalars(ns_run_dir)

        extract_timings = parse_marker_scalars(extract_log, "[F3RM_TIMING]")
        extract_info = parse_marker_values(extract_log, "[F3RM_INFO]")
        extract_gpu = summarize_gpu_samples(extract_gpu_samples, profile.extract_gpu_ids)
        train_gpu = summarize_gpu_samples(train_gpu_samples, profile.train_gpu_ids)
        final_metrics = {
            tag.removeprefix("Final Metrics/"): value
            for tag, value in event_scalars.items()
            if tag.startswith("Final Metrics/")
        }
        final_timings = {
            tag.removeprefix("Final Timing/"): value
            for tag, value in event_scalars.items()
            if tag.startswith("Final Timing/")
        }
        summary.update(
            {
                "extract_wall_s": extract_wall_s,
                "train_wall_s": train_wall_s,
                "end_to_end_wall_s": extract_wall_s + train_wall_s,
                "extract_timings": extract_timings,
                "extract_info": extract_info,
                "extract_visual_check": extract_visual_check,
                "extract_gpu": extract_gpu,
                "train_gpu": train_gpu,
                "train_run_dir": str(ns_run_dir),
                "final_metrics": final_metrics,
                "final_timings": final_timings,
                "writer_scalars": {
                    tag: value
                    for tag, value in event_scalars.items()
                    if tag in {"Train Iter (time)", "Train Total (time)", "ETA (time)", "Train Rays / Sec", "Test Rays / Sec"}
                },
                "extract_worker_init_s": extract_timings.get("extract.worker_init_s"),
                "extract_worker_warmup_s": extract_timings.get("extract.worker_warmup_s"),
                "extract_batch_compute_s": extract_timings.get("extract.batch_compute_s"),
                "extract_per_image_write_s": extract_timings.get("extract.per_image_write_s"),
                "extract_parallel_ok": int(extract_gpu["parallel_ok"]),
                "peak_extract_gpu_mem_mb": peak_memory_mb(extract_gpu),
                "peak_train_gpu_mem_mb": peak_memory_mb(train_gpu),
                "train_feature_cache_load_avg_s": final_timings.get("Train/feature_cache_load/avg_s"),
                "train_feature_window_fetch_avg_s": final_timings.get("Train/feature_window_fetch/avg_s"),
                "train_feature_window_stack_avg_s": final_timings.get("Train/feature_window_stack/avg_s"),
                "train_batch_load_avg_s": final_timings.get("Train/batch_load/avg_s"),
                "train_model_forward_avg_s": final_timings.get("Train/model_forward/avg_s"),
                "final_eval_all_psnr": final_metrics.get("Eval All Images/psnr"),
                "final_eval_all_ssim": final_metrics.get("Eval All Images/ssim"),
                "final_eval_all_lpips": final_metrics.get("Eval All Images/lpips"),
                "final_train_feature_error": final_metrics.get("Train Batch/feature_error"),
            }
        )
        if not extract_gpu["parallel_ok"]:
            summary["status"] = "failure"
            summary["parallel_validation_error"] = "Extraction did not show overlapping activity on all requested GPUs."
        if extract_visual_check["count"] != 4:
            summary["status"] = "failure"
            summary["visual_check_validation_error"] = "Expected exactly 4 extraction visual-check plots."
        if not final_metrics:
            summary["status"] = "failure"
            summary["metrics_validation_error"] = "No Final Metrics/* scalars were found in the TensorBoard event files."
    except subprocess.CalledProcessError as exc:
        failed_log = train_log if train_log.exists() else extract_log
        summary.update(
            {
                "status": "failure",
                "returncode": exc.returncode,
                "failed_log": str(failed_log),
                "failed_log_tail": tail_lines(failed_log),
            }
        )
    finally:
        remove_dir(feature_cache_dir)
        if not args.keep_ns_output and ns_output_dir.exists():
            remove_dir(ns_output_dir)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(SUMMARY_START)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(SUMMARY_END)
    return 0 if summary["status"] == "ok" else 1


def main() -> int:
    args = parse_args()
    if os.environ.get("F3RM_BENCHMARK_INSIDE") != "1":
        return run_in_container(args)
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
