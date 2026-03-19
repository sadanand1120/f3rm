#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import pty
import re
import select
import shlex
import shutil
import struct
import subprocess
import sys
import termios
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
DEFAULT_STREAM_COLUMNS = 120
DEFAULT_STREAM_ROWS = 40
OSC_ESCAPE_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)")
CSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SINGLE_ESCAPE_RE = re.compile(r"\x1b[@-Z\\-_]")
BENCHMARK_EXTRACT_GPU_IDS = (6, 7)
BENCHMARK_TRAIN_GPU_IDS = (6,)


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    dataset: Path
    extract_gpu_ids: tuple[int, ...]
    train_gpu_ids: tuple[int, ...]


PROFILES = {
    "smoke": BenchmarkProfile(
        name="smoke",
        dataset=REPO_ROOT / "datasets/f3rm/test/poster_smoke",
        extract_gpu_ids=BENCHMARK_EXTRACT_GPU_IDS,
        train_gpu_ids=BENCHMARK_TRAIN_GPU_IDS,
    ),
    "measure": BenchmarkProfile(
        name="measure",
        dataset=REPO_ROOT / "datasets/f3rm/test/poster2",
        extract_gpu_ids=BENCHMARK_EXTRACT_GPU_IDS,
        train_gpu_ids=BENCHMARK_TRAIN_GPU_IDS,
    ),
}


class BenchmarkError(RuntimeError):
    pass


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
            "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
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


def count_f3rm_loc(root: Path) -> int:
    total = 0
    for path in sorted(root.rglob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


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


def _to_display_rgb(image) -> "torch.Tensor":
    import torch

    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    image = image.detach().cpu()
    if image.dtype == torch.uint8:
        image = image.float().div_(255.0)
    else:
        image = image.float()
    return image.clamp_(0.0, 1.0)


def create_visual_checks(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    import gc
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    from f3rm.features.utils import apply_pca_colormap

    config_path = run_dir / "config.yml"
    if not config_path.exists():
        raise BenchmarkError(f"Nerfstudio config not found: {config_path}")

    _, pipeline, _, _ = eval_setup(config_path, test_mode="val")
    train_dataset = getattr(pipeline.datamanager, "train_dataset", None)
    train_ray_generator = getattr(pipeline.datamanager, "train_ray_generator", None)
    if train_dataset is None or train_ray_generator is None:
        raise BenchmarkError("Loaded pipeline does not expose train_dataset/train_ray_generator for visual checks")

    output_dir.mkdir(parents=True, exist_ok=True)
    indices = select_visual_check_indices(len(train_dataset))
    paths: list[str] = []

    with torch.no_grad():
        for idx in indices:
            batch = train_dataset.get_data(idx)
            original_rgb = _to_display_rgb(batch["image"])
            camera_ray_bundle = train_ray_generator.cameras.generate_rays(camera_indices=idx, keep_shape=True)
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True)
            render_rgb = _to_display_rgb(outputs["rgb"])
            if "feature" not in outputs:
                raise BenchmarkError("Model outputs did not include rendered features for visual checks")
            render_feature = apply_pca_colormap(outputs["feature"], niter=5, q_min=0.01, q_max=0.99).cpu()

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(original_rgb.numpy())
            axes[0].set_title(f"Original RGB\nidx={idx}")
            axes[1].imshow(render_rgb.numpy())
            axes[1].set_title("Render RGB")
            axes[2].imshow(render_feature.numpy())
            axes[2].set_title("Render Feature")
            for axis in axes:
                axis.axis("off")
            fig.tight_layout()

            out_path = output_dir / f"visual_check_{idx:06d}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(out_path))

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"count": len(paths), "indices": indices, "paths": paths}


def tail_lines(path: Path, n: int = 40) -> list[str]:
    if not path.exists():
        return []
    return read_clean_log_lines(path)[-n:]


def strip_terminal_escapes(text: str) -> str:
    text = OSC_ESCAPE_RE.sub("", text)
    text = CSI_ESCAPE_RE.sub("", text)
    text = SINGLE_ESCAPE_RE.sub("", text)
    return text


def read_clean_log_lines(path: Path) -> list[str]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    text = strip_terminal_escapes(text)
    return text.splitlines()


def open_stream_pty() -> tuple[int, int]:
    master_fd, slave_fd = pty.openpty()
    try:
        size = os.get_terminal_size(sys.stdout.fileno())
        rows, columns = size.lines, size.columns
    except OSError:
        rows, columns = DEFAULT_STREAM_ROWS, DEFAULT_STREAM_COLUMNS
    try:
        winsize = struct.pack("HHHH", rows, columns, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
    except OSError:
        pass
    return master_fd, slave_fd


def stream_process_output(stream_fd: int, log_file) -> None:
    while True:
        ready, _, _ = select.select([stream_fd], [], [], 0.5)
        if not ready:
            continue
        try:
            chunk = os.read(stream_fd, 4096)
        except OSError as exc:
            if exc.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        log_file.write(chunk)
        log_file.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()


def run_logged_subprocess(command: list[str], env: dict[str, str], log_path: Path) -> float:
    start = perf_counter()
    stream_env = env | {"PYTHONUNBUFFERED": "1"}
    with log_path.open("wb") as log_file:
        print(f"[benchmark] running: {shlex.join(command)}", flush=True)
        print(f"[benchmark] logging to: {log_path}", flush=True)
        master_fd, slave_fd = open_stream_pty()
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=stream_env,
                stdin=subprocess.DEVNULL,
                stdout=slave_fd,
                stderr=slave_fd,
            )
        finally:
            os.close(slave_fd)
        try:
            stream_process_output(master_fd, log_file)
        finally:
            os.close(master_fd)
            returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
    return perf_counter() - start


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


def extract_final_metrics(event_scalars: dict[str, float]) -> dict[str, float]:
    return {
        tag.removeprefix("Final Metrics/"): value
        for tag, value in event_scalars.items()
        if tag.startswith("Final Metrics/")
    }


def canonical_metric_summary(final_metrics: dict[str, float]) -> dict[str, float | None]:
    return {
        "final_eval_all_psnr": final_metrics.get("Eval All Images/psnr"),
        "final_eval_all_ssim": final_metrics.get("Eval All Images/ssim"),
        "final_eval_all_lpips": final_metrics.get("Eval All Images/lpips"),
        "final_train_feature_error": final_metrics.get("Train Batch/feature_error"),
    }


def run_benchmark(args: argparse.Namespace) -> int:
    profile = PROFILES[args.profile]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in profile.train_gpu_ids)

    ensure_dataset(profile.dataset)
    dataset_stats = scan_dataset(profile.dataset)

    run_id = args.run_tag or strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "benchmark_runs" / profile.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ns_output_dir = run_dir / "ns_outputs"
    summary_path = run_dir / "summary.json"
    extract_log = run_dir / "extract.log"
    train_log = run_dir / "train.log"
    visual_dir = run_dir / "visual_check"
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
        "--skip-visualization",
    ]
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
        "f3rm_loc": count_f3rm_loc(REPO_ROOT / "f3rm"),
    }

    try:
        remove_dir(feature_cache_dir)

        extract_wall_s = run_logged_subprocess(command=extract_command, env=extract_env, log_path=extract_log)
        train_wall_s = run_logged_subprocess(command=train_command, env=train_env, log_path=train_log)

        ns_run_dir = locate_single_run_dir(ns_output_dir, experiment_name)
        visual_check = create_visual_checks(run_dir=ns_run_dir, output_dir=visual_dir)
        event_scalars = load_event_scalars(ns_run_dir)
        final_metrics = extract_final_metrics(event_scalars)

        summary.update(
            {
                "extract_wall_s": extract_wall_s,
                "train_wall_s": train_wall_s,
                "end_to_end_wall_s": extract_wall_s + train_wall_s,
                "visual_check": visual_check,
                "train_run_dir": str(ns_run_dir),
                "final_metrics": final_metrics,
            }
        )
        summary.update(canonical_metric_summary(final_metrics))

        if visual_check["count"] != 4:
            summary["status"] = "failure"
            summary["visual_check_validation_error"] = "Expected exactly 4 saved visual checks."
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
    except Exception as exc:
        failed_log = train_log if train_log.exists() else extract_log
        summary.update(
            {
                "status": "failure",
                "error": f"{type(exc).__name__}: {exc}",
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
