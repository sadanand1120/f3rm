#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from f3rm.train_schedule import derive_train_schedule


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "f3rm" / "f3rm_config.py"
DATA_PATH = REPO_ROOT / "datasets/f3rm/test/poster"
BENCHMARK_RUNS_ROOT = REPO_ROOT / "benchmark_runs"
BENCHMARK_OUTPUTS_ROOT = REPO_ROOT / "benchmark_outputs"
EXPERIMENT_NAME = "poster-train-time"
METHOD_NAME = "f3rm"
EXPECTED_CONDA_PREFIX = "/opt/miniconda3/envs/f3rm"
EXPECTED_FEATURE_CACHES = ("clip", "foreground_")
BENCHMARK_SEED = 0
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
RESULTS_TSV_FILENAME = "results_row.tsv"
RESULTS_TSV_NUMERIC_PRECISION = 2
SCHEDULE_FLAG_NAMES = (
    "NUM_IMAGES_TOTAL",
    "TRAIN_SPLIT_FRACTION",
    "TRAIN_IMAGE_WH",
    "TRAIN_NUM_RAYS_PER_BATCH",
    "TRAIN_NUM_IMAGES_TO_SAMPLE_FROM",
    "PIXEL_VISITATION",
    "WINDOW_COVERAGE",
)


def load_schedule_flags(config_path: Path) -> dict[str, int | float]:
    module = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    values: dict[str, int | float] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in SCHEDULE_FLAG_NAMES:
            continue
        values[name] = ast.literal_eval(node.value)
    missing = [name for name in SCHEDULE_FLAG_NAMES if name not in values]
    if missing:
        raise RuntimeError(f"missing schedule flags in {config_path}: {missing}")
    return values


ACTIVE_SCHEDULE_FLAGS = load_schedule_flags(CONFIG_PATH)
NUM_IMAGES_TOTAL = int(ACTIVE_SCHEDULE_FLAGS["NUM_IMAGES_TOTAL"])
TRAIN_SPLIT_FRACTION = float(ACTIVE_SCHEDULE_FLAGS["TRAIN_SPLIT_FRACTION"])
TRAIN_IMAGE_WH = tuple(int(value) for value in ACTIVE_SCHEDULE_FLAGS["TRAIN_IMAGE_WH"])
TRAIN_NUM_RAYS_PER_BATCH = int(ACTIVE_SCHEDULE_FLAGS["TRAIN_NUM_RAYS_PER_BATCH"])
TRAIN_NUM_IMAGES_TO_SAMPLE_FROM = int(ACTIVE_SCHEDULE_FLAGS["TRAIN_NUM_IMAGES_TO_SAMPLE_FROM"])
PIXEL_VISITATION = float(ACTIVE_SCHEDULE_FLAGS["PIXEL_VISITATION"])
WINDOW_COVERAGE = float(ACTIVE_SCHEDULE_FLAGS["WINDOW_COVERAGE"])
TOTAL_DATASET_IMAGES = NUM_IMAGES_TOTAL
ACTIVE_TRAIN_SCHEDULE = derive_train_schedule(
    num_images_total=NUM_IMAGES_TOTAL,
    train_split_fraction=TRAIN_SPLIT_FRACTION,
    train_image_wh=TRAIN_IMAGE_WH,
    train_num_rays_per_batch=TRAIN_NUM_RAYS_PER_BATCH,
    train_num_images_to_sample_from=TRAIN_NUM_IMAGES_TO_SAMPLE_FROM,
    pixel_visitation=PIXEL_VISITATION,
    window_coverage=WINDOW_COVERAGE,
)
TRAIN_IMAGE_COUNT = ACTIVE_TRAIN_SCHEDULE.train_image_count
TRAIN_TOTAL_PIXELS = ACTIVE_TRAIN_SCHEDULE.train_total_pixels
BASELINE_MAX_NUM_ITERATIONS = 8100
BASELINE_TRAIN_NUM_RAYS_PER_BATCH = 1 << 13
BASELINE_TRAIN_NUM_IMAGES_TO_SAMPLE_FROM = 32
BASELINE_TRAIN_NUM_TIMES_TO_REPEAT_IMAGES = 512
BASELINE_PIXEL_VISIT_X = (
    BASELINE_MAX_NUM_ITERATIONS * BASELINE_TRAIN_NUM_RAYS_PER_BATCH / TRAIN_TOTAL_PIXELS
)
BASELINE_WINDOW_COVERAGE_W = (
    math.ceil(BASELINE_MAX_NUM_ITERATIONS / BASELINE_TRAIN_NUM_TIMES_TO_REPEAT_IMAGES)
    * BASELINE_TRAIN_NUM_IMAGES_TO_SAMPLE_FROM
    / TRAIN_IMAGE_COUNT
)
SMOKE_MAX_ITERATIONS = 1024
QUALITY_CLEAR_PCT = 10.0
QUALITY_REVIEW_PCT = 15.0

RESULTS_TSV_COLUMNS = (
    "commit",
    "profile",
    "wall_time_s",
    "train_total_time_s",
    "startup_overhead_s",
    "train_iter_time_s",
    "train_rays_per_sec",
    "train_batch_load_s",
    "train_feature_cache_load_s",
    "train_feature_gather_s",
    "train_feature_populate_s",
    "train_model_forward_s",
    "train_metrics_loss_s",
    "train_image_viz_s",
    "eval_batch_load_s",
    "eval_feature_cache_load_s",
    "eval_feature_gather_s",
    "eval_feature_populate_s",
    "eval_image_s",
    "eval_all_images_s",
    "peak_gpu_mem_mb",
    "pixel_visit_x",
    "pixel_visit_x_pct_diff",
    "window_coverage_w",
    "window_coverage_w_pct_diff",
    "train_psnr",
    "train_feature_error",
    "train_foreground_acc",
    "eval_all_psnr",
    "eval_all_ssim",
    "eval_all_lpips",
    "max_quality_regression_pct",
    "status",
    "description",
    "summary_path",
)

RESULTS_TSV_NUMERIC_COLUMNS = frozenset(
    RESULTS_TSV_COLUMNS[2:-3]
)

TIMING_TAGS = {
    "train_total_time_s": "Train Total (time)",
    "train_iter_time_s": "Train Iter (time)",
    "train_rays_per_sec": "Train Rays / Sec",
    "train_batch_load_s": "Timing/Train/batch_load",
    "train_feature_cache_load_s": "Timing/Train/feature_cache_load",
    "train_feature_gather_s": "Timing/Train/feature_gather",
    "train_feature_populate_s": "Timing/Train/feature_populate",
    "train_model_forward_s": "Timing/Train/model_forward",
    "train_metrics_loss_s": "Timing/Train/metrics_loss",
    "train_image_viz_s": "Timing/Train/image_viz",
    "eval_batch_load_s": "Timing/Eval/batch_load",
    "eval_feature_cache_load_s": "Timing/Eval/feature_cache_load",
    "eval_feature_gather_s": "Timing/Eval/feature_gather",
    "eval_feature_populate_s": "Timing/Eval/feature_populate",
    "eval_image_s": "Timing/Eval/image_total",
    "eval_all_images_s": "Timing/Eval/all_images_total",
}

SELECTED_FINAL_METRICS = {
    "train_psnr": "Final Metrics/Train Batch/psnr",
    "train_feature_error": "Final Metrics/Train Batch/feature_error",
    "train_foreground_acc": "Final Metrics/Train Batch/foreground_acc",
    "eval_all_psnr": "Final Metrics/Eval All Images/psnr",
    "eval_all_ssim": "Final Metrics/Eval All Images/ssim",
    "eval_all_lpips": "Final Metrics/Eval All Images/lpips",
}

QUALITY_EXCLUDE_HINTS = ("fps", "num_rays_per_sec")
LOWER_IS_BETTER_HINTS = ("error", "loss", "distortion", "lpips", "mae", "mse", "rmse")
HIGHER_IS_BETTER_HINTS = ("psnr", "ssim", "foreground_acc")


@dataclass(frozen=True)
class Profile:
    name: str
    extra_args: tuple[str, ...] = ()


PROFILES = {
    "smoke": Profile(
        name="smoke",
        extra_args=(
            "--max-num-iterations",
            str(SMOKE_MAX_ITERATIONS),
            "--logging.profiler",
            "none",
            "--steps-per-save",
            "0",
            "--steps-per-eval-batch",
            str(SMOKE_MAX_ITERATIONS - 1),
            "--steps-per-eval-image",
            str(SMOKE_MAX_ITERATIONS - 1),
            "--steps-per-eval-all-images",
            str(SMOKE_MAX_ITERATIONS - 1),
            "--pipeline.steps-per-train-image-viz",
            "0",
        ),
    ),
    "measure": Profile(name="measure"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed F3RM training benchmark harness. Run this inside the `fresh` container with the `f3rm` env.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True, help="Benchmark profile to run.")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="Optional prior summary.json used to compute quality regression tradeoff bands.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command and exit.")
    return parser.parse_args()


def ensure_runtime() -> None:
    errors: list[str] = []
    if os.environ.get("CONDA_PREFIX") != EXPECTED_CONDA_PREFIX:
        errors.append(f"expected CONDA_PREFIX={EXPECTED_CONDA_PREFIX}, got {os.environ.get('CONDA_PREFIX')!r}")
    if not DATA_PATH.exists():
        errors.append(f"missing dataset: {DATA_PATH}")
    for cache_name in EXPECTED_FEATURE_CACHES:
        cache_dir = DATA_PATH / "features" / cache_name
        if not (cache_dir / "meta.pt").exists():
            errors.append(f"missing cached features: {cache_dir / 'meta.pt'}")
            continue
        file_count = len(list(cache_dir.glob("image_*.*")))
        if file_count != TOTAL_DATASET_IMAGES:
            errors.append(
                f"expected {TOTAL_DATASET_IMAGES} cached feature files in {cache_dir}, found {file_count}"
            )
    if shutil.which("ns-train") is None:
        errors.append("`ns-train` is not on PATH inside the active environment")
    if errors:
        raise SystemExit("Runtime validation failed:\n- " + "\n- ".join(errors))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def run_id(profile: Profile) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{profile.name}_{git_commit()}"


def build_command(profile: Profile, run_name: str) -> list[str]:
    command = [
        "ns-train",
        METHOD_NAME,
        "--machine.num-devices",
        "1",
        "--machine.seed",
        str(BENCHMARK_SEED),
        "--vis",
        "tensorboard",
        "--data",
        str(DATA_PATH),
        "--output-dir",
        str(BENCHMARK_OUTPUTS_ROOT),
        "--experiment-name",
        EXPERIMENT_NAME,
        "--timestamp",
        run_name,
    ]
    command.extend(profile.extra_args)
    return command


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(BENCHMARK_SEED)
    env["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def stream_subprocess(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            sys.stdout.flush()
            log_file.flush()
        return process.wait()


def load_event_scalars(run_dir: Path) -> dict[str, list[Any]]:
    from tensorboard.backend.event_processing import event_accumulator

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no tensorboard event files found under {run_dir}")
    accumulator = event_accumulator.EventAccumulator(str(run_dir), size_guidance={event_accumulator.SCALARS: 0})
    accumulator.Reload()
    return {tag: accumulator.Scalars(tag) for tag in accumulator.Tags().get("scalars", [])}


def last_scalar(events: dict[str, list[Any]], tag: str) -> float | None:
    if tag not in events or not events[tag]:
        return None
    return float(events[tag][-1].value)


def max_scalar(events: dict[str, list[Any]], tag: str) -> float | None:
    if tag not in events or not events[tag]:
        return None
    return max(float(event.value) for event in events[tag])


def extract_final_metrics(events: dict[str, list[Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for tag, tag_events in events.items():
        if not tag.startswith("Final Metrics/") or not tag_events:
            continue
        metrics[tag.removeprefix("Final Metrics/")] = float(tag_events[-1].value)
    return metrics


def metric_direction(name: str) -> str:
    lower_name = name.lower()
    if any(hint in lower_name for hint in LOWER_IS_BETTER_HINTS):
        return "lower"
    if any(hint in lower_name for hint in HIGHER_IS_BETTER_HINTS):
        return "higher"
    return "higher"


def is_quality_metric(name: str) -> bool:
    return not any(hint in name.lower() for hint in QUALITY_EXCLUDE_HINTS)


def compute_quality_guardrail(final_metrics: dict[str, float], baseline_summary_path: Path | None) -> dict[str, Any]:
    result = {
        "baseline_summary": None if baseline_summary_path is None else str(baseline_summary_path.resolve()),
        "max_regression_pct": None,
        "comfort_pct": QUALITY_CLEAR_PCT,
        "review_pct": QUALITY_REVIEW_PCT,
        "band": None,
        "passed": None,
        "regressions_pct": {},
    }
    if baseline_summary_path is None:
        return result

    baseline_payload = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    baseline_metrics = baseline_payload.get("final_metrics", {})
    regressions: dict[str, float] = {}
    for name, baseline_value in baseline_metrics.items():
        if name not in final_metrics or not is_quality_metric(name):
            continue
        current_value = final_metrics[name]
        denom = max(abs(float(baseline_value)), 1e-8)
        if metric_direction(name) == "lower":
            regressions[name] = max(0.0, (float(current_value) - float(baseline_value)) / denom * 100.0)
        else:
            regressions[name] = max(0.0, (float(baseline_value) - float(current_value)) / denom * 100.0)

    max_regression = max(regressions.values(), default=0.0)
    if max_regression <= QUALITY_CLEAR_PCT:
        band = "clear"
        passed: bool | None = True
    elif max_regression <= QUALITY_REVIEW_PCT:
        band = "judgment"
        passed = None
    else:
        band = "reject"
        passed = False
    result["max_regression_pct"] = max_regression
    result["band"] = band
    result["passed"] = passed
    result["regressions_pct"] = regressions
    return result


def pct_diff_from_baseline(value: float | None, baseline: float) -> float | None:
    if value is None:
        return None
    return (float(value) - baseline) / baseline * 100.0


def summarize_coverage(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yml"
    coverage = {
        "config_path": str(config_path.resolve()) if config_path.exists() else None,
        "error": None,
        "max_num_iterations": None,
        "train_num_rays_per_batch": None,
        "train_num_images_to_sample_from": None,
        "train_num_times_to_repeat_images": None,
        "pixel_visit_x": None,
        "pixel_visit_x_pct_diff": None,
        "window_coverage_w": None,
        "window_coverage_w_pct_diff": None,
    }

    max_num_iterations = ACTIVE_TRAIN_SCHEDULE.max_num_iterations
    train_num_rays_per_batch = TRAIN_NUM_RAYS_PER_BATCH
    train_num_images_to_sample_from = TRAIN_NUM_IMAGES_TO_SAMPLE_FROM
    train_num_times_to_repeat_images = ACTIVE_TRAIN_SCHEDULE.train_num_times_to_repeat_images
    coverage.update(
        {
            "max_num_iterations": max_num_iterations,
            "train_num_rays_per_batch": train_num_rays_per_batch,
            "train_num_images_to_sample_from": train_num_images_to_sample_from,
            "train_num_times_to_repeat_images": train_num_times_to_repeat_images,
        }
    )

    pixel_visit_x = float(max_num_iterations) * float(train_num_rays_per_batch) / TRAIN_TOTAL_PIXELS
    window_coverage_w = (
        math.ceil(float(max_num_iterations) / float(train_num_times_to_repeat_images))
        * float(train_num_images_to_sample_from)
        / TRAIN_IMAGE_COUNT
    )
    coverage.update(
        {
            "pixel_visit_x": pixel_visit_x,
            "pixel_visit_x_pct_diff": pct_diff_from_baseline(pixel_visit_x, BASELINE_PIXEL_VISIT_X),
            "window_coverage_w": window_coverage_w,
            "window_coverage_w_pct_diff": pct_diff_from_baseline(window_coverage_w, BASELINE_WINDOW_COVERAGE_W),
        }
    )
    return coverage


def summarize_run(
    profile: Profile,
    run_name: str,
    run_dir: Path,
    log_path: Path,
    wall_time_s: float,
    returncode: int,
    baseline_summary_path: Path | None,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    events = load_event_scalars(run_dir)
    final_metrics = extract_final_metrics(events)
    if not final_metrics:
        raise RuntimeError(f"no Final Metrics/* scalars found under {run_dir}")

    timings = {key: last_scalar(events, tag) for key, tag in TIMING_TAGS.items()}
    selected_metrics = {key: last_scalar(events, tag) for key, tag in SELECTED_FINAL_METRICS.items()}
    train_total_time_s = timings.get("train_total_time_s")
    peak_gpu_mem_mb = max_scalar(events, "GPU Memory (MB)")
    quality_guardrail = compute_quality_guardrail(final_metrics, baseline_summary_path)

    return {
        "profile": profile.name,
        "status": "ok" if returncode == 0 else "fail",
        "returncode": returncode,
        "git_commit": git_commit(),
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "log_path": str(log_path.resolve()),
        "wall_time_s": wall_time_s,
        "startup_overhead_s": None if train_total_time_s is None else max(0.0, wall_time_s - train_total_time_s),
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "max_quality_regression_pct": quality_guardrail.get("max_regression_pct"),
        "coverage": coverage,
        "timings": timings,
        "selected_metrics": selected_metrics,
        "final_metrics": final_metrics,
        "quality_guardrail": quality_guardrail,
        "fixed_contract": {
            "data_path": str(DATA_PATH.resolve()),
            "smoke_max_iterations": SMOKE_MAX_ITERATIONS,
            "machine_seed": BENCHMARK_SEED,
            "pythonhashseed": BENCHMARK_SEED,
            "cuda_visible_devices": "1",
            "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
            "config_num_images_total": NUM_IMAGES_TOTAL,
            "config_train_split_fraction": TRAIN_SPLIT_FRACTION,
            "config_train_image_wh": list(TRAIN_IMAGE_WH),
            "train_image_count": TRAIN_IMAGE_COUNT,
            "train_total_pixels": TRAIN_TOTAL_PIXELS,
            "config_train_num_rays_per_batch": TRAIN_NUM_RAYS_PER_BATCH,
            "config_train_num_images_to_sample_from": TRAIN_NUM_IMAGES_TO_SAMPLE_FROM,
            "config_pixel_visitation": PIXEL_VISITATION,
            "config_window_coverage_w": WINDOW_COVERAGE,
            "baseline_reference_max_iterations": BASELINE_MAX_NUM_ITERATIONS,
            "baseline_reference_train_num_rays_per_batch": BASELINE_TRAIN_NUM_RAYS_PER_BATCH,
            "baseline_reference_train_num_images_to_sample_from": BASELINE_TRAIN_NUM_IMAGES_TO_SAMPLE_FROM,
            "baseline_reference_train_num_times_to_repeat_images": BASELINE_TRAIN_NUM_TIMES_TO_REPEAT_IMAGES,
            "baseline_reference_pixel_visit_x": BASELINE_PIXEL_VISIT_X,
            "baseline_reference_window_coverage_w": BASELINE_WINDOW_COVERAGE_W,
            "quality_comfort_pct": QUALITY_CLEAR_PCT,
            "quality_review_pct": QUALITY_REVIEW_PCT,
            "config_owned_hyperparameters": True,
        },
        **coverage,
        **timings,
        **selected_metrics,
    }


def format_results_tsv_value(column: str, value: Any) -> str:
    if value is None:
        return ""
    if column in RESULTS_TSV_NUMERIC_COLUMNS and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{RESULTS_TSV_NUMERIC_PRECISION}f}"
    return str(value)


def build_measure_results_row(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    timings = summary.get("timings", {})
    selected_metrics = summary.get("selected_metrics", {})
    coverage = summary.get("coverage", {})
    status = "crash" if summary.get("status") != "ok" else ""
    row = {
        "commit": summary.get("git_commit"),
        "profile": summary.get("profile"),
        "wall_time_s": summary.get("wall_time_s"),
        "train_total_time_s": timings.get("train_total_time_s"),
        "startup_overhead_s": summary.get("startup_overhead_s"),
        "train_iter_time_s": timings.get("train_iter_time_s"),
        "train_rays_per_sec": timings.get("train_rays_per_sec"),
        "train_batch_load_s": timings.get("train_batch_load_s"),
        "train_feature_cache_load_s": timings.get("train_feature_cache_load_s"),
        "train_feature_gather_s": timings.get("train_feature_gather_s"),
        "train_feature_populate_s": timings.get("train_feature_populate_s"),
        "train_model_forward_s": timings.get("train_model_forward_s"),
        "train_metrics_loss_s": timings.get("train_metrics_loss_s"),
        "train_image_viz_s": timings.get("train_image_viz_s"),
        "eval_batch_load_s": timings.get("eval_batch_load_s"),
        "eval_feature_cache_load_s": timings.get("eval_feature_cache_load_s"),
        "eval_feature_gather_s": timings.get("eval_feature_gather_s"),
        "eval_feature_populate_s": timings.get("eval_feature_populate_s"),
        "eval_image_s": timings.get("eval_image_s"),
        "eval_all_images_s": timings.get("eval_all_images_s"),
        "peak_gpu_mem_mb": summary.get("peak_gpu_mem_mb"),
        "pixel_visit_x": coverage.get("pixel_visit_x"),
        "pixel_visit_x_pct_diff": coverage.get("pixel_visit_x_pct_diff"),
        "window_coverage_w": coverage.get("window_coverage_w"),
        "window_coverage_w_pct_diff": coverage.get("window_coverage_w_pct_diff"),
        "train_psnr": selected_metrics.get("train_psnr"),
        "train_feature_error": selected_metrics.get("train_feature_error"),
        "train_foreground_acc": selected_metrics.get("train_foreground_acc"),
        "eval_all_psnr": selected_metrics.get("eval_all_psnr"),
        "eval_all_ssim": selected_metrics.get("eval_all_ssim"),
        "eval_all_lpips": selected_metrics.get("eval_all_lpips"),
        "max_quality_regression_pct": summary.get("max_quality_regression_pct"),
        "status": status,
        "description": "",
        "summary_path": str(summary_path.resolve()),
    }
    return row


def write_results_row_tsv(path: Path, row: dict[str, Any]) -> None:
    header = "\t".join(RESULTS_TSV_COLUMNS)
    values = "\t".join(format_results_tsv_value(column, row.get(column)) for column in RESULTS_TSV_COLUMNS)
    path.write_text(header + "\n" + values + "\n", encoding="utf-8")


def print_summary(summary: dict[str, Any], summary_path: Path) -> None:
    timings = summary.get("timings", {})
    selected_metrics = summary.get("selected_metrics", {})
    compact = {
        "profile": summary["profile"],
        "status": summary["status"],
        "returncode": summary["returncode"],
        "git_commit": summary["git_commit"],
        "wall_time_s": summary["wall_time_s"],
        "train_total_time_s": timings.get("train_total_time_s"),
        "startup_overhead_s": summary["startup_overhead_s"],
        "train_iter_time_s": timings.get("train_iter_time_s"),
        "train_rays_per_sec": timings.get("train_rays_per_sec"),
        "peak_gpu_mem_mb": summary["peak_gpu_mem_mb"],
        "eval_all_psnr": selected_metrics.get("eval_all_psnr"),
        "eval_all_ssim": selected_metrics.get("eval_all_ssim"),
        "eval_all_lpips": selected_metrics.get("eval_all_lpips"),
        "max_quality_regression_pct": summary.get("max_quality_regression_pct"),
        "quality_tradeoff_band": summary.get("quality_guardrail", {}).get("band"),
        "quality_guardrail_passed": summary.get("quality_guardrail", {}).get("passed"),
        "summary_path": str(summary_path.resolve()),
    }
    print("=== BENCHMARK SUMMARY BEGIN ===")
    print(json.dumps(compact, sort_keys=True))
    print("=== BENCHMARK SUMMARY END ===")


def main() -> None:
    args = parse_args()
    ensure_runtime()

    profile = PROFILES[args.profile]
    run_name = run_id(profile)
    benchmark_run_dir = BENCHMARK_RUNS_ROOT / profile.name / run_name
    run_dir = BENCHMARK_OUTPUTS_ROOT / EXPERIMENT_NAME / METHOD_NAME / run_name
    log_path = benchmark_run_dir / "train.log"
    summary_path = benchmark_run_dir / "summary.json"
    benchmark_run_dir.mkdir(parents=True, exist_ok=False)

    command = build_command(profile, run_name)
    (benchmark_run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")

    if args.dry_run:
        payload = {
            "profile": profile.name,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "command": command,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    wall_start = perf_counter()
    returncode = stream_subprocess(command, log_path)
    wall_time_s = perf_counter() - wall_start
    coverage = summarize_coverage(run_dir)

    try:
        summary = summarize_run(
            profile=profile,
            run_name=run_name,
            run_dir=run_dir,
            log_path=log_path,
            wall_time_s=wall_time_s,
            returncode=returncode,
            baseline_summary_path=args.baseline_summary,
            coverage=coverage,
        )
    except Exception as exc:
        summary = {
            "profile": profile.name,
            "status": "fail",
            "returncode": returncode,
            "git_commit": git_commit(),
            "run_name": run_name,
            "run_dir": str(run_dir.resolve()),
            "log_path": str(log_path.resolve()),
            "wall_time_s": wall_time_s,
            "startup_overhead_s": None,
            "peak_gpu_mem_mb": None,
            "max_quality_regression_pct": None,
            "coverage": coverage,
            "timings": {},
            "selected_metrics": {},
            "quality_guardrail": {},
            "error": str(exc),
            **coverage,
        }

    if profile.name == "measure":
        results_row = build_measure_results_row(summary, summary_path)
        results_row_path = benchmark_run_dir / RESULTS_TSV_FILENAME
        write_results_row_tsv(results_row_path, results_row)
        summary["results_tsv"] = {
            "columns": list(RESULTS_TSV_COLUMNS),
            "numeric_precision_dp": RESULTS_TSV_NUMERIC_PRECISION,
            "path": str(results_row_path.resolve()),
            "row": results_row,
        }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(summary, summary_path)

    if summary.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
