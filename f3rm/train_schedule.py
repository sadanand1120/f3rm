import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DerivedTrainSchedule:
    max_num_iterations: int
    steps_per_eval_all_images: int
    train_num_times_to_repeat_images: int


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int_tuple(name: str, default: tuple[int, int]) -> tuple[int, int]:
    value = os.getenv(name)
    return default if value is None else tuple(int(part.strip()) for part in value.split(",", maxsplit=1))


def _required_iterations(total_pixels: int, rays_per_step: int, pixel_visitation: float) -> int:
    if pixel_visitation <= 0.0:
        return 1
    if rays_per_step <= 0:
        raise ValueError("rays_per_step must be positive when pixel visitation is enabled")
    return max(1, math.ceil(total_pixels * pixel_visitation / rays_per_step))


def derive_train_schedule(
    num_images_total: int,
    train_split_fraction: float,
    train_image_wh: tuple[int, int],
    num_devices: int,
    train_num_rays_per_batch: int,
    train_num_images_to_sample_from: int,
    pixel_visitation: float,
    window_coverage: float,
) -> DerivedTrainSchedule:
    image_width, image_height = train_image_wh
    train_image_count = max(1, math.ceil(num_images_total * train_split_fraction))
    train_total_pixels = train_image_count * image_width * image_height

    global_train_num_rays_per_step = num_devices * train_num_rays_per_batch
    max_num_iterations = _required_iterations(train_total_pixels, global_train_num_rays_per_step, pixel_visitation)
    target_windows = max(1, round(window_coverage * train_image_count / train_num_images_to_sample_from))
    train_num_times_to_repeat_images = max(1, math.ceil((num_devices * max_num_iterations) / target_windows))
    return DerivedTrainSchedule(
        max_num_iterations=max_num_iterations,
        steps_per_eval_all_images=max(0, max_num_iterations - 1),
        train_num_times_to_repeat_images=train_num_times_to_repeat_images,
    )
