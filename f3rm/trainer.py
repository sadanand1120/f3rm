"""
Custom F3RM Trainer.
"""

import functools
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Type

import torch
from rich.panel import Panel
from rich.table import Table

from nerfstudio.utils.decorators import check_eval_enabled


def _explicit_vis_mode(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg.startswith("--vis="):
            return arg.split("=", 1)[1]
        if arg == "--vis" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _install_nonviewer_import_stubs() -> None:
    vis_mode = _explicit_vis_mode(sys.argv[1:])
    if vis_mode not in {"wandb", "tensorboard", "comet"}:
        return

    if "nerfstudio.viewer.viewer" not in sys.modules:
        import nerfstudio.viewer as viewer_pkg

        viewer_module = types.ModuleType("nerfstudio.viewer.viewer")

        class Viewer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Viewer should not be instantiated when --vis disables viewer modes.")

        viewer_module.Viewer = Viewer
        sys.modules["nerfstudio.viewer.viewer"] = viewer_module
        viewer_pkg.viewer = viewer_module

    if "nerfstudio.viewer_legacy.server.viewer_state" not in sys.modules:
        import nerfstudio.viewer_legacy.server as legacy_pkg

        legacy_module = types.ModuleType("nerfstudio.viewer_legacy.server.viewer_state")

        class ViewerLegacyState:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Legacy viewer should not be instantiated when --vis disables viewer modes.")

        legacy_module.ViewerLegacyState = ViewerLegacyState
        sys.modules["nerfstudio.viewer_legacy.server.viewer_state"] = legacy_module
        legacy_pkg.viewer_state = legacy_module


_install_nonviewer_import_stubs()

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE


FINAL_METRIC_METADATA_KEYS = {"image_idx", "num_rays"}


@dataclass
class F3RMTrainerConfig(TrainerConfig):
    """F3RM Trainer Config."""

    _target: Type = field(default_factory=lambda: F3RMTrainer)


class F3RMTrainer(Trainer):
    """F3RM Trainer."""

    config: F3RMTrainerConfig

    def __init__(self, config: F3RMTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        torch.use_deterministic_algorithms(False)

        super().__init__(config, local_rank, world_size)
        self._latest_train_metrics: Dict[str, Any] = {}
        self._latest_eval_batch_metrics: Dict[str, Any] = {}
        self._latest_eval_image_metrics: Dict[str, Any] = {}
        self._latest_eval_all_metrics: Dict[str, Any] = {}

    @staticmethod
    def _extract_scalar_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, float]:
        scalar_metrics: Dict[str, float] = {}
        for key, value in metrics_dict.items():
            if key in FINAL_METRIC_METADATA_KEYS:
                continue
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                scalar_metrics[key] = float(value.detach().cpu())
                continue
            try:
                scalar_metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        return scalar_metrics

    @staticmethod
    def _detach_scalar_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        scalar_metrics: Dict[str, Any] = {}
        for key, value in metrics_dict.items():
            if key in FINAL_METRIC_METADATA_KEYS:
                continue
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                scalar_metrics[key] = value.detach()
                continue
            try:
                scalar_metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        return scalar_metrics

    @staticmethod
    def _sanitize_optimizer_state(optimizer: torch.optim.Optimizer) -> bool:
        did_sanitize = False
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    torch.nan_to_num_(value, nan=0.0, posinf=0.0, neginf=0.0)
                    did_sanitize = True
        return did_sanitize

    @staticmethod
    def _sanitize_parameter_tensors(parameters) -> bool:
        did_sanitize = False
        for param in parameters:
            if torch.is_tensor(param) and not torch.isfinite(param).all():
                with torch.no_grad():
                    torch.nan_to_num_(param, nan=0.0, posinf=0.0, neginf=0.0)
                did_sanitize = True
        return did_sanitize

    def _sanitize_camera_opt_params_if_needed(self) -> None:
        camera_optimizer = getattr(self.pipeline.model, "camera_optimizer", None)
        pose_adjustment = getattr(camera_optimizer, "pose_adjustment", None)
        if pose_adjustment is None or torch.isfinite(pose_adjustment).all():
            return
        with torch.no_grad():
            torch.nan_to_num_(pose_adjustment, nan=0.0, posinf=0.0, neginf=0.0)

    def _sanitize_all_optimizer_groups(self) -> None:
        for group in self.optimizers.parameters.keys():
            params = list(self.optimizers.parameters[group])
            self._sanitize_parameter_tensors(params)
            optimizer = self.optimizers.optimizers[group]
            self._sanitize_optimizer_state(optimizer)

    @profiler.time_function
    def train_iteration(self, step: int):
        self._sanitize_camera_opt_params_if_needed()

        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)

        cpu_or_cuda_str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        self._latest_train_metrics = self._detach_scalar_metrics(metrics_dict)
        if not torch.isfinite(loss):
            self._sanitize_all_optimizer_groups()
            return loss, loss_dict, metrics_dict

        if self.use_grad_scaler:
            self.grad_scaler.scale(loss).backward()  # type: ignore
        else:
            loss.backward()

        needs_step = [
            group
            for group in self.optimizers.parameters.keys()
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]

        if self.use_grad_scaler:
            self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)
        else:
            for group in needs_step:
                optimizer = self.optimizers.optimizers[group]
                max_norm = self.optimizers.config[group]["optimizer"].max_norm
                params = self.optimizers.parameters[group]
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, max_norm)
                if any(any(param.grad is not None for param in pg["params"]) for pg in optimizer.param_groups):
                    optimizer.step()

        if self.config.log_gradients:
            total_grad = 0.0
            for tag, value in self.pipeline.model.named_parameters():
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad
            metrics_dict["Gradients/Total"] = total_grad  # type: ignore

        if self.use_grad_scaler:
            scale = self.grad_scaler.get_scale()
            self.grad_scaler.update()
            if scale <= self.grad_scaler.get_scale():
                self.optimizers.scheduler_step_all(step)
        else:
            self.optimizers.scheduler_step_all(step)
        return loss, loss_dict, metrics_dict

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            self._latest_eval_batch_metrics = self._detach_scalar_metrics(eval_metrics_dict)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        if step_check(step, self.config.steps_per_eval_image):
            with writer.TimeWriter(writer, writer.EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            self._latest_eval_image_metrics = self._detach_scalar_metrics(metrics_dict)
            writer.put_time(
                name=writer.EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            for image_name, image in images_dict.items():
                writer.put_image(name="Eval Images/" + image_name, image=image, step=step)

        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            self._latest_eval_all_metrics = self._detach_scalar_metrics(metrics_dict)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)

    def _get_final_metric_groups(self) -> list[tuple[str, Dict[str, float]]]:
        metric_groups = [
            ("Train Batch", self._extract_scalar_metrics(self._latest_train_metrics)),
            ("Eval Batch", self._extract_scalar_metrics(self._latest_eval_batch_metrics)),
            ("Eval Image", self._extract_scalar_metrics(self._latest_eval_image_metrics)),
            ("Eval All Images", self._extract_scalar_metrics(self._latest_eval_all_metrics)),
        ]
        return [(group_name, metrics) for group_name, metrics in metric_groups if metrics]

    def _queue_final_metrics(self, metric_groups: list[tuple[str, Dict[str, float]]]) -> None:
        for group_name, metrics in metric_groups:
            for metric_name, metric_value in sorted(metrics.items()):
                writer.put_scalar(name=f"Final Metrics/{group_name}/{metric_name}", scalar=metric_value, step=self.step)

    def _print_final_metrics(self, metric_groups: list[tuple[str, Dict[str, float]]]) -> None:
        if not metric_groups:
            return

        table = Table(show_header=True)
        table.add_column("Group")
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        for group_name, metrics in metric_groups:
            for metric_name, metric_value in sorted(metrics.items()):
                table.add_row(group_name, metric_name, f"{metric_value:.6f}")

        CONSOLE.print(Panel(table, title="[bold]Final Unweighted Metrics[/bold]", expand=False))

    def _after_train(self) -> None:
        metric_groups = self._get_final_metric_groups()
        self._queue_final_metrics(metric_groups)
        super()._after_train()
        self._print_final_metrics(metric_groups)
