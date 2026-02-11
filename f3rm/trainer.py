"""
Custom F3RM Trainer.
"""

import functools
import math
import os
from dataclasses import dataclass, field
from typing import Type

import torch
from torch.cuda.amp import GradScaler

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import profiler, writer


@dataclass
class F3RMTrainerConfig(TrainerConfig):
    """F3RM Trainer Config."""

    _target: Type = field(default_factory=lambda: F3RMTrainer)
    deterministic_mode: bool = True
    deterministic_warn_only: bool = False
    amp_min_scale: float = 256.0
    amp_reset_window: int = 2000
    amp_disable_after_resets: int = 3
    amp_disable_on_instability: bool = True


class F3RMTrainer(Trainer):
    """F3RM Trainer."""

    config: F3RMTrainerConfig

    def __init__(self, config: F3RMTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        """Initialize F3RM trainer with deterministic-or-fast backend settings."""
        model_impl = getattr(getattr(config.pipeline, "model", None), "implementation", None)
        strict_determinism = config.deterministic_mode and model_impl != "tcnn"

        if strict_determinism:
            # Required by deterministic cuBLAS kernels for some CUDA paths.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            config.mixed_precision = False
            config.use_grad_scaler = False

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = not strict_determinism
            torch.backends.cudnn.allow_tf32 = not strict_determinism
            torch.backends.cudnn.benchmark = not strict_determinism
            torch.backends.cudnn.deterministic = strict_determinism

        torch.set_float32_matmul_precision("highest" if strict_determinism else "high")
        if strict_determinism:
            torch.use_deterministic_algorithms(True, warn_only=config.deterministic_warn_only)
        else:
            torch.use_deterministic_algorithms(False)

        super().__init__(config, local_rank, world_size)
        self._grad_scaler_init_scale = float(self.grad_scaler.get_scale())
        self._grad_scaler_min_scale = float(config.amp_min_scale)
        self._grad_scaler_reset_steps: list[int] = []
        self._amp_disabled_due_to_instability = False

    def _amp_guardrails(self, step: int, *, loss_finite: bool = True, check_scale: bool = False) -> None:
        """Single utility for AMP/scaler instability handling."""
        if not self.use_grad_scaler:
            return

        if not loss_finite:
            if self.config.amp_disable_on_instability and not self._amp_disabled_due_to_instability:
                self.mixed_precision = False
                self.use_grad_scaler = False
                self.grad_scaler = GradScaler(enabled=False)
                self._amp_disabled_due_to_instability = True
                writer.put_scalar(name="train/amp_disabled_due_to_instability", scalar=1.0, step=step)
            else:
                self.grad_scaler = GradScaler(enabled=True, init_scale=self._grad_scaler_init_scale)
                writer.put_scalar(name="train/grad_scaler_reset", scalar=1.0, step=step)
            return

        if not check_scale:
            return

        scale = float(self.grad_scaler.get_scale())
        if math.isfinite(scale) and scale >= self._grad_scaler_min_scale:
            return

        self.grad_scaler = GradScaler(enabled=True, init_scale=self._grad_scaler_init_scale)
        writer.put_scalar(name="train/grad_scaler_reset", scalar=1.0, step=step)
        if not self._grad_scaler_reset_steps or self._grad_scaler_reset_steps[-1] != step:
            self._grad_scaler_reset_steps.append(step)
        window_start = step - int(self.config.amp_reset_window)
        self._grad_scaler_reset_steps = [s for s in self._grad_scaler_reset_steps if s >= window_start]
        writer.put_scalar(
            name="train/grad_scaler_resets_in_window",
            scalar=float(len(self._grad_scaler_reset_steps)),
            step=step,
        )
        if self.config.amp_disable_on_instability and len(self._grad_scaler_reset_steps) >= self.config.amp_disable_after_resets:
            self.mixed_precision = False
            self.use_grad_scaler = False
            self.grad_scaler = GradScaler(enabled=False)
            self._amp_disabled_due_to_instability = True
            writer.put_scalar(name="train/amp_disabled_due_to_instability", scalar=1.0, step=step)

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

    @staticmethod
    def _has_nonfinite_gradients(parameters) -> bool:
        for param in parameters:
            grad = getattr(param, "grad", None)
            if grad is not None and not torch.isfinite(grad).all():
                return True
        return False

    @staticmethod
    def _sanitize_nonfinite_gradients(parameters) -> bool:
        did_sanitize = False
        for param in parameters:
            grad = getattr(param, "grad", None)
            if grad is not None and not torch.isfinite(grad).all():
                with torch.no_grad():
                    torch.nan_to_num_(grad, nan=0.0, posinf=0.0, neginf=0.0)
                did_sanitize = True
        return did_sanitize

    def _sanitize_camera_opt_params_if_needed(self, step: int) -> None:
        """Prevent a non-finite camera optimizer state from poisoning subsequent steps."""
        camera_optimizer = getattr(self.pipeline.model, "camera_optimizer", None)
        pose_adjustment = getattr(camera_optimizer, "pose_adjustment", None)
        if pose_adjustment is None:
            return
        if torch.isfinite(pose_adjustment).all():
            return
        with torch.no_grad():
            torch.nan_to_num_(pose_adjustment, nan=0.0, posinf=0.0, neginf=0.0)
        writer.put_scalar(name="train/camera_opt_params_sanitized", scalar=1.0, step=step)

    def _sanitize_all_optimizer_groups(self, step: int) -> None:
        """Sanitize params and optimizer states across all groups after a non-finite forward."""
        for group in self.optimizers.parameters.keys():
            params = list(self.optimizers.parameters[group])
            if self._sanitize_parameter_tensors(params):
                writer.put_scalar(name=f"train/nonfinite_param_sanitized/{group}", scalar=1.0, step=step)
            if self._sanitize_nonfinite_gradients(params):
                writer.put_scalar(name=f"train/nonfinite_grad_sanitized/{group}", scalar=1.0, step=step)
            optimizer = self.optimizers.optimizers[group]
            if self._sanitize_optimizer_state(optimizer):
                writer.put_scalar(name=f"train/nonfinite_opt_state_sanitized/{group}", scalar=1.0, step=step)

    @profiler.time_function
    def train_iteration(self, step: int):
        """Run one training iteration with scaler and camera-opt safety guards."""
        self._amp_guardrails(step, check_scale=True)
        self._sanitize_camera_opt_params_if_needed(step)

        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)

        cpu_or_cuda_str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())

        if not torch.isfinite(loss):
            writer.put_scalar(name="train/nonfinite_loss_skip", scalar=1.0, step=step)
            self._sanitize_all_optimizer_groups(step)
            self._amp_guardrails(step, loss_finite=False)
            writer.put_scalar(name="current_step", scalar=float(step), step=step)
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

        for group in needs_step:
            optimizer = self.optimizers.optimizers[group]
            max_norm = self.optimizers.config[group]["optimizer"].max_norm
            params = list(self.optimizers.parameters[group])

            if self._sanitize_parameter_tensors(params):
                writer.put_scalar(name=f"train/nonfinite_param_sanitized/{group}", scalar=1.0, step=step)
            if self._sanitize_optimizer_state(optimizer):
                writer.put_scalar(name=f"train/nonfinite_opt_state_sanitized/{group}", scalar=1.0, step=step)

            has_grad = any(any(param.grad is not None for param in pg["params"]) for pg in optimizer.param_groups)
            if not has_grad:
                continue

            if self.use_grad_scaler:
                # Non-finite checks must run on unscaled grads, otherwise large scaler values create false positives.
                self.grad_scaler.unscale_(optimizer)

            if self._has_nonfinite_gradients(params):
                self._sanitize_nonfinite_gradients(params)
                writer.put_scalar(name=f"train/nonfinite_grad_skip/{group}", scalar=1.0, step=step)
                continue

            if self.use_grad_scaler:
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, max_norm)
                self.grad_scaler.step(optimizer)
            else:
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, max_norm)
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
            scale = float(self.grad_scaler.get_scale())
            self.grad_scaler.update()
            new_scale = float(self.grad_scaler.get_scale())
            writer.put_scalar(name="train/grad_scaler_scale", scalar=new_scale, step=step)
            if scale <= new_scale:
                self.optimizers.scheduler_step_all(step)
        else:
            self.optimizers.scheduler_step_all(step)
        self._amp_guardrails(step, check_scale=True)

        writer.put_scalar(name="current_step", scalar=float(step), step=step)
        return loss, loss_dict, metrics_dict
