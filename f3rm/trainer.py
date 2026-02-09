"""
Custom F3RM Trainer.
"""

from dataclasses import dataclass, field
from typing import Type

import torch

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import writer


@dataclass
class F3RMTrainerConfig(TrainerConfig):
    """F3RM Trainer Config."""

    _target: Type = field(default_factory=lambda: F3RMTrainer)


class F3RMTrainer(Trainer):
    """F3RM Trainer."""

    config: F3RMTrainerConfig

    def __init__(self, config: F3RMTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        """Initialize F3RM trainer with CUDA matmul perf settings."""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        super().__init__(config, local_rank, world_size)

    def train_iteration(self, step: int):
        """Run one training iteration and explicitly log the current global step."""
        loss, loss_dict, metrics_dict = super().train_iteration(step)
        writer.put_scalar(name="current_step", scalar=float(step), step=step)
        return loss, loss_dict, metrics_dict
