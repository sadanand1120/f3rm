"""
Custom F3RM Trainer with comprehensive seeding support.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Type

import torch
import wandb
from rich import box, style
from rich.panel import Panel
from rich.table import Table

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import writer
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.utils.misc import step_check

from f3rm.utils.seeding import (
    print_reproducibility_info,
    set_seed_comprehensive,
    setup_distributed_seeding,
    validate_reproducibility_setup,
)


@dataclass
class F3RMTrainerConfig(TrainerConfig):
    """F3RM Trainer Config with enhanced seeding support."""

    _target: Type = field(default_factory=lambda: F3RMTrainer)

    # Seeding parameters
    enable_comprehensive_seeding: bool = True
    """Whether to enable comprehensive seeding for reproducibility."""

    seed_deterministic_algorithms: bool = True
    """Whether to use deterministic algorithms (may impact performance)."""

    seed_warn_only: bool = False
    """If True, only warn about non-deterministic operations instead of erroring."""

    seed_cublas_workspace: bool = True
    """Whether to configure cuBLAS workspace for deterministic behavior."""

    print_seed_info: bool = True
    """Whether to print detailed seeding information at startup."""


class F3RMTrainer(Trainer):
    """F3RM Trainer with comprehensive seeding support."""

    config: F3RMTrainerConfig

    def __init__(self, config: F3RMTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        """Initialize F3RM trainer with seeding setup."""

        # Setup seeding before calling parent constructor
        if config.enable_comprehensive_seeding:
            self._setup_seeding(config, local_rank, world_size)

        # Call parent constructor
        super().__init__(config, local_rank, world_size)

        # Print reproducibility information if requested
        if config.print_seed_info:
            print_reproducibility_info()
            validate_reproducibility_setup()

    def _setup_seeding(self, config: F3RMTrainerConfig, local_rank: int, world_size: int) -> None:
        """Setup comprehensive seeding for reproducible training."""

        CONSOLE.print("[bold green]🌱 Setting up F3RM comprehensive seeding...")

        if world_size > 1:
            # Multi-GPU/distributed setup
            CONSOLE.print(f"[yellow]🌍 Distributed training detected: rank {local_rank}/{world_size}")
            setup_distributed_seeding(
                rank=local_rank,
                world_size=world_size,
                base_seed=config.machine.seed
            )
        else:
            # Single GPU setup
            set_seed_comprehensive(
                seed=config.machine.seed,
                deterministic_algorithms=config.seed_deterministic_algorithms,
                warn_only=config.seed_warn_only,
                cublas_workspace_config=config.seed_cublas_workspace,
            )

        # Print seeding information
        CONSOLE.print(f"[green]✅ F3RM seeding completed with base seed: {config.machine.seed}")

        # Warn about performance implications
        if config.seed_deterministic_algorithms and not config.seed_warn_only:
            CONSOLE.print("[yellow]⚠️  Deterministic algorithms enabled: training may be slower but more reproducible")

    def setup(self, test_mode: str = "test") -> None:
        """Setup with additional seeding validation."""

        # Call parent setup
        super().setup(test_mode)

        # Additional validation after setup
        if self.config.enable_comprehensive_seeding and self.config.print_seed_info:
            CONSOLE.print("[bold blue]🔍 Post-setup seeding validation:")
            validate_reproducibility_setup()

    def train(self) -> None:
        """Train the model."""

        if self.config.enable_comprehensive_seeding:
            CONSOLE.print("[bold green]🚀 Starting F3RM training with comprehensive seeding enabled")
            CONSOLE.print(f"[blue]📍 Training will be reproducible with seed: {self.config.machine.seed}")

        # tweaked super().train() to add step count logging
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        # don't want to call save_dataparser_transform if pipeline's datamanager does not have a dataparser
        if isinstance(self.pipeline.datamanager, VanillaDataManager):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            step = 0
            for step in range(self._start_step, self._start_step + num_iterations):
                while self.training_state == "paused":
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Emit current step both as scalar metric (shows in Runs table) and summary fallback
                writer.put_scalar(name="current_step", scalar=int(step), step=step)
                if self.config.is_wandb_enabled() and wandb.run is not None:
                    wandb.run.summary["current_step"] = int(step)

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()
