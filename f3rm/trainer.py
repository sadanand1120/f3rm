"""
Custom F3RM Trainer with comprehensive seeding support.
"""

import os
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils.rich_utils import CONSOLE

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
        """Train with seeding information."""

        if self.config.enable_comprehensive_seeding:
            CONSOLE.print("[bold green]🚀 Starting F3RM training with comprehensive seeding enabled")
            CONSOLE.print(f"[blue]📍 Training will be reproducible with seed: {self.config.machine.seed}")

        # Call parent train method
        super().train()
