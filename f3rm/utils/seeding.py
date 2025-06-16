"""
Comprehensive seeding utilities for reproducible F3RM training.

This module provides functions to set random seeds for:
- PyTorch (including CUDA)
- NumPy
- Python's random module
- Hash-based randomness

Handles multi-GPU and multi-process scenarios properly.
"""

import os
import random
import warnings
from typing import Optional

import numpy as np
import torch


def set_seed_comprehensive(
    seed: int = 42,
    deterministic_algorithms: bool = True,
    warn_only: bool = False,
    cublas_workspace_config: bool = True
) -> None:
    """
    Set random seeds for comprehensive reproducibility across all libraries.

    This function handles:
    - PyTorch (CPU and CUDA)
    - NumPy
    - Python random
    - Environment variables for deterministic behavior
    - CUDA deterministic settings

    Args:
        seed: Random seed to use across all libraries
        deterministic_algorithms: If True, use deterministic algorithms where possible
        warn_only: If True, only warn about non-deterministic operations instead of erroring
        cublas_workspace_config: If True, configure cuBLAS workspace for deterministic behavior
    """
    print(f"🌱 Setting comprehensive random seed: {seed}")

    # Set Python's random seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure deterministic behavior
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

        # Configure cuBLAS for deterministic behavior (may impact performance)
        if cublas_workspace_config and torch.cuda.is_available():
            # Set cuBLAS workspace config for deterministic behavior
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Set additional environment variables for deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Configure CUDA for reproducibility
    if torch.cuda.is_available():
        # These settings improve reproducibility but may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For better performance in some cases, you might want:
        # torch.backends.cudnn.benchmark = True  # Can improve performance but reduces reproducibility

    print(f"✅ Comprehensive seeding completed with seed: {seed}")
    if deterministic_algorithms:
        print("🔒 Deterministic algorithms enabled")
        if not warn_only:
            print("⚠️  Training may be slower due to deterministic operations")


def set_worker_seed(worker_id: int = 0, base_seed: int = 42) -> None:
    """
    Set seeds for DataLoader workers in multi-processing scenarios.

    This should be used as the worker_init_fn in DataLoader for reproducible
    data loading across multiple worker processes.

    Args:
        worker_id: Worker process ID (automatically passed by DataLoader)
        base_seed: Base seed to derive worker-specific seeds from
    """
    # Create unique seed for each worker based on worker_id
    worker_seed = base_seed + worker_id

    # Set seeds for this worker
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # Note: We don't set CUDA seeds here as they're global and handled by the main process


def get_seeded_dataloader_kwargs(seed: int = 42) -> dict:
    """
    Get DataLoader kwargs for reproducible data loading.

    Args:
        seed: Base seed for worker initialization

    Returns:
        Dictionary of DataLoader kwargs for reproducible behavior
    """
    return {
        "worker_init_fn": lambda worker_id: set_worker_seed(worker_id, seed),
        "generator": torch.Generator().manual_seed(seed),
    }


def seed_hash_functions(seed: int = 42) -> None:
    """
    Seed hash-based random number generators used in neural networks.

    This is particularly important for hash-based encodings like HashGrid
    which are commonly used in NeRF models.

    Args:
        seed: Seed for hash functions
    """
    # This affects tcnn (tiny-cuda-nn) hash grid initialization if used
    os.environ["TCNN_SEED"] = str(seed)


def print_reproducibility_info() -> None:
    """Print information about current reproducibility settings."""
    print("\n🔍 Reproducibility Status:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN deterministic: {torch.backends.cudnn.deterministic}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    # Check if deterministic algorithms are enabled
    try:
        deterministic = torch.are_deterministic_algorithms_enabled()
        print(f"  Deterministic algorithms: {deterministic}")
    except:
        print("  Deterministic algorithms: Unknown (PyTorch version may not support)")

    # Check environment variables
    env_vars = ["PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "TCNN_SEED"]
    print("  Environment variables:")
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"    {var}: {value}")


def validate_reproducibility_setup() -> None:
    """
    Validate that the environment is properly configured for reproducibility.
    Raises warnings if potential issues are detected.
    """
    issues = []

    # Check if cuDNN benchmark is enabled (can cause non-deterministic behavior)
    if torch.cuda.is_available() and torch.backends.cudnn.benchmark:
        issues.append("cuDNN benchmark is enabled - this may cause non-deterministic behavior")

    # Check if PYTHONHASHSEED is set
    if "PYTHONHASHSEED" not in os.environ:
        issues.append("PYTHONHASHSEED environment variable is not set")

    # Check if deterministic algorithms are enabled
    try:
        if not torch.are_deterministic_algorithms_enabled():
            issues.append("PyTorch deterministic algorithms are not enabled")
    except:
        pass  # Feature not available in older PyTorch versions

    if issues:
        warning_msg = "⚠️  Potential reproducibility issues detected:\n" + "\n".join(f"  - {issue}" for issue in issues)
        warnings.warn(warning_msg, UserWarning)
    else:
        print("✅ Reproducibility setup looks good!")


# Multi-GPU specific considerations
def setup_distributed_seeding(rank: int, world_size: int, base_seed: int = 42) -> None:
    """
    Setup seeding for distributed (multi-GPU) training.

    Args:
        rank: Process rank in distributed setup
        world_size: Total number of processes
        base_seed: Base seed to derive rank-specific seeds from
    """
    # Create unique seed for each process
    process_seed = base_seed + rank

    print(f"🌍 Setting up distributed seeding for rank {rank}/{world_size} with seed {process_seed}")

    # Set comprehensive seed for this process
    set_seed_comprehensive(
        seed=process_seed,
        deterministic_algorithms=True,
        warn_only=True  # Use warn_only in distributed settings to avoid crashes
    )

    # Additional hash function seeding with rank-specific seed
    seed_hash_functions(process_seed)
