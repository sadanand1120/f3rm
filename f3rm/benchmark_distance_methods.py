#!/usr/bin/env python3
"""
Distance Calculation Benchmarking Script

This script benchmarks different distance calculation methods for pointcloud processing:
1. scipy.cdist (CPU only)
2. torch.cdist (CPU/GPU)
3. Simple euclidean distance (CPU/GPU)
4. Dot product similarity (CPU/GPU)
5. 5D dot trick (precomputed) (CPU/GPU)
6. 5D dot trick (on-the-fly) (CPU/GPU)

Tests both 3D coordinates and high-dimensional feature vectors.

Usage:
    python benchmark_distance_methods.py --num-main 10000 --num-floor 50000 --feature-dim 512
"""

import argparse
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from contextlib import contextmanager

console = Console()


# --- Method enable/disable switches ---
METHODS_ENABLED = {
    "scipy_cdist": True,
    "torch_cdist": True,
    "simple_euclidean": True,
    "dot_product": True,
    "5d_dot_precomputed": True,
    "5d_dot_onthefly": True,
}
# --------------------------------------


def generate_test_data(
    num_main: int,
    num_floor: int,
    dim_3d: int = 3,
    dim_high: int = 512,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Generate test data for benchmarking."""
    console.print(f"[yellow]Generating test data...")
    console.print(f"  Main points: {num_main:,}")
    console.print(f"  Floor points: {num_floor:,}")
    console.print(f"  3D dim: {dim_3d}, High dim: {dim_high}")
    console.print(f"  Device: {device}")

    data = {}

    # Generate 3D coordinates (realistic room coordinates)
    main_3d = torch.randn(num_main, dim_3d) * 0.5  # Smaller spread for main points
    floor_3d = torch.randn(num_floor, dim_3d) * 1.0  # Larger spread for floor

    # Generate high-dimensional features (normalized like real features)
    main_high = torch.randn(num_main, dim_high)
    main_high = torch.nn.functional.normalize(main_high, dim=1)

    floor_high = torch.randn(num_floor, dim_high)
    floor_high = torch.nn.functional.normalize(floor_high, dim=1)

    # Store data for both CPU and GPU
    data["main_3d_cpu"] = main_3d
    data["floor_3d_cpu"] = floor_3d
    data["main_high_cpu"] = main_high
    data["floor_high_cpu"] = floor_high

    if device == "cuda" and torch.cuda.is_available():
        data["main_3d_gpu"] = main_3d.cuda()
        data["floor_3d_gpu"] = floor_3d.cuda()
        data["main_high_gpu"] = main_high.cuda()
        data["floor_high_gpu"] = floor_high.cuda()

    return data


def precompute_5d_representations(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute 5D representations for the dot trick.

    For a 3D point p=(x,y,z):
    f(p) = [x, y, z, ‖p‖², 1]
    g(p) = [-2x, -2y, -2z, 1, ‖p‖²]

    Then f(p) · g(q) = ‖p‖² + ‖q‖² - 2 p·q = ‖p - q‖²
    """
    if points.shape[1] != 3:
        raise ValueError("5D dot trick only works with 3D points")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    norm_sq = torch.sum(points**2, dim=1)

    # f representation
    f_repr = torch.stack([x, y, z, norm_sq, torch.ones_like(x)], dim=1)

    # g representation
    g_repr = torch.stack([-2 * x, -2 * y, -2 * z, torch.ones_like(x), norm_sq], dim=1)

    return f_repr, g_repr


def method_scipy_cdist(main_points: np.ndarray, floor_points: np.ndarray, chunk_size: int = 1000, progress_update_fn=None, run_idx=0, num_runs=1, method_name='', device='', data_type='') -> float:
    """Method 1: scipy.cdist with chunking (CPU only)."""
    start_time = time.time()

    min_distances = np.full(len(floor_points), np.inf)
    total_chunks = (len(floor_points) + chunk_size - 1) // chunk_size

    for chunk_idx, i in enumerate(range(0, len(floor_points), chunk_size)):
        end_idx = min(i + chunk_size, len(floor_points))
        chunk_floor = floor_points[i:end_idx]

        chunk_distances = cdist(chunk_floor, main_points)
        chunk_min_distances = chunk_distances.min(axis=1)
        min_distances[i:end_idx] = chunk_min_distances
        if progress_update_fn:
            progress_update_fn(run_idx, num_runs, chunk_idx + 1, total_chunks, method_name, device, data_type)

    return time.time() - start_time


def method_torch_cdist(main_points: torch.Tensor, floor_points: torch.Tensor, chunk_size: int = 1000, progress_update_fn=None, run_idx=0, num_runs=1, method_name='', device='', data_type='') -> float:
    """Method 2: torch.cdist with chunking."""
    start_time = time.time()

    min_distances = torch.full((len(floor_points),), float('inf'), device=floor_points.device)
    total_chunks = (len(floor_points) + chunk_size - 1) // chunk_size

    for chunk_idx, i in enumerate(range(0, len(floor_points), chunk_size)):
        end_idx = min(i + chunk_size, len(floor_points))
        chunk_floor = floor_points[i:end_idx]

        chunk_distances = torch.cdist(chunk_floor, main_points)
        chunk_min_distances = chunk_distances.min(dim=1)[0]
        min_distances[i:end_idx] = chunk_min_distances
        if progress_update_fn:
            progress_update_fn(run_idx, num_runs, chunk_idx + 1, total_chunks, method_name, device, data_type)

    if floor_points.device.type == 'cuda':
        torch.cuda.synchronize()

    return time.time() - start_time


def method_simple_euclidean(main_points: torch.Tensor, floor_points: torch.Tensor, chunk_size: int = 1000, progress_update_fn=None, run_idx=0, num_runs=1, method_name='', device='', data_type='') -> float:
    """Method 3: Simple euclidean distance calculation."""
    start_time = time.time()

    min_distances = torch.full((len(floor_points),), float('inf'), device=floor_points.device)
    total_chunks = (len(floor_points) + chunk_size - 1) // chunk_size

    for chunk_idx, i in enumerate(range(0, len(floor_points), chunk_size)):
        end_idx = min(i + chunk_size, len(floor_points))
        chunk_floor = floor_points[i:end_idx]

        # Compute squared distances manually: ‖a - b‖² = ‖a‖² + ‖b‖² - 2a·b
        chunk_floor_sq = torch.sum(chunk_floor**2, dim=1, keepdim=True)  # (chunk_size, 1)
        main_sq = torch.sum(main_points**2, dim=1, keepdim=True).T  # (1, num_main)
        cross_term = 2 * torch.mm(chunk_floor, main_points.T)  # (chunk_size, num_main)

        chunk_distances_sq = chunk_floor_sq + main_sq - cross_term
        chunk_distances = torch.sqrt(torch.clamp(chunk_distances_sq, min=0))
        chunk_min_distances = chunk_distances.min(dim=1)[0]
        min_distances[i:end_idx] = chunk_min_distances
        if progress_update_fn:
            progress_update_fn(run_idx, num_runs, chunk_idx + 1, total_chunks, method_name, device, data_type)

    if floor_points.device.type == 'cuda':
        torch.cuda.synchronize()

    return time.time() - start_time


def method_dot_product(main_points: torch.Tensor, floor_points: torch.Tensor, chunk_size: int = 1000, progress_update_fn=None, run_idx=0, num_runs=1, method_name='', device='', data_type='') -> float:
    """Method 4: Dot product similarity (note: this is similarity, not distance)."""
    start_time = time.time()

    max_similarities = torch.full((len(floor_points),), float('-inf'), device=floor_points.device)
    total_chunks = (len(floor_points) + chunk_size - 1) // chunk_size

    for chunk_idx, i in enumerate(range(0, len(floor_points), chunk_size)):
        end_idx = min(i + chunk_size, len(floor_points))
        chunk_floor = floor_points[i:end_idx]

        chunk_similarities = torch.mm(chunk_floor, main_points.T)
        chunk_max_similarities = chunk_similarities.max(dim=1)[0]
        max_similarities[i:end_idx] = chunk_max_similarities
        if progress_update_fn:
            progress_update_fn(run_idx, num_runs, chunk_idx + 1, total_chunks, method_name, device, data_type)

    if floor_points.device.type == 'cuda':
        torch.cuda.synchronize()

    return time.time() - start_time


def method_5d_dot_precomputed(main_f: torch.Tensor, floor_g: torch.Tensor, chunk_size: int = 1000, progress_update_fn=None, run_idx=0, num_runs=1, method_name='', device='', data_type='') -> float:
    """Method 5: 5D dot trick with precomputed representations."""
    start_time = time.time()

    min_distances = torch.full((len(floor_g),), float('inf'), device=floor_g.device)
    total_chunks = (len(floor_g) + chunk_size - 1) // chunk_size

    for chunk_idx, i in enumerate(range(0, len(floor_g), chunk_size)):
        end_idx = min(i + chunk_size, len(floor_g))
        chunk_floor_g = floor_g[i:end_idx]

        # Compute squared distances using dot product
        chunk_distances_sq = torch.mm(chunk_floor_g, main_f.T)
        chunk_distances = torch.sqrt(torch.clamp(chunk_distances_sq, min=0))
        chunk_min_distances = chunk_distances.min(dim=1)[0]
        min_distances[i:end_idx] = chunk_min_distances
        if progress_update_fn:
            progress_update_fn(run_idx, num_runs, chunk_idx + 1, total_chunks, method_name, device, data_type)

    if floor_g.device.type == 'cuda':
        torch.cuda.synchronize()

    return time.time() - start_time


def precompute_and_save_5d_representations(
    main_points: torch.Tensor,
    floor_points: torch.Tensor,
    temp_dir: Path,
    chunk_size: int = 10000
) -> Tuple[Path, Path]:
    """
    Precompute and save 5D representations to disk for large datasets.
    This function's execution time is NOT included in benchmarks.
    """
    temp_dir.mkdir(exist_ok=True)

    # Precompute main points (f representation)
    main_f, _ = precompute_5d_representations(main_points)
    main_f_path = temp_dir / "main_f.npy"
    np.save(main_f_path, main_f.cpu().numpy())

    # Precompute floor points (g representation) in chunks and save
    floor_g_path = temp_dir / "floor_g.npy"

    # For very large datasets, process in chunks
    if len(floor_points) > chunk_size:
        # Create memory-mapped array for large datasets
        floor_g_shape = (len(floor_points), 5)  # 5D representation
        floor_g_mmap = np.memmap(floor_g_path, dtype=np.float32, mode='w+', shape=floor_g_shape)

        for i in range(0, len(floor_points), chunk_size):
            end_idx = min(i + chunk_size, len(floor_points))
            chunk_floor = floor_points[i:end_idx]

            _, chunk_floor_g = precompute_5d_representations(chunk_floor)
            floor_g_mmap[i:end_idx] = chunk_floor_g.cpu().numpy()

        # Force write to disk
        del floor_g_mmap
    else:
        # Small dataset, process all at once
        _, floor_g = precompute_5d_representations(floor_points)
        np.save(floor_g_path, floor_g.cpu().numpy())

    return main_f_path, floor_g_path


def method_5d_dot_precomputed_from_disk(
    main_f_path: Path,
    floor_g_path: Path,
    device: torch.device,
    chunk_size: int = 1000,
    load_chunk_size: int = 10000,
    progress_update_fn=None,
    run_idx=0,
    num_runs=1,
    method_name='',
    device_str='',
    data_type=''
) -> float:
    """
    Method 5: 5D dot trick loading precomputed representations from disk.
    Only the actual distance computation time is measured, not loading time.
    """
    # Load main_f (should be small enough to fit in memory)
    main_f = torch.from_numpy(np.load(main_f_path)).to(device)

    # Get floor_g size and shape
    floor_g_mmap = np.memmap(floor_g_path, dtype=np.float32, mode='r')
    total_floor_points = floor_g_mmap.shape[0] // 5

    min_distances = torch.full((total_floor_points,), float('inf'), device=device)

    # Start timing ONLY the distance computation
    start_time = time.time()

    # Process floor_g in chunks, loading from disk as needed
    total_load_chunks = (total_floor_points + load_chunk_size - 1) // load_chunk_size
    for load_chunk_idx, load_start in enumerate(range(0, total_floor_points, load_chunk_size)):
        load_end = min(load_start + load_chunk_size, total_floor_points)
        # Load chunk from disk and reshape to (chunk_size, 5)
        pause_time = time.time()
        chunk_flat = np.array(floor_g_mmap[load_start * 5:load_end * 5])
        floor_g_chunk = torch.from_numpy(chunk_flat.reshape(-1, 5)).to(device)
        start_time += (time.time() - pause_time)

        # Process this loaded chunk in smaller computation chunks
        total_chunks = (len(floor_g_chunk) + chunk_size - 1) // chunk_size
        for chunk_idx, i in enumerate(range(0, len(floor_g_chunk), chunk_size)):
            end_idx = min(i + chunk_size, len(floor_g_chunk))
            chunk_floor_g = floor_g_chunk[i:end_idx]

            # Compute squared distances using dot product
            chunk_distances_sq = torch.mm(chunk_floor_g, main_f.T)
            chunk_distances = torch.sqrt(torch.clamp(chunk_distances_sq, min=0))
            chunk_min_distances = chunk_distances.min(dim=1)[0]

            global_start_idx = load_start + i
            global_end_idx = load_start + end_idx
            min_distances[global_start_idx:global_end_idx] = chunk_min_distances
            if progress_update_fn:
                progress_update_fn(run_idx, num_runs, load_chunk_idx * total_chunks + chunk_idx + 1, total_load_chunks * total_chunks, method_name, device_str, data_type)

        # Clear chunk from GPU memory
        del floor_g_chunk
        torch.cuda.empty_cache()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    return time.time() - start_time


def method_5d_dot_onthefly(main_points: torch.Tensor, floor_points: torch.Tensor, chunk_size: int = 1000, progress_update_fn=None, run_idx=0, num_runs=1, method_name='', device='', data_type='') -> float:
    """Method 6: 5D dot trick with on-the-fly computation."""
    start_time = time.time()

    # Precompute main points representation (f)
    main_f, _ = precompute_5d_representations(main_points)

    min_distances = torch.full((len(floor_points),), float('inf'), device=floor_points.device)
    total_chunks = (len(floor_points) + chunk_size - 1) // chunk_size

    for chunk_idx, i in enumerate(range(0, len(floor_points), chunk_size)):
        end_idx = min(i + chunk_size, len(floor_points))
        chunk_floor = floor_points[i:end_idx]

        # Compute g representation for this chunk
        _, chunk_floor_g = precompute_5d_representations(chunk_floor)

        # Compute squared distances using dot product
        chunk_distances_sq = torch.mm(chunk_floor_g, main_f.T)
        chunk_distances = torch.sqrt(torch.clamp(chunk_distances_sq, min=0))
        chunk_min_distances = chunk_distances.min(dim=1)[0]
        min_distances[i:end_idx] = chunk_min_distances
        if progress_update_fn:
            progress_update_fn(run_idx, num_runs, chunk_idx + 1, total_chunks, method_name, device, data_type)

    if floor_points.device.type == 'cuda':
        torch.cuda.synchronize()

    return time.time() - start_time


def run_benchmarks(
    data: Dict[str, torch.Tensor],
    chunk_size: int = 1000,
    num_runs: int = 3,
    output_dir: Path = Path("benchmark_results"),
    no_cpu: bool = False,
    no_highd: bool = False
) -> pd.DataFrame:
    """Run all benchmarks and return results."""

    results = []

    methods_info = [
        ("scipy_cdist", "CPU", True, False),  # (method_name, device, cpu_only, needs_3d)
        ("torch_cdist", "CPU", False, False),
        ("torch_cdist", "GPU", False, False),
        ("simple_euclidean", "CPU", False, False),
        ("simple_euclidean", "GPU", False, False),
        ("dot_product", "CPU", False, False),
        ("dot_product", "GPU", False, False),
        ("5d_dot_precomputed", "CPU", False, True),
        ("5d_dot_precomputed", "GPU", False, True),
        ("5d_dot_onthefly", "CPU", False, True),
        ("5d_dot_onthefly", "GPU", False, True),
    ]

    # Test both 3D and high-D
    data_types = [("3D", "3d"), ("High-D", "high")]

    total_tests = len(methods_info) * len(data_types)
    console.print(f"[bold blue]Running {total_tests} benchmark combinations ({num_runs} runs each)...")

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as global_progress:

        task = global_progress.add_task("Starting...", total=total_tests)

        for data_type_name, data_suffix in data_types:
            # --- Skip all High-D if requested ---
            if no_highd and data_type_name == "High-D":
                for _ in methods_info:
                    global_progress.advance(task)
                continue
            for method_name, device, cpu_only, needs_3d in methods_info:
                # --- Skip if method is disabled ---
                if not METHODS_ENABLED.get(method_name, True):
                    global_progress.advance(task)
                    continue
                # ---
                # --- Skip all CPU if requested ---
                if no_cpu and device == "CPU":
                    global_progress.advance(task)
                    continue
                # ---
                # Skip GPU tests if CUDA not available
                if device == "GPU" and not torch.cuda.is_available():
                    console.print(f"[yellow]Skipping {method_name} GPU - CUDA not available")
                    global_progress.advance(task)
                    continue
                # Skip high-D tests for 5D methods
                if needs_3d and data_suffix == "high":
                    console.print(f"[yellow]Skipping {method_name} {device} High-D - only works with 3D")
                    global_progress.advance(task)
                    continue
                # Skip CPU-only methods on GPU
                if cpu_only and device == "GPU":
                    global_progress.advance(task)
                    continue
                device_suffix = "cpu" if device == "CPU" else "gpu"
                main_key = f"main_{data_suffix}_{device_suffix}"
                floor_key = f"floor_{data_suffix}_{device_suffix}"
                if main_key not in data or floor_key not in data:
                    console.print(f"[yellow]Skipping {method_name} {device} {data_type_name} - data not available")
                    global_progress.advance(task)
                    continue
                main_points = data[main_key]
                floor_points = data[floor_key]
                # For 5d_dot_precomputed, prepare disk storage (this time is NOT counted)
                precomputed_paths = None
                if method_name == "5d_dot_precomputed" and data_suffix == "3d":
                    temp_dir = output_dir / f"temp_5d_{device.lower()}_{data_suffix}"
                    console.print(f"[dim]Precomputing 5D representations for {method_name} {device} {data_type_name}...")
                    precomputed_paths = precompute_and_save_5d_representations(
                        main_points, floor_points, temp_dir, chunk_size=10000
                    )
                times = []
                # --- Persistent method name in progress bar ---
                persistent_desc = f"[cyan]{method_name} {device} {data_type_name}"
                global_progress.update(task, description=f"{persistent_desc} [Run 1/{num_runs}]")
                try:
                    for run in range(num_runs):
                        if num_runs > 1:
                            global_progress.update(task, description=f"{persistent_desc} [Run {run+1}/{num_runs}]")

                        def progress_update_fn(run_idx, num_runs, chunk_idx, total_chunks, m, d, dt): return global_progress.update(
                            task,
                            description=f"{persistent_desc} [Run {run_idx+1}/{num_runs}, Chunk {chunk_idx}/{total_chunks}]"
                        )
                        if method_name == "scipy_cdist":
                            elapsed = method_scipy_cdist(
                                main_points.numpy(),
                                floor_points.numpy(),
                                chunk_size,
                                progress_update_fn=progress_update_fn,
                                run_idx=run,
                                num_runs=num_runs,
                                method_name=method_name,
                                device=device,
                                data_type=data_type_name
                            )
                        elif method_name == "torch_cdist":
                            elapsed = method_torch_cdist(
                                main_points, floor_points, chunk_size,
                                progress_update_fn=progress_update_fn,
                                run_idx=run,
                                num_runs=num_runs,
                                method_name=method_name,
                                device=device,
                                data_type=data_type_name
                            )
                        elif method_name == "simple_euclidean":
                            elapsed = method_simple_euclidean(
                                main_points, floor_points, chunk_size,
                                progress_update_fn=progress_update_fn,
                                run_idx=run,
                                num_runs=num_runs,
                                method_name=method_name,
                                device=device,
                                data_type=data_type_name
                            )
                        elif method_name == "dot_product":
                            elapsed = method_dot_product(
                                main_points, floor_points, chunk_size,
                                progress_update_fn=progress_update_fn,
                                run_idx=run,
                                num_runs=num_runs,
                                method_name=method_name,
                                device=device,
                                data_type=data_type_name
                            )
                        elif method_name == "5d_dot_precomputed":
                            main_f_path, floor_g_path = precomputed_paths
                            device_obj = torch.device("cuda" if device_suffix == "gpu" else "cpu")
                            elapsed = method_5d_dot_precomputed_from_disk(
                                main_f_path, floor_g_path, device_obj, chunk_size, load_chunk_size=10000,
                                progress_update_fn=progress_update_fn,
                                run_idx=run,
                                num_runs=num_runs,
                                method_name=method_name,
                                device_str=device,
                                data_type=data_type_name
                            )
                        elif method_name == "5d_dot_onthefly":
                            elapsed = method_5d_dot_onthefly(
                                main_points, floor_points, chunk_size,
                                progress_update_fn=progress_update_fn,
                                run_idx=run,
                                num_runs=num_runs,
                                method_name=method_name,
                                device=device,
                                data_type=data_type_name
                            )
                        times.append(elapsed)
                except Exception as e:
                    console.print(f"[red]Error in {method_name} {device} {data_type_name}: {e}")
                    times = [float('inf')]
                avg_time = np.mean(times)
                std_time = np.std(times)
                results.append({
                    'Method': method_name,
                    'Device': device,
                    'Data_Type': data_type_name,
                    'Avg_Time': avg_time,
                    'Std_Time': std_time,
                    'Min_Time': np.min(times),
                    'Max_Time': np.max(times)
                })
                if method_name == "5d_dot_precomputed" and precomputed_paths is not None:
                    temp_dir = output_dir / f"temp_5d_{device.lower()}_{data_suffix}"
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                        console.print(f"[dim]Cleaned up temporary files for {method_name} {device} {data_type_name}")
                global_progress.advance(task)
    return pd.DataFrame(results)


def create_visualization(results_df: pd.DataFrame, output_dir: Path):
    """Create a single 6x4 heatmap (methods x [CPU 3D, GPU 3D, CPU High-D, GPU High-D]),
    with both cell value and color based on global speedup. N/A cells are gray and annotated as 'N/A'."""
    import matplotlib as mpl
    # Filter out failed tests
    valid_results = results_df[results_df['Avg_Time'] != float('inf')].copy()
    if valid_results.empty:
        console.print("[red]No valid results to visualize")
        return

    method_order = [
        "simple_euclidean",
        "scipy_cdist",
        "torch_cdist",
        "dot_product",
        "5d_dot_onthefly",
        "5d_dot_precomputed"
    ]
    col_order = [
        "CPU 3D", "GPU 3D",
        "CPU High-D", "GPU High-D"
    ]
    # Build a 6x4 matrix of times
    time_matrix = np.full((6, 4), np.nan)
    global_speedup_matrix = np.full((6, 4), np.nan)
    annot_matrix = np.full((6, 4), '', dtype=object)
    # Map col index to device/data_type
    col_map = [
        ("CPU", "3D"), ("GPU", "3D"),
        ("CPU", "High-D"), ("GPU", "High-D")
    ]
    for i, method in enumerate(method_order):
        for j, (device, dtype) in enumerate(col_map):
            df = valid_results[(valid_results['Method'] == method) & (valid_results['Device'] == device) & (valid_results['Data_Type'] == dtype)]
            if not df.empty:
                t = df.iloc[0]['Avg_Time']
                time_matrix[i, j] = t
    # Compute global slowest
    valid_times = time_matrix[~np.isnan(time_matrix)]
    global_slowest = np.nanmax(valid_times)
    # Fill matrices
    for i in range(6):
        for j in range(4):
            t = time_matrix[i, j]
            if not np.isnan(t):
                global_speedup = global_slowest / t
                global_speedup_matrix[i, j] = global_speedup
                annot_matrix[i, j] = f"{global_speedup:.1f}x\n({t:.3f}s)"
            else:
                annot_matrix[i, j] = "N/A"
    # Mask for N/A
    mask = np.isnan(global_speedup_matrix)
    # Set up the plot
    plt.figure(figsize=(10, 6))
    # Use a colormap with gray for N/A
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    cmap = cmap.with_extremes(bad="#cccccc")
    # Logarithmic normalization for colorbar (global speedup)
    valid_speedups = global_speedup_matrix[~np.isnan(global_speedup_matrix)]
    vmin = np.nanmin(valid_speedups)
    vmax = np.nanmax(valid_speedups)
    norm = mpl.colors.LogNorm(vmin=max(vmin, 1.01), vmax=vmax)  # avoid vmin=1 for log
    # Plot
    ax = sns.heatmap(
        global_speedup_matrix,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        mask=mask,
        cbar=True,
        norm=norm,
        yticklabels=method_order,
        xticklabels=col_order,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"fontsize": 10}
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label('Global Speedup (relative to slowest overall)', rotation=270, labelpad=20)
    ax.set_title("Distance Method Benchmark (higher is better)\nCell value and color: global speedup")
    ax.set_xlabel('')
    ax.set_ylabel('Method')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    output_path = output_dir / "benchmark_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved visualization: {output_path}")
    plt.show()


def print_summary_table(results_df: pd.DataFrame):
    """Print a summary table of results."""

    # Filter out failed tests
    valid_results = results_df[results_df['Avg_Time'] != float('inf')].copy()

    if valid_results.empty:
        console.print("[red]No valid results to display")
        return

    # Create summary table
    table = Table(title="Benchmark Results Summary")
    table.add_column("Method", style="cyan")
    table.add_column("Device", style="magenta")
    table.add_column("Data Type", style="yellow")
    table.add_column("Avg Time (s)", justify="right", style="green")
    table.add_column("Speedup vs Slowest", justify="right", style="blue")

    # Sort by average time
    sorted_results = valid_results.sort_values('Avg_Time')
    slowest_time = sorted_results['Avg_Time'].max()

    for _, row in sorted_results.iterrows():
        speedup = slowest_time / row['Avg_Time']
        table.add_row(
            row['Method'],
            row['Device'],
            row['Data_Type'],
            f"{row['Avg_Time']:.3f}",
            f"{speedup:.1f}x"
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Benchmark distance calculation methods")
    parser.add_argument("--num-main", type=int, default=10000,
                        help="Number of main points (default: 10000)")
    parser.add_argument("--num-floor", type=int, default=50000,
                        help="Number of floor points (default: 50000)")
    parser.add_argument("--feature-dim", type=int, default=512,
                        help="Dimension of high-D features (default: 512)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Chunk size for processing (default: 1000)")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of runs per test for averaging (default: 3)")
    parser.add_argument("--output-dir", type=Path, default="benchmark_results",
                        help="Output directory for results (default: benchmark_results)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Skip GPU tests")
    parser.add_argument("--no-cpu", action="store_true",
                        help="Skip CPU tests")
    parser.add_argument("--no-highd", action="store_true",
                        help="Skip High-D (high-dimensional) benchmarks")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu"

    console.print(f"[bold blue]Distance Calculation Benchmark")
    console.print(f"[bold blue]Main points: {args.num_main:,}")
    console.print(f"[bold blue]Floor points: {args.num_floor:,}")
    console.print(f"[bold blue]Feature dimension: {args.feature_dim}")
    console.print(f"[bold blue]Chunk size: {args.chunk_size}")
    console.print(f"[bold blue]Runs per test: {args.num_runs}")
    console.print(f"[bold blue]Device: {device}")
    if args.no_cpu:
        console.print(f"[yellow]Skipping all CPU device benchmarks (--no-cpu)")
    if args.no_highd:
        console.print(f"[yellow]Skipping all High-D benchmarks (--no-highd)")

    # Generate test data
    data = generate_test_data(
        args.num_main,
        args.num_floor,
        dim_high=args.feature_dim,
        device=device
    )

    # Run benchmarks
    results_df = run_benchmarks(data, args.chunk_size, args.num_runs, args.output_dir, no_cpu=args.no_cpu, no_highd=args.no_highd)

    # Save results
    results_path = args.output_dir / "benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    console.print(f"[green]✓ Saved detailed results: {results_path}")

    # Print summary
    print_summary_table(results_df)

    # Create visualization
    create_visualization(results_df, args.output_dir)

    console.print(f"[bold green]✓ Benchmarking completed!")


if __name__ == "__main__":
    main()
