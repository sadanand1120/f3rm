#!/usr/bin/env python3
"""
F3RM Feature Pointcloud Exporter

This script exports pointclouds with RGB, features, feature_PCA, and pred_normals data from F3RM models.
It samples points from the NeRF and extracts features, colors, and predicted normals, saving them to optimized formats.

Usage:
    python f3rm/manual/export_feature_pointcloud.py --config path/to/config.yml --output-dir path/to/output/ --num-points 50000000
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from nerfstudio.utils.eval_utils import eval_setup
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from tqdm import tqdm

from f3rm.pca_colormap import apply_pca_colormap_return_proj

console = Console()


def sample_feature_pointcloud(
    pipeline,
    num_points: int,
    bbox_min: Optional[Tuple[float, float, float]] = None,
    bbox_max: Optional[Tuple[float, float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a pointcloud with RGB, features, and consistent PCA features.

    This uses the simple approach from f3rm_robot/utils.py that avoids tensor shape issues.

    Args:
        pipeline: The F3RM pipeline
        num_points: Target number of points to sample
        bbox_min: Optional minimum bounding box coordinates (x, y, z)
        bbox_max: Optional maximum bounding box coordinates (x, y, z)

    Returns:
        Tuple of (points, rgb, features, feature_pca, pred_normals) tensors
    """
    device = pipeline.device

    console.print(f"[bold green]Sampling {num_points:,} points from F3RM model...")

    if bbox_min is not None and bbox_max is not None:
        console.print(f"[bold green]Using bounding box: {bbox_min} to {bbox_max}")

    points_list = []
    rgbs_list = []
    features_list = []
    pred_normals_list = []

    # Initialize PCA for consistent feature_pca across chunks
    pca_proj = None
    pca_min = None
    pca_max = None

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Sampling points", total=num_points)
        collected_points = 0

        while collected_points < num_points:
            with torch.no_grad():
                # Get rays from datamanager
                ray_bundle, _ = pipeline.datamanager.next_train(0)

                # Get model outputs - this is the simple approach that works
                outputs = pipeline.model(ray_bundle)

                rgb = outputs["rgb"]
                features = outputs["feature"]
                depth = outputs["depth"]

                # Get pred_normals if available
                pred_normals = None
                if "pred_normals" in outputs:
                    pred_normals = outputs["pred_normals"]

                # Convert depth to world coordinates
                nerf_points = ray_bundle.origins + ray_bundle.directions * depth

                # Apply bounding box filtering if specified
                if bbox_min is not None and bbox_max is not None:
                    bbox_min_tensor = torch.tensor(bbox_min, device=device)
                    bbox_max_tensor = torch.tensor(bbox_max, device=device)

                    # Create mask for points within bounding box
                    within_bbox = torch.all(
                        (nerf_points >= bbox_min_tensor) & (nerf_points <= bbox_max_tensor),
                        dim=-1
                    )

                    # Filter points, RGB, features, and pred_normals
                    if within_bbox.any():
                        nerf_points = nerf_points[within_bbox]
                        rgb = rgb[within_bbox]
                        features = features[within_bbox]
                        if pred_normals is not None:
                            pred_normals = pred_normals[within_bbox]
                    else:
                        # No points within bbox, skip this batch
                        continue

                points_list.append(nerf_points)
                rgbs_list.append(rgb)
                features_list.append(features)
                if pred_normals is not None:
                    pred_normals_list.append(pred_normals)

                collected_points += len(nerf_points)
                progress.update(task, completed=min(collected_points, num_points))

                # Clear GPU cache periodically to prevent OOM
                if len(points_list) % 10 == 0:  # Every 10 chunks
                    torch.cuda.empty_cache()

                # Break if we have enough points
                if collected_points >= num_points:
                    break

    # Concatenate all collected data
    if not points_list:
        raise RuntimeError("No valid points were sampled. Try expanding the bounding box.")

    console.print(f"[yellow]Concatenating {len(points_list)} chunks...")

    # Move to CPU and concatenate to save GPU memory
    points_cpu = []
    rgbs_cpu = []
    features_cpu = []
    pred_normals_cpu = []

    for points_chunk, rgbs_chunk, features_chunk in zip(points_list, rgbs_list, features_list):
        points_cpu.append(points_chunk.cpu())
        rgbs_cpu.append(rgbs_chunk.cpu())
        features_cpu.append(features_chunk.cpu())

    # Handle pred_normals if available
    has_pred_normals = len(pred_normals_list) > 0
    if has_pred_normals:
        for pred_normals_chunk in pred_normals_list:
            pred_normals_cpu.append(pred_normals_chunk.cpu())

    # Clear GPU lists to free memory
    del points_list, rgbs_list, features_list, pred_normals_list
    torch.cuda.empty_cache()

    # Concatenate on CPU
    points = torch.cat(points_cpu, dim=0)
    rgbs = torch.cat(rgbs_cpu, dim=0)
    features = torch.cat(features_cpu, dim=0)

    # Handle pred_normals concatenation
    pred_normals = None
    if has_pred_normals:
        pred_normals = torch.cat(pred_normals_cpu, dim=0)

    # Clear CPU lists to free memory
    del points_cpu, rgbs_cpu, features_cpu, pred_normals_cpu

    # Subsample to exactly num_points if we have more
    if len(points) > num_points:
        console.print(f"[yellow]Subsampling from {len(points):,} to {num_points:,} points...")
        indices = torch.randperm(len(points))[:num_points]
        points = points[indices]
        rgbs = rgbs[indices]
        features = features[indices]
        if pred_normals is not None:
            pred_normals = pred_normals[indices]

    console.print(f"[bold green]Collected {len(points):,} points")

    # Move back to GPU for PCA computation
    points = points.to(device)
    rgbs = rgbs.to(device)
    features = features.to(device)
    if pred_normals is not None:
        pred_normals = pred_normals.to(device)

    # Apply consistent PCA to all features
    console.print("[bold green]Computing PCA projection for features...")
    feature_pca, pca_proj, pca_min, pca_max = apply_pca_colormap_return_proj(features, pca_proj, pca_min, pca_max)

    return points, rgbs, features, feature_pca, pred_normals, pca_proj, pca_min, pca_max


def save_pointcloud_data(
    output_dir: Path,
    points: torch.Tensor,
    rgbs: torch.Tensor,
    features: torch.Tensor,
    feature_pca: torch.Tensor,
    pred_normals: Optional[torch.Tensor],
    pca_proj: torch.Tensor,
    pca_min: torch.Tensor,
    pca_max: torch.Tensor,
    compress_features: bool = True,
) -> None:
    """
    Save pointcloud data in multiple formats with data type optimization.

    Args:
        output_dir: Output directory
        points: Point coordinates (N, 3)
        rgbs: RGB colors (N, 3)
        features: Feature vectors (N, feature_dim)
        feature_pca: PCA-projected features (N, 3)
        pca_proj: PCA projection matrix
        pca_min: PCA min values for normalization
        pca_max: PCA max values for normalization
        compress_features: Whether to compress features to save space
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy for easier manipulation
    points_np = points.cpu().numpy().astype(np.float32)
    rgbs_np = rgbs.cpu().numpy().astype(np.float32)
    features_np = features.cpu().numpy()
    feature_pca_np = feature_pca.cpu().numpy().astype(np.float32)

    # Handle pred_normals
    pred_normals_np = None
    if pred_normals is not None:
        pred_normals_np = pred_normals.cpu().numpy().astype(np.float32)

    console.print(f"[bold green]Saving pointcloud data to {output_dir}")

    # 1. Save RGB pointcloud (standard PLY format)
    rgb_pcd = o3d.geometry.PointCloud()
    rgb_pcd.points = o3d.utility.Vector3dVector(points_np)
    rgb_pcd.colors = o3d.utility.Vector3dVector(rgbs_np)

    rgb_path = output_dir / "pointcloud_rgb.ply"
    o3d.io.write_point_cloud(str(rgb_path), rgb_pcd)
    console.print(f"[green]✓ Saved RGB pointcloud: {rgb_path}")

    # 2. Save feature_PCA pointcloud (PLY format)
    pca_pcd = o3d.geometry.PointCloud()
    pca_pcd.points = o3d.utility.Vector3dVector(points_np)
    pca_pcd.colors = o3d.utility.Vector3dVector(feature_pca_np)

    pca_path = output_dir / "pointcloud_feature_pca.ply"
    o3d.io.write_point_cloud(str(pca_path), pca_pcd)
    console.print(f"[green]✓ Saved feature_PCA pointcloud: {pca_path}")

    # 3. Save pred_normals pointcloud (PLY format) if available
    if pred_normals_np is not None:
        normals_pcd = o3d.geometry.PointCloud()
        normals_pcd.points = o3d.utility.Vector3dVector(points_np)
        normals_pcd.colors = o3d.utility.Vector3dVector(pred_normals_np)

        normals_path = output_dir / "pointcloud_pred_normals.ply"
        o3d.io.write_point_cloud(str(normals_path), normals_pcd)
        console.print(f"[green]✓ Saved pred_normals pointcloud: {normals_path}")
    else:
        console.print(f"[yellow]No pred_normals available - model may not have predict_normals enabled")

    # 4. Save features with compression options
    if compress_features:
        # Use float16 for features to save space (reduces size by ~50%)
        features_compressed = features_np.astype(np.float16)
        features_path = output_dir / "features_float16.npy"
        np.save(features_path, features_compressed)
        console.print(f"[green]✓ Saved compressed features (float16): {features_path}")
        console.print(f"[dim]Original size: {features_np.nbytes / 1e6:.1f}MB, Compressed: {features_compressed.nbytes / 1e6:.1f}MB")
    else:
        # Save full precision features
        features_path = output_dir / "features_float32.npy"
        np.save(features_path, features_np.astype(np.float32))
        console.print(f"[green]✓ Saved full precision features: {features_path}")

    # 5. Save points separately for easy loading
    points_path = output_dir / "points.npy"
    np.save(points_path, points_np)
    console.print(f"[green]✓ Saved points: {points_path}")

    # 6. Save PCA transformation parameters for consistent visualization
    pca_data = {
        'projection_matrix': pca_proj.cpu().numpy(),
        'min_values': pca_min.cpu().numpy(),
        'max_values': pca_max.cpu().numpy(),
        'feature_dim': features.shape[-1],
        'num_points': len(points)
    }

    pca_path = output_dir / "pca_params.pkl"
    with open(pca_path, 'wb') as f:
        pickle.dump(pca_data, f)
    console.print(f"[green]✓ Saved PCA parameters: {pca_path}")

    # 7. Save metadata
    metadata = {
        'num_points': len(points),
        'feature_dim': features.shape[-1],
        'bbox_min': points_np.min(axis=0).astype(float).tolist(),
        'bbox_max': points_np.max(axis=0).astype(float).tolist(),
        'feature_range': [float(features_np.min()), float(features_np.max())],
        'compressed_features': compress_features,
        'files': {
            'rgb_pointcloud': 'pointcloud_rgb.ply',
            'pca_pointcloud': 'pointcloud_feature_pca.ply',
            'pred_normals_pointcloud': 'pointcloud_pred_normals.ply' if pred_normals_np is not None else None,
            'features': 'features_float16.npy' if compress_features else 'features_float32.npy',
            'points': 'points.npy',
            'pca_params': 'pca_params.pkl'
        },
        'has_pred_normals': pred_normals_np is not None
    }

    metadata_path = output_dir / "metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    console.print(f"[green]✓ Saved metadata: {metadata_path}")

    # Print summary
    total_size = sum((output_dir / f).stat().st_size for f in metadata['files'].values() if f is not None) / 1e6
    console.print(f"\n[bold green]Export Summary:")
    console.print(f"  Points: {len(points):,}")
    console.print(f"  Feature dimension: {features.shape[-1]}")
    console.print(f"  Pred normals: {'Yes' if pred_normals_np is not None else 'No'}")
    console.print(f"  Total size: {total_size:.1f}MB")
    console.print(f"  Files saved in: {output_dir}")


def export_feature_pointcloud(
    config_path: Path,
    output_dir: Path,
    num_points: int = 1000000,
    compress_features: bool = True,
    bbox_min: Optional[Tuple[float, float, float]] = None,
    bbox_max: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Export F3RM pointcloud with RGB, features, and feature_PCA.

    Args:
        config_path: Path to F3RM model config
        output_dir: Output directory
        num_points: Number of points to sample
        compress_features: Whether to compress features to save space
        bbox_min: Optional minimum bounding box coordinates (x, y, z)
        bbox_max: Optional maximum bounding box coordinates (x, y, z)
    """
    console.print(f"[bold blue]F3RM Feature Pointcloud Exporter")
    console.print(f"[bold blue]Config: {config_path}")
    console.print(f"[bold blue]Output: {output_dir}")
    console.print(f"[bold blue]Target points: {num_points:,}")

    # Load model
    console.print(f"[bold green]Loading F3RM model...")
    config, pipeline, checkpoint_path, step = eval_setup(config_path)
    console.print(f"[bold green]Loaded checkpoint from step {step}")

    # Sample pointcloud with features
    points, rgbs, features, feature_pca, pred_normals, pca_proj, pca_min, pca_max = sample_feature_pointcloud(
        pipeline=pipeline,
        num_points=num_points,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )

    # Save all data
    save_pointcloud_data(
        output_dir=output_dir,
        points=points,
        rgbs=rgbs,
        features=features,
        feature_pca=feature_pca,
        pred_normals=pred_normals,
        pca_proj=pca_proj,
        pca_min=pca_min,
        pca_max=pca_max,
        compress_features=compress_features,
    )

    console.print(f"[bold green]✓ Export completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Export F3RM feature pointclouds")
    parser.add_argument("--config", type=Path, required=True, help="Path to F3RM model config.yml")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--num-points", type=int, default=5000000, help="Number of points to sample (default: 1M, try 100K-500K for large scenes)")
    parser.add_argument("--no-compress", action="store_true", help="Don't compress features (use float32)")

    # Bounding box arguments with reasonable defaults for indoor scenes
    parser.add_argument("--bbox-min", type=float, nargs=3, default=[-1.0, -1.0, -1.0],
                        help="Minimum bounding box coordinates (x y z) - default: -1 -1 -1")
    parser.add_argument("--bbox-max", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Maximum bounding box coordinates (x y z) - default: 1 1 1")
    parser.add_argument("--no-bbox", action="store_true", help="Disable bounding box filtering")

    args = parser.parse_args()

    # Validate inputs
    if not args.config.exists():
        console.print(f"[bold red]Config file not found: {args.config}")
        return

    # Set up bounding box
    bbox_min = None if args.no_bbox else tuple(args.bbox_min)
    bbox_max = None if args.no_bbox else tuple(args.bbox_max)

    if not args.no_bbox:
        console.print(f"[bold blue]Bounding box: {bbox_min} to {bbox_max}")
        console.print(f"[dim]Use --no-bbox to disable filtering, or adjust --bbox-min/--bbox-max")

    # Memory usage warning
    if args.num_points > 5000000:
        console.print(f"[yellow]Warning: {args.num_points:,} points may require significant GPU memory")
        console.print(f"[yellow]Consider using fewer points (e.g., --num-points 1000000) if you encounter OOM errors")

    try:
        export_feature_pointcloud(
            config_path=args.config,
            output_dir=args.output_dir,
            num_points=args.num_points,
            compress_features=not args.no_compress,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
