#!/usr/bin/env python3
"""
F3RM Feature Pointcloud Exporter

This script exports pointclouds with RGB, features, feature_PCA, and pred_normals data from F3RM models.
It samples points from the NeRF and extracts features, colors, and predicted normals, saving them to optimized formats.

Usage:
    python f3rm/manual/export_feature_pointcloud.py --config path/to/config.yml --output-dir path/to/output/ --num-points 50000000
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from nerfstudio.utils.eval_utils import eval_setup
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

from f3rm.pca_colormap import apply_pca_colormap_return_proj

console = Console()


def _collect_and_concatenate_tensors(tensor_lists: Dict[str, List[torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tensor lists to CPU, concatenate, and return results."""
    results = {}
    with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console) as progress:
        concat_task = progress.add_task("Concatenating tensors", total=len(tensor_lists))
        for name, tensor_list in tensor_lists.items():
            if not tensor_list:
                results[name] = None
            else:
                results[name] = torch.cat([t.cpu() for t in tensor_list], dim=0).to(device)
            progress.advance(concat_task)
    return results


def _save_ply_pointcloud(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    """Save pointcloud as PLY file."""
    # Ensure data types and shapes are correct for Open3D
    points = points.astype(np.float64)
    colors = colors.astype(np.float64)

    # Ensure colors are in [0, 1] range and have correct shape
    colors = np.clip(colors, 0.0, 1.0)
    if colors.shape[-1] != 3:
        raise ValueError(f"Colors must have 3 channels, got {colors.shape[-1]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(output_path), pcd)


def sample_feature_pointcloud(
    pipeline,
    num_points: int,
    bbox_bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Sample a pointcloud with RGB, features, and consistent PCA features.

    Bbox filtering is applied during sampling to prioritize points within the cube
    defined by bbox_bounds (default: [-1, 1] on each axis).
    """
    device = pipeline.device
    tensor_lists = {"points": [], "rgbs": [], "features": []}
    # Raw outputs (before shaders)
    raw_additional_outputs = {
        "pred_normals": [], "centroid": [], "centroid_spread": [], "foreground_logits": []
    }
    pca_proj = pca_min = pca_max = None
    collected_points = 0

    bbox_min, bbox_max = bbox_bounds
    with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), TimeRemainingColumn(), console=console) as progress:
        sampling_task = progress.add_task("Sampling points (in [-1,1] bbox)", total=num_points)
        while collected_points < num_points:
            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                outputs = pipeline.model(ray_bundle)
                points_batch = ray_bundle.origins + ray_bundle.directions * outputs["depth"]
                # Keep only points within the bbox
                in_bbox_mask = ((points_batch >= bbox_min) & (points_batch <= bbox_max)).all(dim=-1)

                if in_bbox_mask.any():
                    filtered = {
                        "points": points_batch[in_bbox_mask],
                        "rgbs": outputs["rgb"][in_bbox_mask] if outputs.get("rgb") is not None else None,
                        "features": outputs["feature"][in_bbox_mask] if outputs.get("feature") is not None else None,
                    }
                    for key in tensor_lists:
                        if filtered[key] is not None:
                            tensor_lists[key].append(filtered[key])
                    # Filter and collect raw additional outputs
                    for key in raw_additional_outputs:
                        if key in outputs and outputs[key] is not None:
                            raw_additional_outputs[key].append(outputs[key][in_bbox_mask])

                    collected_points += len(filtered["points"])
                    progress.update(sampling_task, completed=min(collected_points, num_points))
                if len(tensor_lists["points"]) % 10 == 0:
                    torch.cuda.empty_cache()

    results = _collect_and_concatenate_tensors(tensor_lists, device)
    del tensor_lists
    torch.cuda.empty_cache()

    indices = None
    if len(results["points"]) > num_points:
        console.print(f"[yellow]Subsampling from {len(results['points']):,} to {num_points:,} points...")
        indices = torch.randperm(len(results["points"]))[:num_points]
        with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console) as progress:
            subsample_task = progress.add_task("Subsampling tensors", total=len(results))
            for key in results:
                if results[key] is not None:
                    results[key] = results[key][indices]
                progress.advance(subsample_task)

    console.print(f"[bold green]Collected {len(results['points']):,} points")
    console.print("[bold green]Computing PCA projection for features...")
    feature_pca, pca_proj, pca_min, pca_max = apply_pca_colormap_return_proj(results["features"], pca_proj, pca_min, pca_max)

    # Process raw additional outputs and apply shaders
    final_additional = {}
    with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console) as progress:
        process_task = progress.add_task("Processing additional outputs", total=len(raw_additional_outputs))
        for key, tensor_list in raw_additional_outputs.items():
            if tensor_list:
                raw_tensor = torch.cat([t.cpu() for t in tensor_list], dim=0).to(device)
                if indices is not None:
                    raw_tensor = raw_tensor[indices]

                # Apply shaders to create RGB visualizations (exactly as in model.py)
                if key == "pred_normals":
                    final_additional["pred_normals"] = raw_tensor
                elif key == "centroid":
                    # Compute global min/max over contracted centroids for consistent coloring across aggregation
                    contracted = pipeline.model.centroid_shader.contract(raw_tensor)
                    min_v = contracted.amin(dim=0, keepdim=True)
                    max_v = contracted.amax(dim=0, keepdim=True)
                    final_additional["centroid_pred_rgb"] = pipeline.model.centroid_shader(raw_tensor, global_min=min_v, global_max=max_v)
                elif key == "centroid_spread":
                    # Apply spread shaders (exactly as in model.py lines 493-494)
                    final_additional["centroid_spread_error_rgb"] = pipeline.model.spread_shader(raw_tensor[..., :1])
                    final_additional["centroid_spread_prob_rgb"] = pipeline.model.prob_shader(raw_tensor[..., 1:2])
                elif key == "foreground_logits":
                    # Convert logits → probs for class-1, then apply shader (matches model full-image viz)
                    probs = torch.softmax(raw_tensor / 1.0, dim=-1)[..., 1:2]  # 1.0 is softmax temp
                    final_additional["foreground_prob_rgb"] = pipeline.model.prob_from_probs_shader(probs)
            progress.advance(process_task)

    return results["points"], results["rgbs"], results["features"], feature_pca, pca_proj, pca_min, pca_max, final_additional


def save_pointcloud_data(output_dir: Path, points: torch.Tensor, rgbs: torch.Tensor, features: torch.Tensor, feature_pca: torch.Tensor, pca_proj: torch.Tensor, pca_min: torch.Tensor, pca_max: torch.Tensor, additional_outputs: Dict[str, torch.Tensor], compress_features: bool = True) -> None:
    """Save pointcloud data in multiple formats with data type optimization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_np = {
        "points": points.cpu().numpy().astype(np.float32),
        "rgbs": rgbs.cpu().numpy().astype(np.float32),
        "features": features.cpu().numpy(),
        "feature_pca": feature_pca.cpu().numpy().astype(np.float32)
    }

    console.print(f"[bold green]Saving pointcloud data to {output_dir}")
    _save_ply_pointcloud(data_np["points"], data_np["rgbs"], output_dir / "pointcloud_rgb.ply")
    console.print(f"[green]✓ Saved RGB pointcloud: {output_dir}/pointcloud_rgb.ply")
    _save_ply_pointcloud(data_np["points"], data_np["feature_pca"], output_dir / "pointcloud_feature_pca.ply")
    console.print(f"[green]✓ Saved feature_PCA pointcloud: {output_dir}/pointcloud_feature_pca.ply")

    additional_files = {}
    with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console) as progress:
        save_task = progress.add_task("Saving additional outputs", total=len(additional_outputs))
        for key, tensor in additional_outputs.items():
            if tensor is not None:
                rgb_data = tensor.cpu().numpy().astype(np.float32)
                filename = f"pointcloud_{key}.ply"
                _save_ply_pointcloud(data_np["points"], rgb_data, output_dir / filename)
                console.print(f"[green]✓ Saved {key} pointcloud: {output_dir}/{filename}")
                additional_files[key] = filename
            progress.advance(save_task)

    np.save(output_dir / "points.npy", data_np["points"])
    console.print(f"[green]✓ Saved points: {output_dir}/points.npy")

    features_file = "features_float16.npy" if compress_features else "features_float32.npy"
    features_data = data_np["features"].astype(np.float16 if compress_features else np.float32)
    np.save(output_dir / features_file, features_data)
    if compress_features:
        console.print(f"[green]✓ Saved compressed features (float16): {output_dir}/{features_file}")
        console.print(f"[dim]Original size: {data_np['features'].nbytes / 1e6:.1f}MB, Compressed: {features_data.nbytes / 1e6:.1f}MB")
    else:
        console.print(f"[green]✓ Saved full precision features: {output_dir}/{features_file}")

    pca_data = {
        'projection_matrix': pca_proj.cpu().numpy(),
        'min_values': pca_min.cpu().numpy(),
        'max_values': pca_max.cpu().numpy(),
        'feature_dim': features.shape[-1],
        'num_points': len(points)
    }
    with open(output_dir / "pca_params.pkl", 'wb') as f:
        pickle.dump(pca_data, f)
    console.print(f"[green]✓ Saved PCA parameters: {output_dir}/pca_params.pkl")

    metadata = {
        'num_points': len(points),
        'feature_dim': features.shape[-1],
        'bbox_min': data_np["points"].min(axis=0).tolist(),
        'bbox_max': data_np["points"].max(axis=0).tolist(),
        'feature_range': [float(data_np["features"].min()), float(data_np["features"].max())],
        'compressed_features': compress_features,
        'files': {
            'rgb_pointcloud': 'pointcloud_rgb.ply',
            'pca_pointcloud': 'pointcloud_feature_pca.ply',
            'features': features_file,
            'points': 'points.npy',
            'pca_params': 'pca_params.pkl',
            **additional_files
        },
        'additional_outputs': list(additional_outputs.keys())
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    console.print(f"[green]✓ Saved metadata: {output_dir}/metadata.json")

    total_size = sum((output_dir / f).stat().st_size for f in metadata['files'].values() if f is not None) / 1e6
    console.print(f"\n[bold green]Export Summary:")
    console.print(f"  Points: {len(points):,}")
    console.print(f"  Feature dimension: {features.shape[-1]}")
    console.print(f"  Additional outputs: {list(additional_outputs.keys())}")
    console.print(f"  Total size: {total_size:.1f}MB")
    console.print(f"  Files saved in: {output_dir}")


def export_feature_pointcloud(config_path: Path, output_dir: Path, num_points: int = 1000000, compress_features: bool = True) -> None:
    """Export F3RM pointcloud with RGB, features, and feature_PCA."""
    console.print(f"[bold blue]F3RM Feature Pointcloud Exporter")
    console.print(f"[bold blue]Config: {config_path}")
    console.print(f"[bold blue]Output: {output_dir}")
    console.print(f"[bold blue]Target points: {num_points:,}")
    console.print(f"[bold green]Loading F3RM model...")
    config, pipeline, checkpoint_path, step = eval_setup(config_path=config_path, test_mode="test")
    console.print(f"[bold green]Loaded checkpoint from step {step}")
    pipeline.eval()
    points, rgbs, features, feature_pca, pca_proj, pca_min, pca_max, additional_outputs = sample_feature_pointcloud(
        pipeline, num_points, bbox_bounds=(-1.0, 1.0)
    )
    save_pointcloud_data(output_dir, points, rgbs, features, feature_pca, pca_proj, pca_min, pca_max, additional_outputs, compress_features=compress_features)
    console.print(f"[bold green]✓ Export completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export F3RM feature pointclouds")
    parser.add_argument("--config", type=Path, required=True, help="Path to F3RM model config.yml")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--num-points", type=int, default=5000000, help="Number of points to sample")
    parser.add_argument("--no-compress", action="store_true", help="Use float32 instead of float16 features")
    args = parser.parse_args()
    export_feature_pointcloud(config_path=args.config, output_dir=args.output_dir, num_points=args.num_points, compress_features=not args.no_compress)
