#!/usr/bin/env python3
"""
F3RM Feature Pointcloud Visualizer

This script provides visualization of F3RM pointclouds with:
- RGB/feature_PCA modes via command line flags
- Open-vocabulary semantic similarity queries
- Simple and reliable Open3D visualization

Usage:
    python f3rm/manual/visualize_feature_pointcloud.py --data-dir path/to/exported/pointcloud/ --mode rgb
    python f3rm/manual/visualize_feature_pointcloud.py --data-dir path/to/exported/pointcloud/ --mode pca
    python f3rm/manual/visualize_feature_pointcloud.py --data-dir path/to/exported/pointcloud/ --mode semantic --query "chair"
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from rich.console import Console

import open_clip
from f3rm.features.clip_extract import CLIPArgs
from f3rm.minimal.utils import compute_similarity_text2vis

console = Console()


class FeaturePointcloudData:
    """Container for feature pointcloud data with lazy loading."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.metadata = self._load_metadata()

        # Lazy loading containers
        self._points = None
        self._features = None
        self._pca_params = None
        self._rgb_pcd = None
        self._pca_pcd = None

    def _load_metadata(self) -> Dict:
        """Load metadata from exported pointcloud."""
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            return json.load(f)

    @property
    def points(self) -> np.ndarray:
        """Load points if not already loaded."""
        if self._points is None:
            points_path = self.data_dir / "points.npy"
            self._points = np.load(points_path)
        return self._points

    @property
    def features(self) -> np.ndarray:
        """Load features if not already loaded."""
        if self._features is None:
            features_file = self.metadata['files']['features']
            features_path = self.data_dir / features_file
            self._features = np.load(features_path)

            # Convert back to float32 if compressed
            if self.metadata['compressed_features']:
                self._features = self._features.astype(np.float32)
        return self._features

    @property
    def pca_params(self) -> Dict:
        """Load PCA parameters if not already loaded."""
        if self._pca_params is None:
            pca_path = self.data_dir / "pca_params.pkl"
            with open(pca_path, 'rb') as f:
                self._pca_params = pickle.load(f)
        return self._pca_params

    @property
    def rgb_pointcloud(self) -> o3d.geometry.PointCloud:
        """Load RGB pointcloud if not already loaded."""
        if self._rgb_pcd is None:
            rgb_path = self.data_dir / self.metadata['files']['rgb_pointcloud']
            self._rgb_pcd = o3d.io.read_point_cloud(str(rgb_path))
        return self._rgb_pcd

    @property
    def pca_pointcloud(self) -> o3d.geometry.PointCloud:
        """Load PCA pointcloud if not already loaded."""
        if self._pca_pcd is None:
            pca_path = self.data_dir / self.metadata['files']['pca_pointcloud']
            self._pca_pcd = o3d.io.read_point_cloud(str(pca_path))
        return self._pca_pcd

    def get_info(self) -> str:
        """Get formatted info about the pointcloud."""
        return f"""
F3RM Feature Pointcloud Info:
  Points: {self.metadata['num_points']:,}
  Feature dimension: {self.metadata['feature_dim']}
  Bounding box: {self.metadata['bbox_min']} to {self.metadata['bbox_max']}
  Compressed features: {self.metadata['compressed_features']}
  Data directory: {self.data_dir}
"""


class SemanticSimilarityUtils:
    """Utilities for open-vocabulary semantic similarity on pointclouds."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.tokenizer = None

    def _load_clip_model(self):
        """Lazy load CLIP model."""
        if self.clip_model is None:
            console.print("[yellow]Loading CLIP model...")
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                CLIPArgs.model_name,
                pretrained=CLIPArgs.model_pretrained,
                device=self.device
            )
            self.clip_model.eval()
            self.tokenizer = open_clip.get_tokenizer(CLIPArgs.model_name)
            console.print("[green]✓ CLIP model loaded")

    def compute_text_similarities(
        self,
        features: np.ndarray,
        text_queries: List[str],
        has_negatives: bool = True,
        softmax_temp: float = 1.0,
        chunk_size: int = 100000
    ) -> np.ndarray:
        """
        Compute similarity between pointcloud features and text queries.

        Args:
            features: Feature vectors (N, feature_dim)
            text_queries: List of text queries. First is positive, rest are negatives
            has_negatives: Whether to use contrastive computation
            softmax_temp: Temperature for softmax
            chunk_size: Number of features to process at once to avoid CUDA OOM

        Returns:
            Similarity scores (N,) for each point
        """
        self._load_clip_model()

        # Encode text queries once
        text_tokens = self.tokenizer(text_queries).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)

        # Process features in chunks to avoid CUDA memory issues
        n_points = features.shape[0]
        all_similarities = []

        console.print(f"[yellow]Processing {n_points:,} points in chunks of {chunk_size:,}...")

        for i in range(0, n_points, chunk_size):
            end_idx = min(i + chunk_size, n_points)
            chunk_features = features[i:end_idx]

            # Convert chunk to torch and move to device
            features_torch = torch.from_numpy(chunk_features).to(self.device)

            with torch.no_grad():
                # Compute similarities for this chunk
                chunk_similarities = compute_similarity_text2vis(
                    features_torch, text_features,
                    has_negatives=has_negatives,
                    softmax_temp=softmax_temp
                )

                # Move to CPU and store
                all_similarities.append(chunk_similarities.cpu().numpy())

            # Clear GPU memory for this chunk
            del features_torch
            torch.cuda.empty_cache()

            if (i // chunk_size + 1) % 10 == 0:
                console.print(f"[yellow]Processed {end_idx:,}/{n_points:,} points...")

        # Concatenate all chunks
        similarities = np.concatenate(all_similarities, axis=0)
        return similarities.squeeze()

    def create_similarity_pointcloud(
        self,
        points: np.ndarray,
        similarities: np.ndarray,
        colormap: str = "turbo",
        threshold: Optional[float] = None
    ) -> o3d.geometry.PointCloud:
        """
        Create a pointcloud colored by similarity scores.

        Args:
            points: Point coordinates (N, 3)
            similarities: Similarity scores (N,)
            colormap: Matplotlib colormap name
            threshold: Optional threshold to filter points

        Returns:
            Colored pointcloud
        """
        original_points = points.copy()
        original_similarities = similarities.copy()

        # Filter by threshold if provided
        if threshold is not None:
            mask = similarities > threshold
            if mask.sum() == 0:
                console.print(f"[yellow]Warning: No points above threshold {threshold}, showing all points")
                # Don't filter, show all points
                pass
            else:
                points = points[mask]
                similarities = similarities[mask]

        # Handle empty arrays
        if len(similarities) == 0:
            console.print("[yellow]Warning: No similarities to visualize")
            return o3d.geometry.PointCloud()

        # Normalize similarities to [0, 1]
        sim_min, sim_max = similarities.min(), similarities.max()
        if sim_max - sim_min < 1e-8:
            # All similarities are the same
            sim_norm = np.ones_like(similarities) * 0.5
        else:
            sim_norm = (similarities - sim_min) / (sim_max - sim_min)

        # Apply colormap
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        colors = cmap(sim_norm)[:, :3]  # Remove alpha channel

        # Create pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd


def create_coordinate_frame(size: float = 0.2) -> List[o3d.geometry.Geometry]:
    """Create coordinate frame with X(red), Y(green), Z(blue) axes at origin."""
    geometries = []

    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    geometries.append(coord_frame)

    return geometries


def create_bounding_box_lines(bbox_min: np.ndarray, bbox_max: np.ndarray) -> List[o3d.geometry.Geometry]:
    """Create wireframe bounding box visualization."""
    geometries = []

    # Create bounding box wireframe
    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    bbox_lines.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color
    geometries.append(bbox_lines)

    return geometries


def create_grid_lines(bbox_min: np.ndarray, bbox_max: np.ndarray, grid_size: float = 0.5) -> List[o3d.geometry.Geometry]:
    """Create grid lines for spatial reference."""
    geometries = []

    # Create grid lines on XY plane at Z=0 (if Z=0 is within bounds)
    if bbox_min[2] <= 0 <= bbox_max[2]:
        points = []
        lines = []

        # Vertical lines (parallel to Y axis)
        x_coords = np.arange(
            np.ceil(bbox_min[0] / grid_size) * grid_size,
            bbox_max[0] + grid_size / 2,
            grid_size
        )
        for x in x_coords:
            if bbox_min[0] <= x <= bbox_max[0]:
                start_idx = len(points)
                points.extend([[x, bbox_min[1], 0], [x, bbox_max[1], 0]])
                lines.append([start_idx, start_idx + 1])

        # Horizontal lines (parallel to X axis)
        y_coords = np.arange(
            np.ceil(bbox_min[1] / grid_size) * grid_size,
            bbox_max[1] + grid_size / 2,
            grid_size
        )
        for y in y_coords:
            if bbox_min[1] <= y <= bbox_max[1]:
                start_idx = len(points)
                points.extend([[bbox_min[0], y, 0], [bbox_max[0], y, 0]])
                lines.append([start_idx, start_idx + 1])

        if points:
            grid = o3d.geometry.LineSet()
            grid.points = o3d.utility.Vector3dVector(points)
            grid.lines = o3d.utility.Vector2iVector(lines)
            grid.paint_uniform_color([0.9, 0.9, 0.9])  # Light gray
            geometries.append(grid)

    return geometries


def create_filter_bbox_lines(bbox_min: np.ndarray, bbox_max: np.ndarray) -> List[o3d.geometry.Geometry]:
    """Create wireframe bounding box visualization for filtering bounds in different color."""
    geometries = []

    # Create bounding box wireframe in different color (orange/yellow for filter bounds)
    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    bbox_lines.paint_uniform_color([1.0, 0.5, 0.0])  # Orange color for filter bounds
    geometries.append(bbox_lines)

    return geometries


def apply_bbox_filter(points: np.ndarray, colors: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply bounding box filter to points and colors."""
    # Create mask for points within bounding box
    within_bbox = np.all(
        (points >= bbox_min) & (points <= bbox_max),
        axis=-1
    )

    return points[within_bbox], colors[within_bbox]


def apply_semantic_filter(
    data: FeaturePointcloudData,
    points: np.ndarray,
    colors: np.ndarray,
    filter_query: str,
    filter_mode: str,  # 'filter-out' or 'filter-in'
    threshold: float = 0.502,
    softmax_temp: float = 1.0,
    negative_queries: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply semantic filtering to points and colors."""
    if negative_queries is None:
        negative_queries = ["object"]

    # Initialize semantic utils
    semantic_utils = SemanticSimilarityUtils()

    # Compute similarities
    console.print(f"[yellow]Computing semantic filter for '{filter_query}' (mode: {filter_mode}, threshold: {threshold})...")
    queries = [filter_query] + negative_queries
    similarities = semantic_utils.compute_text_similarities(
        data.features, queries,
        has_negatives=len(negative_queries) > 0,
        softmax_temp=softmax_temp
    )

    # Create mask based on filter mode
    above_thresh_mask = similarities > threshold

    if filter_mode == 'filter-out':
        # Remove points that match the query (keep points NOT matching)
        keep_mask = ~above_thresh_mask
        console.print(f"[cyan]Filter-out: Removing {above_thresh_mask.sum():,} points matching '{filter_query}'")
    elif filter_mode == 'filter-in':
        # Keep only points that match the query
        keep_mask = above_thresh_mask
        console.print(f"[cyan]Filter-in: Keeping only {above_thresh_mask.sum():,} points matching '{filter_query}'")
    else:
        raise ValueError(f"Invalid filter mode: {filter_mode}. Use 'filter-out' or 'filter-in'")

    return points[keep_mask], colors[keep_mask]


def create_reference_geometries(data: FeaturePointcloudData, filter_bbox_min: Optional[np.ndarray] = None, filter_bbox_max: Optional[np.ndarray] = None) -> List[o3d.geometry.Geometry]:
    """Create all reference geometries (coordinate frame, bounding box, grid, and optional filter bbox)."""
    geometries = []

    # Get bounding box from metadata
    bbox_min = np.array(data.metadata['bbox_min'])
    bbox_max = np.array(data.metadata['bbox_max'])

    # Coordinate frame at origin
    geometries.extend(create_coordinate_frame(size=0.1))

    # Original bounding box wireframe (gray)
    geometries.extend(create_bounding_box_lines(bbox_min, bbox_max))

    # Grid lines for spatial reference
    geometries.extend(create_grid_lines(bbox_min, bbox_max, grid_size=0.2))

    # Add filter bounding box if provided (orange)
    if filter_bbox_min is not None and filter_bbox_max is not None:
        geometries.extend(create_filter_bbox_lines(filter_bbox_min, filter_bbox_max))

    return geometries


def apply_filters(
    data: FeaturePointcloudData,
    points: np.ndarray,
    colors: np.ndarray,
    bbox_filter_min: Optional[List[float]] = None,
    bbox_filter_max: Optional[List[float]] = None,
    semantic_filter_query: Optional[str] = None,
    semantic_filter_mode: Optional[str] = None,
    semantic_threshold: float = 0.502,
    semantic_softmax_temp: float = 1.0,
    semantic_negatives: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply all requested filters to points and colors."""
    original_count = len(points)

    # Apply bbox filter if requested
    if bbox_filter_min is not None and bbox_filter_max is not None:
        bbox_min = np.array(bbox_filter_min)
        bbox_max = np.array(bbox_filter_max)
        points, colors = apply_bbox_filter(points, colors, bbox_min, bbox_max)
        console.print(f"[cyan]Bbox filter: {len(points):,} / {original_count:,} points remaining")

    # Apply semantic filter if requested
    if semantic_filter_query is not None and semantic_filter_mode is not None:
        points, colors = apply_semantic_filter(
            data, points, colors, semantic_filter_query, semantic_filter_mode,
            semantic_threshold, semantic_softmax_temp, semantic_negatives
        )
        console.print(f"[cyan]Semantic filter: {len(points):,} points after filtering")

    if len(points) == 0:
        console.print("[yellow]Warning: All points filtered out!")

    return points, colors


def visualize_rgb(
    data: FeaturePointcloudData,
    show_guides: bool = True,
    bbox_filter_min: Optional[List[float]] = None,
    bbox_filter_max: Optional[List[float]] = None,
    semantic_filter_query: Optional[str] = None,
    semantic_filter_mode: Optional[str] = None,
    semantic_threshold: float = 0.502,
    semantic_softmax_temp: float = 1.0,
    semantic_negatives: Optional[List[str]] = None
):
    """Visualize RGB pointcloud with optional filtering."""
    console.print("[bold green]Visualizing RGB pointcloud...")
    console.print(data.get_info())

    pcd = data.rgb_pointcloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    console.print(f"[green]Loaded pointcloud with {len(points)} points")

    # Apply filters if requested
    points, colors = apply_filters(
        data, points, colors, bbox_filter_min, bbox_filter_max,
        semantic_filter_query, semantic_filter_mode, semantic_threshold,
        semantic_softmax_temp, semantic_negatives
    )

    # Create filtered pointcloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Start with filtered pointcloud
    all_geometries = [filtered_pcd]

    # Add reference geometries if requested
    if show_guides:
        filter_bbox_min = np.array(bbox_filter_min) if bbox_filter_min is not None else None
        filter_bbox_max = np.array(bbox_filter_max) if bbox_filter_max is not None else None
        reference_geoms = create_reference_geometries(data, filter_bbox_min, filter_bbox_max)
        all_geometries.extend(reference_geoms)

        guide_msg = "[dim]Reference guides: Red=X, Green=Y, Blue=Z axes; Gray=original bounds; Light gray=grid"
        if filter_bbox_min is not None and filter_bbox_max is not None:
            guide_msg += "; Orange=filter bounds"
        console.print(guide_msg)

    # Simple visualization
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name="F3RM RGB Pointcloud",
        width=1200,
        height=800,
        left=50,
        top=50
    )


def visualize_pca(
    data: FeaturePointcloudData,
    show_guides: bool = True,
    bbox_filter_min: Optional[List[float]] = None,
    bbox_filter_max: Optional[List[float]] = None,
    semantic_filter_query: Optional[str] = None,
    semantic_filter_mode: Optional[str] = None,
    semantic_threshold: float = 0.502,
    semantic_softmax_temp: float = 1.0,
    semantic_negatives: Optional[List[str]] = None
):
    """Visualize PCA feature pointcloud with optional filtering."""
    console.print("[bold green]Visualizing PCA feature pointcloud...")
    console.print(data.get_info())

    pcd = data.pca_pointcloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    console.print(f"[green]Loaded pointcloud with {len(points)} points")

    # Apply filters if requested
    points, colors = apply_filters(
        data, points, colors, bbox_filter_min, bbox_filter_max,
        semantic_filter_query, semantic_filter_mode, semantic_threshold,
        semantic_softmax_temp, semantic_negatives
    )

    # Create filtered pointcloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Start with filtered pointcloud
    all_geometries = [filtered_pcd]

    # Add reference geometries if requested
    if show_guides:
        filter_bbox_min = np.array(bbox_filter_min) if bbox_filter_min is not None else None
        filter_bbox_max = np.array(bbox_filter_max) if bbox_filter_max is not None else None
        reference_geoms = create_reference_geometries(data, filter_bbox_min, filter_bbox_max)
        all_geometries.extend(reference_geoms)

        guide_msg = "[dim]Reference guides: Red=X, Green=Y, Blue=Z axes; Gray=original bounds; Light gray=grid"
        if filter_bbox_min is not None and filter_bbox_max is not None:
            guide_msg += "; Orange=filter bounds"
        console.print(guide_msg)

    # Simple visualization
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name="F3RM PCA Feature Pointcloud",
        width=1200,
        height=800,
        left=50,
        top=50
    )


def visualize_semantic(
    data: FeaturePointcloudData,
    query: str,
    negative_queries: Optional[List[str]] = None,
    threshold: float = 0.502,  # Same default as opt.py
    softmax_temp: float = 1.0,  # Same default as opt.py
    save_result: bool = False,
    show_guides: bool = True,
    bbox_filter_min: Optional[List[float]] = None,
    bbox_filter_max: Optional[List[float]] = None,
    semantic_filter_query: Optional[str] = None,
    semantic_filter_mode: Optional[str] = None,
    semantic_threshold: float = 0.502,
    semantic_softmax_temp: float = 1.0,
    semantic_negatives: Optional[List[str]] = None,
    background_alpha: float = 0.3
):
    """Visualize semantic similarity pointcloud with RGB background and optional filtering."""
    console.print(f"[bold green]Visualizing semantic similarity for query: '{query}'")
    console.print(data.get_info())

    if negative_queries is None:
        negative_queries = ["object"]  # Same default as opt.py

    # Initialize semantic utils
    semantic_utils = SemanticSimilarityUtils()

    # Compute similarities for the main query
    console.print(f"[yellow]Computing similarities for '{query}' with threshold {threshold}, softmax_temp {softmax_temp}...")
    queries = [query] + negative_queries
    similarities = semantic_utils.compute_text_similarities(
        data.features, queries,
        has_negatives=len(negative_queries) > 0,
        softmax_temp=softmax_temp
    )

    # Print statistics
    above_thresh = (similarities > threshold).sum()
    console.print(f"[cyan]Similarity statistics:")
    console.print(f"  Range: {similarities.min():.3f} - {similarities.max():.3f}")
    console.print(f"  Mean ± std: {similarities.mean():.3f} ± {similarities.std():.3f}")
    console.print(f"  Using threshold: {threshold:.3f}")
    console.print(f"  Points above threshold: {above_thresh:,} / {len(similarities):,}")

    # Get original RGB pointcloud for background
    rgb_pcd = data.rgb_pointcloud
    all_points = np.asarray(rgb_pcd.points)
    all_rgb_colors = np.asarray(rgb_pcd.colors)

    # Create mask for points above threshold
    above_threshold_mask = similarities > threshold

    # Create combined pointcloud with RGB background and semantic heatmap
    combined_points = all_points.copy()
    combined_colors = all_rgb_colors.copy()

    if background_alpha > 0:
        # Make RGB background transparent by reducing intensity
        combined_colors = combined_colors * background_alpha
        console.print(f"[cyan]Applied {background_alpha:.1f} transparency to RGB background")

    # Apply semantic heatmap to points above threshold
    if above_threshold_mask.any():
        # Get similarity scores for points above threshold
        above_thresh_similarities = similarities[above_threshold_mask]
        above_thresh_points = all_points[above_threshold_mask]

        # Create heatmap colors for points above threshold
        sim_min, sim_max = above_thresh_similarities.min(), above_thresh_similarities.max()
        if sim_max - sim_min < 1e-8:
            # All similarities are the same
            sim_norm = np.ones_like(above_thresh_similarities) * 0.5
        else:
            sim_norm = (above_thresh_similarities - sim_min) / (sim_max - sim_min)

        # Apply colormap for heatmap
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("turbo")
        heatmap_colors = cmap(sim_norm)[:, :3]  # Remove alpha channel

        # Replace colors for points above threshold with heatmap colors
        combined_colors[above_threshold_mask] = heatmap_colors

        console.print(f"[green]Applied semantic heatmap to {above_thresh_similarities.shape[0]:,} points above threshold")
    else:
        console.print("[yellow]No points above threshold - showing only RGB background")

    # Apply additional filters if requested (bbox and semantic filters)
    filtered_points, filtered_colors = apply_filters(
        data, combined_points, combined_colors, bbox_filter_min, bbox_filter_max,
        semantic_filter_query, semantic_filter_mode, semantic_threshold,
        semantic_softmax_temp, semantic_negatives
    )

    # Create final combined pointcloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    console.print(f"[green]Created combined pointcloud with {len(filtered_points):,} points")
    console.print(f"[cyan]Background alpha: {background_alpha:.1f} (0.0=invisible, 1.0=full RGB)")

    # Start with combined pointcloud
    all_geometries = [combined_pcd]

    # Add reference geometries if requested
    if show_guides:
        filter_bbox_min = np.array(bbox_filter_min) if bbox_filter_min is not None else None
        filter_bbox_max = np.array(bbox_filter_max) if bbox_filter_max is not None else None
        reference_geoms = create_reference_geometries(data, filter_bbox_min, filter_bbox_max)
        all_geometries.extend(reference_geoms)

        guide_msg = "[dim]Reference guides: Red=X, Green=Y, Blue=Z axes; Gray=original bounds; Light gray=grid"
        if filter_bbox_min is not None and filter_bbox_max is not None:
            guide_msg += "; Orange=filter bounds"
        console.print(guide_msg)

    # Save if requested (save the original similarity result, not the combined one)
    if save_result:
        # Create and save the pure similarity pointcloud (without RGB background)
        sim_pcd = semantic_utils.create_similarity_pointcloud(
            data.points, similarities, threshold=threshold
        )
        output_path = data.data_dir / f"semantic_query_{query.replace(' ', '_')}.ply"
        o3d.io.write_point_cloud(str(output_path), sim_pcd)
        console.print(f"[green]Saved pure similarity pointcloud: {output_path}")

        # Also save the combined version with RGB background
        combined_output_path = data.data_dir / f"semantic_query_{query.replace(' ', '_')}_with_background.ply"
        o3d.io.write_point_cloud(str(combined_output_path), combined_pcd)
        console.print(f"[green]Saved combined pointcloud with RGB background: {combined_output_path}")

    # Simple visualization
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name=f"F3RM Semantic Similarity: '{query}' (α={background_alpha:.1f})",
        width=1200,
        height=800,
        left=50,
        top=50
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize F3RM feature pointclouds")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing exported pointcloud data")
    parser.add_argument("--mode", choices=["rgb", "pca", "semantic"], default="rgb", help="Visualization mode")
    parser.add_argument("--query", type=str, help="Semantic query (required for semantic mode)")
    parser.add_argument("--negative-queries", nargs="*", help="Negative queries for semantic mode (default: ['object'])")
    parser.add_argument("--threshold", type=float, default=0.502, help="Similarity threshold for semantic mode (default: 0.502 like opt.py)")
    parser.add_argument("--softmax-temp", type=float, default=1.0, help="Softmax temperature for semantic mode (default: 1.0 like opt.py)")
    parser.add_argument("--save", action="store_true", help="Save semantic similarity results")
    parser.add_argument("--no-guides", action="store_true", help="Hide coordinate frame and reference guides")

    # Bounding box filtering arguments
    parser.add_argument("--bbox-filter-min", type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help="Filter: minimum bounding box coordinates (x y z) - only show points within this bbox")
    parser.add_argument("--bbox-filter-max", type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help="Filter: maximum bounding box coordinates (x y z) - only show points within this bbox")

    # Semantic filtering arguments
    parser.add_argument("--semantic-filter-query", type=str,
                        help="Filter: semantic query for filtering points (e.g., 'floor', 'wall')")
    parser.add_argument("--semantic-filter-mode", choices=["filter-out", "filter-in"],
                        help="Filter: 'filter-out' removes matching points, 'filter-in' keeps only matching points")
    parser.add_argument("--semantic-threshold", type=float, default=0.502,
                        help="Filter: similarity threshold for semantic filtering (default: 0.502)")
    parser.add_argument("--semantic-softmax-temp", type=float, default=1.0,
                        help="Filter: softmax temperature for semantic filtering (default: 1.0)")
    parser.add_argument("--semantic-negatives", nargs="*", default=["object"],
                        help="Filter: negative queries for semantic filtering (default: ['object'])")

    # Semantic mode background arguments
    parser.add_argument("--background-alpha", type=float, default=0.3,
                        help="Semantic mode: transparency of RGB background (0.0=invisible, 1.0=full RGB, default: 0.3)")

    args = parser.parse_args()

    # Validate inputs
    if not args.data_dir.exists():
        console.print(f"[bold red]Data directory not found: {args.data_dir}")
        return

    if args.mode == "semantic" and not args.query:
        console.print("[bold red]--query is required for semantic mode")
        return

    # Validate bbox filtering arguments
    if (args.bbox_filter_min is None) != (args.bbox_filter_max is None):
        console.print("[bold red]Both --bbox-filter-min and --bbox-filter-max must be provided together")
        return

    # Validate semantic filtering arguments
    if args.semantic_filter_query and not args.semantic_filter_mode:
        console.print("[bold red]--semantic-filter-mode is required when using --semantic-filter-query")
        return
    if args.semantic_filter_mode and not args.semantic_filter_query:
        console.print("[bold red]--semantic-filter-query is required when using --semantic-filter-mode")
        return

    console.print(f"[bold blue]F3RM Feature Pointcloud Visualizer")
    console.print(f"[bold blue]Data directory: {args.data_dir}")
    console.print(f"[bold blue]Mode: {args.mode}")

    # Print active filters
    if args.bbox_filter_min and args.bbox_filter_max:
        console.print(f"[bold yellow]Bbox filter: {args.bbox_filter_min} to {args.bbox_filter_max}")
    if args.semantic_filter_query and args.semantic_filter_mode:
        console.print(f"[bold yellow]Semantic filter: {args.semantic_filter_mode} '{args.semantic_filter_query}' (threshold: {args.semantic_threshold})")

    try:
        # Load pointcloud data
        data = FeaturePointcloudData(args.data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")

        show_guides = not args.no_guides

        # Visualize based on mode
        if args.mode == "rgb":
            visualize_rgb(
                data,
                show_guides,
                args.bbox_filter_min,
                args.bbox_filter_max,
                args.semantic_filter_query,
                args.semantic_filter_mode,
                args.threshold,
                args.softmax_temp,
                args.semantic_negatives
            )
        elif args.mode == "pca":
            visualize_pca(
                data,
                show_guides,
                args.bbox_filter_min,
                args.bbox_filter_max,
                args.semantic_filter_query,
                args.semantic_filter_mode,
                args.threshold,
                args.softmax_temp,
                args.semantic_negatives
            )
        elif args.mode == "semantic":
            visualize_semantic(
                data,
                args.query,
                args.negative_queries,
                args.threshold,
                args.softmax_temp,
                args.save,
                show_guides,
                args.bbox_filter_min,
                args.bbox_filter_max,
                args.semantic_filter_query,
                args.semantic_filter_mode,
                args.semantic_threshold,
                args.semantic_softmax_temp,
                args.semantic_negatives,
                args.background_alpha
            )

    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
