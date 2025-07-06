#!/usr/bin/env python3
"""
F3RM Feature Pointcloud Visualizer

This script provides visualization of F3RM pointclouds with:
- RGB/feature_PCA modes via command line flags
- Open-vocabulary semantic similarity queries
- Simple and reliable Open3D visualization

Usage:
    python visualize_feature_pointcloud.py --data-dir path/to/exported/pointcloud/ --mode rgb
    python visualize_feature_pointcloud.py --data-dir path/to/exported/pointcloud/ --mode pca
    python visualize_feature_pointcloud.py --data-dir path/to/exported/pointcloud/ --mode semantic --query "chair"
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
        softmax_temp: float = 1.0
    ) -> np.ndarray:
        """
        Compute similarity between pointcloud features and text queries.

        Args:
            features: Feature vectors (N, feature_dim)
            text_queries: List of text queries. First is positive, rest are negatives
            has_negatives: Whether to use contrastive computation
            softmax_temp: Temperature for softmax

        Returns:
            Similarity scores (N,) for each point
        """
        self._load_clip_model()

        # Convert features to torch
        features_torch = torch.from_numpy(features).to(self.device)

        # Encode text queries
        text_tokens = self.tokenizer(text_queries).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)

        # Compute similarities using the same method as in opt.py
        similarities = compute_similarity_text2vis(
            features_torch, text_features,
            has_negatives=has_negatives,
            softmax_temp=softmax_temp
        )

        return similarities.cpu().numpy().squeeze()

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


def create_reference_geometries(data: FeaturePointcloudData) -> List[o3d.geometry.Geometry]:
    """Create all reference geometries (coordinate frame, bounding box, grid)."""
    geometries = []

    # Get bounding box from metadata
    bbox_min = np.array(data.metadata['bbox_min'])
    bbox_max = np.array(data.metadata['bbox_max'])

    # Coordinate frame at origin
    geometries.extend(create_coordinate_frame(size=0.1))

    # Bounding box wireframe
    geometries.extend(create_bounding_box_lines(bbox_min, bbox_max))

    # Grid lines for spatial reference
    geometries.extend(create_grid_lines(bbox_min, bbox_max, grid_size=0.2))

    return geometries


def visualize_rgb(data: FeaturePointcloudData, show_guides: bool = True):
    """Visualize RGB pointcloud."""
    console.print("[bold green]Visualizing RGB pointcloud...")
    console.print(data.get_info())

    pcd = data.rgb_pointcloud
    console.print(f"[green]Loaded pointcloud with {len(pcd.points)} points")

    # Start with pointcloud
    all_geometries = [pcd]

    # Add reference geometries if requested
    if show_guides:
        reference_geoms = create_reference_geometries(data)
        all_geometries.extend(reference_geoms)
        console.print("[dim]Reference guides: Red=X, Green=Y, Blue=Z axes; Gray=bounding box; Light gray=grid")

    # Simple visualization
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name="F3RM RGB Pointcloud",
        width=1200,
        height=800,
        left=50,
        top=50
    )


def visualize_pca(data: FeaturePointcloudData, show_guides: bool = True):
    """Visualize PCA feature pointcloud."""
    console.print("[bold green]Visualizing PCA feature pointcloud...")
    console.print(data.get_info())

    pcd = data.pca_pointcloud
    console.print(f"[green]Loaded pointcloud with {len(pcd.points)} points")

    # Start with pointcloud
    all_geometries = [pcd]

    # Add reference geometries if requested
    if show_guides:
        reference_geoms = create_reference_geometries(data)
        all_geometries.extend(reference_geoms)
        console.print("[dim]Reference guides: Red=X, Green=Y, Blue=Z axes; Gray=bounding box; Light gray=grid")

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
    show_guides: bool = True
):
    """Visualize semantic similarity pointcloud."""
    console.print(f"[bold green]Visualizing semantic similarity for query: '{query}'")
    console.print(data.get_info())

    if negative_queries is None:
        negative_queries = ["object"]  # Same default as opt.py

    # Initialize semantic utils
    semantic_utils = SemanticSimilarityUtils()

    # Compute similarities
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

    # Create similarity pointcloud
    sim_pcd = semantic_utils.create_similarity_pointcloud(
        data.points, similarities, threshold=threshold
    )

    console.print(f"[green]Created similarity pointcloud with {len(sim_pcd.points)} points")

    # Start with pointcloud
    all_geometries = [sim_pcd]

    # Add reference geometries if requested
    if show_guides:
        reference_geoms = create_reference_geometries(data)
        all_geometries.extend(reference_geoms)
        console.print("[dim]Reference guides: Red=X, Green=Y, Blue=Z axes; Gray=bounding box; Light gray=grid")

    # Save if requested
    if save_result:
        output_path = data.data_dir / f"semantic_query_{query.replace(' ', '_')}.ply"
        o3d.io.write_point_cloud(str(output_path), sim_pcd)
        console.print(f"[green]Saved similarity pointcloud: {output_path}")

    # Simple visualization
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name=f"F3RM Semantic Similarity: '{query}'",
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

    args = parser.parse_args()

    # Validate inputs
    if not args.data_dir.exists():
        console.print(f"[bold red]Data directory not found: {args.data_dir}")
        return

    if args.mode == "semantic" and not args.query:
        console.print("[bold red]--query is required for semantic mode")
        return

    console.print(f"[bold blue]F3RM Feature Pointcloud Visualizer")
    console.print(f"[bold blue]Data directory: {args.data_dir}")
    console.print(f"[bold blue]Mode: {args.mode}")

    try:
        # Load pointcloud data
        data = FeaturePointcloudData(args.data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")

        show_guides = not args.no_guides

        # Visualize based on mode
        if args.mode == "rgb":
            visualize_rgb(data, show_guides)
        elif args.mode == "pca":
            visualize_pca(data, show_guides)
        elif args.mode == "semantic":
            visualize_semantic(
                data,
                args.query,
                args.negative_queries,
                args.threshold,
                args.softmax_temp,
                args.save,
                show_guides
            )

    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
