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


def visualize_rgb(data: FeaturePointcloudData):
    """Visualize RGB pointcloud."""
    console.print("[bold green]Visualizing RGB pointcloud...")
    console.print(data.get_info())

    pcd = data.rgb_pointcloud
    console.print(f"[green]Loaded pointcloud with {len(pcd.points)} points")

    # Simple visualization
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="F3RM RGB Pointcloud",
        width=1200,
        height=800,
        left=50,
        top=50
    )


def visualize_pca(data: FeaturePointcloudData):
    """Visualize PCA feature pointcloud."""
    console.print("[bold green]Visualizing PCA feature pointcloud...")
    console.print(data.get_info())

    pcd = data.pca_pointcloud
    console.print(f"[green]Loaded pointcloud with {len(pcd.points)} points")

    # Simple visualization
    o3d.visualization.draw_geometries(
        [pcd],
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
    threshold: Optional[float] = None,
    save_result: bool = False
):
    """Visualize semantic similarity pointcloud."""
    console.print(f"[bold green]Visualizing semantic similarity for query: '{query}'")
    console.print(data.get_info())

    if negative_queries is None:
        negative_queries = ["object", "background", "floor"]

    # Initialize semantic utils
    semantic_utils = SemanticSimilarityUtils()

    # Compute similarities
    console.print(f"[yellow]Computing similarities for '{query}'...")
    queries = [query] + negative_queries
    similarities = semantic_utils.compute_text_similarities(
        data.features, queries, has_negatives=len(negative_queries) > 0
    )

    # Compute adaptive threshold if not provided
    if threshold is None:
        sim_mean = similarities.mean()
        sim_std = similarities.std()
        threshold = min(sim_mean + 0.5 * sim_std, similarities.max() * 0.8)
        threshold = max(threshold, sim_mean)

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

    # Save if requested
    if save_result:
        output_path = data.data_dir / f"semantic_query_{query.replace(' ', '_')}.ply"
        o3d.io.write_point_cloud(str(output_path), sim_pcd)
        console.print(f"[green]Saved similarity pointcloud: {output_path}")

    # Simple visualization
    o3d.visualization.draw_geometries(
        [sim_pcd],
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
    parser.add_argument("--negative-queries", nargs="*", help="Negative queries for semantic mode")
    parser.add_argument("--threshold", type=float, help="Similarity threshold for semantic mode (auto if not specified)")
    parser.add_argument("--save", action="store_true", help="Save semantic similarity results")

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

        # Visualize based on mode
        if args.mode == "rgb":
            visualize_rgb(data)
        elif args.mode == "pca":
            visualize_pca(data)
        elif args.mode == "semantic":
            visualize_semantic(
                data,
                args.query,
                args.negative_queries,
                args.threshold,
                args.save
            )

    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
