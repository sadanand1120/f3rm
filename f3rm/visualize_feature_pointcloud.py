#!/usr/bin/env python3
"""
F3RM Feature Pointcloud Visualizer

This script provides interactive visualization of F3RM pointclouds with:
- RGB/feature_PCA switching capability  
- Open-vocabulary semantic similarity queries
- Interactive filtering and selection

Usage:
    python visualize_feature_pointcloud.py --data-dir path/to/exported/pointcloud/
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
        # Filter by threshold if provided
        if threshold is not None:
            mask = similarities > threshold
            points = points[mask]
            similarities = similarities[mask]

        # Normalize similarities to [0, 1]
        sim_norm = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)

        # Apply colormap
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        colors = cmap(sim_norm)[:, :3]  # Remove alpha channel

        # Create pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def find_similar_regions(
        self,
        features: np.ndarray,
        points: np.ndarray,
        text_query: str,
        negative_queries: Optional[List[str]] = None,
        threshold: float = 0.5,
        min_cluster_size: int = 10
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Find regions in the pointcloud that match a text query.

        Args:
            features: Feature vectors (N, feature_dim)
            points: Point coordinates (N, 3)
            text_query: Positive text query
            negative_queries: Optional negative queries for contrast
            threshold: Similarity threshold
            min_cluster_size: Minimum points per cluster

        Returns:
            Tuple of (similarities, list of cluster point indices)
        """
        # Prepare text queries
        queries = [text_query]
        if negative_queries:
            queries.extend(negative_queries)

        # Compute similarities
        similarities = self.compute_text_similarities(
            features, queries,
            has_negatives=len(queries) > 1
        )

        # Filter by threshold
        high_sim_mask = similarities > threshold
        high_sim_points = points[high_sim_mask]

        if len(high_sim_points) < min_cluster_size:
            console.print(f"[yellow]Warning: Only {len(high_sim_points)} points above threshold {threshold}")
            return similarities, []

        # Cluster similar points
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(eps=0.05, min_samples=min_cluster_size)
        cluster_labels = clustering.fit_predict(high_sim_points)

        # Extract clusters
        clusters = []
        high_sim_indices = np.where(high_sim_mask)[0]

        for cluster_id in range(cluster_labels.max() + 1):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = high_sim_indices[cluster_mask]
            if len(cluster_indices) >= min_cluster_size:
                clusters.append(cluster_indices)

        console.print(f"[green]Found {len(clusters)} clusters matching '{text_query}'")

        return similarities, clusters


class InteractivePointcloudVisualizer:
    """Interactive pointcloud visualizer with RGB/PCA switching."""

    def __init__(self, data: FeaturePointcloudData):
        self.data = data
        self.semantic_utils = SemanticSimilarityUtils()
        self.vis = None
        self.current_mode = "rgb"
        self.current_query = None
        self.current_similarities = None

    def start_visualization(self):
        """Start the interactive visualization."""
        console.print(self.data.get_info())
        console.print("[bold green]Starting interactive pointcloud visualizer...")
        console.print("[bold blue]Controls:")
        console.print("  'r' - Switch to RGB mode")
        console.print("  'p' - Switch to PCA mode")
        console.print("  's' - Perform semantic query")
        console.print("  'q' - Quit")

        # Initialize Open3D visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="F3RM Feature Pointcloud Visualizer", width=1200, height=800)

        # Register key callbacks
        self.vis.register_key_callback(ord('R'), self._switch_to_rgb)
        self.vis.register_key_callback(ord('P'), self._switch_to_pca)
        self.vis.register_key_callback(ord('S'), self._semantic_query)
        self.vis.register_key_callback(ord('Q'), self._quit)

        # Start with RGB pointcloud
        self._switch_to_rgb(self.vis)

        # Run visualization loop
        self.vis.run()
        self.vis.destroy_window()

    def _switch_to_rgb(self, vis):
        """Switch to RGB visualization mode."""
        if self.current_mode != "rgb":
            vis.clear_geometries()
            vis.add_geometry(self.data.rgb_pointcloud)
            self.current_mode = "rgb"
            console.print("[green]Switched to RGB mode")
        return False

    def _switch_to_pca(self, vis):
        """Switch to PCA visualization mode."""
        if self.current_mode != "pca":
            vis.clear_geometries()
            vis.add_geometry(self.data.pca_pointcloud)
            self.current_mode = "pca"
            console.print("[green]Switched to PCA mode")
        return False

    def _semantic_query(self, vis):
        """Perform semantic similarity query."""
        console.print("\n[bold blue]Semantic Query Mode")

        # Get query from user
        positive_query = input("Enter positive query (e.g., 'chair'): ").strip()
        if not positive_query:
            console.print("[yellow]No query entered, returning to visualization")
            return False

        negative_input = input("Enter negative queries (comma-separated, optional): ").strip()
        negative_queries = [q.strip() for q in negative_input.split(",")] if negative_input else ["object"]

        threshold = 0.5
        threshold_input = input(f"Enter similarity threshold (default {threshold}): ").strip()
        if threshold_input:
            try:
                threshold = float(threshold_input)
            except ValueError:
                console.print(f"[yellow]Invalid threshold, using default {threshold}")

        try:
            # Compute similarities
            console.print(f"[yellow]Computing similarities for '{positive_query}'...")
            queries = [positive_query] + negative_queries
            similarities = self.semantic_utils.compute_text_similarities(
                self.data.features, queries, has_negatives=len(negative_queries) > 0
            )

            # Create similarity pointcloud
            sim_pcd = self.semantic_utils.create_similarity_pointcloud(
                self.data.points, similarities, threshold=threshold
            )

            # Update visualization
            vis.clear_geometries()
            vis.add_geometry(sim_pcd)
            self.current_mode = "semantic"
            self.current_query = positive_query
            self.current_similarities = similarities

            # Print statistics
            above_thresh = (similarities > threshold).sum()
            console.print(f"[green]✓ Query: '{positive_query}'")
            console.print(f"[green]  Points above threshold {threshold}: {above_thresh:,} / {len(similarities):,}")
            console.print(f"[green]  Similarity range: {similarities.min():.3f} - {similarities.max():.3f}")

            # Find clusters
            _, clusters = self.semantic_utils.find_similar_regions(
                self.data.features, self.data.points, positive_query,
                negative_queries, threshold
            )

            if clusters:
                console.print(f"[green]  Found {len(clusters)} distinct regions")
                for i, cluster in enumerate(clusters):
                    center = self.data.points[cluster].mean(axis=0)
                    console.print(f"    Cluster {i+1}: {len(cluster)} points at {center}")

        except Exception as e:
            console.print(f"[red]Error during semantic query: {e}")

        return False

    def _quit(self, vis):
        """Quit the visualizer."""
        console.print("[yellow]Quitting visualizer...")
        vis.close()
        return True


def demonstrate_semantic_similarity(data: FeaturePointcloudData, queries: List[str]):
    """Demonstrate semantic similarity capabilities."""
    console.print("[bold blue]Demonstrating semantic similarity capabilities...")

    semantic_utils = SemanticSimilarityUtils()

    for query in queries:
        console.print(f"\n[bold yellow]Query: '{query}'")

        # Default negative queries
        negative_queries = ["object", "background", "floor"]

        # Compute similarities
        similarities = semantic_utils.compute_text_similarities(
            data.features, [query] + negative_queries, has_negatives=True
        )

        # Statistics
        threshold = 0.5
        above_thresh = (similarities > threshold).sum()

        console.print(f"  Similarity range: {similarities.min():.3f} - {similarities.max():.3f}")
        console.print(f"  Points above threshold {threshold}: {above_thresh:,} / {len(similarities):,}")

        # Find clusters
        _, clusters = semantic_utils.find_similar_regions(
            data.features, data.points, query, negative_queries, threshold
        )

        if clusters:
            console.print(f"  Found {len(clusters)} distinct regions:")
            for i, cluster in enumerate(clusters):
                center = data.points[cluster].mean(axis=0)
                console.print(f"    Region {i+1}: {len(cluster)} points at [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

        # Save similarity pointcloud
        sim_pcd = semantic_utils.create_similarity_pointcloud(
            data.points, similarities, threshold=threshold
        )

        output_path = data.data_dir / f"semantic_query_{query.replace(' ', '_')}.ply"
        o3d.io.write_point_cloud(str(output_path), sim_pcd)
        console.print(f"  Saved similarity pointcloud: {output_path}")


def visualize_feature_pointcloud(
    data_dir: Path,
    interactive: bool = True,
    demo_queries: Optional[List[str]] = None,
) -> None:
    """
    Main function to visualize feature pointclouds.

    Args:
        data_dir: Directory containing exported pointcloud data
        interactive: Whether to start interactive visualization
        demo_queries: Optional list of queries to demonstrate
    """
    console.print(f"[bold blue]F3RM Feature Pointcloud Visualizer")
    console.print(f"[bold blue]Data directory: {data_dir}")

    # Load pointcloud data
    try:
        data = FeaturePointcloudData(data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")
    except Exception as e:
        console.print(f"[red]Error loading pointcloud data: {e}")
        return

    # Demonstrate semantic similarity if queries provided
    if demo_queries:
        demonstrate_semantic_similarity(data, demo_queries)

    # Start interactive visualization
    if interactive:
        visualizer = InteractivePointcloudVisualizer(data)
        visualizer.start_visualization()
    else:
        console.print(data.get_info())


def main():
    parser = argparse.ArgumentParser(description="Visualize F3RM feature pointclouds")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing exported pointcloud data")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive visualization")
    parser.add_argument("--demo-queries", nargs="*", help="Demonstration queries for semantic similarity")

    args = parser.parse_args()

    # Validate inputs
    if not args.data_dir.exists():
        console.print(f"[bold red]Data directory not found: {args.data_dir}")
        return

    # Default demo queries if none provided
    demo_queries = args.demo_queries or ["chair", "table", "lamp", "book"]

    try:
        visualize_feature_pointcloud(
            data_dir=args.data_dir,
            interactive=not args.no_interactive,
            demo_queries=demo_queries if not args.no_interactive else None,
        )
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
