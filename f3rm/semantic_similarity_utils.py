#!/usr/bin/env python3
"""
Semantic Similarity Utilities for F3RM Feature Pointclouds

This module provides utility functions for open-vocabulary semantic similarity
analysis on F3RM feature pointclouds, following the approach used in opt.py.

Usage:
    from f3rm.semantic_similarity_utils import SemanticPointcloudAnalyzer
    
    analyzer = SemanticPointcloudAnalyzer(features, points)
    similarities = analyzer.query_similarity("chair", negatives=["object"])
    clusters = analyzer.find_object_instances("magazine", threshold=0.502)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from rich.console import Console
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

import open_clip
from f3rm.features.clip_extract import CLIPArgs
from f3rm.minimal.utils import compute_similarity_text2vis

console = Console()


class SemanticPointcloudAnalyzer:
    """
    Semantic analysis utilities for F3RM feature pointclouds.

    This class provides methods for open-vocabulary queries, object detection,
    and spatial clustering based on CLIP features, following the approach
    demonstrated in opt.py.
    """

    def __init__(
        self,
        features: np.ndarray,
        points: np.ndarray,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the semantic analyzer.

        Args:
            features: Feature vectors (N, feature_dim)
            points: Point coordinates (N, 3)
            device: PyTorch device for computations
        """
        self.features = features
        self.points = points
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy loading for CLIP model
        self._clip_model = None
        self._tokenizer = None

        # Cache for computed similarities
        self._similarity_cache = {}

    def _load_clip_model(self):
        """Load CLIP model if not already loaded."""
        if self._clip_model is None:
            console.print("[yellow]Loading CLIP model...")
            self._clip_model, _, _ = open_clip.create_model_and_transforms(
                CLIPArgs.model_name,
                pretrained=CLIPArgs.model_pretrained,
                device=self.device
            )
            self._clip_model.eval()
            self._tokenizer = open_clip.get_tokenizer(CLIPArgs.model_name)
            console.print("[green]✓ CLIP model loaded")

    def query_similarity(
        self,
        positive_query: str,
        negatives: Optional[List[str]] = None,
        softmax_temp: float = 1.0,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Compute similarity scores for a text query using the same method as in opt.py.

        Args:
            positive_query: Main query text (e.g., "magazine")
            negatives: Negative queries for contrastive learning (default: ["object"])
            softmax_temp: Temperature for softmax scaling
            use_cache: Whether to use cached results

        Returns:
            Similarity scores (N,) for each point

        Example:
            # Following the opt.py approach:
            similarities = analyzer.query_similarity("magazine", negatives=["object"])
            mask = similarities > 0.502  # Same threshold as in opt.py
        """
        # Use default negatives if none provided (same as in opt.py)
        if negatives is None:
            negatives = ["object"]

        # Create cache key
        cache_key = f"{positive_query}|{','.join(negatives)}|{softmax_temp}"

        if use_cache and cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        self._load_clip_model()

        # Prepare text queries (positive first, then negatives)
        text_queries = [positive_query] + negatives

        # Convert features to torch
        features_torch = torch.from_numpy(self.features).to(self.device)

        # Encode text queries
        text_tokens = self._tokenizer(text_queries).to(self.device)
        with torch.no_grad():
            text_features = self._clip_model.encode_text(text_tokens)

        # Compute similarities using the same method as in opt.py
        has_negatives = len(negatives) > 0
        similarities = compute_similarity_text2vis(
            features_torch,
            text_features,
            has_negatives=has_negatives,
            softmax_temp=softmax_temp
        )

        result = similarities.cpu().numpy().squeeze()

        if use_cache:
            self._similarity_cache[cache_key] = result

        return result

    def find_object_instances(
        self,
        query: str,
        threshold: float = 0.502,  # Same threshold as used in opt.py
        negatives: Optional[List[str]] = None,
        min_cluster_size: int = 10,
        eps: float = 0.05,
        return_details: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Dict]]:
        """
        Find object instances matching a query, following the opt.py approach.

        Args:
            query: Object query (e.g., "magazine")
            threshold: Similarity threshold (0.502 as used in opt.py)
            negatives: Negative queries for contrast
            min_cluster_size: Minimum points per cluster
            eps: DBSCAN clustering radius
            return_details: Whether to return detailed analysis

        Returns:
            List of cluster point indices, optionally with details dict

        Example:
            # Find magazine instances like in opt.py
            clusters = analyzer.find_object_instances("magazine", threshold=0.502)
            for i, cluster in enumerate(clusters):
                center = points[cluster].mean(axis=0)
                print(f"Magazine {i+1}: {len(cluster)} points at {center}")
        """
        # Compute similarities
        similarities = self.query_similarity(query, negatives)

        # Apply threshold (same as opt.py: magazine_mask = (sims > 0.502))
        high_sim_mask = similarities > threshold
        high_sim_points = self.points[high_sim_mask]

        if len(high_sim_points) < min_cluster_size:
            console.print(f"[yellow]Warning: Only {len(high_sim_points)} points above threshold {threshold}")
            return [] if not return_details else ([], {})

        # Cluster similar points using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size)
        cluster_labels = clustering.fit_predict(high_sim_points)

        # Extract clusters
        clusters = []
        high_sim_indices = np.where(high_sim_mask)[0]

        for cluster_id in range(cluster_labels.max() + 1):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = high_sim_indices[cluster_mask]
            if len(cluster_indices) >= min_cluster_size:
                clusters.append(cluster_indices)

        if not return_details:
            return clusters

        # Compute detailed analysis
        details = {
            'query': query,
            'threshold': threshold,
            'total_above_threshold': len(high_sim_points),
            'num_clusters': len(clusters),
            'similarity_stats': {
                'min': similarities.min(),
                'max': similarities.max(),
                'mean': similarities.mean(),
                'std': similarities.std()
            },
            'cluster_info': []
        }

        for i, cluster in enumerate(clusters):
            cluster_points = self.points[cluster]
            cluster_sims = similarities[cluster]

            cluster_info = {
                'id': i,
                'size': len(cluster),
                'center': cluster_points.mean(axis=0),
                'bbox_min': cluster_points.min(axis=0),
                'bbox_max': cluster_points.max(axis=0),
                'similarity_mean': cluster_sims.mean(),
                'similarity_std': cluster_sims.std()
            }
            details['cluster_info'].append(cluster_info)

        return clusters, details

    def compare_queries(
        self,
        queries: List[str],
        negatives: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> Dict[str, Dict]:
        """
        Compare multiple queries and their similarity patterns.

        Args:
            queries: List of queries to compare
            negatives: Shared negative queries
            threshold: Similarity threshold for analysis

        Returns:
            Dictionary with analysis for each query
        """
        if negatives is None:
            negatives = ["object", "background"]

        results = {}

        for query in queries:
            similarities = self.query_similarity(query, negatives)
            above_threshold = (similarities > threshold).sum()

            clusters, details = self.find_object_instances(
                query, threshold, negatives, return_details=True
            )

            results[query] = {
                'similarities': similarities,
                'above_threshold': above_threshold,
                'clusters': clusters,
                'details': details
            }

        return results

    def spatial_analysis(
        self,
        query: str,
        negatives: Optional[List[str]] = None,
        threshold: float = 0.5,
        grid_resolution: float = 0.1
    ) -> Dict:
        """
        Perform spatial analysis of query matches.

        Args:
            query: Query text
            negatives: Negative queries
            threshold: Similarity threshold
            grid_resolution: Spatial grid resolution

        Returns:
            Dictionary with spatial analysis results
        """
        similarities = self.query_similarity(query, negatives)
        high_sim_mask = similarities > threshold

        if not high_sim_mask.any():
            return {'empty': True}

        high_sim_points = self.points[high_sim_mask]
        high_sim_scores = similarities[high_sim_mask]

        # Compute spatial statistics
        bbox_min = high_sim_points.min(axis=0)
        bbox_max = high_sim_points.max(axis=0)
        volume = np.prod(bbox_max - bbox_min)
        density = len(high_sim_points) / volume if volume > 0 else 0

        # Create spatial grid
        grid_counts = {}
        for point, score in zip(high_sim_points, high_sim_scores):
            grid_idx = tuple((point / grid_resolution).astype(int))
            if grid_idx not in grid_counts:
                grid_counts[grid_idx] = {'count': 0, 'scores': []}
            grid_counts[grid_idx]['count'] += 1
            grid_counts[grid_idx]['scores'].append(score)

        # Summarize grid statistics
        grid_stats = {}
        for grid_idx, data in grid_counts.items():
            grid_stats[grid_idx] = {
                'count': data['count'],
                'mean_score': np.mean(data['scores']),
                'center': np.array(grid_idx) * grid_resolution
            }

        return {
            'query': query,
            'num_points': len(high_sim_points),
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'volume': volume,
            'density': density,
            'grid_resolution': grid_resolution,
            'grid_stats': grid_stats,
            'score_stats': {
                'min': high_sim_scores.min(),
                'max': high_sim_scores.max(),
                'mean': high_sim_scores.mean(),
                'std': high_sim_scores.std()
            }
        }

    def create_similarity_heatmap(
        self,
        query: str,
        negatives: Optional[List[str]] = None,
        resolution: Tuple[int, int, int] = (50, 50, 50)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a 3D similarity heatmap for visualization.

        Args:
            query: Query text
            negatives: Negative queries
            resolution: 3D grid resolution

        Returns:
            Tuple of (grid_coordinates, similarity_values)
        """
        similarities = self.query_similarity(query, negatives)

        # Create 3D grid
        bbox_min = self.points.min(axis=0)
        bbox_max = self.points.max(axis=0)

        x = np.linspace(bbox_min[0], bbox_max[0], resolution[0])
        y = np.linspace(bbox_min[1], bbox_max[1], resolution[1])
        z = np.linspace(bbox_min[2], bbox_max[2], resolution[2])

        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        grid_coords = np.stack([grid_x, grid_y, grid_z], axis=-1)

        # Interpolate similarities to grid
        from scipy.spatial import cKDTree

        tree = cKDTree(self.points)
        grid_flat = grid_coords.reshape(-1, 3)

        # Find nearest neighbors for each grid point
        distances, indices = tree.query(grid_flat, k=5)

        # Weight by inverse distance
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Interpolate similarities
        grid_similarities = (similarities[indices] * weights).sum(axis=1)
        grid_similarities = grid_similarities.reshape(resolution)

        return grid_coords, grid_similarities

    def export_semantic_analysis(
        self,
        queries: List[str],
        output_path: str,
        threshold: float = 0.5
    ) -> None:
        """
        Export comprehensive semantic analysis to file.

        Args:
            queries: List of queries to analyze
            output_path: Output file path
            threshold: Similarity threshold
        """
        import json
        from pathlib import Path

        results = {
            'metadata': {
                'num_points': len(self.points),
                'feature_dim': self.features.shape[1],
                'queries': queries,
                'threshold': threshold,
                'bbox_min': self.points.min(axis=0).tolist(),
                'bbox_max': self.points.max(axis=0).tolist()
            },
            'query_results': {}
        }

        for query in queries:
            console.print(f"[yellow]Analyzing query: '{query}'")

            # Basic similarity analysis
            similarities = self.query_similarity(query)

            # Find clusters
            clusters, details = self.find_object_instances(
                query, threshold, return_details=True
            )

            # Spatial analysis
            spatial_info = self.spatial_analysis(query, threshold=threshold)

            results['query_results'][query] = {
                'similarities_stats': {
                    'min': float(similarities.min()),
                    'max': float(similarities.max()),
                    'mean': float(similarities.mean()),
                    'std': float(similarities.std())
                },
                'clusters': details,
                'spatial_analysis': spatial_info
            }

        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        console.print(f"[green]✓ Semantic analysis exported to: {output_path}")


def demonstrate_semantic_analysis():
    """
    Demonstrate semantic analysis capabilities with example usage.
    """
    console.print("[bold blue]F3RM Semantic Similarity Utilities Demo")
    console.print("This demo shows how to use the semantic analysis tools.")
    console.print("\n[bold yellow]Example Usage:")

    example_code = '''
from f3rm.semantic_similarity_utils import SemanticPointcloudAnalyzer
import numpy as np

# Load your pointcloud data
points = np.load("points.npy")
features = np.load("features.npy")

# Create analyzer
analyzer = SemanticPointcloudAnalyzer(features, points)

# Query for objects (following opt.py approach)
similarities = analyzer.query_similarity("magazine", negatives=["object"])
magazine_mask = similarities > 0.502  # Same threshold as opt.py

# Find object instances
clusters = analyzer.find_object_instances("magazine", threshold=0.502)
for i, cluster in enumerate(clusters):
    center = points[cluster].mean(axis=0)
    print(f"Magazine {i+1}: {len(cluster)} points at {center}")

# Compare multiple queries
queries = ["chair", "table", "book", "magazine"]
results = analyzer.compare_queries(queries)

# Spatial analysis
spatial_info = analyzer.spatial_analysis("chair", threshold=0.5)
print(f"Chair density: {spatial_info['density']:.3f} points per unit volume")

# Export comprehensive analysis
analyzer.export_semantic_analysis(queries, "semantic_analysis.json")
'''

    console.print(example_code)

    console.print("\n[bold yellow]Key Features:")
    console.print("• Same method as opt.py: compute_similarity_text2vis with negatives")
    console.print("• Object instance detection with spatial clustering")
    console.print("• Multi-query comparison and analysis")
    console.print("• Spatial density and distribution analysis")
    console.print("• 3D similarity heatmaps for visualization")
    console.print("• Comprehensive export for further analysis")

    console.print("\n[bold green]Integration with F3RM:")
    console.print("• Compatible with exported feature pointclouds")
    console.print("• Uses same CLIP model and tokenizer as F3RM")
    console.print("• Follows opt.py approach for consistency")
    console.print("• Efficient caching for repeated queries")


if __name__ == "__main__":
    demonstrate_semantic_analysis()
