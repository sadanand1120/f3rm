#!/usr/bin/env python3
"""Semantic Similarity Utilities for F3RM Feature Pointclouds"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open_clip
import torch
from rich.console import Console
from sklearn.cluster import DBSCAN

from f3rm.features.clip_extract import CLIPArgs
from f3rm.minimal.utils import compute_similarity_text2vis

console = Console()


class SemanticSimilarityUtils:
    """Utilities for open-vocabulary semantic similarity on pointclouds."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.tokenizer = None

    def _load_clip_model(self):
        """Lazy load CLIP model."""
        if self.clip_model is None:
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                CLIPArgs.model_name,
                pretrained=CLIPArgs.model_pretrained,
                device=self.device
            )
            self.clip_model.eval()
            self.tokenizer = open_clip.get_tokenizer(CLIPArgs.model_name)

    def compute_text_similarities(
        self,
        features: np.ndarray,
        text_queries: List[str],
        has_negatives: bool = True,
        softmax_temp: float = 1.0,
        chunk_size: int = 100000
    ) -> np.ndarray:
        """Compute similarity between pointcloud features and text queries."""
        self._load_clip_model()
        text_tokens = self.tokenizer(text_queries).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        n_points = features.shape[0]
        all_similarities = []
        for i in range(0, n_points, chunk_size):
            end_idx = min(i + chunk_size, n_points)
            chunk_features = features[i:end_idx]
            features_torch = torch.from_numpy(chunk_features).to(self.device)
            with torch.no_grad():
                chunk_similarities = compute_similarity_text2vis(
                    features_torch, text_features,
                    has_negatives=has_negatives,
                    softmax_temp=softmax_temp
                )
                all_similarities.append(chunk_similarities.cpu().numpy())
            del features_torch
            torch.cuda.empty_cache()
        similarities = np.concatenate(all_similarities, axis=0)
        return similarities.squeeze()

    def create_similarity_pointcloud(
        self,
        points: np.ndarray,
        similarities: np.ndarray,
        colormap: str = "turbo",
        threshold: Optional[float] = None
    ) -> o3d.geometry.PointCloud:
        """Create a pointcloud colored by similarity scores."""
        if threshold is not None:
            mask = similarities > threshold
            if mask.sum() > 0:
                points = points[mask]
                similarities = similarities[mask]

        sim_min, sim_max = similarities.min(), similarities.max()
        if sim_max - sim_min < 1e-8:
            sim_norm = np.ones_like(similarities) * 0.5
        else:
            sim_norm = (similarities - sim_min) / (sim_max - sim_min)

        cmap = plt.get_cmap(colormap)
        colors = cmap(sim_norm)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd


class SemanticPointcloudAnalyzer:
    """Semantic analysis utilities for F3RM feature pointclouds."""

    def __init__(self, features: np.ndarray, points: np.ndarray, device: Optional[torch.device] = None):
        self.features = features
        self.points = points
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_utils = SemanticSimilarityUtils(device=self.device)
        self._similarity_cache = {}

    def query_similarity(self, positive_query: str, negatives: Optional[List[str]] = None, softmax_temp: float = 1.0, use_cache: bool = True) -> np.ndarray:
        """Compute similarity scores for a text query using the same method as in opt.py."""
        if negatives is None:
            negatives = ["object"]
        cache_key = f"{positive_query}|{','.join(negatives)}|{softmax_temp}"
        if use_cache and cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        text_queries = [positive_query] + negatives
        similarities = self.semantic_utils.compute_text_similarities(self.features, text_queries, has_negatives=len(negatives) > 0, softmax_temp=softmax_temp)
        if use_cache:
            self._similarity_cache[cache_key] = similarities
        return similarities

    def find_object_instances(self, query: str, threshold: float = 0.502, negatives: Optional[List[str]] = None, min_cluster_size: int = 10, eps: float = 0.05, softmax_temp: float = 1.0, return_details: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Dict]]:
        """Find object instances matching a query, following the opt.py approach."""
        similarities = self.query_similarity(query, negatives, softmax_temp)
        high_sim_mask = similarities > threshold
        high_sim_points = self.points[high_sim_mask]
        if len(high_sim_points) < min_cluster_size:
            console.print(f"[yellow]Warning: Only {len(high_sim_points)} points above threshold {threshold}")
            return [] if not return_details else ([], {})
        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size)
        cluster_labels = clustering.fit_predict(high_sim_points)
        clusters = []
        high_sim_indices = np.where(high_sim_mask)[0]
        for cluster_id in range(cluster_labels.max() + 1):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = high_sim_indices[cluster_mask]
            if len(cluster_indices) >= min_cluster_size:
                clusters.append(cluster_indices)
        if not return_details:
            return clusters
        details = {
            'query': query, 'threshold': threshold, 'softmax_temp': softmax_temp,
            'total_above_threshold': len(high_sim_points), 'num_clusters': len(clusters),
            'similarity_stats': {'min': similarities.min(), 'max': similarities.max(), 'mean': similarities.mean(), 'std': similarities.std()},
            'cluster_info': []
        }
        for i, cluster in enumerate(clusters):
            cluster_points = self.points[cluster]
            cluster_sims = similarities[cluster]
            cluster_info = {
                'id': i, 'size': len(cluster), 'center': cluster_points.mean(axis=0),
                'bbox_min': cluster_points.min(axis=0), 'bbox_max': cluster_points.max(axis=0),
                'similarity_mean': cluster_sims.mean(), 'similarity_std': cluster_sims.std()
            }
            details['cluster_info'].append(cluster_info)
        return clusters, details

    def compare_queries(self, queries: List[str], negatives: Optional[List[str]] = None, threshold: float = 0.502, softmax_temp: float = 1.0) -> Dict[str, Dict]:
        """Compare multiple queries and their similarity patterns."""
        if negatives is None:
            negatives = ["object"]
        results = {}
        for query in queries:
            similarities = self.query_similarity(query, negatives, softmax_temp)
            above_threshold = (similarities > threshold).sum()
            clusters, details = self.find_object_instances(query, threshold, negatives, softmax_temp=softmax_temp, return_details=True)
            results[query] = {'similarities': similarities, 'above_threshold': above_threshold, 'clusters': clusters, 'details': details}
        return results

    def spatial_analysis(self, query: str, negatives: Optional[List[str]] = None, threshold: float = 0.502, softmax_temp: float = 1.0, grid_resolution: float = 0.1) -> Dict:
        """Perform spatial analysis of query matches."""
        similarities = self.query_similarity(query, negatives, softmax_temp)
        high_sim_mask = similarities > threshold
        if not high_sim_mask.any():
            return {'empty': True}
        high_sim_points = self.points[high_sim_mask]
        high_sim_scores = similarities[high_sim_mask]
        bbox_min = high_sim_points.min(axis=0)
        bbox_max = high_sim_points.max(axis=0)
        volume = np.prod(bbox_max - bbox_min)
        density = len(high_sim_points) / volume if volume > 0 else 0
        grid_counts = {}
        for point, score in zip(high_sim_points, high_sim_scores):
            grid_idx = tuple((point / grid_resolution).astype(int))
            if grid_idx not in grid_counts:
                grid_counts[grid_idx] = {'count': 0, 'scores': []}
            grid_counts[grid_idx]['count'] += 1
            grid_counts[grid_idx]['scores'].append(score)
        grid_stats = {}
        for grid_idx, data in grid_counts.items():
            grid_stats[grid_idx] = {'count': data['count'], 'mean_score': np.mean(data['scores']), 'center': np.array(grid_idx) * grid_resolution}
        return {
            'query': query, 'threshold': threshold, 'softmax_temp': softmax_temp,
            'num_points': len(high_sim_points), 'bbox_min': bbox_min, 'bbox_max': bbox_max,
            'volume': volume, 'density': density, 'grid_resolution': grid_resolution,
            'grid_stats': grid_stats, 'score_stats': {'min': high_sim_scores.min(), 'max': high_sim_scores.max(), 'mean': high_sim_scores.mean(), 'std': high_sim_scores.std()}
        }

    def export_semantic_analysis(self, queries: List[str], output_path: str, threshold: float = 0.502, softmax_temp: float = 1.0) -> None:
        """Export comprehensive semantic analysis to file."""
        results = {
            'metadata': {
                'num_points': len(self.points), 'feature_dim': self.features.shape[1],
                'queries': queries, 'threshold': threshold, 'softmax_temp': softmax_temp,
                'bbox_min': self.points.min(axis=0).tolist(), 'bbox_max': self.points.max(axis=0).tolist()
            },
            'query_results': {}
        }
        for query in queries:
            console.print(f"[yellow]Analyzing query: '{query}'")
            similarities = self.query_similarity(query, softmax_temp=softmax_temp)
            clusters, details = self.find_object_instances(query, threshold, softmax_temp=softmax_temp, return_details=True)
            spatial_info = self.spatial_analysis(query, threshold=threshold, softmax_temp=softmax_temp)
            results['query_results'][query] = {
                'similarities_stats': {'min': float(similarities.min()), 'max': float(similarities.max()), 'mean': float(similarities.mean()), 'std': float(similarities.std())},
                'clusters': details, 'spatial_analysis': spatial_info
            }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[green]✓ Semantic analysis exported to: {output_path}")


if __name__ == "__main__":
    console.print("[bold blue]F3RM Semantic Similarity Utilities Demo")

    try:
        points = np.load("points.npy")
        features = np.load("features.npy")
        analyzer = SemanticPointcloudAnalyzer(features, points)

        similarities = analyzer.query_similarity("magazine", negatives=["object"])

        clusters = analyzer.find_object_instances("magazine", threshold=0.502)
        for i, cluster in enumerate(clusters):
            center = points[cluster].mean(axis=0)
            print(f"Magazine {i+1}: {len(cluster)} points at {center}")

        queries = ["chair", "table", "book", "magazine"]
        results = analyzer.compare_queries(queries, threshold=0.502, softmax_temp=1.0)

        spatial_info = analyzer.spatial_analysis("chair", threshold=0.502, softmax_temp=1.0)
        print(f"Chair density: {spatial_info['density']:.3f} points per unit volume")

        analyzer.export_semantic_analysis(queries, "semantic_analysis.json", threshold=0.502, softmax_temp=1.0)
        console.print("[bold green]Demo completed successfully!")

    except FileNotFoundError:
        console.print("[yellow]Demo files not found. Run after exporting pointcloud data.")
    except Exception as e:
        console.print(f"[red]Demo error: {e}")
