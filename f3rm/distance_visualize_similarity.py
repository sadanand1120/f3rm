#!/usr/bin/env python3
"""
Distance-based Semantic Visualization for F3RM Pointclouds

This script visualizes semantic similarity with distance-based highlighting.
Shows a semantic query (e.g., "chair") as a heatmap with RGB background,
and highlights floor points within a distance threshold in light green.

Usage:
    python distance_visualize_similarity.py --data-dir path/to/exported/pointcloud/ \
        --query "chair" --distance-lower 0.0 --distance-upper 0.5
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from rich.console import Console
from scipy.spatial.distance import cdist

from f3rm.visualize_feature_pointcloud import FeaturePointcloudData, SemanticSimilarityUtils

console = Console()


def visualize_distance_semantic(
    data: FeaturePointcloudData,
    main_query: str,
    distance_lower: float,
    distance_upper: float,
    floor_query: str = "floor",
    background_alpha: float = 0.3,
    semantic_threshold: float = 0.502,
    softmax_temp: float = 1.0,
    save_result: bool = False
):
    """
    Visualize semantic similarity with distance-based floor highlighting.

    Args:
        data: Pointcloud data
        main_query: Main semantic query (e.g., "chair")
        distance_lower: Lower distance bound for floor highlighting
        distance_upper: Upper distance bound for floor highlighting
        floor_query: Floor semantic query (default: "floor")
        background_alpha: RGB background transparency
        semantic_threshold: Similarity threshold
        softmax_temp: Softmax temperature
        save_result: Whether to save result
    """
    console.print(f"[bold green]Distance-based semantic visualization:")
    console.print(f"  Main query: '{main_query}'")
    console.print(f"  Floor query: '{floor_query}'")
    console.print(f"  Distance bounds: [{distance_lower:.3f}, {distance_upper:.3f}]")
    console.print(f"  Background alpha: {background_alpha:.1f}")

    # Initialize semantic utils
    semantic_utils = SemanticSimilarityUtils()

    # Get RGB pointcloud for background
    rgb_pcd = data.rgb_pointcloud
    all_points = np.asarray(rgb_pcd.points)
    all_rgb_colors = np.asarray(rgb_pcd.colors)

    # Compute similarities for main query
    console.print(f"[yellow]Computing similarities for main query '{main_query}'...")
    main_queries = [main_query, "object"]
    main_similarities = semantic_utils.compute_text_similarities(
        data.features, main_queries,
        has_negatives=True,
        softmax_temp=softmax_temp
    )

    # Compute similarities for floor query
    console.print(f"[yellow]Computing similarities for floor query '{floor_query}'...")
    floor_queries = [floor_query, "object"]
    floor_similarities = semantic_utils.compute_text_similarities(
        data.features, floor_queries,
        has_negatives=True,
        softmax_temp=softmax_temp
    )

    # Create masks
    main_mask = main_similarities > semantic_threshold
    floor_mask = floor_similarities > semantic_threshold

    console.print(f"[cyan]Main query '{main_query}': {main_mask.sum():,} points above threshold")
    console.print(f"[cyan]Floor query '{floor_query}': {floor_mask.sum():,} points above threshold")

    # Start with RGB background (with transparency)
    combined_points = all_points.copy()
    combined_colors = all_rgb_colors.copy() * background_alpha

    # Apply main query heatmap to points above threshold
    if main_mask.any():
        main_sims = main_similarities[main_mask]

        # Normalize similarities for colormap
        sim_min, sim_max = main_sims.min(), main_sims.max()
        if sim_max - sim_min > 1e-8:
            sim_norm = (main_sims - sim_min) / (sim_max - sim_min)
        else:
            sim_norm = np.ones_like(main_sims) * 0.5

        # Apply turbo colormap for main query
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("turbo")
        heatmap_colors = cmap(sim_norm)[:, :3]
        combined_colors[main_mask] = heatmap_colors

        console.print(f"[green]Applied heatmap to {main_mask.sum():,} points for '{main_query}'")

    # Find floor points within distance threshold of main query points
    if main_mask.any() and floor_mask.any():
        main_points = all_points[main_mask]
        floor_points = all_points[floor_mask]
        floor_indices = np.where(floor_mask)[0]

        console.print(f"[yellow]Computing distances between {len(main_points):,} main points and {len(floor_points):,} floor points...")

        # Process in chunks to avoid memory issues
        chunk_size = 10000  # Process 10k floor points at a time
        min_distances = np.full(len(floor_points), np.inf)

        for i in range(0, len(floor_points), chunk_size):
            end_idx = min(i + chunk_size, len(floor_points))
            chunk_floor_points = floor_points[i:end_idx]

            # Compute distances for this chunk
            chunk_distances = cdist(chunk_floor_points, main_points)
            chunk_min_distances = chunk_distances.min(axis=1)

            # Store minimum distances
            min_distances[i:end_idx] = chunk_min_distances

            if (i // chunk_size + 1) % 10 == 0:
                console.print(f"[yellow]Processed {end_idx:,}/{len(floor_points):,} floor points...")

        # Find floor points within distance bounds
        near_floor_mask = (min_distances >= distance_lower) & (min_distances <= distance_upper)
        near_floor_indices = floor_indices[near_floor_mask]

        # Color near floor points in light green
        light_green = np.array([0.6, 1.0, 0.6])  # Light green
        combined_colors[near_floor_indices] = light_green

        console.print(f"[green]Highlighted {near_floor_mask.sum():,} floor points within distance bounds [{distance_lower:.3f}, {distance_upper:.3f}] in light green")
    else:
        console.print(f"[yellow]No valid points for distance computation")

    # Create final pointcloud
    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(combined_points)
    result_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Save if requested
    if save_result:
        output_path = data.data_dir / f"distance_semantic_{main_query.replace(' ', '_')}_dist{distance_lower:.2f}-{distance_upper:.2f}.ply"
        o3d.io.write_point_cloud(str(output_path), result_pcd)
        console.print(f"[green]Saved result: {output_path}")

    # Visualize
    console.print(f"[cyan]Legend:")
    console.print(f"  Turbo heatmap: '{main_query}' similarity")
    console.print(f"  Light green: '{floor_query}' within distance bounds [{distance_lower:.3f}, {distance_upper:.3f}]")
    console.print(f"  Transparent RGB: background context (alpha={background_alpha:.1f})")

    o3d.visualization.draw_geometries(
        [result_pcd, coord_frame],
        window_name=f"Distance Semantic: '{main_query}' + '{floor_query}' (d∈[{distance_lower:.2f}, {distance_upper:.2f}])",
        width=1200,
        height=800,
        left=50,
        top=50
    )


def main():
    parser = argparse.ArgumentParser(description="Distance-based semantic similarity visualization")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing exported pointcloud data")
    parser.add_argument("--query", type=str, required=True,
                        help="Main semantic query (e.g., 'chair')")
    parser.add_argument("--distance-lower", type=float, default=0.0,
                        help="Lower distance bound for floor highlighting (default: 0.0)")
    parser.add_argument("--distance-upper", type=float, required=True,
                        help="Upper distance bound for floor highlighting")
    parser.add_argument("--floor-query", type=str, default="floor",
                        help="Floor semantic query (default: 'floor')")
    parser.add_argument("--background-alpha", type=float, default=0.3,
                        help="RGB background transparency (default: 0.3)")
    parser.add_argument("--threshold", type=float, default=0.502,
                        help="Similarity threshold (default: 0.502)")
    parser.add_argument("--softmax-temp", type=float, default=1.0,
                        help="Softmax temperature (default: 1.0)")
    parser.add_argument("--save", action="store_true",
                        help="Save visualization result")

    args = parser.parse_args()

    # Validate inputs
    if not args.data_dir.exists():
        console.print(f"[bold red]Data directory not found: {args.data_dir}")
        return

    console.print(f"[bold blue]F3RM Distance-Based Semantic Visualizer")
    console.print(f"[bold blue]Data directory: {args.data_dir}")

    try:
        # Load pointcloud data
        data = FeaturePointcloudData(args.data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")

        # Run visualization
        visualize_distance_semantic(
            data,
            args.query,
            args.distance_lower,
            args.distance_upper,
            args.floor_query,
            args.background_alpha,
            args.threshold,
            args.softmax_temp,
            args.save
        )

    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
