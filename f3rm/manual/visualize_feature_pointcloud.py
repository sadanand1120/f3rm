
"""F3RM Feature Pointcloud Visualizer"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from rich.console import Console

from f3rm.manual.semantic_similarity_utils import SemanticSimilarityUtils

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
        self._additional_pcds = {}

    def _load_metadata(self) -> Dict:
        """Load metadata from exported pointcloud."""
        metadata_path = self.data_dir / "metadata.json"
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

    def get_additional_pointcloud(self, output_name: str) -> Optional[o3d.geometry.PointCloud]:
        """Load additional pointcloud by name if not already loaded."""
        if output_name not in self._additional_pcds:
            filename = self.metadata['files'].get(output_name)
            if filename is not None:
                file_path = self.data_dir / filename
                if file_path.exists():
                    self._additional_pcds[output_name] = o3d.io.read_point_cloud(str(file_path))
                else:
                    console.print(f"[yellow]Warning: File not found: {file_path}")
                    self._additional_pcds[output_name] = None
            else:
                self._additional_pcds[output_name] = None
        return self._additional_pcds[output_name]


def create_coordinate_frame(size: float = 0.2) -> List[o3d.geometry.Geometry]:
    """Create coordinate frame with X(red), Y(green), Z(blue) axes at origin."""
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return [coord_frame]


def create_bounding_box_lines(bbox_min: np.ndarray, bbox_max: np.ndarray) -> List[o3d.geometry.Geometry]:
    """Create wireframe bounding box visualization."""
    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    bbox_lines.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color
    return [bbox_lines]


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
    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    bbox_lines.paint_uniform_color([1.0, 0.5, 0.0])  # Orange color for filter bounds
    return [bbox_lines]


def apply_bbox_filter(points: np.ndarray, colors: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply bounding box filter to points and colors."""
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
    queries = [filter_query] + negative_queries
    similarities = semantic_utils.compute_text_similarities(
        data.features, queries,
        has_negatives=len(negative_queries) > 0,
        softmax_temp=softmax_temp
    )

    above_thresh_mask = similarities > threshold

    if filter_mode == 'filter-out':
        # Remove points that match the query (keep points NOT matching)
        keep_mask = ~above_thresh_mask
    elif filter_mode == 'filter-in':
        # Keep only points that match the query
        keep_mask = above_thresh_mask
    else:
        raise ValueError(f"Invalid filter mode: {filter_mode}. Use 'filter-out' or 'filter-in'")

    return points[keep_mask], colors[keep_mask]


def create_reference_geometries(data: FeaturePointcloudData, filter_bbox_min: Optional[np.ndarray] = None, filter_bbox_max: Optional[np.ndarray] = None) -> List[o3d.geometry.Geometry]:
    """Create all reference geometries (coordinate frame, bounding box, grid, and optional filter bbox)."""
    bbox_min = np.array(data.metadata['bbox_min'])
    bbox_max = np.array(data.metadata['bbox_max'])

    geometries = []
    geometries.extend(create_coordinate_frame(size=0.1))
    geometries.extend(create_bounding_box_lines(bbox_min, bbox_max))
    geometries.extend(create_grid_lines(bbox_min, bbox_max, grid_size=0.2))

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
    if bbox_filter_min is not None and bbox_filter_max is not None:
        bbox_min = np.array(bbox_filter_min)
        bbox_max = np.array(bbox_filter_max)
        points, colors = apply_bbox_filter(points, colors, bbox_min, bbox_max)

    if semantic_filter_query is not None and semantic_filter_mode is not None:
        points, colors = apply_semantic_filter(
            data, points, colors, semantic_filter_query, semantic_filter_mode,
            semantic_threshold, semantic_softmax_temp, semantic_negatives
        )

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
    pcd = data.rgb_pointcloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Apply filters if requested
    points, colors = apply_filters(
        data, points, colors, bbox_filter_min, bbox_filter_max,
        semantic_filter_query, semantic_filter_mode, semantic_threshold,
        semantic_softmax_temp, semantic_negatives
    )

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
    pcd = data.pca_pointcloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Apply filters if requested
    points, colors = apply_filters(
        data, points, colors, bbox_filter_min, bbox_filter_max,
        semantic_filter_query, semantic_filter_mode, semantic_threshold,
        semantic_softmax_temp, semantic_negatives
    )

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

    # Simple visualization
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name="F3RM PCA Feature Pointcloud",
        width=1200,
        height=800,
        left=50,
        top=50
    )


def visualize_additional(
    data: FeaturePointcloudData,
    output_name: str,
    show_guides: bool = True,
    bbox_filter_min: Optional[List[float]] = None,
    bbox_filter_max: Optional[List[float]] = None,
    semantic_filter_query: Optional[str] = None,
    semantic_filter_mode: Optional[str] = None,
    semantic_threshold: float = 0.502,
    semantic_softmax_temp: float = 1.0,
    semantic_negatives: Optional[List[str]] = None
):
    """Visualize additional pointcloud with optional filtering."""
    pcd = data.get_additional_pointcloud(output_name)
    if pcd is None:
        console.print(f"[bold red]Error: Additional output '{output_name}' not available")
        console.print(f"[yellow]Available additional outputs: {data.metadata.get('additional_outputs', [])}")
        return

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Apply filters if requested
    points, colors = apply_filters(
        data, points, colors, bbox_filter_min, bbox_filter_max,
        semantic_filter_query, semantic_filter_mode, semantic_threshold,
        semantic_softmax_temp, semantic_negatives
    )

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

    # Simple visualization
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name=f"F3RM {output_name.replace('_', ' ').title()} Pointcloud",
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
    if negative_queries is None:
        negative_queries = ["object"]  # Same default as opt.py

    # Initialize semantic utils
    semantic_utils = SemanticSimilarityUtils()

    # Compute similarities for the main query
    queries = [query] + negative_queries
    similarities = semantic_utils.compute_text_similarities(
        data.features, queries,
        has_negatives=len(negative_queries) > 0,
        softmax_temp=softmax_temp
    )

    # Get original RGB pointcloud for background
    rgb_pcd = data.rgb_pointcloud
    all_points = np.asarray(rgb_pcd.points)
    all_rgb_colors = np.asarray(rgb_pcd.colors)

    above_threshold_mask = similarities > threshold

    combined_points = all_points.copy()
    combined_colors = all_rgb_colors.copy()

    if background_alpha > 0:
        # Make RGB background transparent by reducing intensity
        combined_colors = combined_colors * background_alpha

    # Apply semantic heatmap to points above threshold
    if above_threshold_mask.any():
        above_thresh_similarities = similarities[above_threshold_mask]

        sim_min, sim_max = above_thresh_similarities.min(), above_thresh_similarities.max()
        if sim_max - sim_min < 1e-8:
            sim_norm = np.ones_like(above_thresh_similarities) * 0.5
        else:
            sim_norm = (above_thresh_similarities - sim_min) / (sim_max - sim_min)

        cmap = plt.get_cmap("turbo")
        heatmap_colors = cmap(sim_norm)[:, :3]

        combined_colors[above_threshold_mask] = heatmap_colors

    # Apply additional filters if requested (bbox and semantic filters)
    filtered_points, filtered_colors = apply_filters(
        data, combined_points, combined_colors, bbox_filter_min, bbox_filter_max,
        semantic_filter_query, semantic_filter_mode, semantic_threshold,
        semantic_softmax_temp, semantic_negatives
    )

    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # Start with combined pointcloud
    all_geometries = [combined_pcd]

    # Add reference geometries if requested
    if show_guides:
        filter_bbox_min = np.array(bbox_filter_min) if bbox_filter_min is not None else None
        filter_bbox_max = np.array(bbox_filter_max) if bbox_filter_max is not None else None
        reference_geoms = create_reference_geometries(data, filter_bbox_min, filter_bbox_max)
        all_geometries.extend(reference_geoms)

    # Save if requested (save the original similarity result, not the combined one)
    if save_result:
        sim_pcd = semantic_utils.create_similarity_pointcloud(
            data.points, similarities, threshold=threshold
        )
        output_path = data.data_dir / f"semantic_query_{query.replace(' ', '_')}.ply"
        o3d.io.write_point_cloud(str(output_path), sim_pcd)

        combined_output_path = data.data_dir / f"semantic_query_{query.replace(' ', '_')}_with_background.ply"
        o3d.io.write_point_cloud(str(combined_output_path), combined_pcd)

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

    # Safely compute available modes without triggering --help early exit
    tmp = argparse.ArgumentParser(add_help=False)
    tmp.add_argument("--data-dir", type=Path)
    known_args, _ = tmp.parse_known_args()
    try:
        data = FeaturePointcloudData(Path(known_args.data_dir)) if known_args.data_dir else None
    except Exception:
        data = None
    additional = data.metadata.get('additional_outputs', []) if data else []
    available_modes = ["rgb", "pca", "semantic", "LISTALL"] + additional

    parser.add_argument("--mode", choices=available_modes, default="rgb", help="Visualization mode")
    parser.add_argument("--query", type=str, default=None, help="Semantic query")
    parser.add_argument("--negative-queries", nargs="*", default=None, help="Negative queries (default: ['object'])")
    parser.add_argument("--threshold", type=float, default=0.502, help="Similarity threshold (default: 0.502)")
    parser.add_argument("--softmax-temp", type=float, default=1.0, help="Softmax temperature (default: 1.0)")
    parser.add_argument("--save", action="store_true", help="Save semantic similarity results")
    parser.add_argument("--no-guides", action="store_true", help="Hide coordinate frame and reference guides")

    # Bounding box filtering arguments
    parser.add_argument("--bbox-filter-min", type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=None,
                        help="Filter: minimum bounding box coordinates (x y z) - only show points within this bbox")
    parser.add_argument("--bbox-filter-max", type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=None,
                        help="Filter: maximum bounding box coordinates (x y z) - only show points within this bbox")

    # Semantic filtering arguments
    parser.add_argument("--semantic-filter-query", type=str, default=None,
                        help="Filter: semantic query for filtering points (e.g., 'floor', 'wall')")
    parser.add_argument("--semantic-filter-mode", choices=["filter-out", "filter-in"], default=None,
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

    # Load pointcloud data (reuse if already loaded)
    if 'data' not in locals():
        data = FeaturePointcloudData(args.data_dir)
        additional_outputs = data.metadata.get('additional_outputs', [])
        if additional_outputs:
            console.print(f"[cyan]Available additional outputs: {additional_outputs}")
        else:
            console.print("[dim]No additional outputs available")

    show_guides = not args.no_guides

    # Show available outputs if requested
    if args.mode == "LISTALL":
        console.print(f"[bold blue]Available visualization modes:")
        console.print(f"  [green]Core modes: rgb, pca, semantic")
        console.print(f"  [cyan]Additional outputs: {data.metadata.get('additional_outputs', [])}")
        console.print(f"  [yellow]Total available: {len(['rgb', 'pca', 'semantic'] + data.metadata.get('additional_outputs', []))} modes")
        return

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
    elif args.mode in data.metadata.get('additional_outputs', []):
        visualize_additional(
            data,
            args.mode,
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
        # Allow using --query or fallback to --semantic-filter-query for semantic mode
        semantic_query = args.query if args.query is not None else args.semantic_filter_query
        if semantic_query is None:
            console.print("[bold red]Error: --query is required for semantic mode (or provide --semantic-filter-query as fallback)")
            return
        visualize_semantic(
            data,
            semantic_query,
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
    else:
        console.print(f"[bold red]Error: Visualization mode '{args.mode}' not available")
        console.print(f"[yellow]Available modes: rgb, pca, semantic, {', '.join(data.metadata.get('additional_outputs', []))}")
        console.print(f"[cyan]Use --mode LISTALL to see all available modes")


if __name__ == "__main__":
    main()
