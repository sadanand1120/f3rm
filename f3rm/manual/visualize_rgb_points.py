#!/usr/bin/env python3
"""
F3RM RGB Pointcloud with Custom Points Tool

This script provides interactive addition of custom red points at multiple locations
and visualization of RGB pointclouds with the saved points.

Usage:
    python f3rm/manual/visualize_rgb_points.py --data-dir exports/pointcloud_data/ --mode add
    python f3rm/manual/visualize_rgb_points.py --data-dir exports/pointcloud_data/ --mode visualize

Add Mode Controls:
    - Mouse: Normal Open3D viewing controls (camera only)
    - I/K: Move forward/back (±Y)
    - J/L: Move left/right (±X)  
    - U/O: Move up/down (±Z)
    - 1/2: Scale point size (decrease/increase)
    - 'A': Add current point and start new one
    - 'R': Reset current point position
    - 'Q': Quit and save all points

Visualize Mode:
    - Simple visualization of RGB pointcloud with all saved custom points
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import open3d as o3d
from rich.console import Console

# Import reference geometry functions from visualize script
from f3rm.manual.visualize_feature_pointcloud import (
    FeaturePointcloudData,
    create_coordinate_frame,
    create_bounding_box_lines,
    create_grid_lines
)

console = Console()


class CustomPoint:
    """Represents a custom red point with position and scale."""

    def __init__(self, position: np.ndarray, scale: float = 1.0, point_id: int = 0):
        self.position = position.copy()  # 3D position
        self.scale = scale
        self.point_id = point_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'position': self.position.tolist(),
            'scale': self.scale,
            'point_id': self.point_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomPoint':
        """Create from dictionary."""
        return cls(
            position=np.array(data['position']),
            scale=data['scale'],
            point_id=data['point_id']
        )

    def create_geometry(self) -> o3d.geometry.TriangleMesh:
        """Create Open3D geometry for this point."""
        # Create a small red sphere instead of a point for better scaling
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.scale * 0.01)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red color

        # Move sphere to the target position
        sphere.translate(self.position)

        return sphere


class InteractivePointsAdder:
    """Interactive tool for adding custom red points at multiple locations."""

    def __init__(self, data: FeaturePointcloudData, translation_step: float = 0.1, scale_step: float = 0.1, transparency: float = 1.0):
        self.data = data
        self.vis = None

        # Step sizes for transformations
        self.translation_step = translation_step  # units
        self.scale_step = scale_step  # units
        self.transparency = transparency  # RGB pointcloud transparency

        # Current point being positioned
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_scale = 0.2  # Default scale

        # All saved points
        self.saved_points: List[CustomPoint] = []
        self.next_point_id = 0

        # Store geometries
        self.original_pcd = None
        self.reference_geoms = []
        self.current_point_geom = None
        self.saved_points_geoms = []

        # Load existing points if available
        self.load_existing_points()

    def create_reference_geometries(self) -> list:
        """Create coordinate frame, bounding box, and grid."""
        geometries = []

        # Get bounding box from metadata
        bbox_min = np.array(self.data.metadata['bbox_min'])
        bbox_max = np.array(self.data.metadata['bbox_max'])

        # Bounding box wireframe
        geometries.extend(create_bounding_box_lines(bbox_min, bbox_max))

        # Grid lines for spatial reference
        geometries.extend(create_grid_lines(bbox_min, bbox_max, grid_size=0.2))

        return geometries

    def update_current_point_geometry(self):
        """Update the current point geometry based on position and scale."""
        if self.current_point_geom is None:
            self.current_point_geom = o3d.geometry.TriangleMesh()

        # Create a small red sphere for the current point
        self.current_point_geom = o3d.geometry.TriangleMesh.create_sphere(radius=self.current_scale * 0.01)
        self.current_point_geom.paint_uniform_color([1.0, 0.0, 0.0])  # Red color

        # Move sphere to the current position
        self.current_point_geom.translate(self.current_position)

    def update_visualization(self):
        """Update the visualization with current state."""
        if self.vis is None:
            return

        # Save current camera parameters
        view_control = self.vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()

        # Update current point geometry
        self.update_current_point_geometry()

        # Clear and re-add geometries
        self.vis.clear_geometries()

        # Add RGB pointcloud
        self.vis.add_geometry(self.original_pcd)

        # Add reference geometries
        for geom in self.reference_geoms:
            self.vis.add_geometry(geom)

        # Add saved points
        for geom in self.saved_points_geoms:
            self.vis.add_geometry(geom)

        # Add current point (highlighted)
        self.vis.add_geometry(self.current_point_geom)

        # Restore camera parameters to maintain view
        view_control.convert_from_pinhole_camera_parameters(camera_params)

        # Print status
        pos = self.current_position
        console.print(f"[cyan]Current point {self.next_point_id}: "
                      f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                      f"scale={self.current_scale:.3f}, "
                      f"saved={len(self.saved_points)}")

    def reset_current_position(self):
        """Reset current point position to origin."""
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_scale = 0.2
        self.update_visualization()
        console.print("[green]Current point reset to origin")

    def apply_translation(self, dx_direction: int, dy_direction: int, dz_direction: int):
        """Apply translation.

        Args:
            dx_direction: direction multiplier for X (-1, 0, or 1)
            dy_direction: direction multiplier for Y (-1, 0, or 1) 
            dz_direction: direction multiplier for Z (-1, 0, or 1)
        """
        dx = self.translation_step * dx_direction
        dy = self.translation_step * dy_direction
        dz = self.translation_step * dz_direction

        self.current_position += np.array([dx, dy, dz])
        self.update_visualization()

    def apply_scale(self, direction: int):
        """Apply scale change.

        Args:
            direction: 1 for increase, -1 for decrease
        """
        scale_change = self.scale_step * direction
        new_scale = self.current_scale + scale_change

        # Ensure scale stays positive
        if new_scale > 0.01:
            self.current_scale = new_scale
            self.update_visualization()

    def add_current_point(self):
        """Add current point to saved list and start new one."""
        # Create new point from current state
        new_point = CustomPoint(
            position=self.current_position.copy(),
            scale=self.current_scale,
            point_id=self.next_point_id
        )

        # Add to saved list
        self.saved_points.append(new_point)

        # Create geometry for saved point
        saved_geom = new_point.create_geometry()
        self.saved_points_geoms.append(saved_geom)

        # Start next point at the position and scale of the current one (keep position and scale)
        self.current_position = self.current_position.copy()  # Keep the position
        # Keep the current scale (don't reset to 0.2)
        self.next_point_id += 1

        console.print(f"[green]Added point {new_point.point_id} at position {new_point.position}")
        self.update_visualization()

    def load_existing_points(self):
        """Load existing points from JSON file if available."""
        points_path = self.data.data_dir / "custom_points.json"
        if points_path.exists():
            try:
                with open(points_path, 'r') as f:
                    points_data = json.load(f)

                loaded_points = [CustomPoint.from_dict(point_dict) for point_dict in points_data['points']]
                self.saved_points = loaded_points
                self.next_point_id = len(loaded_points)

                # Create geometries for loaded points
                for point in loaded_points:
                    point_geom = point.create_geometry()
                    self.saved_points_geoms.append(point_geom)

                console.print(f"[green]Loaded {len(loaded_points)} existing points from {points_path}")

                # Start new point at the position of the last loaded point if any exist
                if loaded_points:
                    last_point = loaded_points[-1]
                    self.current_position = last_point.position.copy()
                    self.current_scale = last_point.scale
                    console.print(f"[cyan]Starting new point at position of last loaded point")

            except Exception as e:
                console.print(f"[yellow]Warning: Could not load existing points: {e}")
                console.print("[yellow]Starting with empty points list")

    def save_points_to_file(self):
        """Save all points to JSON file."""
        points_data = {
            'points': [point.to_dict() for point in self.saved_points],
            'num_points': len(self.saved_points)
        }

        output_path = self.data.data_dir / "custom_points.json"
        with open(output_path, 'w') as f:
            json.dump(points_data, f, indent=2)

        console.print(f"[green]Saved {len(self.saved_points)} points to {output_path}")

    def start_adding_points(self):
        """Start the interactive points addition tool."""
        console.print("[bold green]Starting F3RM Custom Points Addition Tool")
        console.print(self.data.get_info())

        console.print("\n[bold blue]Controls:")
        console.print("  Mouse: Normal Open3D viewing controls (camera only)")
        console.print("  'I'/'K': Move forward/back (±Y)")
        console.print("  'J'/'L': Move left/right (±X)")
        console.print("  'U'/'O': Move up/down (±Z)")
        console.print("  ↑/↓ Arrow: Scale point size (decrease/increase)")
        console.print("  'A': Add current point and start new one")
        console.print("  'R': Reset current point position")
        console.print("  'Q': Quit and save all points")
        console.print("\n[bold yellow]Goal: Add custom red points at multiple locations")
        console.print("[bold yellow]Note: Camera view is preserved during transforms")

        # Load RGB pointcloud and apply transparency
        self.original_pcd = self.data.rgb_pointcloud
        if self.transparency < 1.0:
            # Apply transparency by modifying colors
            colors = np.asarray(self.original_pcd.colors)
            colors = colors * self.transparency
            self.original_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create reference geometries
        self.reference_geoms = self.create_reference_geometries()

        # Initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="F3RM Custom Points Addition Tool",
            width=1400, height=900
        )

        # Register key callbacks
        self.vis.register_key_callback(ord('A'), lambda vis: self.add_current_point())
        self.vis.register_key_callback(ord('R'), lambda vis: self.reset_current_position())
        self.vis.register_key_callback(ord('Q'), lambda vis: self.save_and_exit(vis))

        # IJKL controls for translation
        self.vis.register_key_callback(ord('I'), lambda vis: self.apply_translation(0, 1, 0))   # I: +Y (forward)
        self.vis.register_key_callback(ord('K'), lambda vis: self.apply_translation(0, -1, 0))  # K: -Y (back)
        self.vis.register_key_callback(ord('J'), lambda vis: self.apply_translation(-1, 0, 0))  # J: -X (left)
        self.vis.register_key_callback(ord('L'), lambda vis: self.apply_translation(1, 0, 0))   # L: +X (right)

        # U/O for up/down translation
        self.vis.register_key_callback(ord('U'), lambda vis: self.apply_translation(0, 0, 1))   # U: +Z (up)
        self.vis.register_key_callback(ord('O'), lambda vis: self.apply_translation(0, 0, -1))  # O: -Z (down)

        # Arrow keys for scaling
        self.vis.register_key_callback(265, lambda vis: self.apply_scale(1))    # Up Arrow: increase scale
        self.vis.register_key_callback(264, lambda vis: self.apply_scale(-1))   # Down Arrow: decrease scale

        # Initial visualization
        self.update_visualization()

        console.print("\n[dim]Detailed Controls:")
        console.print("  I/K: Move forward/back (±Y)")
        console.print("  J/L: Move left/right (±X)")
        console.print("  U/O: Move up/down (±Z)")
        console.print("  ↑/↓ Arrow: Scale point size (decrease/increase)")
        console.print("  A: Add current point and start new one")
        console.print("  R: Reset current point position")
        console.print("  Q: Quit and save all points")
        console.print(f"\n[dim]Step sizes: Translation={self.translation_step}, Scale={self.scale_step}")

        # Run visualization
        self.vis.run()
        self.vis.destroy_window()

    def save_and_exit(self, vis):
        """Save points and exit."""
        if self.saved_points:
            self.save_points_to_file()
        else:
            console.print("[yellow]No points to save")
        vis.close()
        return True


def load_custom_points(data_dir: Path) -> List[CustomPoint]:
    """Load custom points from JSON file."""
    points_path = data_dir / "custom_points.json"
    if not points_path.exists():
        return []

    with open(points_path, 'r') as f:
        points_data = json.load(f)

    points = [CustomPoint.from_dict(point_dict) for point_dict in points_data['points']]
    console.print(f"[green]Loaded {len(points)} custom points from {points_path}")
    return points


def add_custom_points(data_dir: Path, translation_step: float = 0.1, scale_step: float = 0.1, transparency: float = 1.0) -> None:
    """
    Main function to add custom points.

    Args:
        data_dir: Directory containing exported pointcloud data
        translation_step: Translation step size in units (default: 0.1)
        scale_step: Scale step size in units (default: 0.1)
        transparency: Transparency of RGB pointcloud (0.0=transparent, 1.0=opaque, default: 1.0)
    """
    console.print(f"[bold blue]F3RM Custom Points Addition Tool")
    console.print(f"[bold blue]Data directory: {data_dir}")
    console.print(f"[bold blue]Translation step: {translation_step} | Scale step: {scale_step} | Transparency: {transparency}")

    # Load pointcloud data
    try:
        data = FeaturePointcloudData(data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")
    except Exception as e:
        console.print(f"[red]Error loading pointcloud data: {e}")
        return

    # Start points addition tool
    points_adder = InteractivePointsAdder(data, translation_step, scale_step, transparency)
    points_adder.start_adding_points()


def visualize_with_points(data_dir: Path, transparency: float = 1.0) -> None:
    """
    Main function to visualize RGB pointcloud with custom points.

    Args:
        data_dir: Directory containing exported pointcloud data
        transparency: Transparency of RGB pointcloud (0.0=transparent, 1.0=opaque, default: 1.0)
    """
    console.print(f"[bold blue]F3RM RGB Pointcloud with Custom Points Visualizer")
    console.print(f"[bold blue]Data directory: {data_dir}")
    console.print(f"[bold blue]Transparency: {transparency}")

    # Load pointcloud data
    try:
        data = FeaturePointcloudData(data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")
    except Exception as e:
        console.print(f"[red]Error loading pointcloud data: {e}")
        return

    # Load custom points
    custom_points = load_custom_points(data_dir)
    if not custom_points:
        console.print("[yellow]No custom points found. Run in 'add' mode first to create points.")

    console.print("[bold green]Visualizing RGB pointcloud with custom points...")
    console.print(data.get_info())

    # Get RGB pointcloud and apply transparency
    pcd = data.rgb_pointcloud
    if transparency < 1.0:
        # Apply transparency by modifying colors
        colors = np.asarray(pcd.colors)
        colors = colors * transparency
        pcd.colors = o3d.utility.Vector3dVector(colors)
    console.print(f"[green]Loaded pointcloud with {len(np.asarray(pcd.points))} points")

    # Create reference geometries
    reference_geoms = []

    # Get bounding box from metadata
    bbox_min = np.array(data.metadata['bbox_min'])
    bbox_max = np.array(data.metadata['bbox_max'])

    # Bounding box wireframe
    reference_geoms.extend(create_bounding_box_lines(bbox_min, bbox_max))

    # Grid lines for spatial reference
    reference_geoms.extend(create_grid_lines(bbox_min, bbox_max, grid_size=0.2))

    # Create custom points geometries
    points_geometries = []
    for point in custom_points:
        point_geom = point.create_geometry()
        points_geometries.append(point_geom)

    # Combine all geometries
    all_geometries = [pcd] + reference_geoms + points_geometries

    console.print(f"[cyan]Showing {len(custom_points)} custom points")
    console.print("[dim]Reference guides: Gray=original bounds; Light gray=grid")

    # Simple visualization
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name="F3RM RGB Pointcloud with Custom Points",
        width=1200,
        height=800,
        left=50,
        top=50
    )


def main():
    parser = argparse.ArgumentParser(description="Add or visualize custom points with F3RM RGB pointclouds")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing exported pointcloud data")
    parser.add_argument("--mode", choices=["add", "visualize"], default="add",
                        help="Tool mode: 'add' for interactive points placement, 'visualize' for viewing with points")

    # Add mode arguments
    parser.add_argument("--translation-step", type=float, default=0.02,
                        help="Translation step size in units (default: 0.1, add mode)")
    parser.add_argument("--scale-step", type=float, default=0.2,
                        help="Scale step size in units (default: 0.1, add mode)")
    parser.add_argument("--transparency", type=float, default=0.5,
                        help="Transparency of RGB pointcloud (0.0=transparent, 1.0=opaque, default: 1.0)")

    args = parser.parse_args()

    # Validate inputs
    if not args.data_dir.exists():
        console.print(f"[bold red]Data directory not found: {args.data_dir}")
        return

    if not (args.data_dir / "metadata.json").exists():
        console.print(f"[bold red]No metadata.json found. Run f3rm/manual/export_feature_pointcloud.py first.")
        return

    # Check for required files
    required_files = ["pointcloud_rgb.ply", "pointcloud_feature_pca.ply", "points.npy"]
    missing_files = []
    for file in required_files:
        if not (args.data_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        console.print(f"[bold red]Missing required files: {', '.join(missing_files)}")
        console.print("[bold red]Run f3rm/manual/export_feature_pointcloud.py first to generate all required files.")
        return

    try:
        if args.mode == "add":
            add_custom_points(args.data_dir, args.translation_step, args.scale_step, args.transparency)
        elif args.mode == "visualize":
            visualize_with_points(args.data_dir, args.transparency)
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
