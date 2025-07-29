#!/usr/bin/env python3
"""
F3RM RGB Pointcloud with Custom Axes Tool

This script provides interactive addition of custom coordinate axes at multiple locations
and visualization of RGB pointclouds with the saved axes.

Usage:
    python f3rm/manual/visualize_rgb_add_axes.py --data-dir exports/pointcloud_data/ --mode add
    python f3rm/manual/visualize_rgb_add_axes.py --data-dir exports/pointcloud_data/ --mode visualize

Add Mode Controls:
    - Mouse: Normal Open3D viewing controls (camera only)
    - Arrow Keys: Rotate around X/Y axes (pitch/yaw)
    - Z/X: Rotate around Z-axis (roll)
    - I/K: Move forward/back (±Y)
    - J/L: Move left/right (±X)  
    - U/O: Move up/down (±Z)
    - 1/2: Scale axes length (decrease/increase)
    - 'A': Add current axes and start new one
    - 'R': Reset current axes transform
    - 'Q': Quit and save all axes

Visualize Mode:
    - Simple visualization of RGB pointcloud with all saved custom axes
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


class CustomAxis:
    """Represents a custom coordinate axis with position, orientation, and scale."""

    def __init__(self, transform: np.ndarray, scale: float = 1.0, axis_id: int = 0):
        self.transform = transform.copy()  # 4x4 transformation matrix
        self.scale = scale
        self.axis_id = axis_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'transform': self.transform.tolist(),
            'scale': self.scale,
            'axis_id': self.axis_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomAxis':
        """Create from dictionary."""
        return cls(
            transform=np.array(data['transform']),
            scale=data['scale'],
            axis_id=data['axis_id']
        )

    def create_geometry(self) -> o3d.geometry.TriangleMesh:
        """Create Open3D geometry for this axis."""
        # Create coordinate frame at origin with specified scale
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.scale)

        # Apply transformation
        coord_frame.transform(self.transform)

        return coord_frame


class InteractiveAxesAdder:
    """Interactive tool for adding custom axes at multiple locations."""

    def __init__(self, data: FeaturePointcloudData, rotation_step: float = 10.0,
                 translation_step: float = 0.1, scale_step: float = 0.1):
        self.data = data
        self.vis = None

        # Step sizes for transformations
        self.rotation_step = rotation_step  # degrees
        self.translation_step = translation_step  # units
        self.scale_step = scale_step  # units

        # Current axes being positioned
        self.current_transform = np.eye(4)
        self.current_scale = 0.2  # Default scale

        # All saved axes
        self.saved_axes: List[CustomAxis] = []
        self.next_axis_id = 0

        # Store geometries
        self.original_pcd = None
        self.reference_geoms = []
        self.current_axis_geom = None
        self.saved_axes_geoms = []

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

    def update_current_axis_geometry(self):
        """Update the current axis geometry based on transform and scale."""
        if self.current_axis_geom is None:
            self.current_axis_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.current_scale)

        # Reset to identity and apply current transform
        self.current_axis_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.current_scale)
        self.current_axis_geom.transform(self.current_transform)

    def update_visualization(self):
        """Update the visualization with current state."""
        if self.vis is None:
            return

        # Save current camera parameters
        view_control = self.vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()

        # Update current axis geometry
        self.update_current_axis_geometry()

        # Clear and re-add geometries
        self.vis.clear_geometries()

        # Add RGB pointcloud
        self.vis.add_geometry(self.original_pcd)

        # Add reference geometries
        for geom in self.reference_geoms:
            self.vis.add_geometry(geom)

        # Add saved axes
        for geom in self.saved_axes_geoms:
            self.vis.add_geometry(geom)

        # Add current axis (highlighted)
        self.vis.add_geometry(self.current_axis_geom)

        # Restore camera parameters to maintain view
        view_control.convert_from_pinhole_camera_parameters(camera_params)

        # Print status
        pos = self.current_transform[:3, 3]
        console.print(f"[cyan]Current axis {self.next_axis_id}: "
                      f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                      f"scale={self.current_scale:.3f}, "
                      f"saved={len(self.saved_axes)}")

    def reset_current_transform(self):
        """Reset current axes transform to identity."""
        self.current_transform = np.eye(4)
        self.current_scale = 0.2
        self.update_visualization()
        console.print("[green]Current axes reset to identity")

    def apply_rotation(self, axis: str, direction: int = 1):
        """Apply rotation around specified axis.

        Args:
            axis: 'x', 'y', or 'z'
            direction: 1 for positive rotation, -1 for negative
        """
        angle_deg = self.rotation_step * direction
        angle_rad = np.radians(angle_deg)

        if axis == 'x':
            rot_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            rot_matrix = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'z':
            rot_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        else:
            return

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix

        # Apply to current transform
        self.current_transform = transform @ self.current_transform
        self.update_visualization()

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

        translation = np.eye(4)
        translation[:3, 3] = [dx, dy, dz]

        self.current_transform = translation @ self.current_transform
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

    def add_current_axis(self):
        """Add current axis to saved list and start new one."""
        # Create new axis from current state
        new_axis = CustomAxis(
            transform=self.current_transform.copy(),
            scale=self.current_scale,
            axis_id=self.next_axis_id
        )

        # Add to saved list
        self.saved_axes.append(new_axis)

        # Create geometry for saved axis
        saved_geom = new_axis.create_geometry()
        self.saved_axes_geoms.append(saved_geom)

        # Reset for next axis
        self.current_transform = np.eye(4)
        self.current_scale = 0.2
        self.next_axis_id += 1

        console.print(f"[green]Added axis {new_axis.axis_id} at position {new_axis.transform[:3, 3]}")
        self.update_visualization()

    def save_axes_to_file(self):
        """Save all axes to JSON file."""
        axes_data = {
            'axes': [axis.to_dict() for axis in self.saved_axes],
            'num_axes': len(self.saved_axes)
        }

        output_path = self.data.data_dir / "custom_axes.json"
        with open(output_path, 'w') as f:
            json.dump(axes_data, f, indent=2)

        console.print(f"[green]Saved {len(self.saved_axes)} axes to {output_path}")

    def start_adding_axes(self):
        """Start the interactive axes addition tool."""
        console.print("[bold green]Starting F3RM Custom Axes Addition Tool")
        console.print(self.data.get_info())

        console.print("\n[bold blue]Controls:")
        console.print("  Mouse: Normal Open3D viewing controls (camera only)")
        console.print("  Arrow Keys: Rotate around X/Y axes (pitch/yaw)")
        console.print("  'Z'/'X': Rotate around Z-axis (roll)")
        console.print("  'I'/'K': Move forward/back (±Y)")
        console.print("  'J'/'L': Move left/right (±X)")
        console.print("  'U'/'O': Move up/down (±Z)")
        console.print("  '1'/'2': Scale axes length (decrease/increase)")
        console.print("  'A': Add current axes and start new one")
        console.print("  'R': Reset current axes transform")
        console.print("  'Q': Quit and save all axes")
        console.print("\n[bold yellow]Goal: Add custom coordinate axes at multiple locations")
        console.print("[bold yellow]Note: Camera view is preserved during transforms")

        # Load RGB pointcloud
        self.original_pcd = self.data.rgb_pointcloud

        # Create reference geometries
        self.reference_geoms = self.create_reference_geometries()

        # Initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="F3RM Custom Axes Addition Tool",
            width=1400, height=900
        )

        # Register key callbacks
        self.vis.register_key_callback(ord('A'), lambda vis: self.add_current_axis())
        self.vis.register_key_callback(ord('R'), lambda vis: self.reset_current_transform())
        self.vis.register_key_callback(ord('Q'), lambda vis: self.save_and_exit(vis))

        # Arrow key rotation controls
        self.vis.register_key_callback(265, lambda vis: self.apply_rotation('x', 1))    # Up Arrow: +X rotation
        self.vis.register_key_callback(264, lambda vis: self.apply_rotation('x', -1))   # Down Arrow: -X rotation
        self.vis.register_key_callback(263, lambda vis: self.apply_rotation('y', 1))    # Left Arrow: +Y rotation
        self.vis.register_key_callback(262, lambda vis: self.apply_rotation('y', -1))   # Right Arrow: -Y rotation

        # Z/X keys for Z rotation
        self.vis.register_key_callback(ord('Z'), lambda vis: self.apply_rotation('z', 1))    # Z: +Z rotation
        self.vis.register_key_callback(ord('X'), lambda vis: self.apply_rotation('z', -1))   # X: -Z rotation

        # IJKL controls for translation
        self.vis.register_key_callback(ord('I'), lambda vis: self.apply_translation(0, 1, 0))   # I: +Y (forward)
        self.vis.register_key_callback(ord('K'), lambda vis: self.apply_translation(0, -1, 0))  # K: -Y (back)
        self.vis.register_key_callback(ord('J'), lambda vis: self.apply_translation(-1, 0, 0))  # J: -X (left)
        self.vis.register_key_callback(ord('L'), lambda vis: self.apply_translation(1, 0, 0))   # L: +X (right)

        # U/O for up/down translation
        self.vis.register_key_callback(ord('U'), lambda vis: self.apply_translation(0, 0, 1))   # U: +Z (up)
        self.vis.register_key_callback(ord('O'), lambda vis: self.apply_translation(0, 0, -1))  # O: -Z (down)

        # 1/2 for scaling
        self.vis.register_key_callback(ord('1'), lambda vis: self.apply_scale(-1))  # 1: decrease scale
        self.vis.register_key_callback(ord('2'), lambda vis: self.apply_scale(1))   # 2: increase scale

        # Initial visualization
        self.update_visualization()

        console.print("\n[dim]Detailed Controls:")
        console.print("  ↑/↓ Arrow: Rotate around X-axis (pitch)")
        console.print("  ←/→ Arrow: Rotate around Y-axis (yaw)")
        console.print("  Z/X: Rotate around Z-axis (roll)")
        console.print("  I/K: Move forward/back (±Y)")
        console.print("  J/L: Move left/right (±X)")
        console.print("  U/O: Move up/down (±Z)")
        console.print("  1/2: Scale axes length (decrease/increase)")
        console.print("  A: Add current axes and start new one")
        console.print("  R: Reset current axes transform")
        console.print("  Q: Quit and save all axes")
        console.print(f"\n[dim]Step sizes: Rotation={self.rotation_step}°, Translation={self.translation_step}, Scale={self.scale_step}")

        # Run visualization
        self.vis.run()
        self.vis.destroy_window()

    def save_and_exit(self, vis):
        """Save axes and exit."""
        if self.saved_axes:
            self.save_axes_to_file()
        else:
            console.print("[yellow]No axes to save")
        vis.close()
        return True


def load_custom_axes(data_dir: Path) -> List[CustomAxis]:
    """Load custom axes from JSON file."""
    axes_path = data_dir / "custom_axes.json"
    if not axes_path.exists():
        return []

    with open(axes_path, 'r') as f:
        axes_data = json.load(f)

    axes = [CustomAxis.from_dict(axis_dict) for axis_dict in axes_data['axes']]
    console.print(f"[green]Loaded {len(axes)} custom axes from {axes_path}")
    return axes


def add_custom_axes(data_dir: Path, rotation_step: float = 10.0,
                    translation_step: float = 0.1, scale_step: float = 0.1) -> None:
    """
    Main function to add custom axes.

    Args:
        data_dir: Directory containing exported pointcloud data
        rotation_step: Rotation step size in degrees (default: 10.0)
        translation_step: Translation step size in units (default: 0.1)
        scale_step: Scale step size in units (default: 0.1)
    """
    console.print(f"[bold blue]F3RM Custom Axes Addition Tool")
    console.print(f"[bold blue]Data directory: {data_dir}")
    console.print(f"[bold blue]Rotation step: {rotation_step}° | Translation step: {translation_step} | Scale step: {scale_step}")

    # Load pointcloud data
    try:
        data = FeaturePointcloudData(data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")
    except Exception as e:
        console.print(f"[red]Error loading pointcloud data: {e}")
        return

    # Start axes addition tool
    axes_adder = InteractiveAxesAdder(data, rotation_step, translation_step, scale_step)
    axes_adder.start_adding_axes()


def visualize_with_axes(data_dir: Path) -> None:
    """
    Main function to visualize RGB pointcloud with custom axes.

    Args:
        data_dir: Directory containing exported pointcloud data
    """
    console.print(f"[bold blue]F3RM RGB Pointcloud with Custom Axes Visualizer")
    console.print(f"[bold blue]Data directory: {data_dir}")

    # Load pointcloud data
    try:
        data = FeaturePointcloudData(data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")
    except Exception as e:
        console.print(f"[red]Error loading pointcloud data: {e}")
        return

    # Load custom axes
    custom_axes = load_custom_axes(data_dir)
    if not custom_axes:
        console.print("[yellow]No custom axes found. Run in 'add' mode first to create axes.")

    console.print("[bold green]Visualizing RGB pointcloud with custom axes...")
    console.print(data.get_info())

    # Get RGB pointcloud
    pcd = data.rgb_pointcloud
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

    # Create custom axes geometries
    axes_geometries = []
    for axis in custom_axes:
        axis_geom = axis.create_geometry()
        axes_geometries.append(axis_geom)

    # Combine all geometries
    all_geometries = [pcd] + reference_geoms + axes_geometries

    console.print(f"[cyan]Showing {len(custom_axes)} custom axes")
    console.print("[dim]Reference guides: Gray=original bounds; Light gray=grid")

    # Simple visualization
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name="F3RM RGB Pointcloud with Custom Axes",
        width=1200,
        height=800,
        left=50,
        top=50
    )


def main():
    parser = argparse.ArgumentParser(description="Add or visualize custom axes with F3RM RGB pointclouds")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing exported pointcloud data")
    parser.add_argument("--mode", choices=["add", "visualize"], default="add",
                        help="Tool mode: 'add' for interactive axes placement, 'visualize' for viewing with axes")

    # Add mode arguments
    parser.add_argument("--rotation-step", type=float, default=10.0,
                        help="Rotation step size in degrees (default: 10.0, add mode)")
    parser.add_argument("--translation-step", type=float, default=0.1,
                        help="Translation step size in units (default: 0.1, add mode)")
    parser.add_argument("--scale-step", type=float, default=0.1,
                        help="Scale step size in units (default: 0.1, add mode)")

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
            add_custom_axes(args.data_dir, args.rotation_step, args.translation_step, args.scale_step)
        elif args.mode == "visualize":
            visualize_with_axes(args.data_dir)
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
