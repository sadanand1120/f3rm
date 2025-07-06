#!/usr/bin/env python3
"""
F3RM Pointcloud Alignment Tool

This script provides interactive alignment of F3RM pointclouds with coordinate axes.
Use this after export_feature_pointcloud.py and before visualize_feature_pointcloud.py
to properly align your pointcloud with the coordinate system.

Usage:
    python align_pointcloud.py --data-dir exports/pointcloud_data/

Controls:
    - Mouse: Normal Open3D viewing controls (camera only)
    - Arrow Keys: Rotate around X/Y axes (pitch/yaw)
    - Z/X: Rotate around Z-axis (roll)
    - I/K: Move forward/back (±Y)
    - J/L: Move left/right (±X)  
    - U/O: Move up/down (±Z)
    - Press 'R' to reset transform
    - Press 'S' to save transform and exit
    - Press 'Q' to quit without saving
    
Note: Camera view is preserved during transforms for real-time visual feedback.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from rich.console import Console

# Import reference geometry functions from visualize script
from f3rm.visualize_feature_pointcloud import (
    FeaturePointcloudData,
    create_coordinate_frame,
    create_bounding_box_lines,
    create_grid_lines
)

console = Console()


class InteractiveAlignmentTool:
    """Interactive tool for aligning pointclouds with coordinate axes."""

    def __init__(self, data: FeaturePointcloudData):
        self.data = data
        self.transform = np.eye(4)  # Current transformation matrix
        self.vis = None

        # Store original geometries
        self.original_pcd = None
        self.transformed_pcd = None
        self.reference_geoms = []

    def create_reference_geometries(self) -> list:
        """Create coordinate frame, bounding box, and grid."""
        geometries = []

        # Get bounding box from metadata
        bbox_min = np.array(self.data.metadata['bbox_min'])
        bbox_max = np.array(self.data.metadata['bbox_max'])

        # Coordinate frame at origin (larger for visibility)
        geometries.extend(create_coordinate_frame(size=0.3))

        # Bounding box wireframe
        geometries.extend(create_bounding_box_lines(bbox_min, bbox_max))

        # Grid lines for spatial reference
        geometries.extend(create_grid_lines(bbox_min, bbox_max, grid_size=0.2))

        return geometries

    def update_transformed_pointcloud(self):
        """Update the transformed pointcloud based on current transform while preserving camera view."""
        if self.original_pcd is None or self.vis is None:
            return

        # Save current camera parameters
        view_control = self.vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()

        # Apply transform to pointcloud
        self.transformed_pcd = o3d.geometry.PointCloud(self.original_pcd)
        self.transformed_pcd.transform(self.transform)

        # Clear and re-add geometries
        self.vis.clear_geometries()

        # Add transformed pointcloud
        self.vis.add_geometry(self.transformed_pcd)

        # Add reference geometries
        for geom in self.reference_geoms:
            self.vis.add_geometry(geom)

        # Restore camera parameters to maintain view
        view_control.convert_from_pinhole_camera_parameters(camera_params)

        # Update the visualization
        self.vis.poll_events()
        self.vis.update_renderer()

    def key_callback(self, vis, key, action):
        """Handle key press events."""
        if action == 1:  # Key press
            if key == 82:  # 'R' key - reset transform
                self.reset_transform()
                return True
            elif key == 83:  # 'S' key - save and exit
                self.save_transform()
                vis.close()
                return True
            elif key == 81:  # 'Q' key - quit without saving
                console.print("[yellow]Exiting without saving transform")
                vis.close()
                return True
        elif action == 0:  # Key release
            return False

        return False

    def reset_transform(self):
        """Reset transformation to identity."""
        self.transform = np.eye(4)
        self.update_transformed_pointcloud()
        console.print("[green]Transform reset to identity")

    def apply_rotation(self, axis: str, angle_deg: float):
        """Apply rotation around specified axis."""
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
        self.transform = transform @ self.transform
        self.update_transformed_pointcloud()

    def apply_translation(self, dx: float, dy: float, dz: float):
        """Apply translation."""
        translation = np.eye(4)
        translation[:3, 3] = [dx, dy, dz]

        self.transform = translation @ self.transform
        self.update_transformed_pointcloud()

    def start_alignment(self):
        """Start the interactive alignment tool."""
        console.print("[bold green]Starting F3RM Pointcloud Alignment Tool")
        console.print(self.data.get_info())

        console.print("\n[bold blue]Controls:")
        console.print("  Mouse: Normal Open3D viewing controls (camera only)")
        console.print("  'R': Reset transform to identity")
        console.print("  'S': Save transform and exit")
        console.print("  'Q': Quit without saving")
        console.print("  Arrow Keys: Rotate around X/Y axes (pitch/yaw)")
        console.print("  'Z'/'X': Rotate around Z-axis (roll)")
        console.print("  'I'/'K': Move forward/back (±Y)")
        console.print("  'J'/'L': Move left/right (±X)")
        console.print("  'U'/'O': Move up/down (±Z)")
        console.print("\n[bold yellow]Goal: Align pointcloud so floor is on XY plane and objects are properly oriented")
        console.print("[bold yellow]Note: Camera view is preserved during transforms for real-time feedback")

        # Load RGB pointcloud
        self.original_pcd = self.data.rgb_pointcloud
        self.transformed_pcd = o3d.geometry.PointCloud(self.original_pcd)

        # Create reference geometries
        self.reference_geoms = self.create_reference_geometries()

        # Initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="F3RM Pointcloud Alignment Tool",
            width=1400, height=900
        )

        # Register callbacks for save/reset/quit
        self.vis.register_key_callback(ord('R'), lambda vis: self.reset_transform())
        self.vis.register_key_callback(ord('S'), lambda vis: self.save_and_exit(vis))
        self.vis.register_key_callback(ord('Q'), lambda vis: self.quit_without_save(vis))

        # Arrow key rotation controls (Open3D key codes)
        # Up Arrow = 265, Down Arrow = 264, Left Arrow = 263, Right Arrow = 262
        self.vis.register_key_callback(265, lambda vis: self.apply_rotation('x', 10))    # Up Arrow: +X rotation
        self.vis.register_key_callback(264, lambda vis: self.apply_rotation('x', -10))   # Down Arrow: -X rotation
        self.vis.register_key_callback(263, lambda vis: self.apply_rotation('y', 10))    # Left Arrow: +Y rotation
        self.vis.register_key_callback(262, lambda vis: self.apply_rotation('y', -10))   # Right Arrow: -Y rotation

        # Z/X keys for Z rotation (much more convenient than Page Up/Down)
        self.vis.register_key_callback(ord('Z'), lambda vis: self.apply_rotation('z', 10))    # Z: +Z rotation
        self.vis.register_key_callback(ord('X'), lambda vis: self.apply_rotation('z', -10))   # X: -Z rotation

        # IJKL controls for translation (like vim navigation)
        self.vis.register_key_callback(ord('I'), lambda vis: self.apply_translation(0, 0.1, 0))   # I: +Y (forward)
        self.vis.register_key_callback(ord('K'), lambda vis: self.apply_translation(0, -0.1, 0))  # K: -Y (back)
        self.vis.register_key_callback(ord('J'), lambda vis: self.apply_translation(-0.1, 0, 0))  # J: -X (left)
        self.vis.register_key_callback(ord('L'), lambda vis: self.apply_translation(0.1, 0, 0))   # L: +X (right)

        # U/O for up/down translation
        self.vis.register_key_callback(ord('U'), lambda vis: self.apply_translation(0, 0, 0.1))   # U: +Z (up)
        self.vis.register_key_callback(ord('O'), lambda vis: self.apply_translation(0, 0, -0.1))  # O: -Z (down)

        # Add initial geometries
        self.update_transformed_pointcloud()

        console.print("\n[dim]Detailed Controls:")
        console.print("  ↑/↓ Arrow: Rotate around X-axis (pitch)")
        console.print("  ←/→ Arrow: Rotate around Y-axis (yaw)")
        console.print("  Z/X: Rotate around Z-axis (roll)")
        console.print("  I/K: Move forward/back (±Y)")
        console.print("  J/L: Move left/right (±X)")
        console.print("  U/O: Move up/down (±Z)")
        console.print("  R: Reset transform")
        console.print("  S: Save and exit")
        console.print("  Q: Quit without saving")
        console.print("\n[dim]Note: Use small incremental adjustments for precise alignment")

        # Run visualization
        self.vis.run()
        self.vis.destroy_window()

    def save_and_exit(self, vis):
        """Save transform and exit."""
        self.save_transform()
        vis.close()
        return True

    def quit_without_save(self, vis):
        """Quit without saving."""
        console.print("[yellow]Exiting without saving transform")
        vis.close()
        return True

    def save_transform(self):
        """Save the current transform and apply it to all data."""
        console.print(f"\n[bold green]Saving alignment transform...")

        # Print transform matrix
        console.print("[cyan]Applied transformation matrix:")
        for i, row in enumerate(self.transform):
            console.print(f"  [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")

        # Apply transform to all pointcloud data
        self.apply_transform_to_data()

        console.print("[bold green]✓ Alignment transform saved and applied to all data")

    def apply_transform_to_data(self):
        """Apply the transformation to all pointcloud data and save."""
        data_dir = self.data.data_dir

        # 1. Transform and save RGB pointcloud
        rgb_pcd = o3d.io.read_point_cloud(str(data_dir / "pointcloud_rgb.ply"))
        rgb_pcd.transform(self.transform)
        o3d.io.write_point_cloud(str(data_dir / "pointcloud_rgb.ply"), rgb_pcd)
        console.print("[green]✓ Updated RGB pointcloud")

        # 2. Transform and save PCA pointcloud
        pca_pcd = o3d.io.read_point_cloud(str(data_dir / "pointcloud_feature_pca.ply"))
        pca_pcd.transform(self.transform)
        o3d.io.write_point_cloud(str(data_dir / "pointcloud_feature_pca.ply"), pca_pcd)
        console.print("[green]✓ Updated PCA pointcloud")

        # 3. Transform points array
        points = np.load(data_dir / "points.npy")

        # Apply transform to points (homogeneous coordinates)
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_transformed = (self.transform @ points_homo.T).T[:, :3]

        np.save(data_dir / "points.npy", points_transformed.astype(np.float32))
        console.print("[green]✓ Updated points array")

        # 4. Update metadata with transform and new bounding box
        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Add transform to metadata
        metadata['alignment_transform'] = self.transform.tolist()
        metadata['alignment_applied'] = True

        # Update bounding box
        new_bbox_min = points_transformed.min(axis=0).tolist()
        new_bbox_max = points_transformed.max(axis=0).tolist()
        metadata['bbox_min'] = new_bbox_min
        metadata['bbox_max'] = new_bbox_max

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        console.print("[green]✓ Updated metadata with transform and new bounding box")

        # 5. Note: Features don't need transformation as they're view-invariant
        console.print("[dim]Note: Feature vectors unchanged (view-invariant)")


def align_pointcloud(data_dir: Path) -> None:
    """
    Main function to align pointcloud data.

    Args:
        data_dir: Directory containing exported pointcloud data
    """
    console.print(f"[bold blue]F3RM Pointcloud Alignment Tool")
    console.print(f"[bold blue]Data directory: {data_dir}")

    # Load pointcloud data
    try:
        data = FeaturePointcloudData(data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")
    except Exception as e:
        console.print(f"[red]Error loading pointcloud data: {e}")
        return

    # Check if already aligned
    if 'alignment_applied' in data.metadata and data.metadata['alignment_applied']:
        console.print("[yellow]Warning: This pointcloud has already been aligned")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            console.print("[yellow]Alignment cancelled")
            return

    # Start alignment tool
    alignment_tool = InteractiveAlignmentTool(data)
    alignment_tool.start_alignment()


def main():
    parser = argparse.ArgumentParser(description="Align F3RM feature pointclouds with coordinate axes")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing exported pointcloud data")

    args = parser.parse_args()

    # Validate inputs
    if not args.data_dir.exists():
        console.print(f"[bold red]Data directory not found: {args.data_dir}")
        return

    if not (args.data_dir / "metadata.json").exists():
        console.print(f"[bold red]No metadata.json found. Run export_feature_pointcloud.py first.")
        return

    # Check for required files
    required_files = ["pointcloud_rgb.ply", "pointcloud_feature_pca.ply", "points.npy"]
    missing_files = []
    for file in required_files:
        if not (args.data_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        console.print(f"[bold red]Missing required files: {', '.join(missing_files)}")
        console.print("[bold red]Run export_feature_pointcloud.py first to generate all required files.")
        return

    try:
        align_pointcloud(args.data_dir)
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
