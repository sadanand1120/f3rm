#!/usr/bin/env python3
"""
F3RM Pointcloud Alignment Tool

This script provides interactive alignment of F3RM pointclouds with coordinate axes,
and interactive bounding box filtering with visual feedback.

Use this after f3rm/manual/export_feature_pointcloud.py and before f3rm/manual/visualize_feature_pointcloud.py
to properly align your pointcloud with the coordinate system and/or filter out unwanted regions.

Usage:
    python f3rm/manual/align_pointcloud.py --data-dir exports/pointcloud_data/ --mode align
    python f3rm/manual/align_pointcloud.py --data-dir exports/pointcloud_data/ --mode filter

Align Mode Controls:
    - Mouse: Normal Open3D viewing controls (camera only)
    - Arrow Keys: Rotate around X/Y axes (pitch/yaw)
    - Z/X: Rotate around Z-axis (roll)
    - I/K: Move forward/back (±Y)
    - J/L: Move left/right (±X)  
    - U/O: Move up/down (±Z)
    - Press 'R' to reset transform
    - Press 'S' to save transform and exit
    - Press 'Q' to quit without saving

Filter Mode Controls:
    - Mouse: Normal Open3D viewing controls
    - 'N': Cycle between X, Y, Z axis filtering
    - ↑/↓ Arrow: Adjust max bound for current axis
    - ←/→ Arrow: Adjust min bound for current axis
    - 'R': Reset all filters to original bounds
    - 'S': Save filter bounds and exit
    - 'Q': Quit without saving
    
Note: Camera view is preserved during transforms for real-time visual feedback.
Colored planes show the current filtering bounds (red=X, green=Y, blue=Z).
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
from f3rm.manual.visualize_feature_pointcloud import (
    FeaturePointcloudData,
    create_coordinate_frame,
    create_bounding_box_lines,
    create_grid_lines
)

console = Console()


def create_filter_plane(axis: int, value: float, bbox_min: np.ndarray, bbox_max: np.ndarray,
                        is_min: bool = True, alpha: float = 0.3) -> o3d.geometry.TriangleMesh:
    """
    Create a colored plane to visualize filtering bounds.

    Args:
        axis: 0=X, 1=Y, 2=Z
        value: Position of the plane along the axis
        bbox_min, bbox_max: Overall bounding box for plane size
        is_min: True for min bound plane, False for max bound plane
        alpha: Transparency (not used in Open3D mesh, but for reference)

    Returns:
        Colored triangle mesh plane
    """
    # Colors for each axis (red=X, green=Y, blue=Z)
    axis_colors = [
        [1.0, 0.3, 0.3],  # Red for X
        [0.3, 1.0, 0.3],  # Green for Y
        [0.3, 0.3, 1.0],  # Blue for Z
    ]

    # Make min planes slightly darker
    color = axis_colors[axis]
    if is_min:
        color = [c * 0.7 for c in color]

    # Create plane vertices based on axis
    if axis == 0:  # X axis plane (YZ plane)
        vertices = [
            [value, bbox_min[1], bbox_min[2]],
            [value, bbox_max[1], bbox_min[2]],
            [value, bbox_max[1], bbox_max[2]],
            [value, bbox_min[1], bbox_max[2]]
        ]
    elif axis == 1:  # Y axis plane (XZ plane)
        vertices = [
            [bbox_min[0], value, bbox_min[2]],
            [bbox_max[0], value, bbox_min[2]],
            [bbox_max[0], value, bbox_max[2]],
            [bbox_min[0], value, bbox_max[2]]
        ]
    else:  # Z axis plane (XY plane)
        vertices = [
            [bbox_min[0], bbox_min[1], value],
            [bbox_max[0], bbox_min[1], value],
            [bbox_max[0], bbox_max[1], value],
            [bbox_min[0], bbox_max[1], value]
        ]

    # Create triangles (two triangles per quad)
    triangles = [
        [0, 1, 2],
        [0, 2, 3]
    ]

    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)

    return mesh


class InteractiveAlignmentTool:
    """Interactive tool for aligning pointclouds with coordinate axes."""

    def __init__(self, data: FeaturePointcloudData, rotation_step: float = 10.0, translation_step: float = 0.1):
        self.data = data
        self.transform = np.eye(4)  # Current transformation matrix
        self.vis = None

        # Step sizes for transformations
        self.rotation_step = rotation_step  # degrees
        self.translation_step = translation_step  # units

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
        self.transform = transform @ self.transform
        self.update_transformed_pointcloud()

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
        self.vis.register_key_callback(265, lambda vis: self.apply_rotation('x', 1))    # Up Arrow: +X rotation
        self.vis.register_key_callback(264, lambda vis: self.apply_rotation('x', -1))   # Down Arrow: -X rotation
        self.vis.register_key_callback(263, lambda vis: self.apply_rotation('y', 1))    # Left Arrow: +Y rotation
        self.vis.register_key_callback(262, lambda vis: self.apply_rotation('y', -1))   # Right Arrow: -Y rotation

        # Z/X keys for Z rotation (much more convenient than Page Up/Down)
        self.vis.register_key_callback(ord('Z'), lambda vis: self.apply_rotation('z', 1))    # Z: +Z rotation
        self.vis.register_key_callback(ord('X'), lambda vis: self.apply_rotation('z', -1))   # X: -Z rotation

        # IJKL controls for translation (like vim navigation)
        self.vis.register_key_callback(ord('I'), lambda vis: self.apply_translation(0, 1, 0))   # I: +Y (forward)
        self.vis.register_key_callback(ord('K'), lambda vis: self.apply_translation(0, -1, 0))  # K: -Y (back)
        self.vis.register_key_callback(ord('J'), lambda vis: self.apply_translation(-1, 0, 0))  # J: -X (left)
        self.vis.register_key_callback(ord('L'), lambda vis: self.apply_translation(1, 0, 0))   # L: +X (right)

        # U/O for up/down translation
        self.vis.register_key_callback(ord('U'), lambda vis: self.apply_translation(0, 0, 1))   # U: +Z (up)
        self.vis.register_key_callback(ord('O'), lambda vis: self.apply_translation(0, 0, -1))  # O: -Z (down)

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
        console.print(f"\n[dim]Step sizes: Rotation={self.rotation_step}°, Translation={self.translation_step} units")
        console.print("[dim]Note: Use small incremental adjustments for precise alignment")

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


class InteractiveFilterTool:
    """Interactive tool for setting bounding box filters with visual feedback."""

    def __init__(self, data: FeaturePointcloudData, filter_step: float = 0.05):
        self.data = data
        self.vis = None
        self.filter_step = filter_step

        # Initialize filter bounds to original bounding box
        self.original_bbox_min = np.array(self.data.metadata['bbox_min'])
        self.original_bbox_max = np.array(self.data.metadata['bbox_max'])
        self.filter_min = self.original_bbox_min.copy()
        self.filter_max = self.original_bbox_max.copy()

        # Current axis being filtered (0=X, 1=Y, 2=Z)
        self.current_axis = 0
        self.axis_names = ['X', 'Y', 'Z']
        self.axis_colors = ['red', 'green', 'blue']

        # Store geometries
        self.original_pcd = None
        self.filtered_pcd = None
        self.reference_geoms = []
        self.filter_planes = []

    def create_reference_geometries(self) -> list:
        """Create coordinate frame, bounding box, and grid."""
        geometries = []

        # Coordinate frame at origin (larger for visibility)
        geometries.extend(create_coordinate_frame(size=0.3))

        # Original bounding box wireframe (gray)
        geometries.extend(create_bounding_box_lines(self.original_bbox_min, self.original_bbox_max))

        # Grid lines for spatial reference
        geometries.extend(create_grid_lines(self.original_bbox_min, self.original_bbox_max, grid_size=0.2))

        return geometries

    def create_filter_planes(self) -> list:
        """Create colored planes showing current filter bounds."""
        planes = []

        for axis in range(3):
            # Min plane
            min_plane = create_filter_plane(
                axis, self.filter_min[axis],
                self.original_bbox_min, self.original_bbox_max,
                is_min=True
            )
            planes.append(min_plane)

            # Max plane
            max_plane = create_filter_plane(
                axis, self.filter_max[axis],
                self.original_bbox_min, self.original_bbox_max,
                is_min=False
            )
            planes.append(max_plane)

        return planes

    def apply_filter(self):
        """Apply current filter bounds to the pointcloud."""
        points = np.asarray(self.original_pcd.points)
        colors = np.asarray(self.original_pcd.colors)

        # Create mask for points within filter bounds
        within_bounds = np.all(
            (points >= self.filter_min) & (points <= self.filter_max),
            axis=1
        )

        # Apply filter
        filtered_points = points[within_bounds]
        filtered_colors = colors[within_bounds]

        # Update filtered pointcloud
        self.filtered_pcd = o3d.geometry.PointCloud()
        self.filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        self.filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        return len(filtered_points), len(points)

    def update_visualization(self):
        """Update the visualization with current filter state."""
        if self.vis is None:
            return

        # Save current camera parameters
        view_control = self.vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()

        # Apply current filter
        filtered_count, total_count = self.apply_filter()

        # Create new filter planes
        self.filter_planes = self.create_filter_planes()

        # Clear and re-add geometries
        self.vis.clear_geometries()

        # Add filtered pointcloud
        self.vis.add_geometry(self.filtered_pcd)

        # Add reference geometries
        for geom in self.reference_geoms:
            self.vis.add_geometry(geom)

        # Add filter planes
        for plane in self.filter_planes:
            self.vis.add_geometry(plane)

        # Restore camera parameters to maintain view
        view_control.convert_from_pinhole_camera_parameters(camera_params)

        # Update status
        axis_name = self.axis_names[self.current_axis]
        axis_color = self.axis_colors[self.current_axis]
        console.print(f"[{axis_color}]Current axis: {axis_name} | "
                      f"Min: {self.filter_min[self.current_axis]:.3f} | "
                      f"Max: {self.filter_max[self.current_axis]:.3f} | "
                      f"Points: {filtered_count:,}/{total_count:,}")

    def cycle_axis(self):
        """Cycle to the next axis (X -> Y -> Z -> X)."""
        self.current_axis = (self.current_axis + 1) % 3
        console.print(f"[bold {self.axis_colors[self.current_axis]}]Switched to {self.axis_names[self.current_axis]} axis filtering")
        self.update_visualization()

    def adjust_bound(self, is_max: bool, direction: int):
        """Adjust min or max bound for current axis.

        Args:
            is_max: True to adjust max bound, False for min bound
            direction: 1 for increase, -1 for decrease
        """
        axis = self.current_axis
        step = self.filter_step * direction

        if is_max:
            new_value = self.filter_max[axis] + step
            # Ensure max >= min
            if new_value >= self.filter_min[axis]:
                self.filter_max[axis] = new_value
        else:
            new_value = self.filter_min[axis] + step
            # Ensure min <= max
            if new_value <= self.filter_max[axis]:
                self.filter_min[axis] = new_value

        self.update_visualization()

    def reset_filters(self):
        """Reset all filter bounds to original bounding box."""
        self.filter_min = self.original_bbox_min.copy()
        self.filter_max = self.original_bbox_max.copy()
        console.print("[green]Filter bounds reset to original bounding box")
        self.update_visualization()

    def start_filtering(self):
        """Start the interactive filter tool."""
        console.print("[bold green]Starting F3RM Pointcloud Filter Tool")
        console.print(self.data.get_info())

        console.print("\n[bold blue]Filter Mode Controls:")
        console.print("  Mouse: Normal Open3D viewing controls")
        console.print("  'N': Cycle between X, Y, Z axis filtering")
        console.print("  ↑/↓ Arrow: Adjust max bound for current axis")
        console.print("  ←/→ Arrow: Adjust min bound for current axis")
        console.print("  'R': Reset all filters to original bounds")
        console.print("  'S': Save filter bounds and exit")
        console.print("  'Q': Quit without saving")
        console.print("\n[bold yellow]Goal: Set bounding box filters to remove unwanted regions")
        console.print("[bold yellow]Colored planes show filter bounds: Red=X, Green=Y, Blue=Z")
        console.print("[bold yellow]Darker planes = min bounds, Brighter planes = max bounds")

        # Load RGB pointcloud
        self.original_pcd = self.data.rgb_pointcloud

        # Create reference geometries
        self.reference_geoms = self.create_reference_geometries()

        # Initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="F3RM Pointcloud Filter Tool",
            width=1400, height=900
        )

        # Register key callbacks
        self.vis.register_key_callback(ord('N'), lambda vis: self.cycle_axis())
        self.vis.register_key_callback(ord('R'), lambda vis: self.reset_filters())
        self.vis.register_key_callback(ord('S'), lambda vis: self.save_and_exit(vis))
        self.vis.register_key_callback(ord('Q'), lambda vis: self.quit_without_save(vis))

        # Arrow key controls for adjusting bounds
        # Up Arrow = 265, Down Arrow = 264, Left Arrow = 263, Right Arrow = 262
        self.vis.register_key_callback(265, lambda vis: self.adjust_bound(True, 1))    # Up: increase max
        self.vis.register_key_callback(264, lambda vis: self.adjust_bound(True, -1))   # Down: decrease max
        self.vis.register_key_callback(262, lambda vis: self.adjust_bound(False, 1))   # Right: increase min
        self.vis.register_key_callback(263, lambda vis: self.adjust_bound(False, -1))  # Left: decrease min

        # Initial visualization update
        self.update_visualization()

        console.print("\n[dim]Detailed Controls:")
        console.print("  N: Cycle axis (X → Y → Z → X)")
        console.print("  ↑/↓: Adjust max bound (+/-)")
        console.print("  ←/→: Adjust min bound (-/+)")
        console.print("  R: Reset filters")
        console.print("  S: Save and exit")
        console.print("  Q: Quit without saving")
        console.print(f"\n[dim]Filter step size: {self.filter_step} units")
        console.print(f"[bold {self.axis_colors[self.current_axis]}]Starting with {self.axis_names[self.current_axis]} axis")

        # Run visualization
        self.vis.run()
        self.vis.destroy_window()

    def save_and_exit(self, vis):
        """Save filter bounds and exit."""
        self.save_filter_bounds()
        vis.close()
        return True

    def quit_without_save(self, vis):
        """Quit without saving."""
        console.print("[yellow]Exiting without saving filter bounds")
        vis.close()
        return True

    def save_filter_bounds(self):
        """Apply the current filter bounds to all data and save."""
        console.print(f"\n[bold green]Applying filter bounds to all data...")

        # Print filter bounds
        console.print("[cyan]Filter bounds:")
        for i, axis_name in enumerate(self.axis_names):
            color = self.axis_colors[i]
            console.print(f"  [{color}]{axis_name}: {self.filter_min[i]:.3f} to {self.filter_max[i]:.3f}")

        # Apply filter to all pointcloud data
        self.apply_filter_to_data()

        console.print("[bold green]✓ Filter bounds applied and data updated")

    def apply_filter_to_data(self):
        """Apply the filter bounds to all pointcloud data and save."""
        data_dir = self.data.data_dir

        # Load original points array to create the filter mask
        points = np.load(data_dir / "points.npy")

        # Create mask for points within filter bounds
        within_bounds = np.all(
            (points >= self.filter_min) & (points <= self.filter_max),
            axis=1
        )

        original_count = len(points)
        filtered_count = within_bounds.sum()

        console.print(f"[cyan]Filtering {original_count:,} points...")
        console.print(f"[cyan]Keeping {filtered_count:,} points ({100*filtered_count/original_count:.1f}%)")
        console.print(f"[cyan]Removing {original_count - filtered_count:,} points ({100*(original_count - filtered_count)/original_count:.1f}%)")

        # 1. Filter and save RGB pointcloud
        rgb_pcd = o3d.io.read_point_cloud(str(data_dir / "pointcloud_rgb.ply"))
        rgb_points = np.asarray(rgb_pcd.points)
        rgb_colors = np.asarray(rgb_pcd.colors)

        filtered_rgb_pcd = o3d.geometry.PointCloud()
        filtered_rgb_pcd.points = o3d.utility.Vector3dVector(rgb_points[within_bounds])
        filtered_rgb_pcd.colors = o3d.utility.Vector3dVector(rgb_colors[within_bounds])

        o3d.io.write_point_cloud(str(data_dir / "pointcloud_rgb.ply"), filtered_rgb_pcd)
        console.print("[green]✓ Updated RGB pointcloud")

        # 2. Filter and save PCA pointcloud
        pca_pcd = o3d.io.read_point_cloud(str(data_dir / "pointcloud_feature_pca.ply"))
        pca_points = np.asarray(pca_pcd.points)
        pca_colors = np.asarray(pca_pcd.colors)

        filtered_pca_pcd = o3d.geometry.PointCloud()
        filtered_pca_pcd.points = o3d.utility.Vector3dVector(pca_points[within_bounds])
        filtered_pca_pcd.colors = o3d.utility.Vector3dVector(pca_colors[within_bounds])

        o3d.io.write_point_cloud(str(data_dir / "pointcloud_feature_pca.ply"), filtered_pca_pcd)
        console.print("[green]✓ Updated PCA pointcloud")

        # 3. Filter and save points array
        filtered_points = points[within_bounds]
        np.save(data_dir / "points.npy", filtered_points.astype(np.float32))
        console.print("[green]✓ Updated points array")

        # 4. Filter and save features array
        features_file = self.data.metadata['files']['features']
        features = np.load(data_dir / features_file)
        filtered_features = features[within_bounds]
        np.save(data_dir / features_file, filtered_features)
        console.print("[green]✓ Updated features array")

        # 5. Update metadata with new counts and bounding box
        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Update metadata
        metadata['num_points'] = int(filtered_count)
        metadata['bbox_min'] = filtered_points.min(axis=0).tolist()
        metadata['bbox_max'] = filtered_points.max(axis=0).tolist()
        metadata['filter_applied'] = {
            'bounds': {
                'min': self.filter_min.tolist(),
                'max': self.filter_max.tolist()
            },
            'original_count': int(original_count),
            'filtered_count': int(filtered_count),
            'applied': True
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        console.print("[green]✓ Updated metadata with new counts and bounding box")

        console.print(f"[bold green]✓ Filtering complete: {filtered_count:,} points saved")


def align_pointcloud(data_dir: Path, rotation_step: float = 10.0, translation_step: float = 0.1) -> None:
    """
    Main function to align pointcloud data.

    Args:
        data_dir: Directory containing exported pointcloud data
        rotation_step: Rotation step size in degrees (default: 10.0)
        translation_step: Translation step size in units (default: 0.1)
    """
    console.print(f"[bold blue]F3RM Pointcloud Alignment Tool")
    console.print(f"[bold blue]Data directory: {data_dir}")
    console.print(f"[bold blue]Rotation step: {rotation_step}° | Translation step: {translation_step}")

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
    alignment_tool = InteractiveAlignmentTool(data, rotation_step, translation_step)
    alignment_tool.start_alignment()


def filter_pointcloud(data_dir: Path, filter_step: float = 0.05) -> None:
    """
    Main function to set bounding box filters for pointcloud data.

    Args:
        data_dir: Directory containing exported pointcloud data
        filter_step: Filter adjustment step size in units (default: 0.05)
    """
    console.print(f"[bold blue]F3RM Pointcloud Filter Tool")
    console.print(f"[bold blue]Data directory: {data_dir}")
    console.print(f"[bold blue]Filter step: {filter_step} units")

    # Load pointcloud data
    try:
        data = FeaturePointcloudData(data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")
    except Exception as e:
        console.print(f"[red]Error loading pointcloud data: {e}")
        return

    # Check if filters already exist
    if 'filter_bounds' in data.metadata and data.metadata['filter_bounds'].get('applied', False):
        console.print("[yellow]Warning: This pointcloud already has filter bounds set")
        console.print(f"[yellow]Current bounds: {data.metadata['filter_bounds']}")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            console.print("[yellow]Filtering cancelled")
            return

    # Start filter tool
    filter_tool = InteractiveFilterTool(data, filter_step)
    filter_tool.start_filtering()


def main():
    parser = argparse.ArgumentParser(description="Align or filter F3RM feature pointclouds")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing exported pointcloud data")
    parser.add_argument("--mode", choices=["align", "filter"], default="align",
                        help="Tool mode: 'align' for alignment, 'filter' for bounding box filtering")

    # Alignment mode arguments
    parser.add_argument("--rotation-step", type=float, default=10.0,
                        help="Rotation step size in degrees (default: 10.0, align mode)")
    parser.add_argument("--translation-step", type=float, default=0.1,
                        help="Translation step size in units (default: 0.1, align mode)")

    # Filter mode arguments
    parser.add_argument("--filter-step", type=float, default=0.05,
                        help="Filter adjustment step size in units (default: 0.05, filter mode)")

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
        if args.mode == "align":
            align_pointcloud(args.data_dir, args.rotation_step, args.translation_step)
        elif args.mode == "filter":
            filter_pointcloud(args.data_dir, args.filter_step)
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
