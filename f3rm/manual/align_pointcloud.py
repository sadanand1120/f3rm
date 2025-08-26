#!/usr/bin/env python3
"""F3RM Pointcloud Alignment Tool - Interactive alignment and filtering with visual feedback."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
from rich.console import Console

from f3rm.manual.visualize_feature_pointcloud import (
    FeaturePointcloudData, create_coordinate_frame, create_bounding_box_lines, create_grid_lines
)

console = Console()


def create_filter_plane(axis: int, value: float, bbox_min: np.ndarray, bbox_max: np.ndarray, is_min: bool = True) -> o3d.geometry.TriangleMesh:
    """Create colored plane for filtering bounds visualization."""
    axis_colors = [[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]]  # Red=X, Green=Y, Blue=Z
    color = [c * 0.7 for c in axis_colors[axis]] if is_min else axis_colors[axis]
    if axis == 0:  # X axis plane (YZ plane)
        vertices = [[value, bbox_min[1], bbox_min[2]], [value, bbox_max[1], bbox_min[2]], [value, bbox_max[1], bbox_max[2]], [value, bbox_min[1], bbox_max[2]]]
    elif axis == 1:  # Y axis plane (XZ plane)
        vertices = [[bbox_min[0], value, bbox_min[2]], [bbox_max[0], value, bbox_min[2]], [bbox_max[0], value, bbox_max[2]], [bbox_min[0], value, bbox_max[2]]]
    else:  # Z axis plane (XY plane)
        vertices = [[bbox_min[0], bbox_min[1], value], [bbox_max[0], bbox_min[1], value], [bbox_max[0], bbox_max[1], value], [bbox_min[0], bbox_max[1], value]]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.paint_uniform_color(color)
    return mesh


class InteractiveAlignmentTool:
    """Interactive tool for aligning pointclouds with coordinate axes."""

    def __init__(self, data: FeaturePointcloudData, rotation_step: float = 5.0, translation_step: float = 0.05):
        self.data = data
        self.transform = np.eye(4)
        self.vis = None
        self.rotation_step = rotation_step
        self.translation_step = translation_step
        self.original_pcd = None
        self.transformed_pcd = None
        self.reference_geoms = []

    def create_reference_geometries(self) -> list:
        """Create coordinate frame, bounding box, and grid."""
        bbox_min, bbox_max = np.array(self.data.metadata['bbox_min']), np.array(self.data.metadata['bbox_max'])
        geometries = []
        geometries.extend(create_coordinate_frame(size=0.3))
        geometries.extend(create_bounding_box_lines(bbox_min, bbox_max))
        geometries.extend(create_grid_lines(bbox_min, bbox_max, grid_size=0.2))
        return geometries

    def update_transformed_pointcloud(self):
        """Update transformed pointcloud while preserving camera view."""
        if self.original_pcd is None or self.vis is None:
            return
        view_control = self.vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        self.transformed_pcd = o3d.geometry.PointCloud(self.original_pcd)
        self.transformed_pcd.transform(self.transform)
        self.vis.clear_geometries()
        self.vis.add_geometry(self.transformed_pcd)
        for geom in self.reference_geoms:
            self.vis.add_geometry(geom)
        view_control.convert_from_pinhole_camera_parameters(camera_params)

    def reset_transform(self):
        """Reset transformation to identity."""
        self.transform = np.eye(4)
        self.update_transformed_pointcloud()
        console.print("[green]Transform reset")

    def apply_rotation(self, axis: str, direction: int = 1):
        """Apply rotation around specified axis."""
        angle_rad = np.radians(self.rotation_step * direction)
        if axis == 'x':
            rot_matrix = np.array([[1, 0, 0], [0, np.cos(angle_rad), -np.sin(angle_rad)], [0, np.sin(angle_rad), np.cos(angle_rad)]])
        elif axis == 'y':
            rot_matrix = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)], [0, 1, 0], [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
        elif axis == 'z':
            rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0], [np.sin(angle_rad), np.cos(angle_rad), 0], [0, 0, 1]])
        else:
            return
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        self.transform = transform @ self.transform
        self.update_transformed_pointcloud()

    def apply_translation(self, dx_direction: int, dy_direction: int, dz_direction: int):
        """Apply translation."""
        dx, dy, dz = self.translation_step * dx_direction, self.translation_step * dy_direction, self.translation_step * dz_direction
        translation = np.eye(4)
        translation[:3, 3] = [dx, dy, dz]
        self.transform = translation @ self.transform
        self.update_transformed_pointcloud()

    def start_alignment(self):
        """Start the interactive alignment tool."""
        console.print("[bold green]F3RM Pointcloud Alignment Tool")
        console.print("\n[bold blue]Controls:")
        console.print("  Mouse: Normal Open3D viewing controls (camera only)")
        console.print("  Arrow Keys: Rotate around X/Y axes (pitch/yaw)")
        console.print("  Z/X: Rotate around Z-axis (roll)")
        console.print("  I/K: Move forward/back (±Y)")
        console.print("  J/L: Move left/right (±X)")
        console.print("  U/O: Move up/down (±Z)")
        console.print("  R: Reset transform")
        console.print("  S: Save transform and exit")
        console.print("  Q: Quit without saving")
        console.print("\n[bold yellow]Goal: Align pointcloud so floor is on XY plane and objects are properly oriented")
        console.print("[bold yellow]Note: Camera view is preserved during transforms for real-time feedback")
        self.original_pcd = self.data.rgb_pointcloud
        self.transformed_pcd = o3d.geometry.PointCloud(self.original_pcd)
        self.reference_geoms = self.create_reference_geometries()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="F3RM Pointcloud Alignment Tool", width=1400, height=900)
        # Register callbacks
        self.vis.register_key_callback(ord('R'), lambda vis: self.reset_transform())
        self.vis.register_key_callback(ord('S'), lambda vis: self.save_and_exit(vis))
        self.vis.register_key_callback(ord('Q'), lambda vis: self.quit_without_save(vis))
        self.vis.register_key_callback(265, lambda vis: self.apply_rotation('x', 1))    # Up Arrow: +X rotation
        self.vis.register_key_callback(264, lambda vis: self.apply_rotation('x', -1))   # Down Arrow: -X rotation
        self.vis.register_key_callback(263, lambda vis: self.apply_rotation('y', 1))    # Left Arrow: +Y rotation
        self.vis.register_key_callback(262, lambda vis: self.apply_rotation('y', -1))   # Right Arrow: -Y rotation
        self.vis.register_key_callback(ord('Z'), lambda vis: self.apply_rotation('z', 1))    # Z: +Z rotation
        self.vis.register_key_callback(ord('X'), lambda vis: self.apply_rotation('z', -1))   # X: -Z rotation
        self.vis.register_key_callback(ord('I'), lambda vis: self.apply_translation(0, 1, 0))   # I: +Y (forward)
        self.vis.register_key_callback(ord('K'), lambda vis: self.apply_translation(0, -1, 0))  # K: -Y (back)
        self.vis.register_key_callback(ord('J'), lambda vis: self.apply_translation(-1, 0, 0))  # J: -X (left)
        self.vis.register_key_callback(ord('L'), lambda vis: self.apply_translation(1, 0, 0))   # L: +X (right)
        self.vis.register_key_callback(ord('U'), lambda vis: self.apply_translation(0, 0, 1))   # U: +Z (up)
        self.vis.register_key_callback(ord('O'), lambda vis: self.apply_translation(0, 0, -1))  # O: -Z (down)
        self.update_transformed_pointcloud()
        console.print(f"[dim]Step sizes: Rotation={self.rotation_step}°, Translation={self.translation_step} units")
        self.vis.run()
        self.vis.destroy_window()

    def save_and_exit(self, vis):
        """Save transform and exit."""
        self.save_transform()
        vis.close()
        return True

    def quit_without_save(self, vis):
        """Quit without saving."""
        console.print("[yellow]Exiting without saving")
        vis.close()
        return True

    def save_transform(self):
        """Save the current transform and apply it to all data."""
        console.print(f"\n[bold green]Saving alignment transform...")
        console.print("[cyan]Applied transformation matrix:")
        for row in self.transform:
            console.print(f"  [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")
        self.apply_transform_to_data()
        console.print("[bold green]✓ Alignment transform saved")

    def apply_transform_to_data(self):
        """Apply the transformation to all pointcloud data and save."""
        data_dir = self.data.data_dir
        rgb_pcd = o3d.io.read_point_cloud(str(data_dir / "pointcloud_rgb.ply"))
        rgb_pcd.transform(self.transform)
        o3d.io.write_point_cloud(str(data_dir / "pointcloud_rgb.ply"), rgb_pcd)
        console.print("[green]✓ Updated RGB pointcloud")

        pca_pcd = o3d.io.read_point_cloud(str(data_dir / "pointcloud_feature_pca.ply"))
        pca_pcd.transform(self.transform)
        o3d.io.write_point_cloud(str(data_dir / "pointcloud_feature_pca.ply"), pca_pcd)
        console.print("[green]✓ Updated PCA pointcloud")

        additional_outputs = self.data.metadata.get('additional_outputs', [])
        if additional_outputs:
            console.print(f"[cyan]Transforming {len(additional_outputs)} additional outputs: {additional_outputs}")
            for output_name in additional_outputs:
                filename = self.data.metadata['files'].get(output_name)
                if filename is not None:
                    file_path = data_dir / filename
                    if file_path.exists():
                        pcd = o3d.io.read_point_cloud(str(file_path))
                        pcd.transform(self.transform)
                        o3d.io.write_point_cloud(str(file_path), pcd)
                        console.print(f"[green]✓ Updated {output_name} pointcloud")
                    else:
                        console.print(f"[yellow]Warning: {output_name} file not found: {file_path}")
                else:
                    console.print(f"[yellow]Warning: No filename mapping for {output_name}")
        else:
            console.print("[dim]No additional outputs to transform")

        points = np.load(data_dir / "points.npy")
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_transformed = (self.transform @ points_homo.T).T[:, :3]
        np.save(data_dir / "points.npy", points_transformed.astype(np.float32))
        console.print("[green]✓ Updated points array")

        features_file = self.data.metadata['files']['features']
        features = np.load(data_dir / features_file)
        np.save(data_dir / features_file, features)
        console.print("[green]✓ Updated features array")

        pca_path = data_dir / "pca_params.pkl"
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                pca_data = pickle.load(f)
            pca_data['num_points'] = int(len(points_transformed))
            with open(pca_path, 'wb') as f:
                pickle.dump(pca_data, f)
            console.print("[green]✓ Updated PCA parameters")

        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata['alignment_transform'] = self.transform.tolist()
        metadata['alignment_applied'] = True
        metadata['bbox_min'] = points_transformed.min(axis=0).tolist()
        metadata['bbox_max'] = points_transformed.max(axis=0).tolist()
        if 'additional_outputs' not in metadata:
            metadata['additional_outputs'] = self.data.metadata.get('additional_outputs', [])
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        console.print("[green]✓ Updated metadata")


class InteractiveFilterTool:
    """Interactive tool for setting bounding box filters with visual feedback."""

    def __init__(self, data: FeaturePointcloudData, filter_step: float = 0.01):
        self.data = data
        self.vis = None
        self.filter_step = filter_step
        self.original_bbox_min = np.array(self.data.metadata['bbox_min'])
        self.original_bbox_max = np.array(self.data.metadata['bbox_max'])
        self.filter_min = self.original_bbox_min.copy()
        self.filter_max = self.original_bbox_max.copy()
        self.current_axis = 0
        self.axis_names = ['X', 'Y', 'Z']
        self.axis_colors = ['red', 'green', 'blue']
        self.original_pcd = None
        self.filtered_pcd = None
        self.reference_geoms = []
        self.filter_planes = []

    def create_reference_geometries(self) -> list:
        """Create coordinate frame, bounding box, and grid."""
        geometries = []
        geometries.extend(create_coordinate_frame(size=0.3))
        geometries.extend(create_bounding_box_lines(self.original_bbox_min, self.original_bbox_max))
        geometries.extend(create_grid_lines(self.original_bbox_min, self.original_bbox_max, grid_size=0.2))
        return geometries

    def create_filter_planes(self) -> list:
        """Create colored planes showing current filter bounds."""
        planes = []
        for axis in range(3):
            min_plane = create_filter_plane(axis, self.filter_min[axis], self.original_bbox_min, self.original_bbox_max, is_min=True)
            max_plane = create_filter_plane(axis, self.filter_max[axis], self.original_bbox_min, self.original_bbox_max, is_min=False)
            planes.extend([min_plane, max_plane])
        return planes

    def apply_filter(self):
        """Apply current filter bounds to the pointcloud."""
        points = np.asarray(self.original_pcd.points)
        colors = np.asarray(self.original_pcd.colors)
        within_bounds = np.all((points >= self.filter_min) & (points <= self.filter_max), axis=1)
        filtered_points = points[within_bounds]
        filtered_colors = colors[within_bounds]
        self.filtered_pcd = o3d.geometry.PointCloud()
        self.filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        self.filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        return len(filtered_points), len(points)

    def update_visualization(self):
        """Update the visualization with current filter state."""
        if self.vis is None:
            return
        view_control = self.vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        filtered_count, total_count = self.apply_filter()
        self.filter_planes = self.create_filter_planes()
        self.vis.clear_geometries()
        self.vis.add_geometry(self.filtered_pcd)
        for geom in self.reference_geoms:
            self.vis.add_geometry(geom)
        for plane in self.filter_planes:
            self.vis.add_geometry(plane)
        view_control.convert_from_pinhole_camera_parameters(camera_params)
        axis_name, axis_color = self.axis_names[self.current_axis], self.axis_colors[self.current_axis]
        console.print(f"[{axis_color}]{axis_name}: Min={self.filter_min[self.current_axis]:.3f} Max={self.filter_max[self.current_axis]:.3f} Points={filtered_count:,}/{total_count:,}")

    def cycle_axis(self):
        """Cycle to the next axis (X -> Y -> Z -> X)."""
        self.current_axis = (self.current_axis + 1) % 3
        console.print(f"[bold {self.axis_colors[self.current_axis]}]Switched to {self.axis_names[self.current_axis]} axis")
        self.update_visualization()

    def adjust_bound(self, is_max: bool, direction: int):
        """Adjust min or max bound for current axis."""
        axis = self.current_axis
        step = self.filter_step * direction
        if is_max:
            new_value = self.filter_max[axis] + step
            if new_value >= self.filter_min[axis]:
                self.filter_max[axis] = new_value
        else:
            new_value = self.filter_min[axis] + step
            if new_value <= self.filter_max[axis]:
                self.filter_min[axis] = new_value
        self.update_visualization()

    def reset_filters(self):
        """Reset all filter bounds to original bounding box."""
        self.filter_min = self.original_bbox_min.copy()
        self.filter_max = self.original_bbox_max.copy()
        console.print("[green]Filter bounds reset")
        self.update_visualization()

    def start_filtering(self):
        """Start the interactive filter tool."""
        console.print("[bold green]F3RM Pointcloud Filter Tool")
        console.print("\n[bold blue]Filter Mode Controls:")
        console.print("  Mouse: Normal Open3D viewing controls")
        console.print("  N: Cycle between X, Y, Z axis filtering")
        console.print("  ↑/↓ Arrow: Adjust max bound for current axis")
        console.print("  ←/→ Arrow: Adjust min bound for current axis")
        console.print("  R: Reset all filters to original bounds")
        console.print("  S: Save filter bounds and exit")
        console.print("  Q: Quit without saving")
        console.print("\n[bold yellow]Goal: Set bounding box filters to remove unwanted regions")
        console.print("[bold yellow]Colored planes show filter bounds: Red=X, Green=Y, Blue=Z")
        console.print("[bold yellow]Darker planes = min bounds, Brighter planes = max bounds")
        self.original_pcd = self.data.rgb_pointcloud
        self.reference_geoms = self.create_reference_geometries()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="F3RM Pointcloud Filter Tool", width=1400, height=900)
        # Register key callbacks
        self.vis.register_key_callback(ord('N'), lambda vis: self.cycle_axis())
        self.vis.register_key_callback(ord('R'), lambda vis: self.reset_filters())
        self.vis.register_key_callback(ord('S'), lambda vis: self.save_and_exit(vis))
        self.vis.register_key_callback(ord('Q'), lambda vis: self.quit_without_save(vis))
        self.vis.register_key_callback(265, lambda vis: self.adjust_bound(True, 1))    # Up: increase max
        self.vis.register_key_callback(264, lambda vis: self.adjust_bound(True, -1))   # Down: decrease max
        self.vis.register_key_callback(262, lambda vis: self.adjust_bound(False, 1))   # Right: increase min
        self.vis.register_key_callback(263, lambda vis: self.adjust_bound(False, -1))  # Left: decrease min
        self.update_visualization()
        console.print(f"[dim]Filter step: {self.filter_step} units")
        console.print(f"[bold {self.axis_colors[self.current_axis]}]Starting with {self.axis_names[self.current_axis]} axis")
        self.vis.run()
        self.vis.destroy_window()

    def save_and_exit(self, vis):
        """Save filter bounds and exit."""
        self.save_filter_bounds()
        vis.close()
        return True

    def quit_without_save(self, vis):
        """Quit without saving."""
        console.print("[yellow]Exiting without saving")
        vis.close()
        return True

    def save_filter_bounds(self):
        """Apply the current filter bounds to all data and save."""
        console.print(f"\n[bold green]Applying filter bounds...")
        console.print("[cyan]Filter bounds:")
        for i, axis_name in enumerate(self.axis_names):
            color = self.axis_colors[i]
            console.print(f"  [{color}]{axis_name}: {self.filter_min[i]:.3f} to {self.filter_max[i]:.3f}")
        self.apply_filter_to_data()
        console.print("[bold green]✓ Filter bounds applied")

    def apply_filter_to_data(self):
        """Apply the filter bounds to all pointcloud data and save."""
        data_dir = self.data.data_dir
        points = np.load(data_dir / "points.npy")
        within_bounds = np.all((points >= self.filter_min) & (points <= self.filter_max), axis=1)
        original_count, filtered_count = len(points), within_bounds.sum()
        console.print(f"[cyan]Filtering {original_count:,} points...")
        console.print(f"[cyan]Keeping {filtered_count:,} points ({100*filtered_count/original_count:.1f}%)")

        rgb_pcd = o3d.io.read_point_cloud(str(data_dir / "pointcloud_rgb.ply"))
        rgb_points, rgb_colors = np.asarray(rgb_pcd.points), np.asarray(rgb_pcd.colors)
        filtered_rgb_pcd = o3d.geometry.PointCloud()
        filtered_rgb_pcd.points = o3d.utility.Vector3dVector(rgb_points[within_bounds])
        filtered_rgb_pcd.colors = o3d.utility.Vector3dVector(rgb_colors[within_bounds])
        o3d.io.write_point_cloud(str(data_dir / "pointcloud_rgb.ply"), filtered_rgb_pcd)
        console.print("[green]✓ Updated RGB pointcloud")

        pca_pcd = o3d.io.read_point_cloud(str(data_dir / "pointcloud_feature_pca.ply"))
        pca_points, pca_colors = np.asarray(pca_pcd.points), np.asarray(pca_pcd.colors)
        filtered_pca_pcd = o3d.geometry.PointCloud()
        filtered_pca_pcd.points = o3d.utility.Vector3dVector(pca_points[within_bounds])
        filtered_pca_pcd.colors = o3d.utility.Vector3dVector(pca_colors[within_bounds])
        o3d.io.write_point_cloud(str(data_dir / "pointcloud_feature_pca.ply"), filtered_pca_pcd)
        console.print("[green]✓ Updated PCA pointcloud")

        additional_outputs = self.data.metadata.get('additional_outputs', [])
        if additional_outputs:
            console.print(f"[cyan]Filtering {len(additional_outputs)} additional outputs: {additional_outputs}")
            for output_name in additional_outputs:
                filename = self.data.metadata['files'].get(output_name)
                if filename is not None:
                    file_path = data_dir / filename
                    if file_path.exists():
                        pcd = o3d.io.read_point_cloud(str(file_path))
                        pcd_points, pcd_colors = np.asarray(pcd.points), np.asarray(pcd.colors)
                        filtered_pcd = o3d.geometry.PointCloud()
                        filtered_pcd.points = o3d.utility.Vector3dVector(pcd_points[within_bounds])
                        filtered_pcd.colors = o3d.utility.Vector3dVector(pcd_colors[within_bounds])
                        o3d.io.write_point_cloud(str(file_path), filtered_pcd)
                        console.print(f"[green]✓ Updated {output_name} pointcloud")
                    else:
                        console.print(f"[yellow]Warning: {output_name} file not found: {file_path}")
                else:
                    console.print(f"[yellow]Warning: No filename mapping for {output_name}")
        else:
            console.print("[dim]No additional outputs to filter")

        filtered_points = points[within_bounds]
        np.save(data_dir / "points.npy", filtered_points.astype(np.float32))
        console.print("[green]✓ Updated points array")

        features_file = self.data.metadata['files']['features']
        features = np.load(data_dir / features_file)
        filtered_features = features[within_bounds]
        np.save(data_dir / features_file, filtered_features)
        console.print("[green]✓ Updated features array")

        pca_path = data_dir / "pca_params.pkl"
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                pca_data = pickle.load(f)
            pca_data['num_points'] = int(filtered_count)
            with open(pca_path, 'wb') as f:
                pickle.dump(pca_data, f)
            console.print("[green]✓ Updated PCA parameters")

        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata['num_points'] = int(filtered_count)
        metadata['bbox_min'] = filtered_points.min(axis=0).tolist()
        metadata['bbox_max'] = filtered_points.max(axis=0).tolist()
        metadata['filter_applied'] = {'bounds': {'min': self.filter_min.tolist(), 'max': self.filter_max.tolist()}, 'original_count': int(original_count), 'filtered_count': int(filtered_count), 'applied': True}
        if 'additional_outputs' not in metadata:
            metadata['additional_outputs'] = self.data.metadata.get('additional_outputs', [])
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        console.print("[green]✓ Updated metadata")
        console.print(f"[bold green]✓ Filtering complete: {filtered_count:,} points saved")


def align_pointcloud(data_dir: Path, rotation_step: float = 5.0, translation_step: float = 0.05) -> None:
    """Main function to align pointcloud data."""
    console.print(f"[bold blue]F3RM Pointcloud Alignment Tool")
    console.print(f"[bold blue]Data directory: {data_dir}")
    console.print(f"[bold blue]Rotation step: {rotation_step}° | Translation step: {translation_step}")
    data = FeaturePointcloudData(data_dir)
    console.print("[green]✓ Pointcloud data loaded")
    additional_outputs = data.metadata.get('additional_outputs', [])
    if additional_outputs:
        console.print(f"[cyan]Additional outputs to process: {additional_outputs}")
    else:
        console.print("[dim]No additional outputs to process")
    if 'alignment_applied' in data.metadata and data.metadata['alignment_applied']:
        console.print("[yellow]Warning: Already aligned")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            console.print("[yellow]Alignment cancelled")
            return
    alignment_tool = InteractiveAlignmentTool(data, rotation_step, translation_step)
    alignment_tool.start_alignment()


def filter_pointcloud(data_dir: Path, filter_step: float = 0.01) -> None:
    """Main function to set bounding box filters for pointcloud data."""
    console.print(f"[bold blue]F3RM Pointcloud Filter Tool")
    console.print(f"[bold blue]Data directory: {data_dir}")
    console.print(f"[bold blue]Filter step: {filter_step} units")
    data = FeaturePointcloudData(data_dir)
    console.print("[green]✓ Pointcloud data loaded")
    additional_outputs = data.metadata.get('additional_outputs', [])
    if additional_outputs:
        console.print(f"[cyan]Additional outputs to process: {additional_outputs}")
    else:
        console.print("[dim]No additional outputs to process")
    if 'filter_applied' in data.metadata and data.metadata['filter_applied'].get('applied', False):
        console.print("[yellow]Warning: Already has filter bounds")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            console.print("[yellow]Filtering cancelled")
            return
    filter_tool = InteractiveFilterTool(data, filter_step)
    filter_tool.start_filtering()


def main():
    parser = argparse.ArgumentParser(description="Align or filter F3RM feature pointclouds")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing exported pointcloud data")
    parser.add_argument("--mode", choices=["align", "filter"], default="align", help="Tool mode: 'align' for alignment, 'filter' for bounding box filtering")
    parser.add_argument("--rotation-step", type=float, default=5.0, help="Rotation step size in degrees (default: 5.0, align mode)")
    parser.add_argument("--translation-step", type=float, default=0.05, help="Translation step size in units (default: 0.05, align mode)")
    parser.add_argument("--filter-step", type=float, default=0.01, help="Filter adjustment step size in units (default: 0.01, filter mode)")
    args = parser.parse_args()
    if not args.data_dir.exists():
        console.print(f"[bold red]Data directory not found: {args.data_dir}")
        return
    if not (args.data_dir / "metadata.json").exists():
        console.print(f"[bold red]No metadata.json found. Run f3rm/manual/export_feature_pointcloud.py first.")
        return
    required_files = ["pointcloud_rgb.ply", "pointcloud_feature_pca.ply", "points.npy"]
    missing_files = [file for file in required_files if not (args.data_dir / file).exists()]
    if missing_files:
        console.print(f"[bold red]Missing required files: {', '.join(missing_files)}")
        console.print("[bold red]Run f3rm/manual/export_feature_pointcloud.py first.")
        return
    if args.mode == "align":
        align_pointcloud(args.data_dir, args.rotation_step, args.translation_step)
    elif args.mode == "filter":
        filter_pointcloud(args.data_dir, args.filter_step)


if __name__ == "__main__":
    main()
