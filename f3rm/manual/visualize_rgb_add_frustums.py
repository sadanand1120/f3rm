#!/usr/bin/env python3
"""
F3RM RGB Pointcloud with Custom Frustums Tool

This script provides interactive addition of custom camera frustums at multiple locations
and visualization of RGB pointclouds with the saved frustums.

Usage:
    python f3rm/manual/visualize_rgb_add_frustums.py --data-dir exports/pointcloud_data/ --mode add
    python f3rm/manual/visualize_rgb_add_frustums.py --data-dir exports/pointcloud_data/ --mode visualize

Add Mode Controls:
    - Mouse: Normal Open3D viewing controls (camera only)
    - Arrow Keys: Rotate around X/Y axes (pitch/yaw)
    - Z/X: Rotate around Z-axis (roll)
    - I/K: Move forward/back (±Y)
    - J/L: Move left/right (±X)  
    - U/O: Move up/down (±Z)
    - 1/2: Scale frustum size (decrease/increase)
    - 'A': Add current frustum and start new one
    - 'R': Reset current frustum transform
    - 'Q': Quit and save all frustums

Visualize Mode:
    - Simple visualization of RGB pointcloud with all saved custom frustums
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


class CustomFrustum:
    """Represents a custom camera frustum with position, orientation, and scale."""

    def __init__(self, transform: np.ndarray, scale: float = 1.0, frustum_id: int = 0):
        self.transform = transform.copy()  # 4x4 transformation matrix
        self.scale = scale
        self.frustum_id = frustum_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'transform': self.transform.tolist(),
            'scale': self.scale,
            'frustum_id': self.frustum_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomFrustum':
        """Create from dictionary."""
        return cls(
            transform=np.array(data['transform']),
            scale=data['scale'],
            frustum_id=data['frustum_id']
        )

    def create_geometry(self) -> o3d.geometry.TriangleMesh:
        """Create Open3D geometry for this frustum."""
        # Create camera frustum with thick edges
        frustum_mesh = self.create_frustum_mesh(self.scale)

        # Apply transformation
        frustum_mesh.transform(self.transform)

        return frustum_mesh

    def create_frustum_mesh(self, scale: float) -> o3d.geometry.TriangleMesh:
        """Create a camera frustum as a mesh with thick edges."""
        # Define frustum geometry (pyramid shape)
        # Camera is at origin, looking down -Z axis
        near_plane = 0.1 * scale  # Shorter near plane
        far_plane = 0.2 * scale    # Much shorter far plane (was 1.0)
        fov_h = 0.9 * scale        # Wider field of view width (was 0.5)
        fov_v = 0.75 * scale        # Wider field of view height (was 0.4)
        edge_thickness = 0.01 * scale  # Thickness of the edges

        # Create a mesh by extruding lines into cylinders
        mesh = o3d.geometry.TriangleMesh()

        # Define the 8 corner points of the frustum
        points = np.array([
            # Camera center (apex of pyramid)
            [0, 0, 0],
            # Near plane corners
            [-fov_h * near_plane, -fov_v * near_plane, -near_plane],
            [fov_h * near_plane, -fov_v * near_plane, -near_plane],
            [fov_h * near_plane, fov_v * near_plane, -near_plane],
            [-fov_h * near_plane, fov_v * near_plane, -near_plane],
            # Far plane corners
            [-fov_h * far_plane, -fov_v * far_plane, -far_plane],
            [fov_h * far_plane, -fov_v * far_plane, -far_plane],
            [fov_h * far_plane, fov_v * far_plane, -far_plane],
            [-fov_h * far_plane, fov_v * far_plane, -far_plane],
        ])

        # Define the edges as line segments
        edges = [
            # Lines from camera center to near plane
            (0, 1), (0, 2), (0, 3), (0, 4),
            # Near plane rectangle
            (1, 2), (2, 3), (3, 4), (4, 1),
            # Lines from near plane to far plane
            (1, 5), (2, 6), (3, 7), (4, 8),
            # Far plane rectangle
            (5, 6), (6, 7), (7, 8), (8, 5),
        ]

        # Create thick edges by adding small spheres at each point and cylinders along edges
        for i, point in enumerate(points):
            # Add a small sphere at each vertex
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=edge_thickness)
            sphere.translate(point)
            mesh += sphere

        # Add cylinders along each edge for thickness
        for start_idx, end_idx in edges:
            start_point = points[start_idx]
            end_point = points[end_idx]

            # Create a cylinder along the edge
            direction = end_point - start_point
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length

                # Create cylinder
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=edge_thickness, height=length)

                # Rotate cylinder to align with edge direction
                # Default cylinder is along Z-axis, so we need to rotate it
                z_axis = np.array([0, 0, 1])
                if not np.allclose(direction, z_axis):
                    # Find rotation axis and angle
                    rotation_axis = np.cross(z_axis, direction)
                    if np.linalg.norm(rotation_axis) > 0:
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        cos_angle = np.dot(z_axis, direction)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)

                        # Create rotation matrix
                        rotation_matrix = self.rotation_matrix_from_axis_angle(rotation_axis, angle)
                        cylinder.rotate(rotation_matrix)

                # Position cylinder at midpoint of edge
                midpoint = (start_point + end_point) / 2
                cylinder.translate(midpoint)
                mesh += cylinder

        # Color the frustum (cyan/blue gradient)
        mesh.paint_uniform_color([0.0, 0.8, 1.0])  # Cyan color

        return mesh

    def rotation_matrix_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Create rotation matrix from axis-angle representation."""
        # Rodrigues' rotation formula
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2)
        b, c, d = axis * np.sin(angle / 2)

        return np.array([
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]
        ])


class InteractiveFrustumsAdder:
    """Interactive tool for adding custom frustums at multiple locations."""

    def __init__(self, data: FeaturePointcloudData, rotation_step: float = 10.0,
                 translation_step: float = 0.1, scale_step: float = 0.1, transparency: float = 1.0):
        self.data = data
        self.vis = None

        # Step sizes for transformations
        self.rotation_step = rotation_step  # degrees
        self.translation_step = translation_step  # units
        self.scale_step = scale_step  # units
        self.transparency = transparency  # RGB pointcloud transparency

        # Current frustum being positioned
        self.current_transform = np.eye(4)
        self.current_scale = 0.2  # Default scale

        # All saved frustums
        self.saved_frustums: List[CustomFrustum] = []
        self.next_frustum_id = 0

        # Store geometries
        self.original_pcd = None
        self.reference_geoms = []
        self.current_frustum_geom = None
        self.saved_frustums_geoms = []

        # Load existing frustums if available
        self.load_existing_frustums()

    def load_existing_frustums(self):
        """Load existing frustums from JSON file if available."""
        frustums_path = self.data.data_dir / "custom_frustums.json"
        if frustums_path.exists():
            try:
                with open(frustums_path, 'r') as f:
                    frustums_data = json.load(f)

                loaded_frustums = [CustomFrustum.from_dict(frustum_dict) for frustum_dict in frustums_data['frustums']]
                self.saved_frustums = loaded_frustums
                self.next_frustum_id = len(loaded_frustums)

                # Create geometries for loaded frustums
                for frustum in loaded_frustums:
                    frustum_geom = frustum.create_geometry()
                    self.saved_frustums_geoms.append(frustum_geom)

                console.print(f"[green]Loaded {len(loaded_frustums)} existing frustums from {frustums_path}")

                # Start new frustum at the position of the last loaded frustum if any exist
                if loaded_frustums:
                    last_frustum = loaded_frustums[-1]
                    self.current_transform = last_frustum.transform.copy()
                    self.current_scale = last_frustum.scale
                    console.print(f"[cyan]Starting new frustum at position of last loaded frustum")

            except Exception as e:
                console.print(f"[yellow]Warning: Could not load existing frustums: {e}")
                console.print("[yellow]Starting with empty frustums list")

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

    def update_current_frustum_geometry(self):
        """Update the current frustum geometry based on transform and scale."""
        if self.current_frustum_geom is None:
            self.current_frustum_geom = CustomFrustum(np.eye(4), self.current_scale).create_geometry()

        # Reset to identity and apply current transform
        current_frustum = CustomFrustum(self.current_transform, self.current_scale)
        self.current_frustum_geom = current_frustum.create_geometry()

    def update_visualization(self):
        """Update the visualization with current state."""
        if self.vis is None:
            return

        # Save current camera parameters
        view_control = self.vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()

        # Update current frustum geometry
        self.update_current_frustum_geometry()

        # Clear and re-add geometries
        self.vis.clear_geometries()

        # Add RGB pointcloud
        self.vis.add_geometry(self.original_pcd)

        # Add reference geometries
        for geom in self.reference_geoms:
            self.vis.add_geometry(geom)

        # Add saved frustums
        for geom in self.saved_frustums_geoms:
            self.vis.add_geometry(geom)

        # Add current frustum (highlighted)
        self.vis.add_geometry(self.current_frustum_geom)

        # Restore camera parameters to maintain view
        view_control.convert_from_pinhole_camera_parameters(camera_params)

        # Print status
        pos = self.current_transform[:3, 3]
        console.print(f"[cyan]Current frustum {self.next_frustum_id}: "
                      f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                      f"scale={self.current_scale:.3f}, "
                      f"saved={len(self.saved_frustums)}")

    def reset_current_transform(self):
        """Reset current frustum transform to identity."""
        self.current_transform = np.eye(4)
        self.current_scale = 0.2
        self.update_visualization()
        console.print("[green]Current frustum reset to identity")

    def apply_rotation(self, axis: str, direction: int = 1):
        """Apply rotation around specified axis in the current frustum's own frame.

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
        local_rot = np.eye(4)
        local_rot[:3, :3] = rot_matrix

        # Apply rotation in the current frustum's own frame
        # This means: translate to origin, rotate, translate back
        current_position = self.current_transform[:3, 3]

        # Translate to origin
        to_origin = np.eye(4)
        to_origin[:3, 3] = -current_position

        # Translate back to original position
        to_position = np.eye(4)
        to_position[:3, 3] = current_position

        # Apply the sequence: translate to origin -> rotate -> translate back
        self.current_transform = to_position @ local_rot @ to_origin @ self.current_transform
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

    def add_current_frustum(self):
        """Add current frustum to saved list and start new one."""
        # Create new frustum from current state
        new_frustum = CustomFrustum(
            transform=self.current_transform.copy(),
            scale=self.current_scale,
            frustum_id=self.next_frustum_id
        )

        # Add to saved list
        self.saved_frustums.append(new_frustum)

        # Create geometry for saved frustum
        saved_geom = new_frustum.create_geometry()
        self.saved_frustums_geoms.append(saved_geom)

        # Start next frustum at the position and scale of the current one (keep position and scale, reset orientation)
        current_position = self.current_transform[:3, 3]
        self.current_transform = np.eye(4)
        self.current_transform[:3, 3] = current_position  # Keep the position
        # Keep the current scale (don't reset to 0.2)
        self.next_frustum_id += 1

        console.print(f"[green]Added frustum {new_frustum.frustum_id} at position {new_frustum.transform[:3, 3]}")
        self.update_visualization()

    def save_frustums_to_file(self):
        """Save all frustums to JSON file."""
        frustums_data = {
            'frustums': [frustum.to_dict() for frustum in self.saved_frustums],
            'num_frustums': len(self.saved_frustums)
        }

        output_path = self.data.data_dir / "custom_frustums.json"
        with open(output_path, 'w') as f:
            json.dump(frustums_data, f, indent=2)

        console.print(f"[green]Saved {len(self.saved_frustums)} frustums to {output_path}")

    def start_adding_frustums(self):
        """Start the interactive frustums addition tool."""
        console.print("[bold green]Starting F3RM Custom Frustums Addition Tool")
        console.print(self.data.get_info())

        console.print("\n[bold blue]Controls:")
        console.print("  Mouse: Normal Open3D viewing controls (camera only)")
        console.print("  Arrow Keys: Rotate around X/Y axes (pitch/yaw)")
        console.print("  'Z'/'X': Rotate around Z-axis (roll)")
        console.print("  'I'/'K': Move forward/back (±Y)")
        console.print("  'J'/'L': Move left/right (±X)")
        console.print("  'U'/'O': Move up/down (±Z)")
        console.print("  '1'/'2': Scale frustum size (decrease/increase)")
        console.print("  'A': Add current frustum and start new one")
        console.print("  'R': Reset current frustum transform")
        console.print("  'Q': Quit and save all frustums")
        console.print("\n[bold yellow]Goal: Add custom camera frustums at multiple locations")
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
            window_name="F3RM Custom Frustums Addition Tool",
            width=1400, height=900
        )

        # Register key callbacks
        self.vis.register_key_callback(ord('A'), lambda vis: self.add_current_frustum())
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
        console.print("  1/2: Scale frustum size (decrease/increase)")
        console.print("  A: Add current frustum and start new one")
        console.print("  R: Reset current frustum transform")
        console.print("  Q: Quit and save all frustums")
        console.print(f"\n[dim]Step sizes: Rotation={self.rotation_step}°, Translation={self.translation_step}, Scale={self.scale_step}")

        # Run visualization
        self.vis.run()
        self.vis.destroy_window()

    def save_and_exit(self, vis):
        """Save frustums and exit."""
        if self.saved_frustums:
            self.save_frustums_to_file()
        else:
            console.print("[yellow]No frustums to save")
        vis.close()
        return True


def load_custom_frustums(data_dir: Path) -> List[CustomFrustum]:
    """Load custom frustums from JSON file."""
    frustums_path = data_dir / "custom_frustums.json"
    if not frustums_path.exists():
        return []

    with open(frustums_path, 'r') as f:
        frustums_data = json.load(f)

    frustums = [CustomFrustum.from_dict(frustum_dict) for frustum_dict in frustums_data['frustums']]
    console.print(f"[green]Loaded {len(frustums)} custom frustums from {frustums_path}")
    return frustums


def add_custom_frustums(data_dir: Path, rotation_step: float = 10.0,
                        translation_step: float = 0.1, scale_step: float = 0.1, transparency: float = 1.0) -> None:
    """
    Main function to add custom frustums.

    Args:
        data_dir: Directory containing exported pointcloud data
        rotation_step: Rotation step size in degrees (default: 10.0)
        translation_step: Translation step size in units (default: 0.1)
        scale_step: Scale step size in units (default: 0.1)
        transparency: Transparency of RGB pointcloud (0.0=transparent, 1.0=opaque, default: 1.0)
    """
    console.print(f"[bold blue]F3RM Custom Frustums Addition Tool")
    console.print(f"[bold blue]Data directory: {data_dir}")
    console.print(f"[bold blue]Rotation step: {rotation_step}° | Translation step: {translation_step} | Scale step: {scale_step} | Transparency: {transparency}")

    # Load pointcloud data
    try:
        data = FeaturePointcloudData(data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")
    except Exception as e:
        console.print(f"[red]Error loading pointcloud data: {e}")
        return

    # Start frustums addition tool
    frustums_adder = InteractiveFrustumsAdder(data, rotation_step, translation_step, scale_step, transparency)
    frustums_adder.start_adding_frustums()


def visualize_with_frustums(data_dir: Path, transparency: float = 1.0) -> None:
    """
    Main function to visualize RGB pointcloud with custom frustums.

    Args:
        data_dir: Directory containing exported pointcloud data
        transparency: Transparency of RGB pointcloud (0.0=transparent, 1.0=opaque, default: 1.0)
    """
    console.print(f"[bold blue]F3RM RGB Pointcloud with Custom Frustums Visualizer")
    console.print(f"[bold blue]Data directory: {data_dir}")
    console.print(f"[bold blue]Transparency: {transparency}")

    # Load pointcloud data
    try:
        data = FeaturePointcloudData(data_dir)
        console.print("[green]✓ Pointcloud data loaded successfully")
    except Exception as e:
        console.print(f"[red]Error loading pointcloud data: {e}")
        return

    # Load custom frustums
    custom_frustums = load_custom_frustums(data_dir)
    if not custom_frustums:
        console.print("[yellow]No custom frustums found. Run in 'add' mode first to create frustums.")

    console.print("[bold green]Visualizing RGB pointcloud with custom frustums...")
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

    # Create custom frustums geometries
    frustums_geometries = []
    for frustum in custom_frustums:
        frustum_geom = frustum.create_geometry()
        frustums_geometries.append(frustum_geom)

    # Combine all geometries
    all_geometries = [pcd] + reference_geoms + frustums_geometries

    console.print(f"[cyan]Showing {len(custom_frustums)} custom frustums")
    console.print("[dim]Reference guides: Gray=original bounds; Light gray=grid")

    # Simple visualization
    o3d.visualization.draw_geometries(
        all_geometries,
        window_name="F3RM RGB Pointcloud with Custom Frustums",
        width=1200,
        height=800,
        left=50,
        top=50
    )


def main():
    parser = argparse.ArgumentParser(description="Add or visualize custom frustums with F3RM RGB pointclouds")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing exported pointcloud data")
    parser.add_argument("--mode", choices=["add", "visualize"], default="add",
                        help="Tool mode: 'add' for interactive frustums placement, 'visualize' for viewing with frustums")

    # Add mode arguments
    parser.add_argument("--rotation-step", type=float, default=2.0,
                        help="Rotation step size in degrees (default: 10.0, add mode)")
    parser.add_argument("--translation-step", type=float, default=0.01,
                        help="Translation step size in units (default: 0.1, add mode)")
    parser.add_argument("--scale-step", type=float, default=0.02,
                        help="Scale step size in units (default: 0.1, add mode)")
    parser.add_argument("--transparency", type=float, default=1.0,
                        help="Transparency of RGB pointcloud (0.0=transparent, 1.0=opaque, default: 0.5)")

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
            add_custom_frustums(args.data_dir, args.rotation_step, args.translation_step, args.scale_step, args.transparency)
        elif args.mode == "visualize":
            visualize_with_frustums(args.data_dir, args.transparency)
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
