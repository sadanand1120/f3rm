#!/usr/bin/env python3
"""
Instance Axes Annotator

Integrates SAM2 instance segmentation with 3D axes annotation.
1. First run interactive SAM2 segmentation to get instance masks
2. Then annotate 3D axes (RGB = XYZ) for each instance using arrow keys
3. Save both instance masks and axes annotations

Usage:
    python f3rm/manual/instance_axes_annotator.py --image_path /path/to/image.jpg
"""

import argparse
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import time
import math
from scipy.spatial.distance import cdist

# Import our SAM2 segmentation tool
from f3rm.manual.interactive_sam2_segmentation import InteractiveSAM2Segmenter, Colors, print_colored, print_status


class AxesAnnotator:
    """
    3D Axes Annotator for Instance Segmentation

    COORDINATE SYSTEM AND REFERENCE FRAME:
    =====================================

    The saved orientations are defined in the CAMERA COORDINATE SYSTEM with the following convention:

    **Viewpoint: You are looking at the camera head-on from its front**

    Camera Coordinate System (Right-Handed):
    - X-axis (RED): Points to the RIGHT of the camera view
    - Y-axis (GREEN): Points DOWNWARD in the camera view  
    - Z-axis (BLUE): Points INTO the scene (away from camera)

    **Important Notes:**
    1. This follows the standard computer vision camera coordinate system
    2. Y-axis points DOWN (not up) - this is the IMAGE coordinate convention
    3. Z-axis points FORWARD into the scene (positive depth)
    4. Origin is at the camera optical center

    **Object Orientation Interpretation:**
    When you annotate an object's axes, you are defining:
    - RED axis (X): Object's "right" direction in camera view
    - GREEN axis (Y): Object's "down" direction in camera view  
    - BLUE axis (Z): Object's "forward" direction (into scene)

    **Rotation Matrix Interpretation:**
    The saved rotation matrix R transforms vectors from the object's local coordinate
    system to the camera coordinate system:

    camera_vector = R @ object_vector

    **For Neural Network Training:**
    When training orientation heads, the network predicts rotations in this camera
    coordinate system. The ground truth orientations are consistent with this frame.

    """

    def __init__(self):
        """Initialize the axes annotator with camera coordinate system"""
        self.axes_data = {}  # Store axes for each instance: {instance_id: {'phi': float, 'theta': float, 'gamma': float}}
        self.current_instance_id = 1
        self.rotation_step = 5.0  # degrees per arrow key press

        # Axes visualization parameters
        self.axis_length = 80  # pixels
        self.axis_thickness = 6

        # Color coding follows camera coordinate system convention
        self.axis_colors = {
            'x': (0, 0, 255),    # RED for X-axis (camera right)
            'y': (0, 255, 0),    # GREEN for Y-axis (camera down)
            'z': (255, 0, 0)     # BLUE for Z-axis (camera forward/into scene)
        }

    def compute_instance_center(self, mask):
        """Compute the center of an instance mask"""
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            return None
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        return (center_x, center_y)

    @staticmethod
    def euler_to_rotation_matrix(phi, theta, gamma):
        """
        Convert Euler angles to rotation matrix using ZYX convention in camera coordinates

        CAMERA COORDINATE SYSTEM ROTATIONS:
        - phi (yaw): Rotation around Z-axis (camera forward) - left/right turn
        - theta (pitch): Rotation around Y-axis (camera down) - up/down tilt  
        - gamma (roll): Rotation around X-axis (camera right) - clockwise/counterclockwise spin

        **Reference Frame: Camera coordinate system**
        - Looking at camera from front: X=right, Y=down, Z=forward
        - All rotations follow right-hand rule
        - Rotation order: Z(yaw) -> Y(pitch) -> X(roll) 

        **Interpretation for Objects:**
        - phi=0°: Object facing camera direction (Z-forward)
        - theta=0°: Object level with camera horizontal plane
        - gamma=0°: Object upright relative to camera orientation

        Args:
            phi (float): Yaw angle in degrees (rotation around Z/forward axis)
            theta (float): Pitch angle in degrees (rotation around Y/down axis)
            gamma (float): Roll angle in degrees (rotation around X/right axis)

        Returns:
            np.ndarray: 3x3 rotation matrix in camera coordinate system
        """
        phi_rad = np.radians(phi)
        theta_rad = np.radians(theta)
        gamma_rad = np.radians(gamma)

        # Rotation matrices for each axis in camera coordinates
        # Z-axis rotation (yaw) - around camera forward axis
        Rz = np.array([
            [np.cos(phi_rad), -np.sin(phi_rad), 0],
            [np.sin(phi_rad), np.cos(phi_rad), 0],
            [0, 0, 1]
        ])

        # Y-axis rotation (pitch) - around camera down axis
        Ry = np.array([
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0, 1, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)]
        ])

        # X-axis rotation (roll) - around camera right axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(gamma_rad), -np.sin(gamma_rad)],
            [0, np.sin(gamma_rad), np.cos(gamma_rad)]
        ])

        # Combined rotation: R = Rz * Ry * Rx (ZYX convention)
        # Transforms from object coordinates to camera coordinates
        R = Rz @ Ry @ Rx
        return R

    def get_3d_axes_vectors(self, phi, theta, gamma):
        """Get 3D axis vectors by applying rotation to standard basis"""
        # Standard basis vectors
        x_axis = np.array([1, 0, 0])  # Red axis (X)
        y_axis = np.array([0, 1, 0])  # Green axis (Y)
        z_axis = np.array([0, 0, 1])  # Blue axis (Z)

        # Get rotation matrix
        R = self.euler_to_rotation_matrix(phi, theta, gamma)

        # Apply rotation to get oriented axes
        x_3d = R @ x_axis
        y_3d = R @ y_axis
        z_3d = R @ z_axis

        return x_3d, y_3d, z_3d

    def project_3d_to_2d(self, vector_3d, scale=1.0):
        """Project 3D vector to 2D screen coordinates using perspective projection"""
        # Use perspective projection with camera at (0, 0, 3) looking at origin
        camera_distance = 3.0

        # Simple perspective: x' = x*f/(z+f), y' = y*f/(z+f) where f is focal length
        focal_length = camera_distance
        z_offset = camera_distance  # Offset to avoid division by zero

        x_2d = (vector_3d[0] * focal_length) / (vector_3d[2] + z_offset) * scale
        y_2d = -(vector_3d[1] * focal_length) / (vector_3d[2] + z_offset) * scale  # Flip Y for screen coords

        return np.array([x_2d, y_2d])

    def draw_axes_on_image(self, image, center, phi, theta, gamma, axis_length=None, alpha=1.0):
        """Draw 3D axes on image at specified center point"""
        if axis_length is None:
            axis_length = self.axis_length

        # Get 3D vectors
        x_vec, y_vec, z_vec = self.get_3d_axes_vectors(phi, theta, gamma)

        # Project to 2D and scale
        x_2d = self.project_3d_to_2d(x_vec, axis_length)
        y_2d = self.project_3d_to_2d(y_vec, axis_length)
        z_2d = self.project_3d_to_2d(z_vec, axis_length)

        # Calculate end points
        center = np.array(center)
        x_end = center + x_2d
        y_end = center + y_2d
        z_end = center + z_2d

        # Draw axes with proper depth ordering
        axes_data = [
            ('z', z_end, z_vec[2]),  # Z (blue/up)
            ('y', y_end, y_vec[2]),  # Y (green/right)
            ('x', x_end, x_vec[2])   # X (red/front)
        ]

        # Sort by depth (Z component) - draw back-to-front
        axes_data.sort(key=lambda x: x[2])

        # Draw directly on image - NO BLENDING
        for axis_name, end_point, depth in axes_data:
            base_color = self.axis_colors[axis_name]

            # Adjust color brightness based on alpha (for dimmed previous axes)
            if alpha < 1.0:
                # Dim the color for previous axes
                adjusted_color = tuple(int(c * alpha * 0.7) for c in base_color)
                thickness = max(2, int(self.axis_thickness * 0.7))
            else:
                # Full color for current axes
                adjusted_color = base_color
                thickness = self.axis_thickness

            # Draw directly on the image
            cv2.arrowedLine(
                image,
                tuple(center.astype(int)),
                tuple(end_point.astype(int)),
                adjusted_color,
                thickness,
                tipLength=0.3
            )

            # Add axis label
            label_pos = end_point + (end_point - center) * 0.2
            cv2.putText(
                image,
                axis_name.upper(),
                tuple(label_pos.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                adjusted_color,
                2
            )

        return image

    def rotation_matrix_to_6d(self, R):
        """
        Convert 3x3 rotation matrix to 6D representation (Zhou et al. CVPR 2019)

        This implements the continuous 6D rotation representation from:
        "On the Continuity of Rotation Representations in Neural Networks" 
        by Zhou et al., CVPR 2019

        Key advantages for neural network training:
        - CONTINUOUS representation (vs discontinuous quaternions/Euler angles)
        - Lower approximation error for neural networks
        - 6-14x lower mean errors compared to discontinuous representations
        - Suitable for gradient-based optimization

        Format: Takes first two columns of 3x3 rotation matrix and flattens to 6D vector
        Output: [a1_x, a1_y, a1_z, a2_x, a2_y, a2_z] where a1, a2 are first two columns

        For neural network training:
        1. Network outputs 6D vector
        2. Convert back to rotation matrix using Gram-Schmidt:
           b1 = normalize(a1)
           b2 = normalize(a2 - (b1·a2)*b1)  
           b3 = b1 × b2
           R = [b1, b2, b3]
        3. Use geodesic loss: θ = arccos((trace(R_gt^T * R_pred) - 1) / 2)

        Args:
            R (np.ndarray): 3x3 rotation matrix

        Returns:
            np.ndarray: 6D vector [a1_x, a1_y, a1_z, a2_x, a2_y, a2_z]
        """
        # Take first two columns of rotation matrix
        return R[:, :2].flatten()  # Returns [a1_x, a1_y, a1_z, a2_x, a2_y, a2_z]

    def create_6d_rotation_field(self, instance_mask, image_shape):
        """
        Create pixel-wise 6D rotation field for neural network training

        This creates a dense 6D rotation field where each pixel contains the 6D rotation
        representation of the instance it belongs to. This is perfect for training 
        orientation heads in neural networks.

        Usage for neural network training:
        1. Use this as ground truth target: shape (H, W, 6)
        2. Network predicts per-pixel 6D vectors
        3. Apply loss only on pixels belonging to instances (non-zero mask)
        4. Convert 6D predictions back to rotation matrices using Gram-Schmidt
        5. Use geodesic angular loss for supervision

        Example PyTorch training code:
        ```python
        # Network prediction: (B, H, W, 6)
        pred_6d = model(image)

        # Ground truth from this function: (H, W, 6) 
        gt_6d = torch.from_numpy(rotation_field)

        # Convert to rotation matrices
        pred_R = rot6d_to_matrix(pred_6d)  # (B, H, W, 3, 3)
        gt_R = rot6d_to_matrix(gt_6d)      # (H, W, 3, 3)

        # Geodesic loss (only on instance pixels)
        mask = (instance_mask > 0)
        loss = orientation_loss_6d(mask, gt_R, pred_6d)
        ```

        Args:
            instance_mask (np.ndarray): Instance segmentation mask
            image_shape (tuple): Shape of the image (H, W, C)

        Returns:
            np.ndarray: Dense 6D rotation field of shape (H, W, 6)
        """
        h, w = image_shape[:2]
        rotation_field = np.zeros((h, w, 6), dtype=np.float32)

        # For each instance with annotated axes
        for instance_id, axes_data in self.axes_data.items():
            # Get instance mask
            current_mask = (instance_mask == instance_id)

            if np.any(current_mask):
                # Get rotation matrix for this instance
                R = self.euler_to_rotation_matrix(
                    axes_data['phi'],
                    axes_data['theta'],
                    axes_data['gamma']
                )

                # Convert to 6D representation
                rot_6d = self.rotation_matrix_to_6d(R)

                # Assign to all pixels belonging to this instance
                rotation_field[current_mask] = rot_6d

        return rotation_field

    def create_visualization(self, image, instance_mask, current_instance_id, phi, theta, gamma, display_scale=1.0):
        """Create visualization with instance mask and axes, showing previous annotations"""
        # Start with clean RGB image - NO overlays
        vis_image = image.copy()

        # Draw contour for current instance only
        current_mask = (instance_mask == current_instance_id)
        if np.any(current_mask):
            contours, _ = cv2.findContours(current_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)  # Green contour for current

        # Draw all previous axes
        for prev_id in range(1, current_instance_id):
            if prev_id in self.axes_data:
                prev_mask = (instance_mask == prev_id)
                if np.any(prev_mask):
                    prev_center = self.compute_instance_center(prev_mask)
                    if prev_center:
                        prev_axes = self.axes_data[prev_id]
                        vis_image = self.draw_axes_on_image(
                            vis_image, prev_center,
                            prev_axes['phi'], prev_axes['theta'], prev_axes['gamma'],
                            alpha=0.6  # Dimmed previous axes
                        )

        # Draw current axes (full opacity)
        if np.any(current_mask):
            center = self.compute_instance_center(current_mask)
            if center:
                vis_image = self.draw_axes_on_image(vis_image, center, phi, theta, gamma, alpha=1.0)

        # Resize for display if needed
        if display_scale != 1.0:
            new_width = int(vis_image.shape[1] * display_scale)
            new_height = int(vis_image.shape[0] * display_scale)
            vis_image = cv2.resize(vis_image, (new_width, new_height))

        return vis_image

    def annotate_axes_for_instances(self, image, instance_mask, total_instances):
        """Interactive axes annotation for all instances"""
        print_colored("\n" + "=" * 60, Colors.MAGENTA, bold=True)
        print_colored("3D Axes Annotation Phase", Colors.MAGENTA, bold=True)
        print_colored("=" * 60, Colors.MAGENTA, bold=True)
        print_colored("Controls:", Colors.WHITE, bold=True)
        print_colored("  ↑/↓ arrows:  Pitch rotation (θ) - tilt up/down", Colors.CYAN)
        print_colored("  ←/→ arrows:  Yaw rotation (φ) - turn left/right", Colors.CYAN)
        print_colored("  Z/X keys:    Roll rotation (γ) - rotate clockwise/counterclockwise", Colors.CYAN)
        print_colored("  'n' key:     Save current axes & move to next", Colors.GREEN)
        print_colored("  'r' key:     Reset current axes to default", Colors.YELLOW)
        print_colored("  ESC key:     Exit without saving", Colors.RED)
        print_colored("=" * 60, Colors.MAGENTA, bold=True)

        # Calculate display scale for HD viewing
        display_height, display_width = image.shape[:2]
        max_display_height = 720
        max_display_width = 1280

        if display_height > max_display_height or display_width > max_display_width:
            scale_h = max_display_height / display_height
            scale_w = max_display_width / display_width
            display_scale = min(scale_h, scale_w)
        else:
            display_scale = 1.0

        window_name = "3D Axes Annotation"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        for instance_id in range(1, total_instances + 1):
            # Get current instance mask
            current_mask = (instance_mask == instance_id)

            if not np.any(current_mask):
                continue

            # Initialize axes angles for this instance
            phi = 0.0    # azimuth (0-360°)
            theta = 0.0  # polar (-90 to 90°)
            gamma = 0.0  # rotation (-180 to 180°)

            print_colored(f"\n🎯 Annotating axes for instance {instance_id}/{total_instances}", Colors.MAGENTA, bold=True)
            print_colored(f"   Azimuth (φ): {phi:.1f}°, Polar (θ): {theta:.1f}°, Rotation (γ): {gamma:.1f}°", Colors.CYAN)

            while True:
                # Create visualization (pass full instance_mask, not just current_mask)
                vis_image = self.create_visualization(image, instance_mask, instance_id, phi, theta, gamma, display_scale)

                # Display
                cv2.imshow(window_name, vis_image)
                key = cv2.waitKey(1) & 0xFF

                update_display = False

                if key == 82 or key == ord('w'):  # Up arrow or W
                    theta = (theta + self.rotation_step) % 360.0
                    update_display = True
                elif key == 84 or key == ord('s'):  # Down arrow or S
                    theta = (theta - self.rotation_step) % 360.0
                    update_display = True
                elif key == 81 or key == ord('a'):  # Left arrow or A
                    phi = (phi - self.rotation_step) % 360.0
                    update_display = True
                elif key == 83 or key == ord('d'):  # Right arrow or D
                    phi = (phi + self.rotation_step) % 360.0
                    update_display = True
                elif key == ord('z'):  # Z - rotate counter-clockwise
                    gamma = (gamma - self.rotation_step) % 360.0
                    update_display = True
                elif key == ord('x'):  # X - rotate clockwise
                    gamma = (gamma + self.rotation_step) % 360.0
                    update_display = True
                elif key == ord('r'):  # Reset axes
                    phi, theta, gamma = 0.0, 0.0, 0.0
                    update_display = True
                    print_colored("🔄 Reset axes to default orientation", Colors.YELLOW)
                elif key == ord('n'):  # Next instance
                    # Save current axes
                    self.axes_data[instance_id] = {
                        'phi': phi,
                        'theta': theta,
                        'gamma': gamma
                    }
                    print_colored(f"✓ Saved axes for instance {instance_id}", Colors.GREEN)
                    print_colored(f"   Final: φ={phi:.1f}°, θ={theta:.1f}°, γ={gamma:.1f}°", Colors.CYAN)
                    break
                elif key == 27:  # ESC
                    print_status("Exiting axes annotation...", "warning")
                    cv2.destroyAllWindows()
                    return False

                if update_display:
                    print_colored(f"   Azimuth (φ): {phi:.1f}°, Polar (θ): {theta:.1f}°, Rotation (γ): {gamma:.1f}°", Colors.CYAN)

        cv2.destroyAllWindows()
        return True

    def create_final_visualization(self, image, instance_mask, total_instances):
        """Create final visualization with all instances and their axes"""
        vis_image = image.copy()

        # Color palette for instances
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (255, 150, 100), (150, 255, 100), (100, 150, 255)
        ]

        for instance_id in range(1, total_instances + 1):
            current_mask = (instance_mask == instance_id)

            if not np.any(current_mask) or instance_id not in self.axes_data:
                continue

            # Get instance color
            color = colors[(instance_id - 1) % len(colors)]

            # Create overlay
            overlay = np.zeros_like(vis_image)
            overlay[current_mask] = color

            # Blend overlay
            alpha = 0.25
            vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)

            # Draw contour
            contours, _ = cv2.findContours(current_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (255, 255, 255), 2)

            # Draw axes
            center = self.compute_instance_center(current_mask)
            if center:
                axes_data = self.axes_data[instance_id]
                vis_image = self.draw_axes_on_image(
                    vis_image, center,
                    axes_data['phi'], axes_data['theta'], axes_data['gamma']
                )

                # Add instance ID label
                label_pos = (center[0] - 15, center[1] - 90)
                cv2.putText(
                    vis_image, f"ID:{instance_id}",
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

        return vis_image

    def save_annotations(self, output_base_path, instance_mask, image_shape):
        """
        Save instance masks, axes annotations, and 6D rotation representations

        This function saves multiple formats optimized for different use cases:

        1. Instance masks (.npy): Standard segmentation masks for object detection
        2. Euler angles (.npy): Human-interpretable rotation angles 
        3. Rotation matrices (.npy): Full 3x3 rotation matrices
        4. 6D rotations (.npy): Zhou et al. continuous representation for neural networks
        5. Pixel-wise 6D field (.npy): Dense ground truth for training orientation heads
        6. Text summary (.txt): Human-readable summary of all annotations

        The 6D outputs follow Zhou et al. CVPR 2019 "On the Continuity of Rotation 
        Representations in Neural Networks" for optimal neural network training:

        - 6D representation is CONTINUOUS (vs discontinuous quaternions/Euler)
        - Enables gradient-based optimization without singularities
        - Proven 6-14x lower errors in rotation regression tasks

        Neural Network Training Usage:
        ```python
        # Load the pixel-wise 6D field as ground truth
        gt_rotation_field = np.load("image_rotation_field_6d.npy")  # (H, W, 6)
        gt_instances = np.load("image_instances.npy")               # (H, W)

        # Network architecture: image -> per-pixel 6D rotations
        orientation_head = nn.Linear(feat_dim, 6)
        pred_6d = orientation_head(features)  # (B, H, W, 6)

        # Convert to rotation matrices using Gram-Schmidt
        pred_R = rot6d_to_matrix(pred_6d)
        gt_R = rot6d_to_matrix(gt_rotation_field)

        # Geodesic loss (angular error on SO(3))
        loss = orientation_loss_6d(instance_mask, gt_R, pred_6d)
        ```

        Args:
            output_base_path (str): Base path for output files (without extension)
            instance_mask (np.ndarray): Instance segmentation mask
            image_shape (tuple): Shape of original image

        Returns:
            dict: Paths to all generated files with descriptive keys
        """
        # Save instance mask as .npy
        mask_path = f"{output_base_path}_instances.npy"
        np.save(mask_path, instance_mask)
        print_status(f"Instance masks saved to: {mask_path}", "success")

        # Prepare axes data for saving
        axes_array_data = []
        rotation_matrices_data = []

        total_instances = len(self.axes_data)
        for instance_id in range(1, total_instances + 1):
            if instance_id in self.axes_data:
                # Get instance center
                current_mask = (instance_mask == instance_id)
                center = self.compute_instance_center(current_mask)

                axes_data = self.axes_data[instance_id]

                # Store: [instance_id, center_x, center_y, phi, theta, gamma]
                axes_array_data.append([
                    instance_id,
                    center[0] if center else 0,
                    center[1] if center else 0,
                    axes_data['phi'],
                    axes_data['theta'],
                    axes_data['gamma']
                ])

                # Store rotation matrix for 6D conversion
                R = self.euler_to_rotation_matrix(
                    axes_data['phi'],
                    axes_data['theta'],
                    axes_data['gamma']
                )
                rotation_matrices_data.append({
                    'instance_id': instance_id,
                    'rotation_matrix': R,
                    'rotation_6d': self.rotation_matrix_to_6d(R)
                })

        # Save axes annotations as .npy (Euler angles)
        axes_path = f"{output_base_path}_axes.npy"
        axes_array = np.array(axes_array_data)
        np.save(axes_path, axes_array)
        print_status(f"Axes annotations (Euler) saved to: {axes_path}", "success")

        # Save rotation matrices as .npy
        matrices_path = f"{output_base_path}_rotation_matrices.npy"
        matrices_array = np.array([item['rotation_matrix'] for item in rotation_matrices_data])
        np.save(matrices_path, matrices_array)
        print_status(f"Rotation matrices saved to: {matrices_path}", "success")

        # Save 6D rotation representations per instance
        rot6d_path = f"{output_base_path}_rotation_6d.npy"
        rot6d_array = np.array([item['rotation_6d'] for item in rotation_matrices_data])
        np.save(rot6d_path, rot6d_array)
        print_status(f"6D rotation vectors saved to: {rot6d_path}", "success")

        # Create and save pixel-wise 6D rotation field
        rotation_field = self.create_6d_rotation_field(instance_mask, image_shape)
        field_path = f"{output_base_path}_rotation_field_6d.npy"
        np.save(field_path, rotation_field)
        print_status(f"Pixel-wise 6D rotation field saved to: {field_path}", "success")

        # Also save as human-readable text
        txt_path = f"{output_base_path}_axes.txt"
        with open(txt_path, 'w') as f:
            f.write("# Instance Axes Annotations\n")
            f.write("# Generated by Instance Axes Annotator\n")
            f.write("# Compatible with Zhou et al. CVPR 2019 6D rotation representation\n\n")

            f.write("# COORDINATE SYSTEM: Camera Coordinate System\n")
            f.write("# Reference Frame: Looking at camera head-on from its front\n")
            f.write("# X-axis (RED): Points RIGHT in camera view\n")
            f.write("# Y-axis (GREEN): Points DOWN in camera view\n")
            f.write("# Z-axis (BLUE): Points FORWARD into scene (away from camera)\n")
            f.write("# Rotation Matrix: R transforms object coordinates -> camera coordinates\n")
            f.write("# camera_vector = R @ object_vector\n\n")

            f.write("# EULER ANGLES (Human Readable)\n")
            f.write("# Format: instance_id, center_x, center_y, phi(yaw), theta(pitch), gamma(roll)\n")
            f.write("# Angles in degrees, ZYX rotation convention\n")
            f.write("# phi (yaw): Rotation around Z-axis (camera forward) - left/right turn\n")
            f.write("# theta (pitch): Rotation around Y-axis (camera down) - up/down tilt\n")
            f.write("# gamma (roll): Rotation around X-axis (camera right) - clockwise/counterclockwise\n")
            for row in axes_array:
                f.write(f"{int(row[0])}, {row[1]:.1f}, {row[2]:.1f}, {row[3]:.1f}, {row[4]:.1f}, {row[5]:.1f}\n")

            f.write("\n# 6D ROTATION VECTORS (Neural Network Training)\n")
            f.write("# Format: instance_id, a1_x, a1_y, a1_z, a2_x, a2_y, a2_z\n")
            f.write("# First two columns of rotation matrix (Zhou et al. CVPR 2019)\n")
            f.write("# All vectors are in CAMERA COORDINATE SYSTEM\n")
            f.write("# Usage: orientation_head = nn.Linear(feat_dim, 6)\n")
            f.write("# Reconstruction: b1=normalize(a1), b2=normalize(a2-(b1·a2)*b1), b3=b1×b2\n")
            f.write("# Loss: geodesic angle θ = arccos((trace(R_gt^T * R_pred) - 1) / 2)\n")
            for i, item in enumerate(rotation_matrices_data):
                rot6d = item['rotation_6d']
                f.write(f"{item['instance_id']}, {rot6d[0]:.6f}, {rot6d[1]:.6f}, {rot6d[2]:.6f}, {rot6d[3]:.6f}, {rot6d[4]:.6f}, {rot6d[5]:.6f}\n")

            f.write(f"\n# OUTPUT FILES SUMMARY\n")
            f.write(f"# {output_base_path}_instances.npy: Instance segmentation masks\n")
            f.write(f"# {output_base_path}_axes.npy: Euler angles per instance\n")
            f.write(f"# {output_base_path}_rotation_matrices.npy: Full 3x3 rotation matrices\n")
            f.write(f"# {output_base_path}_rotation_6d.npy: 6D vectors per instance\n")
            f.write(f"# {output_base_path}_rotation_field_6d.npy: Dense (H,W,6) field for training\n")
            f.write(f"# Use rotation_field_6d.npy as ground truth for neural orientation heads\n")
            f.write(f"# ALL ORIENTATIONS ARE IN CAMERA COORDINATE SYSTEM\n")

        print_status(f"Axes annotations (text) saved to: {txt_path}", "success")

        return {
            'instances': mask_path,
            'axes_euler': axes_path,
            'rotation_matrices': matrices_path,
            'rotation_6d': rot6d_path,
            'rotation_field_6d': field_path,
            'text_summary': txt_path
        }

    @staticmethod
    def visualize_rotation_matrix(image, center, rotation_matrix, axis_length=80, axis_thickness=6):
        """
        Simple utility to draw 3D axes on an image given a rotation matrix

        Args:
            image (np.ndarray): Image to draw on (will be modified in-place)
            center (tuple): (x, y) center point for axes
            rotation_matrix (np.ndarray): 3x3 rotation matrix
            axis_length (int): Length of axes in pixels
            axis_thickness (int): Thickness of axes lines

        Returns:
            np.ndarray: Image with axes drawn
        """
        # Standard basis vectors
        x_axis = np.array([1, 0, 0])  # Red axis (X)
        y_axis = np.array([0, 1, 0])  # Green axis (Y)
        z_axis = np.array([0, 0, 1])  # Blue axis (Z)

        # Apply rotation to get oriented axes
        x_3d = rotation_matrix @ x_axis
        y_3d = rotation_matrix @ y_axis
        z_3d = rotation_matrix @ z_axis

        # Simple 2D projection (ignoring perspective)
        def project_to_2d(vec_3d):
            return np.array([vec_3d[0], -vec_3d[1]]) * axis_length  # Flip Y for screen coords

        # Project to 2D
        x_2d = project_to_2d(x_3d)
        y_2d = project_to_2d(y_3d)
        z_2d = project_to_2d(z_3d)

        # Calculate end points
        center = np.array(center)
        x_end = center + x_2d
        y_end = center + y_2d
        z_end = center + z_2d

        # Axis colors: Red=X, Green=Y, Blue=Z
        axis_colors = {
            'x': (0, 0, 255),    # RED for X-axis
            'y': (0, 255, 0),    # GREEN for Y-axis
            'z': (255, 0, 0)     # BLUE for Z-axis
        }

        # Draw axes (back to front based on Z-depth)
        axes_data = [
            ('z', z_end, z_3d[2]),
            ('y', y_end, y_3d[2]),
            ('x', x_end, x_3d[2])
        ]
        axes_data.sort(key=lambda x: x[2])  # Sort by depth

        for axis_name, end_point, _ in axes_data:
            cv2.arrowedLine(
                image,
                tuple(center.astype(int)),
                tuple(end_point.astype(int)),
                axis_colors[axis_name],
                axis_thickness,
                tipLength=0.3
            )

            # Add axis label
            label_pos = end_point + (end_point - center) * 0.2
            cv2.putText(
                image,
                axis_name.upper(),
                tuple(label_pos.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                axis_colors[axis_name],
                2
            )

        return image


def main():
    parser = argparse.ArgumentParser(description="Interactive SAM2 Instance Segmentation + 3D Axes Annotation")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for all generated files (default: same directory as input image)")
    parser.add_argument("--sam2_checkpoint",
                        default="//robodata/smodak/repos/sam2/checkpoints/sam2.1_hiera_large.pt",
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg",
                        default="//robodata/smodak/repos/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="Path to SAM2 model config")
    parser.add_argument("--skip_segmentation", action="store_true",
                        help="Skip segmentation and load existing instance mask")
    parser.add_argument("--instance_mask_path", type=str,
                        help="Path to existing instance mask .npy file (if skip_segmentation=True)")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable optimizations for CUDA
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        print_status("Warning: CUDA not available, using CPU (will be slow)", "warning")

    print_status(f"Using device: {device}", "info")

    # Load image
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not load image: {args.image_path}")

    # Setup output paths
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        image_basename = os.path.splitext(os.path.basename(args.image_path))[0]
        output_base = os.path.join(args.output_dir, image_basename)
    else:
        output_base = os.path.splitext(args.image_path)[0]

    # Phase 1: Instance Segmentation
    if args.skip_segmentation:
        if not args.instance_mask_path or not os.path.exists(args.instance_mask_path):
            raise FileNotFoundError("Instance mask path required when skipping segmentation")

        print_status("Loading existing instance mask...", "info")
        instance_mask = np.load(args.instance_mask_path)
        print_status(f"Loaded instance mask with {np.max(instance_mask)} instances", "success")
    else:
        print_colored("\n" + "=" * 60, Colors.BLUE, bold=True)
        print_colored("PHASE 1: INSTANCE SEGMENTATION", Colors.BLUE, bold=True)
        print_colored("=" * 60, Colors.BLUE, bold=True)

        # Create segmenter and run
        segmenter = InteractiveSAM2Segmenter(args.sam2_checkpoint, args.model_cfg, device)
        mask_output_path = segmenter.run_interactive_segmentation(args.image_path)

        if not mask_output_path:
            print_status("Segmentation cancelled. Exiting.", "error")
            return

        # Load the generated instance mask
        instance_mask = np.load(mask_output_path)
        print_status(f"Loaded instance mask with {np.max(instance_mask)} instances", "success")

    total_instances = int(np.max(instance_mask))

    if total_instances == 0:
        print_status("No instances found in mask. Exiting.", "error")
        return

    # Phase 2: Axes Annotation
    print_colored("\n" + "=" * 60, Colors.BLUE, bold=True)
    print_colored("PHASE 2: 3D AXES ANNOTATION", Colors.BLUE, bold=True)
    print_colored("=" * 60, Colors.BLUE, bold=True)

    axes_annotator = AxesAnnotator()

    # Run axes annotation
    success = axes_annotator.annotate_axes_for_instances(image, instance_mask, total_instances)

    if not success:
        print_status("Axes annotation cancelled. Exiting.", "error")
        return

    # Phase 3: Final Visualization and Saving
    print_colored("\n" + "=" * 60, Colors.BLUE, bold=True)
    print_colored("PHASE 3: FINAL RESULTS", Colors.BLUE, bold=True)
    print_colored("=" * 60, Colors.BLUE, bold=True)

    # Create final visualization
    final_vis = axes_annotator.create_final_visualization(image, instance_mask, total_instances)

    # Display final result
    print_status("Displaying final visualization...", "info")

    # Calculate display scale
    display_height, display_width = final_vis.shape[:2]
    max_display_height = 800
    max_display_width = 1200

    if display_height > max_display_height or display_width > max_display_width:
        scale_h = max_display_height / display_height
        scale_w = max_display_width / display_width
        display_scale = min(scale_h, scale_w)
        display_vis = cv2.resize(final_vis,
                                 (int(display_width * display_scale), int(display_height * display_scale)))
    else:
        display_vis = final_vis

    cv2.imshow("Final Instance Segmentation + 3D Axes", display_vis)
    print_colored("Press any key to save and exit...", Colors.CYAN)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save all annotations
    output_files = axes_annotator.save_annotations(output_base, instance_mask, image.shape)

    # Save final visualization image
    final_vis_path = f"{output_base}_final_visualization.jpg"
    cv2.imwrite(final_vis_path, final_vis)
    print_status(f"Final visualization saved to: {final_vis_path}", "success")

    # Print summary
    print_colored(f"\n📊 ANNOTATION COMPLETE!", Colors.GREEN, bold=True)
    print_colored(f"   Total instances: {total_instances}", Colors.CYAN)
    print_colored(f"   📂 Output Files:", Colors.WHITE, bold=True)
    print_colored(f"      Instance masks: {output_files['instances']}", Colors.CYAN)
    print_colored(f"      Euler angles: {output_files['axes_euler']}", Colors.CYAN)
    print_colored(f"      Rotation matrices: {output_files['rotation_matrices']}", Colors.CYAN)
    print_colored(f"      6D rotations: {output_files['rotation_6d']}", Colors.CYAN)
    print_colored(f"      6D pixel field: {output_files['rotation_field_6d']}", Colors.CYAN)
    print_colored(f"      Text summary: {output_files['text_summary']}", Colors.CYAN)
    print_colored(f"      Final visualization: {final_vis_path}", Colors.CYAN)


if __name__ == "__main__":
    main()
