"""
final conclusion: optimized and non-optimized frustums are pretty much the same. Very small refinement in the camera pose (ie, extrinsics).
"""

import torch
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from pathlib import Path
import open3d as o3d
from tqdm import tqdm
from debug_clipsam_seg import CLIPSAMSegmenter


def create_frustum_mesh(scale: float = 1.0) -> o3d.geometry.TriangleMesh:
    """Create a camera frustum as a mesh with thick edges."""
    near_plane, far_plane = 0.1 * scale, 0.2 * scale
    fov_h, fov_v = 0.9 * scale, 0.75 * scale
    edge_thickness = 0.002 * scale
    mesh = o3d.geometry.TriangleMesh()

    points = np.array([
        [0, 0, 0], [-fov_h * near_plane, -fov_v * near_plane, -near_plane], [fov_h * near_plane, -fov_v * near_plane, -near_plane],
        [fov_h * near_plane, fov_v * near_plane, -near_plane], [-fov_h * near_plane, fov_v * near_plane, -near_plane],
        [-fov_h * far_plane, -fov_v * far_plane, -far_plane], [fov_h * far_plane, -fov_v * far_plane, -far_plane],
        [fov_h * far_plane, fov_v * far_plane, -far_plane], [-fov_h * far_plane, fov_v * far_plane, -far_plane],
    ])

    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1), (1, 5), (2, 6), (3, 7), (4, 8), (5, 6), (6, 7), (7, 8), (8, 5)]

    for i, point in enumerate(points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=edge_thickness)
        sphere.translate(point)
        mesh += sphere

    for start_idx, end_idx in edges:
        start_point, end_point = points[start_idx], points[end_idx]
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=edge_thickness, height=length)
            z_axis = np.array([0, 0, 1])
            if not np.allclose(direction, z_axis):
                rotation_axis = np.cross(z_axis, direction)
                if np.linalg.norm(rotation_axis) > 0:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    cos_angle = np.dot(z_axis, direction)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    axis = rotation_axis
                    a = np.cos(angle / 2)
                    b, c, d = axis * np.sin(angle / 2)
                    rotation_matrix = np.array([
                        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]
                    ])
                    cylinder.rotate(rotation_matrix)
            midpoint = (start_point + end_point) / 2
            cylinder.translate(midpoint)
            mesh += cylinder
    return mesh


# Hardcoded inputs
INPUT_IMAGES = [
    "datasets/f3rm/custom/betamulti1/small/images/frame_00001.png",
    # "datasets/f3rm/custom/betamulti1/small/images/frame_00071.png",
    # "datasets/f3rm/custom/betamulti1/small/images/frame_00091.png",
    # "datasets/f3rm/custom/betamulti1/small/images/frame_00101.png",
]
CONFIG_PATH = "cent7_outputs/betam1_small_cstext_lang32_loss2e3_trunk64F_fg64x1/f3rm/2025-08-27_152340/config.yml"
DATA_DIR = Path(INPUT_IMAGES[0]).parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRUSTUM_SCALE = 0.4

# Setup
segmenter = CLIPSAMSegmenter(data_dir=DATA_DIR, config_path=CONFIG_PATH, debug=False)

# Colors: green for optimized, red for non-optimized
GREEN_COLOR = np.array([0.0, 1.0, 0.0])  # Green for optimized
RED_COLOR = np.array([1.0, 0.0, 0.0])    # Red for non-optimized

all_frustums = []
all_colors = []

# Main logic
for i, image_path in enumerate(tqdm(INPUT_IMAGES, desc="Processing images")):
    feat_image_index, image_path, split, local_cam_idx = segmenter.get_cam_info(image_path=image_path)

    # Get camera poses from the segmenter's pipeline
    cams = segmenter.pipeline.datamanager.eval_ray_generator.cameras if split == 'eval' else segmenter.pipeline.datamanager.train_ray_generator.cameras

    # Original camera pose (non-optimized)
    original_c2w_34 = cams.camera_to_worlds[local_cam_idx].cpu().numpy()
    original_c2w = np.vstack([original_c2w_34, np.array([[0, 0, 0, 1]])])
    print(f"Original pose: \n{original_c2w}")

    # Optimized camera pose
    c_tensor = torch.tensor([local_cam_idx], device=cams.device)
    if split == 'eval':
        camera_opt_to_camera_4x4 = np.eye(4)  # No optimization for eval cameras
    else:
        camera_opt_to_camera = segmenter.pipeline.model.camera_optimizer(c_tensor)
        camera_opt_to_camera_4x4 = np.vstack([camera_opt_to_camera[0].detach().cpu().numpy(), np.array([[0, 0, 0, 1]])])
    optimized_c2w = original_c2w @ camera_opt_to_camera_4x4

    # Create frustums for both poses
    # Non-optimized frustum (red)
    frustum_original = create_frustum_mesh(FRUSTUM_SCALE)
    frustum_original.paint_uniform_color(RED_COLOR)
    frustum_original.transform(original_c2w)
    all_frustums.append(frustum_original)
    all_colors.append(RED_COLOR)

    # Optimized frustum (green)
    frustum_optimized = create_frustum_mesh(FRUSTUM_SCALE)
    frustum_optimized.paint_uniform_color(GREEN_COLOR)
    frustum_optimized.transform(optimized_c2w)
    all_frustums.append(frustum_optimized)
    all_colors.append(GREEN_COLOR)

    print(f"Image {i+1}: {Path(image_path).name}")
    print(f"  Original pose: {original_c2w[:3, 3]}")
    print(f"  Optimized pose: {optimized_c2w[:3, 3]}")
    print(f"  Translation diff: {np.linalg.norm(optimized_c2w[:3, 3] - original_c2w[:3, 3]):.4f}")

# Output
print(f"\nTotal frustums: {len(all_frustums)} (4 images × 2 poses each)")
print("Green = optimized poses, Red = non-optimized poses")

# Visualize
complete_mesh = o3d.geometry.TriangleMesh()
for frustum in all_frustums:
    complete_mesh += frustum

o3d.visualization.draw_geometries([complete_mesh])
