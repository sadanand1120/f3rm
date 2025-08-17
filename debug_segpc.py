import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.cm as cm
import open3d as o3d
import json
from sam2.features.utils import SAM2utils
from sam2.features.clip_main import CLIPfeatures
from f3rm.features.sam2_extract import SAM2Args
from debug_clipsam_seg import CLIPSAMSegmenter


class CLIPSAMPointcloudData:
    def __init__(self, points: np.ndarray, colors: np.ndarray, mean_points: list, mean_colors: list, camera_frustums: list, frustum_colors: list, camera_poses: list):
        self.points, self.colors = points, colors
        self.mean_points, self.mean_colors = mean_points, mean_colors
        self.camera_frustums, self.frustum_colors = camera_frustums, frustum_colors
        self.camera_poses = camera_poses

    def to_dict(self) -> dict:
        return {
            'points': self.points.tolist(), 'colors': self.colors.tolist(),
            'mean_points': [point.tolist() for point in self.mean_points],
            'mean_colors': [color.tolist() for color in self.mean_colors],
            'camera_poses': [pose.tolist() for pose in self.camera_poses],
            'num_points': len(self.points), 'num_means': len(self.mean_points), 'num_frustums': len(self.camera_frustums)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CLIPSAMPointcloudData':
        return cls(
            points=np.array(data['points']), colors=np.array(data['colors']),
            mean_points=[np.array(point) for point in data['mean_points']],
            mean_colors=[np.array(color) for color in data['mean_colors']],
            camera_frustums=[], frustum_colors=[np.array(color) for color in data['mean_colors']],
            camera_poses=[np.array(pose) for pose in data.get('camera_poses', [])]
        )

    def save_to_directory(self, output_dir: Path, mean_point_scale: float, frustum_scale: float, pointcloud_point_size: float, pointcloud_opacity: float):
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "points.npy", self.points)
        np.save(output_dir / "colors.npy", self.colors)
        metadata = {
            'num_points': len(self.points), 'num_means': len(self.mean_points), 'num_frustums': len(self.camera_frustums),
            'bbox_min': self.points.min(axis=0).tolist(), 'bbox_max': self.points.max(axis=0).tolist(),
            'mean_points': [point.tolist() for point in self.mean_points],
            'mean_colors': [color.tolist() for color in self.mean_colors],
            'frustum_colors': [color.tolist() for color in self.frustum_colors],
            'camera_poses': [pose.tolist() for pose in self.camera_poses],
            'mean_point_scale': mean_point_scale, 'frustum_scale': frustum_scale,
            'pointcloud_point_size': pointcloud_point_size, 'pointcloud_opacity': pointcloud_opacity
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        o3d.io.write_point_cloud(str(output_dir / "pointcloud.pcd"), pcd)
        print(f"Saved CLIP-SAM pointcloud data to: {output_dir}")


def downsample_points(points: np.ndarray, colors: np.ndarray, num_images: int) -> tuple[np.ndarray, np.ndarray]:
    """Downsample points to contribute 1/N of original points per image."""
    target_points = max(1, len(points) // num_images)
    if len(points) <= target_points:
        return points, colors
    indices = np.random.choice(len(points), target_points, replace=False)
    return points[indices], colors[indices]


def create_frustum_mesh(scale: float = 1.0) -> o3d.geometry.TriangleMesh:
    """Create a camera frustum as a mesh with thick edges."""
    near_plane, far_plane = 0.1 * scale, 0.2 * scale
    fov_h, fov_v = 0.9 * scale, 0.75 * scale
    edge_thickness = 0.01 * scale
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


@torch.no_grad()
def process_single_image_pointcloud(segmenter, clip_model, image_path, text_prompt, negative_text, softmax_temp, min_instance_percent, top_mean_percent, pc_thresh, device):
    feat_image_index, image_path, split, local_cam_idx = segmenter.get_cam_info(image_path=image_path)
    clip_patch_feats = segmenter.clip_features[feat_image_index].to(device)
    text_emb = clip_model.encode_text(text_prompt)
    neg_text_embs = torch.stack([clip_model.encode_text(negative_text)], dim=0)
    sim_map = clip_model.compute_similarity(clip_patch_feats, text_emb, neg_text_embs=neg_text_embs, softmax_temp=softmax_temp, normalize=True)
    auto_masks = segmenter.sam2_masks[feat_image_index]
    inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(auto_masks, min_iou=float(SAM2Args.pred_iou_thresh), min_area=float(SAM2Args.min_mask_region_area), assign_by="area", start_from="low")
    inst_mask = segmenter.filter_sam2_inst_mask(inst_mask, min_instance_percent)
    sim_map_upscaled = np.array(Image.fromarray(sim_map.cpu().numpy()).resize((inst_mask.shape[1], inst_mask.shape[0]), Image.BILINEAR))
    segment_sim_map = segmenter.compute_segment_similarity(sim_map_upscaled, inst_mask, top_mean_percent)
    pipeline_outputs, camera_ray_bundle, optimized_c2w = segmenter.get_pipeline_outputs(split, local_cam_idx, render_features=False)
    points_3d, colors_3d = segmenter.generate_pointcloud_from_clipsam(segment_sim_map, pipeline_outputs["depth_raw"], camera_ray_bundle, pipeline_outputs["pred_rgb"], pc_thresh=pc_thresh)
    return points_3d, colors_3d, optimized_c2w


def visualize_clipsam_pointcloud(data_dir: str):
    """Standalone function to visualize CLIP-SAM pointcloud data. Copy-paste this function to any machine."""
    import numpy as np
    import pyvista as pv
    from pathlib import Path
    import json

    def create_frustum_mesh_pv(scale: float = 1.0) -> pv.PolyData:
        near_plane, far_plane = 0.1 * scale, 0.2 * scale
        fov_h, fov_v = 0.9 * scale, 0.75 * scale
        points = np.array([
            [0, 0, 0], [-fov_h * near_plane, -fov_v * near_plane, -near_plane], [fov_h * near_plane, -fov_v * near_plane, -near_plane],
            [fov_h * near_plane, fov_v * near_plane, -near_plane], [-fov_h * near_plane, fov_v * near_plane, -near_plane],
            [-fov_h * far_plane, -fov_v * far_plane, -far_plane], [fov_h * far_plane, -fov_v * far_plane, -far_plane],
            [fov_h * far_plane, fov_v * far_plane, -far_plane], [-fov_h * far_plane, fov_v * far_plane, -far_plane],
        ])
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1), (1, 5), (2, 6), (3, 7), (4, 8), (5, 6), (6, 7), (7, 8), (8, 5)]
        all_lines = []
        for start_idx, end_idx in edges:
            line = pv.Line(points[start_idx], points[end_idx])
            all_lines.append(line)
        if all_lines:
            frustum = all_lines[0]
            for line in all_lines[1:]:
                frustum = frustum.merge(line)
            return frustum
        else:
            return pv.PolyData()

    data_dir = Path(data_dir)
    points = np.load(data_dir / "points.npy")
    colors = np.load(data_dir / "colors.npy")
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    pcd = pv.PolyData(points)
    pcd.point_data["colors"] = colors
    mean_point_scale = metadata['mean_point_scale']
    frustum_scale = metadata['frustum_scale']
    pointcloud_point_size = metadata['pointcloud_point_size']
    pointcloud_opacity = metadata['pointcloud_opacity']
    mean_points = [np.array(point) for point in metadata['mean_points']]
    mean_colors = [np.array(color) for color in metadata['mean_colors']]

    plotter = pv.Plotter()
    plotter.add_points(pcd, scalars="colors", rgb=True, point_size=pointcloud_point_size, opacity=pointcloud_opacity)

    for point, color in zip(mean_points, mean_colors):
        sphere = pv.Sphere(radius=mean_point_scale, center=point)
        plotter.add_mesh(sphere, color=color, opacity=1.0)

    if 'camera_poses' in metadata:
        camera_poses = [np.array(pose) for pose in metadata['camera_poses']]
        frustum_colors = [np.array(color) for color in metadata['frustum_colors']]
        for pose, color in zip(camera_poses, frustum_colors):
            frustum = create_frustum_mesh_pv(scale=frustum_scale)
            frustum.transform(pose)
            plotter.add_mesh(frustum, color=color, line_width=3)

    plotter.show()


if __name__ == "__main__":
    INPUT_IMAGES = [
        "datasets/f3rm/custom/betaipad/small/images/frame_00043.png",  # 1, 71, 91, 101 for book
        "datasets/f3rm/custom/betaipad/small/images/frame_00074.png",
        "datasets/f3rm/custom/betaipad/small/images/frame_00091.png",
        "datasets/f3rm/custom/betaipad/small/images/frame_00115.png",
    ]
    CONFIG_PATH = "bugfix_outputs/betaipad_small_pipe_default/f3rm/2025-08-15_155758/config.yml"
    DATA_DIR = Path(INPUT_IMAGES[0]).parent.parent
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TEXT_PROMPT = "ipad"
    NEGATIVE_TEXT = "object"
    SOFTMAX_TEMP = 0.01
    MIN_INSTANCE_PERCENT = 1.0
    TOP_MEAN_PERCENT = 5
    
    PC_THRESH = 0.4  # 0.7 for book, 0.4 for ipad
    OUTPUT_DIR = Path("clipsam_pointcloud_output")
    MEAN_POINT_SCALE, FRUSTUM_SCALE = 0.04, 0.4
    POINTCLOUD_POINT_SIZE, POINTCLOUD_OPACITY = 1.0, 0.6

    segmenter = CLIPSAMSegmenter(data_dir=DATA_DIR, config_path=CONFIG_PATH, debug=False)
    clip_model = CLIPfeatures(device=DEVICE)
    tab20 = cm.get_cmap('tab20')
    num_images = len(INPUT_IMAGES)
    colors = [np.array(tab20(i / num_images)[:3]) for i in range(num_images)]

    all_points, all_colors = [], []
    mean_points, mean_colors = [], []
    camera_frustums, frustum_colors = [], []
    camera_poses = []

    for i, image_path in enumerate(tqdm(INPUT_IMAGES, desc="Processing images")):
        points_3d, colors_3d, camera_pose = process_single_image_pointcloud(
            segmenter, clip_model, image_path, TEXT_PROMPT, NEGATIVE_TEXT,
            SOFTMAX_TEMP, MIN_INSTANCE_PERCENT, TOP_MEAN_PERCENT, PC_THRESH, DEVICE
        )
        if len(points_3d) > 0:
            sampled_points, sampled_colors = downsample_points(points_3d, colors_3d, len(INPUT_IMAGES))
            all_points.append(sampled_points)
            all_colors.append(sampled_colors)
            mean_point = np.mean(points_3d, axis=0)
            mean_points.append(mean_point)
            mean_colors.append(colors[i])
            frustum_mesh = create_frustum_mesh(FRUSTUM_SCALE)
            frustum_mesh.paint_uniform_color(colors[i])
            frustum_mesh.transform(camera_pose)
            camera_frustums.append(frustum_mesh)
            frustum_colors.append(colors[i])
            camera_poses.append(camera_pose)
            tqdm.write(f"Generated {len(points_3d)} points from {os.path.basename(image_path)}")
        else:
            tqdm.write(f"No points generated from {os.path.basename(image_path)}")

    if all_points:
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        print(f"Total merged pointcloud: {len(merged_points)} points")

        pointcloud_data = CLIPSAMPointcloudData(
            points=merged_points, colors=merged_colors, mean_points=mean_points, mean_colors=mean_colors,
            camera_frustums=camera_frustums, frustum_colors=frustum_colors, camera_poses=camera_poses
        )
        pointcloud_data.save_to_directory(OUTPUT_DIR, MEAN_POINT_SCALE, FRUSTUM_SCALE, POINTCLOUD_POINT_SIZE, POINTCLOUD_OPACITY)

        complete_mesh = o3d.geometry.TriangleMesh()
        for i, (mean_point, mean_color) in enumerate(tqdm(zip(mean_points, mean_colors), desc="Adding mean points", total=len(mean_points))):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=MEAN_POINT_SCALE)
            sphere.paint_uniform_color(mean_color)
            sphere.translate(mean_point)
            complete_mesh += sphere
        for frustum_mesh in tqdm(camera_frustums, desc="Adding camera frustums", total=len(camera_frustums)):
            complete_mesh += frustum_mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)
        geometries = [pcd, complete_mesh]
        print(f"Added {len(mean_points)} mean points and {len(camera_frustums)} camera frustums")
        print(f"Color scheme: {num_images} colors from tab20 palette")
        print(f"Data saved to: {OUTPUT_DIR}")
    else:
        print("No points generated from any image")
