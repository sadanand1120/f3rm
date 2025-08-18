import os
import shutil
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
    def __init__(self, points: np.ndarray, colors: np.ndarray, mean_points: list, mean_colors: list, camera_frustums: list, frustum_colors: list, camera_poses: list, pred_points: np.ndarray | None = None, pred_colors: np.ndarray | None = None):
        self.points, self.colors = points, colors
        self.mean_points, self.mean_colors = mean_points, mean_colors
        self.camera_frustums, self.frustum_colors = camera_frustums, frustum_colors
        self.camera_poses = camera_poses
        self.pred_points = pred_points if pred_points is not None else np.empty((0, 3))
        self.pred_colors = pred_colors if pred_colors is not None else np.empty((0, 3))

    def to_dict(self) -> dict:
        return {
            'points': self.points.tolist(), 'colors': self.colors.tolist(),
            'mean_points': [point.tolist() for point in self.mean_points],
            'mean_colors': [color.tolist() for color in self.mean_colors],
            'camera_poses': [pose.tolist() for pose in self.camera_poses],
            'num_points': len(self.points), 'num_means': len(self.mean_points), 'num_frustums': len(self.camera_frustums),
            'pred_points': self.pred_points.tolist(), 'pred_colors': self.pred_colors.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CLIPSAMPointcloudData':
        return cls(
            points=np.array(data['points']), colors=np.array(data['colors']),
            mean_points=[np.array(point) for point in data['mean_points']],
            mean_colors=[np.array(color) for color in data['mean_colors']],
            camera_frustums=[], frustum_colors=[np.array(color) for color in data['mean_colors']],
            camera_poses=[np.array(pose) for pose in data.get('camera_poses', [])],
            pred_points=np.array(data.get('pred_points', [])), pred_colors=np.array(data.get('pred_colors', []))
        )

    def save_to_directory(self, output_dir: Path, mean_point_scale: float, frustum_scale: float, pointcloud_point_size: float, pointcloud_opacity: float):
        # delete output dir if it exists
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save point clouds only if they have data
        if len(self.points) > 0:
            np.save(output_dir / "points.npy", self.points)
            np.save(output_dir / "colors.npy", self.colors)

        if len(self.pred_points) > 0:
            np.save(output_dir / "pred_points.npy", self.pred_points)
            np.save(output_dir / "pred_colors.npy", self.pred_colors)

        # Compute bbox from available points
        all_points = []
        if len(self.points) > 0:
            all_points.append(self.points)
        if len(self.pred_points) > 0:
            all_points.append(self.pred_points)

        if all_points:
            combined_points = np.vstack(all_points)
            bbox_min = combined_points.min(axis=0).tolist()
            bbox_max = combined_points.max(axis=0).tolist()
        else:
            # Default bbox if no points
            bbox_min = [-1.0, -1.0, -1.0]
            bbox_max = [1.0, 1.0, 1.0]

        metadata = {
            'num_points': len(self.points), 'num_means': len(self.mean_points), 'num_frustums': len(self.camera_frustums),
            'bbox_min': bbox_min, 'bbox_max': bbox_max,
            'mean_points': [point.tolist() for point in self.mean_points],
            'mean_colors': [color.tolist() for color in self.mean_colors],
            'frustum_colors': [color.tolist() for color in self.frustum_colors],
            'camera_poses': [pose.tolist() for pose in self.camera_poses],
            'mean_point_scale': mean_point_scale, 'frustum_scale': frustum_scale,
            'pointcloud_point_size': pointcloud_point_size, 'pointcloud_opacity': pointcloud_opacity,
            'has_pred': bool(len(self.pred_points) > 0)
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save main pointcloud only if it has data
        if len(self.points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
            o3d.io.write_point_cloud(str(output_dir / "pointcloud.pcd"), pcd)
        if len(self.pred_points) > 0:
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(self.pred_points)
            pcd_pred.colors = o3d.utility.Vector3dVector(self.pred_colors)
            o3d.io.write_point_cloud(str(output_dir / "pred_pointcloud.pcd"), pcd_pred)
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
    pipeline_outputs, camera_ray_bundle, optimized_c2w = segmenter.get_pipeline_outputs(split, local_cam_idx, render_features=True)
    points_3d, colors_3d = segmenter.generate_pointcloud_from_clipsam(segment_sim_map, pipeline_outputs["depth_raw"], camera_ray_bundle, pipeline_outputs["pred_rgb"], pc_thresh=pc_thresh)
    centroid_pred = pipeline_outputs.get("centroid_pred", None)
    pred_centroids_segment = None
    if centroid_pred is not None:
        binary_mask = (segment_sim_map > pc_thresh)
        pred_centroids_segment = centroid_pred[binary_mask].cpu().numpy()
    return points_3d, colors_3d, optimized_c2w, pred_centroids_segment


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
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    mean_point_scale = metadata['mean_point_scale']
    frustum_scale = metadata['frustum_scale']
    pointcloud_point_size = metadata['pointcloud_point_size']
    pointcloud_opacity = metadata['pointcloud_opacity']
    mean_points = [np.array(point) for point in metadata['mean_points']]
    mean_colors = [np.array(color) for color in metadata['mean_colors']]

    plotter = pv.Plotter()
    # GT RGB pointcloud if present
    pts_path, cols_path = data_dir / "points.npy", data_dir / "colors.npy"
    if pts_path.exists() and cols_path.exists():
        points = np.load(pts_path)
        colors = np.load(cols_path)
        if len(points) > 0:
            pcd = pv.PolyData(points)
            pcd.point_data["colors"] = colors
            plotter.add_points(pcd, scalars="colors", rgb=True, point_size=pointcloud_point_size, opacity=pointcloud_opacity)
    # Predicted centroids pointcloud if present
    pred_pts_path, pred_cols_path = data_dir / "pred_points.npy", data_dir / "pred_colors.npy"
    if pred_pts_path.exists() and pred_cols_path.exists():
        pred_points = np.load(pred_pts_path)
        pred_colors = np.load(pred_cols_path)
        if len(pred_points) > 0:
            pcd_pred = pv.PolyData(pred_points)
            pcd_pred.point_data["colors"] = pred_colors
            plotter.add_points(pcd_pred, scalars="colors", rgb=True, point_size=pointcloud_point_size, opacity=1.0)

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
        "datasets/f3rm/custom/betabook/small/images/frame_00001.png",  # 1, 71, 91, 101 for book
        "datasets/f3rm/custom/betabook/small/images/frame_00071.png",
        "datasets/f3rm/custom/betabook/small/images/frame_00091.png",
        "datasets/f3rm/custom/betabook/small/images/frame_00101.png",
    ]
    CONFIG_PATH = "cent_outputs/betabook_small_pipe_cent_cold15k_nodensemb/f3rm/2025-08-17_211558/config.yml"
    DATA_DIR = Path(INPUT_IMAGES[0]).parent.parent
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TEXT_PROMPT = "book"
    NEGATIVE_TEXT = "object"
    SOFTMAX_TEMP = 0.01
    MIN_INSTANCE_PERCENT = 1.0
    TOP_MEAN_PERCENT = 5

    PC_THRESH = 0.7  # 0.7 for book, 0.4 for ipad
    OUTPUT_DIR = Path("clipsam_pointcloud_output")
    MEAN_POINT_SCALE, FRUSTUM_SCALE = 0.04, 0.4
    POINTCLOUD_POINT_SIZE, POINTCLOUD_OPACITY = 1.0, 0.6
    SHOW_PC = True
    SHOW_PRED_CENTROID = True

    segmenter = CLIPSAMSegmenter(data_dir=DATA_DIR, config_path=CONFIG_PATH, debug=False)
    clip_model = CLIPfeatures(device=DEVICE)
    tab20 = cm.get_cmap('tab20')
    num_images = len(INPUT_IMAGES)
    colors = [np.array(tab20(i / num_images)[:3]) for i in range(num_images)]

    all_points, all_colors = [], []
    mean_points, mean_colors = [], []
    camera_frustums, frustum_colors = [], []
    all_pred_centroids, all_pred_colors = [], []
    camera_poses = []

    for i, image_path in enumerate(tqdm(INPUT_IMAGES, desc="Processing images")):
        points_3d, colors_3d, camera_pose, pred_centroids_segment = process_single_image_pointcloud(
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
            if pred_centroids_segment is not None and len(pred_centroids_segment) > 0:
                # Downsample predicted segment centroids to match GT point count for this image
                pred_pts = pred_centroids_segment
                max_preds = min(len(pred_pts), len(sampled_points))
                if max_preds < len(pred_pts):
                    idx = np.random.choice(len(pred_pts), max_preds, replace=False)
                    pred_pts = pred_pts[idx]
                all_pred_centroids.append(pred_pts)
                pred_color = colors[i]
                pred_cols = np.tile(pred_color[None, :], (len(pred_pts), 1))
                all_pred_colors.append(pred_cols)
            tqdm.write(f"Generated {len(points_3d)} points from {os.path.basename(image_path)}")
        else:
            tqdm.write(f"No points generated from {os.path.basename(image_path)}")

    if all_points:
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        print(f"Total merged pointcloud: {len(merged_points)} points")

        pred_points = np.vstack(all_pred_centroids) if (all_pred_centroids and SHOW_PRED_CENTROID) else np.empty((0, 3))
        pred_colors = np.vstack(all_pred_colors) if (all_pred_colors and SHOW_PRED_CENTROID) else np.empty((0, 3))
        # Always save an output dir with metadata; include pc/pred only if flags enabled
        pointcloud_data = CLIPSAMPointcloudData(
            points=merged_points if SHOW_PC else np.empty((0, 3)),
            colors=merged_colors if SHOW_PC else np.empty((0, 3)),
            mean_points=mean_points, mean_colors=mean_colors,
            camera_frustums=camera_frustums, frustum_colors=frustum_colors, camera_poses=camera_poses,
            pred_points=pred_points, pred_colors=pred_colors
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
        geometries = [complete_mesh]
        if SHOW_PC:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged_points)
            pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            geometries.insert(0, pcd)
        if all_pred_centroids and SHOW_PRED_CENTROID:
            pred_points = np.vstack(all_pred_centroids)
            pred_colors = np.vstack(all_pred_colors)
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(pred_points)
            pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
            geometries.append(pcd_pred)
        print(f"Added {len(mean_points)} mean points and {len(camera_frustums)} camera frustums")
        print(f"Color scheme: {num_images} colors from tab20 palette")
        print(f"Data saved to: {OUTPUT_DIR}")
    else:
        print("No points generated from any image")
