import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from sam2.features.utils import AsyncMultiWrapper

from f3rm.features.orientany.homography import Homography
from f3rm.features.orientany.orientany_main import OrientAny
from f3rm.features.utils import (
    BatchFeatureLoader,
    build_transform_lookup,
    get_nerf_ccs_to_normal_ccs_T,
    get_nerf_ccs_to_orig_nerf_world,
    get_orig_to_final_nerf_world_transform_scale,
    resolve_devices_and_workers,
    run_async_in_any_context,
)
from f3rm.manual.instance_axes_annotator import AxesAnnotator
from f3rm.shaders import VectorShader


class ORIENTANYArgs:
    min_instance_percent: float = 1.0
    batch_size_per_gpu: int = 8

    @classmethod
    def id_dict(cls):
        return {
            "min_instance_percent": float(cls.min_instance_percent),
        }


def _filter_masks_by_size(masks: np.ndarray, min_percent: float) -> List[np.ndarray]:
    """Filter masks by minimum size percentage."""
    if masks.ndim < 2:
        return []

    h, w = masks.shape[-2:]
    total_pixels = max(1, h * w)
    kept_masks = []

    for mask in masks:
        if mask.shape != (h, w) or not np.any(mask):
            continue
        percent = (np.sum(mask) / total_pixels) * 100.0
        if percent >= min_percent:
            kept_masks.append(mask)

    return kept_masks


def _get_camera_transforms(image_path: str, transforms_lookup: Dict, T_orig_to_final_nerf_world: np.ndarray,
                           orig_to_final_nerf_world_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get camera transforms for coordinate conversion."""
    nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(Path(image_path).name, transforms_lookup)
    nerf_ccs1_to_final_nerf_world = T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
    nerf_ccs1_to_final_nerf_world[:3, 3] *= orig_to_final_nerf_world_scale
    R_final_nerf_world_to_nerf_ccs1 = nerf_ccs1_to_final_nerf_world[:3, :3].T
    return nerf_ccs1_to_final_nerf_world, R_final_nerf_world_to_nerf_ccs1


def _compute_object_orientation_features(phi: float, theta_elev: float, delta: float,
                                         nerf_ccs1_to_final_nerf_world: np.ndarray) -> np.ndarray:
    """Compute object orientation features in final NeRF world coordinates."""
    # Get rotation from object to normal camera coordinate system
    R_objw_to_normal_ccs1 = OrientAny.get_R_objw2cam(phi, theta_elev, delta)

    # Transform to NeRF coordinate system
    R_objw_to_nerf_ccs1 = get_nerf_ccs_to_normal_ccs_T()[:3, :3].T @ R_objw_to_normal_ccs1

    # Transform to final NeRF world coordinates
    R_objw_to_final_nerf_world = nerf_ccs1_to_final_nerf_world[:3, :3] @ R_objw_to_nerf_ccs1

    # Create transform matrix and project reference points
    T_objw_to_final_nerf_world = OrientAny.get_T_from_R(R_objw_to_final_nerf_world)
    objw_pts = [
        np.array([0, 0, 0]),  # origin
        np.array([1, 0, 0]),  # +X
        np.array([0, 1, 0]),  # +Y (computed via cross product)
        np.array([0, 0, 1]),  # +Z
    ]
    final_nerf_world_pts = Homography.general_project_A_to_B(objw_pts, T_objw_to_final_nerf_world)

    # Extract and normalize orientation vectors
    u_x = final_nerf_world_pts[1] - final_nerf_world_pts[0]  # X-axis vector
    u_z = final_nerf_world_pts[3] - final_nerf_world_pts[0]  # Z-axis vector
    u_x = u_x / np.linalg.norm(u_x)
    u_z = u_z / np.linalg.norm(u_z)

    return np.concatenate([u_x, u_z], axis=0)


def _create_pixel_data(h: int, w: int, obj_masks: List[np.ndarray]) -> np.ndarray:
    """Create pixel data array with object masks and instance assignments."""
    pixel_data = np.zeros((h, w, 3), dtype=np.float16)

    # Create combined object mask (any object = True, background = False)
    if obj_masks:
        obj_mask_combined = np.logical_or.reduce(obj_masks)
        pixel_data[..., 1] = obj_mask_combined.astype(np.float16)  # object channel
        pixel_data[..., 0] = (~obj_mask_combined).astype(np.float16)  # background channel

        # Assign instance IDs to pixels
        next_instance_id = 1
        for mask in obj_masks:
            pixel_data[mask, 2] = float(next_instance_id)
            next_instance_id += 1
    else:
        # No objects - all background
        pixel_data[..., 0] = 1.0  # background channel

    return pixel_data


def parse_orientany_feature_type(feature_type: str) -> List[str]:
    if not feature_type.startswith("ORIENTANY_"):
        raise ValueError(f"Invalid ORIENTANY feature type: {feature_type}. Must start with 'ORIENTANY_'")
    prompts_part = feature_type[len("ORIENTANY_"):]
    if prompts_part == "":
        return []
    return [w.lower() for w in prompts_part.split("_") if w.strip()]


def _build_orientany_full_features(pixel_data: np.ndarray, instance_features: Dict[Any, Any]) -> np.ndarray:
    """Reconstruct dense ORIENTANY feature map from pixel ids and per-instance vectors."""
    h, w, _ = pixel_data.shape
    full_features = np.zeros((h, w, 9), dtype=np.float16)  # 7D features + 2D foreground
    full_features[..., 7:9] = pixel_data[..., :2]  # background/object one-hot

    for instance_id, instance_feat in instance_features.items():
        mask = pixel_data[..., 2] == int(instance_id)
        if np.any(mask) and isinstance(instance_feat, list):
            full_features[mask, :7] = np.asarray(instance_feat, dtype=np.float16)

    return full_features


def _draw_orientany_instance_axes(
    axes_frame: np.ndarray,
    pixel_data: np.ndarray,
    instance_features: Dict[Any, Any],
    R_final_nerf_world_to_nerf_ccs1: np.ndarray,
    axis_length: int = 70,
    axis_thickness: int = 4,
) -> np.ndarray:
    """Draw per-instance axes from ORIENTANY instance features."""
    for instance_id, instance_feat in instance_features.items():
        instance_mask = pixel_data[..., 2] == int(instance_id)
        if not np.any(instance_mask):
            continue

        ys, xs = np.where(instance_mask)
        instance_center = (int(xs.mean()), int(ys.mean()))

        instance_feat_arr = np.asarray(instance_feat, dtype=np.float32)
        u_x_world = instance_feat_arr[:3]
        u_z_world = instance_feat_arr[3:6]
        if np.linalg.norm(u_x_world) < 1e-6 or np.linalg.norm(u_z_world) < 1e-6:
            continue

        u_y_world = np.cross(u_z_world, u_x_world)
        if np.linalg.norm(u_y_world) < 1e-6:
            continue
        u_y_world = u_y_world / np.linalg.norm(u_y_world)
        R_objw_to_final_nerf_world = np.column_stack([u_x_world, u_y_world, u_z_world])
        R_objw_to_nerf_ccs1 = R_final_nerf_world_to_nerf_ccs1 @ R_objw_to_final_nerf_world

        axes_frame = AxesAnnotator.visualize_rotation_matrix(
            axes_frame,
            instance_center,
            R_objw_to_nerf_ccs1,
            axis_length=axis_length,
            axis_thickness=axis_thickness,
        )

    return axes_frame


class ORIENTANYWorker:
    def __init__(self, device: torch.device, data_dir: Path, sam3_feature_type: str):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.data_dir = Path(data_dir)
        self.sam3_feature_type = sam3_feature_type

        # Load SAM3 metadata and create loader
        sam3_root = self.data_dir / "features" / self.sam3_feature_type.lower()
        if not (sam3_root / "meta.pt").exists():
            raise FileNotFoundError(f"Missing SAM3 meta: {sam3_root / 'meta.pt'}")

        sam3_meta = torch.load(sam3_root / "meta.pt")
        self.feat_image_fnames = [str(p) for p in sam3_meta["image_fnames"]]
        self.sam3_loader = BatchFeatureLoader(self.data_dir, self.sam3_feature_type, self.feat_image_fnames, self.device)

        # Initialize orientation estimation model
        self.orient_any = OrientAny("f3rm/features/orientany/ckpts", "ronormsigma1_dino_weight.pt", device=self.device)

        # Load camera transforms for coordinate system conversion
        transforms_path = self.data_dir / "transforms.json"
        if transforms_path.exists():
            T_orig_to_final_nerf_world, orig_to_final_nerf_world_scale = get_orig_to_final_nerf_world_transform_scale(str(transforms_path))
            dataset_transforms_data = json.load(open(transforms_path, "r"))
            self.transforms_lookup = build_transform_lookup(dataset_transforms_data["frames"])
            self.T_orig_to_final_nerf_world = T_orig_to_final_nerf_world
            self.orig_to_final_nerf_world_scale = orig_to_final_nerf_world_scale
        else:
            raise ValueError("transforms.json not found")

    async def compute_orientany_for_image_async(self, image_path: str, debug: bool = False) -> Dict[str, Any]:
        try:
            idx = self.feat_image_fnames.index(str(image_path))
        except ValueError:
            raise ValueError(f"Image path not found in SAM3 meta order: {image_path}")

        # Load and standardize SAM3 masks
        masks = np.asarray(self.sam3_loader[idx])
        if masks.ndim == 4:
            masks = masks[:, 0, ...]  # Remove channel dimension if present
        elif masks.ndim == 2:
            masks = masks[None, ...]  # Add batch dimension if missing
        masks = masks.astype(bool)

        h, w = masks.shape[-2:]
        obj_masks = _filter_masks_by_size(masks, ORIENTANYArgs.min_instance_percent)

        if not obj_masks:
            # No valid objects found
            pixel_data = _create_pixel_data(h, w, [])
            return {"pixel_data": pixel_data, "instance_features": {}}

        # Process each detected object
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # Get camera coordinate transforms for this image
        nerf_ccs1_to_final_nerf_world, _ = _get_camera_transforms(
            image_path, self.transforms_lookup, self.T_orig_to_final_nerf_world, self.orig_to_final_nerf_world_scale
        )

        instance_features = {}
        for mask in obj_masks:
            # Create instance image with mask
            instance_img_array = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
            instance_img_array[..., :3] = img_array * mask[..., None]
            instance_img_array[..., 3] = mask * 255
            instance_img = Image.fromarray(instance_img_array, "RGBA")

            # Estimate object orientation
            rm_bkg_img = self.orient_any.preprocess_remove_bkg(instance_img, do_remove_background=False)
            outs = self.orient_any.get_model_outputs(rm_bkg_img, viz_distn=False)

            # Compute orientation features in final NeRF world coordinates
            orientation_vectors = _compute_object_orientation_features(
                outs["phi"], outs["theta_elev"], outs["delta"], nerf_ccs1_to_final_nerf_world
            )
            confidence = float(outs["confidence"])
            instance_feat = np.concatenate([orientation_vectors, [confidence]], axis=0).astype(np.float16)

            instance_features[len(instance_features) + 1] = instance_feat.tolist()

            # Clean up intermediate objects
            del instance_img, rm_bkg_img, outs

        # Create final pixel data with object masks and instance assignments
        pixel_data = _create_pixel_data(h, w, obj_masks)

        return {"pixel_data": pixel_data, "instance_features": instance_features}


class ORIENTANYExtractor:
    def __init__(
        self,
        device: torch.device,
        data_dir: Optional[Path] = None,
        text_prompts: Optional[List[str]] = None,
        sam3_feature_type: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.verbose = verbose
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.text_prompts = text_prompts
        self.sam3_feature_type = sam3_feature_type or self._infer_sam3_feature_type(text_prompts)

        if self.data_dir is None:
            raise ValueError("ORIENTANYExtractor requires data_dir to locate precomputed SAM3 shards")

        # Validate prerequisites for precomputed SAM3 masks
        sam3_root = self.data_dir / "features" / self.sam3_feature_type.lower()
        if not (sam3_root / "meta.pt").exists():
            raise FileNotFoundError(
                f"Missing SAM3 meta: {sam3_root / 'meta.pt'} (expected for ORIENTANY). "
                f"Run SAM3 extraction for feature type '{self.sam3_feature_type}' first."
            )
        if not list(sam3_root.glob("image_*.npz")):
            raise FileNotFoundError(f"Missing SAM3 per-image features under {sam3_root}")

        devices_param, num_workers = resolve_devices_and_workers(device, ORIENTANYArgs.batch_size_per_gpu)
        if verbose:
            print(f"Initializing ORIENTANY workers (using {self.sam3_feature_type} masks)")
        self.client = AsyncMultiWrapper(
            ORIENTANYWorker,
            num_objects=num_workers,
            devices=devices_param,
            data_dir=self.data_dir,
            sam3_feature_type=self.sam3_feature_type,
        )
        self.num_workers = num_workers

    @staticmethod
    def _infer_sam3_feature_type(text_prompts: Optional[List[str]]) -> str:
        if text_prompts is None or len(text_prompts) == 0:
            return "SAM3_"
        joined = "_".join([p.lower() for p in text_prompts])
        return f"SAM3_{joined}"

    async def extract_batch_async(self, image_paths: List[str], debug: bool = False) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting ORIENTANY features", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_orientany_async(path, self.client, debug=debug) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="ORIENTANY", leave=False)
            results.extend(batch_results)
            gc.collect()
        return results


async def process_single_image_orientany_async(image_path: str, orientany_client: AsyncMultiWrapper, debug: bool = False) -> Dict[str, Any]:
    return await orientany_client.compute_orientany_for_image_async(image_path, debug=debug)


def examine_saved(orientany_feat_dir: str):
    """Create .mp4 video of saved ORIENTANY features with side-by-side visualization."""
    meta_path = os.path.join(orientany_feat_dir, "meta.pt")
    assert os.path.exists(meta_path), f"ORIENTANY meta not found at {meta_path}"

    meta = torch.load(meta_path)
    image_fnames = meta["image_fnames"]
    n_images = len(image_fnames)

    # Get data directory from feature directory
    data_dir = Path(orientany_feat_dir).parent.parent  # features/orientany_ -> data_dir

    # Load transforms for coordinate conversion (same as in debug script)
    transforms_path = data_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found at {transforms_path}")

    T_orig_to_final_nerf_world, scale = get_orig_to_final_nerf_world_transform_scale(str(transforms_path))
    dataset_transforms_data = json.load(open(transforms_path, "r"))
    transforms_lookup = build_transform_lookup(dataset_transforms_data["frames"])

    # Load first image to get dimensions
    first_pixel_path = os.path.join(orientany_feat_dir, "image_000000_pixel.npy")
    first_inst_path = os.path.join(orientany_feat_dir, "image_000000_instances.json")
    if not (os.path.exists(first_pixel_path) and os.path.exists(first_inst_path)):
        raise FileNotFoundError(f"ORIENTANY feature files not found in {orientany_feat_dir}")

    pixel_data = np.load(first_pixel_path)
    H, W = pixel_data.shape[:2]

    video_path = os.path.join(orientany_feat_dir, "features_viz.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Side-by-side: axes + rx + rz = 3 * width
    out = cv2.VideoWriter(video_path, fourcc, 2.0, (W * 3, H))
    vector_shader = VectorShader()

    for i in tqdm(range(n_images), desc="Creating ORIENTANY features video"):
        # Load pixel data and instance features
        pixel_path = os.path.join(orientany_feat_dir, f"image_{i:06d}_pixel.npy")
        inst_path = os.path.join(orientany_feat_dir, f"image_{i:06d}_instances.json")

        if not (os.path.exists(pixel_path) and os.path.exists(inst_path)):
            continue

        pixel_data = np.load(pixel_path)
        with open(inst_path, 'r') as f:
            instance_features = json.load(f)

        # Get original image
        image_path = image_fnames[i]
        if not os.path.exists(image_path):
            continue
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # Get object mask (any detected object pixels)
        obj_mask = pixel_data[..., 1] > 0.5

        if not np.any(obj_mask):
            # No objects detected, create empty frame
            empty_frame = np.zeros((H, W * 3, 3), dtype=np.uint8)
            frame_bgr = cv2.cvtColor(empty_frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            continue

        # Reconstruct full feature map for visualization
        full_features = _build_orientany_full_features(pixel_data, instance_features)

        # Extract orientation vectors
        R_x = full_features[..., :3]  # X-axis orientation vectors
        R_z = full_features[..., 3:6]  # Z-axis orientation vectors

        # Create three visualizations
        # 1. Axes visualization (per instance)
        axes_frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR once at start

        # Get camera transform for this image (same for all instances)
        nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(Path(image_path).name, transforms_lookup)
        nerf_ccs1_to_final_nerf_world = T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
        nerf_ccs1_to_final_nerf_world[:3, 3] *= scale
        R_final_nerf_world_to_nerf_ccs1 = nerf_ccs1_to_final_nerf_world[:3, :3].T

        axes_frame = _draw_orientany_instance_axes(
            axes_frame=axes_frame,
            pixel_data=pixel_data,
            instance_features=instance_features,
            R_final_nerf_world_to_nerf_ccs1=R_final_nerf_world_to_nerf_ccs1,
            axis_length=70,
            axis_thickness=4,
        )

        axes_frame = cv2.cvtColor(axes_frame, cv2.COLOR_BGR2RGB)  # Convert back to RGB once at end

        # 2. X-axis orientation vector visualization
        R_x_tensor = torch.from_numpy(R_x).float()
        obj_tensor = torch.from_numpy(obj_mask).float().unsqueeze(-1)
        rx_rgb = vector_shader(R_x_tensor, valid_mask=obj_tensor)
        rx_rgb = (rx_rgb * 255).clamp(0, 255).byte().numpy()

        # 3. Z-axis orientation vector visualization
        R_z_tensor = torch.from_numpy(R_z).float()
        rz_rgb = vector_shader(R_z_tensor, valid_mask=obj_tensor)
        rz_rgb = (rz_rgb * 255).clamp(0, 255).byte().numpy()

        # Stack frames side by side
        side_by_side = np.hstack([axes_frame, rx_rgb, rz_rgb])

        # Convert to BGR for video
        frame_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    assert os.path.exists(video_path), f"Video not created at {video_path}"


if __name__ == "__main__":
    # examine_saved("datasets/f3rm/opt/objaverse/car2/features/orientany_")

    data_root = Path("datasets/f3rm/opt/objaverse/car2")
    image_dir = data_root / "images"
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    image_paths = [str(p) for p in image_paths[10:13]]  # Just 3 for demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract orientation features
    extractor = ORIENTANYExtractor(device=device, data_dir=data_root, text_prompts=None, verbose=True)
    features_data = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths, debug=True))
    print(f"Extracted {len(features_data)} feature maps")

    # Visualize results with instance-based axes
    vis_count = min(1, len(image_paths))
    fig, axes = plt.subplots(vis_count, 3, figsize=(18, 4 * vis_count))

    # Load transforms for coordinate conversion
    transforms_path = data_root / "transforms.json"
    T_orig_to_final_nerf_world, scale = get_orig_to_final_nerf_world_transform_scale(str(transforms_path))
    dataset_transforms_data = json.load(open(transforms_path, "r"))
    transforms_lookup = build_transform_lookup(dataset_transforms_data["frames"])
    vector_shader = VectorShader()

    for i in tqdm(range(vis_count), desc="Visualizing results"):
        data = features_data[i]
        pixel_data = data['pixel_data']
        instance_features = data['instance_features']

        # RGB image
        rgb = Image.open(image_paths[i]).convert("RGB")
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB {i+1}")
        axes[i, 0].axis('off')

        # Object mask
        obj_mask = pixel_data[..., 1]
        axes[i, 1].imshow(obj_mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Object Mask")
        axes[i, 1].axis('off')

        # Orientation RGB with instance axes
        if instance_features:
            full_features = _build_orientany_full_features(pixel_data, instance_features)

            # Get X-axis orientation vector for visualization
            R_x = full_features[..., :3]
            R_x_tensor = torch.from_numpy(R_x).float()
            obj_tensor = torch.from_numpy(obj_mask).float().unsqueeze(-1)
            orient_rgb = vector_shader(R_x_tensor, valid_mask=obj_tensor)
            orient_rgb = (orient_rgb * 255).clamp(0, 255).byte().numpy()

            # Draw instance axes on orientation visualization
            orient_bgr = cv2.cvtColor(orient_rgb, cv2.COLOR_RGB2BGR)

            nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(Path(image_paths[i]).name, transforms_lookup)
            nerf_ccs1_to_final_nerf_world = T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
            nerf_ccs1_to_final_nerf_world[:3, 3] *= scale
            R_final_nerf_world_to_nerf_ccs1 = nerf_ccs1_to_final_nerf_world[:3, :3].T

            orient_bgr = _draw_orientany_instance_axes(
                axes_frame=orient_bgr,
                pixel_data=pixel_data,
                instance_features=instance_features,
                R_final_nerf_world_to_nerf_ccs1=R_final_nerf_world_to_nerf_ccs1,
                axis_length=70,
                axis_thickness=4,
            )

            # Convert back to RGB for display
            orient_rgb = cv2.cvtColor(orient_bgr, cv2.COLOR_BGR2RGB)

            axes[i, 2].imshow(orient_rgb)
            axes[i, 2].set_title("Orientation RGB + Instance Axes")
        else:
            axes[i, 2].text(0.5, 0.5, 'No instances', ha='center', va='center', transform=axes[i, 2].transAxes)
            axes[i, 2].set_title("No Instances Found")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
