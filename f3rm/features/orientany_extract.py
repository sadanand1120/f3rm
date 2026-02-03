import gc
import asyncio
import json
import os
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sam2.features.utils import SAM2utils
from sam2.features.clip_main import CLIPfeatures
from sam2.features.utils import AsyncMultiWrapper

from f3rm.features.utils import (
    run_async_in_any_context,
    resolve_devices_and_workers,
    get_nerf_ccs_to_normal_ccs_T,
    build_transform_lookup,
    get_nerf_ccs_to_orig_nerf_world,
    get_orig_to_final_nerf_world_transform_scale
)
from f3rm.features.utils import BatchFeatureLoader
from f3rm.features.sam2_extract import SAM2Args
from f3rm.features.orientany.orientany_main import OrientAny
from f3rm.features.orientany.homography import Homography
from f3rm.shaders import VectorShader
from f3rm.manual.instance_axes_annotator import AxesAnnotator


class ORIENTANYArgs:
    negative_texts: List[str] = ["object", "floor", "wall"]
    softmax_temp: float = 0.01
    top_mean_percent: float = 15.0
    sim_thresh: float = 0.7   # 0.7, make it 0.1 for debugging
    min_instance_percent: float = 1.0
    batch_size_per_gpu: int = 4

    @classmethod
    def id_dict(cls):
        return {
            "negative_texts": list(cls.negative_texts),
            "softmax_temp": float(cls.softmax_temp),
            "top_mean_percent": float(cls.top_mean_percent),
            "sim_thresh": float(cls.sim_thresh),
            "min_instance_percent": float(cls.min_instance_percent),
        }


def parse_orientany_feature_type(feature_type: str) -> List[str]:
    if not feature_type.startswith("ORIENTANY_"):
        raise ValueError(f"Invalid ORIENTANY feature type: {feature_type}. Must start with 'ORIENTANY_'")
    prompts_part = feature_type[len("ORIENTANY_"):]
    if prompts_part == "":
        return []
    return [w.lower() for w in prompts_part.split("_") if w.strip()]


class ORIENTANYWorker:
    def __init__(self, device: torch.device, data_dir: Path, text_prompts: Optional[List[str]] = None):
        self.device = device
        self.data_dir = Path(data_dir)
        self.text_prompts = text_prompts

        feat_root = self.data_dir / "features"
        clip_root = feat_root / "clip"
        sam2_root = feat_root / "sam2"
        text_root = feat_root / "text"

        # Load meta to align indices
        clip_meta = torch.load(clip_root / "meta.pt")
        self.feat_image_fnames = [str(p) for p in clip_meta["image_fnames"]]

        # Create batch feature loaders for the new system
        self.clip_loader = BatchFeatureLoader(self.data_dir, "CLIP", self.feat_image_fnames, device)
        self.sam2_loader = BatchFeatureLoader(self.data_dir, "SAM2", self.feat_image_fnames, device)

        if self.text_prompts is None:
            self.text_loader = BatchFeatureLoader(self.data_dir, "TEXT", self.feat_image_fnames, device)
        else:
            self.text_loader = None

        self.clip_model = CLIPfeatures(device=self.device)
        self.orient_any = OrientAny("f3rm/features/orientany/ckpts", "ronormsigma1_dino_weight.pt")

        # Load camera transforms for distribution propagation
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
        # map image → index
        try:
            idx = self.feat_image_fnames.index(str(image_path))
        except ValueError:
            raise ValueError(f"Image path not found in CLIP meta order: {image_path}")

        clip_patch_feats = self.clip_loader[idx]
        raw_auto_masks = self.sam2_loader[idx]

        # prompts
        if self.text_prompts is None:
            text_prompts = self.text_loader[idx]
        else:
            text_prompts = self.text_prompts

        if not raw_auto_masks or not text_prompts:
            # Return all background with zero orientation features
            h, w = (raw_auto_masks[0]['segmentation'].shape if raw_auto_masks else (Image.open(image_path).size[1], Image.open(image_path).size[0]))
            fg = np.zeros((h, w), dtype=bool)
            # Return new format: pixel_data + empty instance_features
            pixel_data = np.zeros((h, w, 3), dtype=np.float16)  # Use fp16 for VRAM efficiency
            pixel_data[..., :2] = np.stack([~fg, fg], axis=-1).astype(np.float16)  # foreground one-hot
            return {
                'pixel_data': pixel_data,
                'instance_features': {}  # No instances
            }

        # Instance mask from SAM2
        inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
            raw_auto_masks,
            min_iou=float(SAM2Args.pred_iou_thresh) if SAM2Args.pred_iou_thresh is not None else 0.0,
            min_area=float(SAM2Args.min_mask_region_area) if SAM2Args.min_mask_region_area is not None else 0.0,
            assign_by="area",
            start_from="low",
        )
        if inst_mask is None:
            if raw_auto_masks:
                h, w = raw_auto_masks[0]['segmentation'].shape
            else:
                img = Image.open(image_path)
                h, w = img.height, img.width
            inst_mask = np.zeros((h, w), dtype=np.uint16)

        # Remove tiny instances
        unique_ids = np.unique(inst_mask)
        total_pixels = inst_mask.size
        for inst_id in unique_ids:
            if inst_id > 0:
                seg = (inst_mask == inst_id)
                percent = (np.sum(seg) / total_pixels) * 100.0
                if percent < ORIENTANYArgs.min_instance_percent:
                    inst_mask[seg] = 0

        # Build auto_masks list back
        auto_masks = []
        for inst_id in np.unique(inst_mask):
            if inst_id <= 0:
                continue
            seg = (inst_mask == inst_id)
            if not np.any(seg):
                continue
            ys, xs = np.where(seg)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
            auto_masks.append({
                "segmentation": seg,
                "bbox": bbox,
                "predicted_iou": np.float16("inf"),  # Use fp16 for VRAM efficiency
                "area": np.float16("inf"),           # Use fp16 for VRAM efficiency
            })

        if not auto_masks:
            h, w = inst_mask.shape
            fg = np.zeros((h, w), dtype=bool)
            pixel_data = np.zeros((h, w, 3), dtype=np.float16)  # Use fp16 for VRAM efficiency
            pixel_data[..., :2] = np.stack([~fg, fg], axis=-1).astype(np.float16)  # foreground one-hot
            return {
                'pixel_data': pixel_data,
                'instance_features': {}  # No instances
            }

        h, w = auto_masks[0]["segmentation"].shape

        # Per-prompt sim maps and combine
        segment_sim_maps: List[np.ndarray] = []
        for text in text_prompts:
            text_emb = self.clip_model.encode_text(text).half()
            neg_text_embs = torch.stack([self.clip_model.encode_text(neg).half() for neg in ORIENTANYArgs.negative_texts], dim=0)
            sim_map = self.clip_model.compute_similarity(
                clip_patch_feats,
                text_emb,
                neg_text_embs=neg_text_embs,
                softmax_temp=ORIENTANYArgs.softmax_temp,
                normalize=True,
            )
            sim_map_up = np.array(Image.fromarray(sim_map.cpu().float().numpy()).resize((w, h), Image.BILINEAR)).astype(np.float16)

            seg_map = np.zeros_like(sim_map_up)
            for m in auto_masks:
                seg = m["segmentation"]
                if seg.shape != (h, w):
                    continue
                vals = sim_map_up[seg]
                k = max(1, int(len(vals) * ORIENTANYArgs.top_mean_percent / 100.0))
                seg_map[seg] = float(np.mean(np.sort(vals)[-k:]))
            segment_sim_maps.append(seg_map)

        combined_sim = np.maximum.reduce(segment_sim_maps) if len(segment_sim_maps) > 0 else np.zeros((h, w))

        # Foreground map: any pixel belonging to any kept mask (threshold on combined similarity)
        fg_mask = np.zeros((h, w), dtype=bool)
        for m in auto_masks:
            seg = m["segmentation"]
            if seg.shape != (h, w):
                continue
            if np.any(combined_sim[seg] > ORIENTANYArgs.sim_thresh):
                fg_mask |= seg

        # Initialize per-pixel data: (H, W, 3) - [fg_one_hot, instance_id]
        pixel_data = np.zeros((h, w, 3), dtype=np.float16)  # Use fp16 for VRAM efficiency
        pixel_data[..., :2] = np.stack([~fg_mask, fg_mask], axis=-1).astype(np.float16)  # foreground one-hot

        # Initialize instance features mapping
        instance_features = {}
        next_instance_id = 1

        # For each foreground instance, compute orientation features
        for m in auto_masks:
            seg = m["segmentation"]
            if seg.shape != (h, w):
                continue
            if not np.any(combined_sim[seg] > ORIENTANYArgs.sim_thresh):
                continue

            # Create instance image for OrientAny
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            instance_img_array = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
            instance_img_array[..., :3] = img_array * seg[..., None]
            instance_img_array[..., 3] = seg * 255
            instance_img = Image.fromarray(instance_img_array, 'RGBA')

            # Get OrientAny predictions for cam1 (current image camera)
            rm_bkg_img = self.orient_any.preprocess_remove_bkg(instance_img, do_remove_background=False)
            outs = self.orient_any.get_model_outputs(rm_bkg_img, viz_distn=False)

            # Extract logits
            gaus_ax_logits = torch.from_numpy(outs['gaus_ax_logits']).to(self.device)  # 360D
            gaus_pl_logits = torch.from_numpy(outs['gaus_pl_logits']).to(self.device)  # 180D
            gaus_ro_logits = torch.from_numpy(outs['gaus_ro_logits']).to(self.device)  # 360D
            conf_logits = torch.from_numpy(outs['conf_logits']).to(self.device)

            # Get rotation matrix using argmax (no distribution propagation)
            ax_pred = torch.argmax(gaus_ax_logits).item()
            pl_pred = torch.argmax(gaus_pl_logits).item()
            ro_pred = torch.argmax(gaus_ro_logits).item()

            # Convert to OrientAny angles
            phi = float(ax_pred)
            theta_elev = float(pl_pred) - 90.0
            delta = float(ro_pred) - self.orient_any.model_config['ro_offset']

            # Get R_objw_to_cam1 (camera 1 coordinate system)
            R_objw_to_normal_ccs1 = self.orient_any.get_R_objw2cam(phi, theta_elev, delta)
            R_objw_to_nerf_ccs1 = get_nerf_ccs_to_normal_ccs_T()[:3, :3].T @ R_objw_to_normal_ccs1

            # Get cam1 to world transforms (all in NeRF CCS)
            nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(os.path.basename(image_path), self.transforms_lookup)
            nerf_ccs1_to_final_nerf_world = self.T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
            nerf_ccs1_to_final_nerf_world[:3, 3] *= self.orig_to_final_nerf_world_scale

            R_objw_to_final_nerf_world = nerf_ccs1_to_final_nerf_world[:3, :3] @ R_objw_to_nerf_ccs1
            # dont care about translation, so assume at final nerf world origin
            T_objw_to_final_nerf_world = self.orient_any.get_T_from_R(R_objw_to_final_nerf_world)
            objw_pts = [
                np.array([0, 0, 0]),
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1])
            ]
            final_nerf_world_pts = Homography.general_project_A_to_B(objw_pts, T_objw_to_final_nerf_world)
            # Store u_x and u_z vectors plus confidence (7D vector)
            u_x = final_nerf_world_pts[1] - final_nerf_world_pts[0]
            u_z = final_nerf_world_pts[3] - final_nerf_world_pts[0]
            u_x = u_x / np.linalg.norm(u_x)
            u_z = u_z / np.linalg.norm(u_z)
            conf = outs['confidence']  # Single confidence value
            instance_feat = np.concatenate([u_x, u_z, [conf]], axis=0).astype(np.float16)

            instance_features[next_instance_id] = instance_feat.tolist()  # Convert to list for JSON serialization

            # Assign instance ID to foreground pixels
            pixel_data[seg, 2] = float(next_instance_id)
            next_instance_id += 1

            # Clean up instance-specific memory
            del instance_img, rm_bkg_img, outs, gaus_ax_logits, gaus_pl_logits, gaus_ro_logits, conf_logits

        result = {
            'pixel_data': pixel_data,  # (H, W, 3) - [fg_one_hot, instance_id]
            'instance_features': instance_features  # {instance_id: 7D_R_x_R_z_confidence}
        }

        return result


class ORIENTANYExtractor:
    def __init__(self, device: torch.device, data_dir: Optional[Path] = None, text_prompts: Optional[List[str]] = None, verbose: bool = False) -> None:
        self.device = device
        self.verbose = verbose
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.text_prompts = text_prompts

        if self.data_dir is None:
            raise ValueError("ORIENTANYExtractor requires data_dir to locate precomputed CLIP and SAM2 shards")

        # Validate prerequisites for new per-image system
        feat_root = self.data_dir / "features"
        clip_root = feat_root / "clip"
        sam2_root = feat_root / "sam2"
        text_root = feat_root / "text"

        if not (clip_root / "meta.pt").exists():
            raise FileNotFoundError(f"Missing CLIP meta: {clip_root / 'meta.pt'}")
        if not list(clip_root.glob("image_*.npy")):
            raise FileNotFoundError(f"Missing CLIP per-image features under {clip_root}")
        if not list(sam2_root.glob("image_*.npz")):
            raise FileNotFoundError(f"Missing SAM2 per-image features under {sam2_root}")
        if text_prompts is None and not list(text_root.glob("image_*.json")):
            raise FileNotFoundError(f"Missing TEXT per-image features under {text_root} (required when using per-image text prompts)")

        devices_param, num_workers = resolve_devices_and_workers(device, ORIENTANYArgs.batch_size_per_gpu)
        if verbose:
            print("Initializing ORIENTANY workers")
        self.client = AsyncMultiWrapper(ORIENTANYWorker, num_objects=num_workers, devices=devices_param, data_dir=self.data_dir, text_prompts=self.text_prompts)
        self.num_workers = num_workers

    async def extract_batch_async(self, image_paths: List[str], debug: bool = False) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting ORIENTANY features", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_orientany_async(path, self.client, debug=debug) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="ORIENTANY", leave=False)
            results.extend(batch_results)
            gc.collect()
        return results


async def extract_orientany_batch(image_paths: List[str], device: torch.device, data_dir: Path, verbose: bool = False, text_prompts: Optional[List[str]] = None, debug: bool = False):
    extractor = ORIENTANYExtractor(device=device, data_dir=data_dir, text_prompts=text_prompts, verbose=verbose)
    return await extractor.extract_batch_async(image_paths, debug=debug)


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

        # Get foreground mask
        fg_mask = pixel_data[..., 1] > 0.5

        if not np.any(fg_mask):
            # No foreground, create empty frame
            empty_frame = np.zeros((H, W * 3, 3), dtype=np.uint8)
            frame_bgr = cv2.cvtColor(empty_frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            continue

        # Reconstruct full features for visualization
        full_features = np.zeros((H, W, 9), dtype=np.float16)
        full_features[..., 7:9] = pixel_data[..., :2]  # foreground one-hot

        # Assign instance features to pixels
        for instance_id, instance_feat in instance_features.items():
            instance_id = int(instance_id)
            mask = (pixel_data[..., 2] == instance_id)
            if np.any(mask) and isinstance(instance_feat, list):
                instance_feat = np.array(instance_feat, dtype=np.float16)
                full_features[mask, :7] = instance_feat

        # Extract R_x and R_z vectors
        R_x = full_features[..., :3]
        R_z = full_features[..., 3:6]

        # Create three visualizations
        # 1. Axes visualization (per instance)
        axes_frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR once at start

        # Get camera transform for this image (same for all instances)
        nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(Path(image_path).name, transforms_lookup)
        nerf_ccs1_to_final_nerf_world = T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
        nerf_ccs1_to_final_nerf_world[:3, 3] *= scale
        R_final_nerf_world_to_nerf_ccs1 = nerf_ccs1_to_final_nerf_world[:3, :3].T

        # Draw axes for each instance
        for instance_id, instance_feat in instance_features.items():
            instance_id = int(instance_id)
            if isinstance(instance_feat, list):
                instance_feat = np.array(instance_feat, dtype=np.float16)

            # Get instance mask
            instance_mask = (pixel_data[..., 2] == instance_id)
            if not np.any(instance_mask):
                continue

            # Calculate instance center
            ys, xs = np.where(instance_mask)
            instance_center = (int(xs.mean()), int(ys.mean()))

            # Stored features are in final NeRF world coordinates
            u_x_world = instance_feat[:3]
            u_z_world = instance_feat[3:6]
            u_y_world = np.cross(u_z_world, u_x_world)
            u_y_world = u_y_world / np.linalg.norm(u_y_world)
            R_objw_to_final_nerf_world = np.column_stack([u_x_world, u_y_world, u_z_world])

            # Transform to camera coordinate system
            R_objw_to_nerf_ccs1 = R_final_nerf_world_to_nerf_ccs1 @ R_objw_to_final_nerf_world

            # Draw axes for this instance (axes_frame is already in BGR format)
            axes_frame = AxesAnnotator.visualize_rotation_matrix(
                axes_frame,  # Already BGR, no conversion needed
                instance_center,
                R_objw_to_nerf_ccs1,
                axis_length=70,  # Smaller axes to avoid crowding
                axis_thickness=4
            )

        axes_frame = cv2.cvtColor(axes_frame, cv2.COLOR_BGR2RGB)  # Convert back to RGB once at end

        # 2. R_x vector visualization
        R_x_tensor = torch.from_numpy(R_x).float()
        fg_tensor = torch.from_numpy(fg_mask).float().unsqueeze(-1)
        rx_rgb = vector_shader(R_x_tensor, valid_mask=fg_tensor)
        rx_rgb = (rx_rgb * 255).clamp(0, 255).byte().numpy()

        # 3. R_z vector visualization
        R_z_tensor = torch.from_numpy(R_z).float()
        rz_rgb = vector_shader(R_z_tensor, valid_mask=fg_tensor)
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
    image_paths = [str(p) for p in image_paths[:3]]  # Just 3 for demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract orientation features using global prompts (with debug info)
    extractor = ORIENTANYExtractor(device=device, data_dir=data_root, text_prompts=None, verbose=True)
    features_data = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths, debug=True))
    print(f"Extracted {len(features_data)} feature maps")

    # Visualize results with instance-based axes
    vis_count = min(2, len(image_paths))
    fig, axes = plt.subplots(3, vis_count, figsize=(6 * vis_count, 12))
    if vis_count == 1:
        axes = axes.reshape(3, 1)

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
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"RGB {i+1}")
        axes[0, i].axis('off')

        # Foreground mask
        fg = pixel_data[..., 1]
        axes[1, i].imshow(fg, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title("Foreground Mask")
        axes[1, i].axis('off')

        # Orientation RGB with instance axes
        if instance_features:
            # Reconstruct full features for visualization
            h, w, _ = pixel_data.shape
            full_features = np.zeros((h, w, 9), dtype=np.float16)
            full_features[..., 7:9] = pixel_data[..., :2]  # foreground one-hot

            for instance_id, instance_feat in instance_features.items():
                instance_id = int(instance_id)
                mask = (pixel_data[..., 2] == instance_id)
                if np.any(mask) and isinstance(instance_feat, list):
                    instance_feat = np.array(instance_feat, dtype=np.float16)
                    full_features[mask, :7] = instance_feat

            # Get R_x vector for visualization
            R_x = full_features[..., :3]
            R_x_tensor = torch.from_numpy(R_x).float()
            fg_tensor = torch.from_numpy(fg).float().unsqueeze(-1)
            orient_rgb = vector_shader(R_x_tensor, valid_mask=fg_tensor)
            orient_rgb = (orient_rgb * 255).clamp(0, 255).byte().numpy()

            # Draw instance axes on orientation visualization
            # Convert orient_rgb to BGR for axes drawing
            orient_bgr = cv2.cvtColor(orient_rgb, cv2.COLOR_RGB2BGR)

            nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(Path(image_paths[i]).name, transforms_lookup)
            nerf_ccs1_to_final_nerf_world = T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
            nerf_ccs1_to_final_nerf_world[:3, 3] *= scale
            R_final_nerf_world_to_nerf_ccs1 = nerf_ccs1_to_final_nerf_world[:3, :3].T

            for instance_id, instance_feat in instance_features.items():
                instance_id = int(instance_id)
                if isinstance(instance_feat, list):
                    instance_feat = np.array(instance_feat, dtype=np.float16)

                instance_mask = (pixel_data[..., 2] == instance_id)
                if not np.any(instance_mask):
                    continue

                ys, xs = np.where(instance_mask)
                instance_center = (int(xs.mean()), int(ys.mean()))

                u_x_world = instance_feat[:3]
                u_z_world = instance_feat[3:6]
                u_y_world = np.cross(u_z_world, u_x_world)
                u_y_world = u_y_world / np.linalg.norm(u_y_world)
                R_objw_to_final_nerf_world = np.column_stack([u_x_world, u_y_world, u_z_world])
                R_objw_to_nerf_ccs1 = R_final_nerf_world_to_nerf_ccs1 @ R_objw_to_final_nerf_world

                # Draw axes in BGR space
                orient_bgr = AxesAnnotator.visualize_rotation_matrix(
                    orient_bgr,
                    instance_center,
                    R_objw_to_nerf_ccs1,
                    axis_length=70,  # Even smaller for visualization
                    axis_thickness=4
                )

            # Convert back to RGB for display
            orient_rgb = cv2.cvtColor(orient_bgr, cv2.COLOR_BGR2RGB)

            axes[2, i].imshow(orient_rgb)
            axes[2, i].set_title("Orientation RGB + Instance Axes")
        else:
            axes[2, i].text(0.5, 0.5, 'No instances', ha='center', va='center', transform=axes[2, i].transAxes)
            axes[2, i].set_title("No Instances Found")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
