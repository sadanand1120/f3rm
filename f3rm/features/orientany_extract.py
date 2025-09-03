import gc
import asyncio
import json
import os
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sam2.features.utils import SAM2utils
from sam2.features.clip_main import CLIPfeatures
from sam2.features.utils import AsyncMultiWrapper

from f3rm.features.utils import (
    LazyFeatures, SAM2LazyAutoMasks, TextLazyFeatures,
    run_async_in_any_context,
    resolve_devices_and_workers,
    get_nerf_ccs_to_normal_ccs_T,
    build_transform_lookup,
    get_nerf_ccs_to_orig_nerf_world,
    get_orig_to_final_nerf_world_transform_scale,
    get_conf_temp_scaled_logits,
    probs_to_von_mises,
    von_mises_to_probs,
    probs_to_normal,
    normal_to_probs
)
from f3rm.features.sam2_extract import SAM2Args
from f3rm.features.orientany.orientany_main import OrientAny


class ORIENTANYArgs:
    negative_texts: List[str] = ["object", "floor", "wall"]
    softmax_temp: float = 0.01
    top_mean_percent: float = 15.0
    sim_thresh: float = 0.7
    min_instance_percent: float = 1.0
    batch_size_per_gpu: int = 4
    conf_exp_scaling: float = 4
    batch_phi: int = 32
    batch_theta: int = 32

    @classmethod
    def id_dict(cls):
        return {
            "negative_texts": list(cls.negative_texts),
            "softmax_temp": float(cls.softmax_temp),
            "top_mean_percent": float(cls.top_mean_percent),
            "sim_thresh": float(cls.sim_thresh),
            "min_instance_percent": float(cls.min_instance_percent),
            "conf_exp_scaling": float(cls.conf_exp_scaling),
            "batch_phi": int(cls.batch_phi),
            "batch_theta": int(cls.batch_theta),
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

        # Load caches
        clip_shards = sorted(clip_root.glob("chunk_*.npy"))
        assert clip_shards, f"No CLIP shards under {clip_root}"
        self.clip_features = LazyFeatures(clip_shards)

        sam2_shards = sorted(sam2_root.glob("chunk_*.npz"))
        assert sam2_shards, f"No SAM2 shards under {sam2_root}"
        self.sam2_masks = SAM2LazyAutoMasks(sam2_shards)

        if self.text_prompts is None:
            text_shards = sorted(text_root.glob("chunk_*.json"))
            assert text_shards, f"No TEXT shards under {text_root}"
            self.text_features = TextLazyFeatures(text_shards)
        else:
            self.text_features = None

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

        clip_patch_feats = self.clip_features[idx].to(self.device)
        raw_auto_masks = self.sam2_masks[idx]

        # prompts
        if self.text_prompts is None:
            text_prompts = self.text_features[idx]
        else:
            text_prompts = self.text_prompts

        if not raw_auto_masks or not text_prompts:
            # Return all background with zero orientation features
            h, w = (raw_auto_masks[0]['segmentation'].shape if raw_auto_masks else (Image.open(image_path).size[1], Image.open(image_path).size[0]))
            fg = np.zeros((h, w), dtype=bool)
            # Return new format: pixel_data + empty instance_features
            pixel_data = np.zeros((h, w, 3), dtype=np.float32)
            pixel_data[..., :2] = np.stack([~fg, fg], axis=-1).astype(np.float32)  # foreground one-hot
            return {
                'pixel_data': pixel_data,
                'instance_features': {}  # No instances
            }

        # Instance mask from SAM2
        inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
            raw_auto_masks,
            min_iou=float(SAM2Args.pred_iou_thresh),
            min_area=float(SAM2Args.min_mask_region_area),
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
                "predicted_iou": float("inf"),
                "area": float("inf"),
            })

        if not auto_masks:
            h, w = inst_mask.shape
            fg = np.zeros((h, w), dtype=bool)
            pixel_data = np.zeros((h, w, 3), dtype=np.float32)
            pixel_data[..., :2] = np.stack([~fg, fg], axis=-1).astype(np.float32)  # foreground one-hot
            return {
                'pixel_data': pixel_data,
                'instance_features': {}  # No instances
            }

        h, w = auto_masks[0]["segmentation"].shape

        # Per-prompt sim maps and combine
        segment_sim_maps: List[np.ndarray] = []
        for text in text_prompts:
            text_emb = self.clip_model.encode_text(text)
            neg_text_embs = torch.stack([self.clip_model.encode_text(neg) for neg in ORIENTANYArgs.negative_texts], dim=0)
            sim_map = self.clip_model.compute_similarity(
                clip_patch_feats,
                text_emb,
                neg_text_embs=neg_text_embs,
                softmax_temp=ORIENTANYArgs.softmax_temp,
                normalize=True,
            )
            sim_map_up = np.array(Image.fromarray(sim_map.cpu().numpy()).resize((w, h), Image.BILINEAR))

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
        pixel_data = np.zeros((h, w, 3), dtype=np.float32)
        pixel_data[..., :2] = np.stack([~fg_mask, fg_mask], axis=-1).astype(np.float32)  # foreground one-hot

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

            # Apply confidence scaling
            conf_scaled_ax_logits = get_conf_temp_scaled_logits(gaus_ax_logits, outs['confidence'], drop_exp_factor=ORIENTANYArgs.conf_exp_scaling)
            conf_scaled_pl_logits = get_conf_temp_scaled_logits(gaus_pl_logits, outs['confidence'], drop_exp_factor=ORIENTANYArgs.conf_exp_scaling)
            conf_scaled_ro_logits = get_conf_temp_scaled_logits(gaus_ro_logits, outs['confidence'], drop_exp_factor=ORIENTANYArgs.conf_exp_scaling)

            # Convert to probabilities
            probs_ax = F.softmax(conf_scaled_ax_logits, dim=0)
            probs_pl = F.softmax(conf_scaled_pl_logits, dim=0)
            probs_ro = F.softmax(conf_scaled_ro_logits, dim=0)

            # Propagate distributions to world camera (cam2 = unit transformation)
            # Get cam1 transforms (following notebook logic exactly)
            nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(os.path.basename(image_path), self.transforms_lookup)
            nerf_ccs1_to_final_nerf_world = self.T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
            nerf_ccs1_to_final_nerf_world[:3, 3] *= self.orig_to_final_nerf_world_scale

            # Define world coordinate system as 4x4 identity (explicit)
            nerf_ccs2_to_final_nerf_world = np.eye(4)
            normal_ccs2_to_final_nerf_world = nerf_ccs2_to_final_nerf_world @ np.linalg.inv(get_nerf_ccs_to_normal_ccs_T())

            # Convert to normal CCS (following notebook Cell 6 logic)
            normal_ccs1_to_final_nerf_world = nerf_ccs1_to_final_nerf_world @ np.linalg.inv(get_nerf_ccs_to_normal_ccs_T())

            # Get normal_ccs1_to_normal_ccs2 (cam1 to world)
            normal_ccs1_to_normal_ccs2 = np.linalg.inv(normal_ccs2_to_final_nerf_world) @ normal_ccs1_to_final_nerf_world
            R_normal_cam1_to_normal_world = normal_ccs1_to_normal_ccs2[:3, :3]

            # Propagate distributions
            probs_ax_world, probs_pl_world, probs_ro_world = self.orient_any.push_distributions_to_new_view(
                probs_ax, probs_pl, probs_ro,
                torch.from_numpy(R_normal_cam1_to_normal_world.astype(np.float32)).to(self.device),
                batch_phi=ORIENTANYArgs.batch_phi,
                batch_theta=ORIENTANYArgs.batch_theta,
                device=self.device,
                show_progress=True
            )

            # Fit appropriate distributions to propagated distributions and store compact params
            # Azimuth and roll: von Mises (circular, 0-360°)
            ax_mean, ax_kappa = probs_to_von_mises(probs_ax_world, n_bins=360, angle_min_deg=0.0, period_deg=360.0)
            ro_mean, ro_kappa = probs_to_von_mises(probs_ro_world, n_bins=360, angle_min_deg=0.0, period_deg=360.0)
            # Polar: normal (linear, 0-180°)
            pl_mean, pl_std = probs_to_normal(probs_pl_world, n_bins=180, angle_min_deg=0.0, period_deg=180.0)

            # Store compact representation: 8D vector (2 von Mises means, 2 von Mises kappas, 1 normal mean, 1 normal std, 2 confidence)
            instance_feat = np.array([
                ax_mean.cpu().item(), ax_kappa.cpu().item(),  # azimuth von Mises params
                pl_mean.cpu().item(), pl_std.cpu().item(),    # polar normal params
                ro_mean.cpu().item(), ro_kappa.cpu().item(),  # roll von Mises params
                conf_logits[0].cpu().item(), conf_logits[1].cpu().item()  # confidence logits
            ], dtype=np.float32)

            instance_features[next_instance_id] = instance_feat.tolist()  # Convert to list for JSON serialization

            # Assign instance ID to foreground pixels
            pixel_data[seg, 2] = float(next_instance_id)
            next_instance_id += 1

            # Store debug info if requested
            if debug:
                if 'debug_distributions' not in locals():
                    debug_distributions = {}
                debug_distributions[next_instance_id - 1] = {
                    'probs_ax_world': probs_ax_world.cpu().numpy(),
                    'probs_pl_world': probs_pl_world.cpu().numpy(),
                    'probs_ro_world': probs_ro_world.cpu().numpy()
                }

            # Clean up instance-specific memory
            del instance_img, rm_bkg_img, outs, gaus_ax_logits, gaus_pl_logits, gaus_ro_logits, conf_logits
            del conf_scaled_ax_logits, conf_scaled_pl_logits, conf_scaled_ro_logits
            del probs_ax, probs_pl, probs_ro
            del probs_ax_world, probs_pl_world, probs_ro_world

        result = {
            'pixel_data': pixel_data,  # (H, W, 3) - [fg_one_hot, instance_id]
            'instance_features': instance_features  # {instance_id: 8D_mixed_distribution_params}
        }

        if debug and 'debug_distributions' in locals():
            result['debug_distributions'] = debug_distributions

        return result


class ORIENTANYExtractor:
    def __init__(self, device: torch.device, data_dir: Optional[Path] = None, text_prompts: Optional[List[str]] = None, verbose: bool = False) -> None:
        self.device = device
        self.verbose = verbose
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.text_prompts = text_prompts

        if self.data_dir is None:
            raise ValueError("ORIENTANYExtractor requires data_dir to locate precomputed CLIP and SAM2 shards")

        # Validate prerequisites
        feat_root = self.data_dir / "features"
        clip_root = feat_root / "clip"
        sam2_root = feat_root / "sam2"
        text_root = feat_root / "text"

        if not (clip_root / "meta.pt").exists():
            raise FileNotFoundError(f"Missing CLIP meta: {clip_root / 'meta.pt'}")
        if not list(clip_root.glob("chunk_*.npy")):
            raise FileNotFoundError(f"Missing CLIP shards under {clip_root}")
        if not list(sam2_root.glob("chunk_*.npz")):
            raise FileNotFoundError(f"Missing SAM2 shards under {sam2_root}")
        if text_prompts is None and not list(text_root.glob("chunk_*.json")):
            raise FileNotFoundError(f"Missing TEXT shards under {text_root} (required when using per-image text prompts)")

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


def make_orientany_extractor(device: torch.device, verbose: bool = False, data_dir: Optional[Path] = None, text_prompts: Optional[List[str]] = None) -> "ORIENTANYExtractor":
    if data_dir is None:
        raise ValueError("make_orientany_extractor requires data_dir")
    return ORIENTANYExtractor(device=device, data_dir=data_dir, text_prompts=text_prompts, verbose=verbose)


async def extract_orientany_batch(image_paths: List[str], device: torch.device, data_dir: Path, verbose: bool = False, text_prompts: Optional[List[str]] = None, debug: bool = False):
    extractor = make_orientany_extractor(device=device, verbose=verbose, data_dir=data_dir, text_prompts=text_prompts)
    return await extractor.extract_batch_async(image_paths, debug=debug)


async def process_single_image_orientany_async(image_path: str, orientany_client: AsyncMultiWrapper, debug: bool = False) -> Dict[str, Any]:
    return await orientany_client.compute_orientany_for_image_async(image_path, debug=debug)


if __name__ == "__main__":
    data_root = Path("datasets/f3rm/panda_demos/caterpillar")
    image_dir = data_root / "images"
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    image_paths = [str(p) for p in image_paths[:3]]  # Just 3 for demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract orientation features using global prompts (with debug info)
    extractor = make_orientany_extractor(device=device, data_dir=data_root, text_prompts=["toy"], verbose=True)
    features_data = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths, debug=True))
    print(f"Extracted {len(features_data)} feature maps")

    # Convert to full feature arrays for visualization
    features = []
    for data in features_data:
        pixel_data = data['pixel_data']  # (H, W, 3)
        instance_features = data['instance_features']  # {instance_id: 8D_mixed_distribution_params}

        # Reconstruct full feature array (H, W, 10) - compact representation
        h, w, _ = pixel_data.shape
        full_features = np.zeros((h, w, 10), dtype=np.float32)

        # Set foreground one-hot at the end
        full_features[..., 8:10] = pixel_data[..., :2]  # foreground one-hot

        # For each foreground pixel, get its instance features (8D mixed distribution params)
        for instance_id, instance_feat in instance_features.items():
            instance_id = int(instance_id)
            mask = (pixel_data[..., 2] == instance_id)
            if np.any(mask):
                # Convert list back to numpy array if needed
                if isinstance(instance_feat, list):
                    instance_feat = np.array(instance_feat, dtype=np.float32)
                full_features[mask, :8] = instance_feat

        features.append(full_features)

    print(f"Sample shape: {features[0].shape if features else None}")

    # Visualize results
    vis_count = min(2, len(image_paths))
    fig, axes = plt.subplots(3, vis_count, figsize=(6 * vis_count, 12))
    if vis_count == 1:
        axes = axes.reshape(3, 1)

    for i in tqdm(range(vis_count), desc="Visualizing results"):
        # RGB image
        rgb = Image.open(image_paths[i]).convert("RGB")
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"RGB {i+1}")
        axes[0, i].axis('off')

        # Foreground mask
        fg = features[i][..., -1] if i < len(features) else None
        if fg is not None:
            axes[1, i].imshow(fg, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title("Foreground Mask")
        axes[1, i].axis('off')

        # Orientation RGB
        if fg is not None:
            # Get distribution means for the three angles (from compact representation)
            ax_mean = features[i][..., 0]  # azimuth mean
            pl_mean = features[i][..., 2]  # polar mean
            ro_mean = features[i][..., 4]  # roll mean

            # Convert to RGB: azimuth->R, polar->G, roll->B
            orient_rgb = np.zeros((*fg.shape, 3), dtype=np.uint8)
            orient_rgb[..., 0] = np.clip(ax_mean / 359.0 * 255, 0, 255).astype(np.uint8)  # R: azimuth
            orient_rgb[..., 1] = np.clip(pl_mean / 179.0 * 255, 0, 255).astype(np.uint8)  # G: polar
            orient_rgb[..., 2] = np.clip(ro_mean / 359.0 * 255, 0, 255).astype(np.uint8)  # B: roll

            # Only show colors for foreground pixels
            orient_rgb[~fg.astype(bool)] = 0
            axes[2, i].imshow(orient_rgb)
            axes[2, i].set_title("Orientation RGB (Distribution Means)")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Demo: Distribution comparison using debug info
    if features_data and features_data[0].get('debug_distributions'):
        print("\n=== Distribution Comparison Demo ===")

        # Get first instance from first image
        first_img_data = features_data[0]
        first_instance_id = next(iter(first_img_data['instance_features'].keys()))
        first_instance_feat = first_img_data['instance_features'][first_instance_id]
        debug_distributions = first_img_data['debug_distributions'][int(first_instance_id)]

        print(f"Analyzing instance {first_instance_id} from first image")
        print(f"Stored distribution params: {first_instance_feat}")

        # Get original world-propagated distributions (from debug info)
        probs_ax_world_orig = debug_distributions['probs_ax_world']
        probs_pl_world_orig = debug_distributions['probs_pl_world']
        probs_ro_world_orig = debug_distributions['probs_ro_world']

        # Extract stored distribution parameters
        ax_mean_stored, ax_kappa_stored = first_instance_feat[0], first_instance_feat[1]
        pl_mean_stored, pl_std_stored = first_instance_feat[2], first_instance_feat[3]
        ro_mean_stored, ro_kappa_stored = first_instance_feat[4], first_instance_feat[5]

        # Expand stored parameters back to distributions (EXACT same logic as loss)
        ax_mean_tensor = torch.tensor(ax_mean_stored, dtype=torch.float32)
        ax_kappa_tensor = torch.tensor(ax_kappa_stored, dtype=torch.float32)
        pl_mean_tensor = torch.tensor(pl_mean_stored, dtype=torch.float32)
        pl_std_tensor = torch.tensor(pl_std_stored, dtype=torch.float32)
        ro_mean_tensor = torch.tensor(ro_mean_stored, dtype=torch.float32)
        ro_kappa_tensor = torch.tensor(ro_kappa_stored, dtype=torch.float32)

        ax_expanded = von_mises_to_probs(ax_mean_tensor, ax_kappa_tensor, n_bins=360, angle_min_deg=0.0, period_deg=360.0).squeeze().cpu().numpy()
        pl_expanded = normal_to_probs(pl_mean_tensor, pl_std_tensor, n_bins=180, angle_min_deg=0.0, period_deg=180.0).squeeze().cpu().numpy()
        ro_expanded = von_mises_to_probs(ro_mean_tensor, ro_kappa_tensor, n_bins=360, angle_min_deg=0.0, period_deg=360.0).squeeze().cpu().numpy()

        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Azimuth comparison
        x_ax = np.arange(360)
        axes[0].plot(x_ax, probs_ax_world_orig, 'k-', label='Original World-Propagated', linewidth=2)
        axes[0].plot(x_ax, ax_expanded, 'g-', label='Expanded von Mises (for loss)', linewidth=1.5)
        axes[0].axvline(ax_mean_stored, color='r', linestyle='--', label=f'Stored von Mises (μ={ax_mean_stored:.1f}, κ={ax_kappa_stored:.2f})')
        axes[0].set_title('Azimuth Distribution')
        axes[0].set_xlabel('Degrees')
        axes[0].set_ylabel('Probability')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Polar comparison
        x_pl = np.arange(180)
        axes[1].plot(x_pl, probs_pl_world_orig, 'k-', label='Original World-Propagated', linewidth=2)
        axes[1].plot(x_pl, pl_expanded, 'g-', label='Expanded Normal (for loss)', linewidth=1.5)
        axes[1].axvline(pl_mean_stored, color='r', linestyle='--', label=f'Stored Normal (μ={pl_mean_stored:.1f}, σ={pl_std_stored:.2f})')
        axes[1].set_title('Polar Distribution')
        axes[1].set_xlabel('Degrees')
        axes[1].set_ylabel('Probability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Roll comparison
        x_ro = np.arange(360)
        axes[2].plot(x_ro, probs_ro_world_orig, 'k-', label='Original World-Propagated', linewidth=2)
        axes[2].plot(x_ro, ro_expanded, 'g-', label='Expanded von Mises (for loss)', linewidth=1.5)
        axes[2].axvline(ro_mean_stored, color='r', linestyle='--', label=f'Stored von Mises (μ={ro_mean_stored:.1f}, κ={ro_kappa_stored:.2f})')
        axes[2].set_title('Roll Distribution')
        axes[2].set_xlabel('Degrees')
        axes[2].set_ylabel('Probability')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('OrientAny Distribution Compression/Expansion Demo', fontsize=14)
        plt.tight_layout()
        plt.show()

        print("Distribution comparison complete!")
    else:
        print("No debug distributions available for demo")
