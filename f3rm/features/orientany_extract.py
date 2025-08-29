import gc
import asyncio
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sam2.features.utils import SAM2utils
from sam2.features.clip_main import CLIPfeatures
from sam2.features.utils import AsyncMultiWrapper

from f3rm.features.utils import LazyFeatures, SAM2LazyAutoMasks, TextLazyFeatures, run_async_in_any_context, resolve_devices_and_workers
from f3rm.features.sam2_extract import SAM2Args
from f3rm.features.orientany.orientany_main import OrientAny


class ORIENTANYArgs:
    negative_texts: List[str] = ["object", "floor", "wall"]
    softmax_temp: float = 0.01
    top_mean_percent: float = 15.0
    sim_thresh: float = 0.7
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

    async def compute_orientany_for_image_async(self, image_path: str) -> Dict[str, Any]:
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

            # Get OrientAny predictions
            rm_bkg_img = self.orient_any.preprocess_remove_bkg(instance_img, do_remove_background=False)
            outs = self.orient_any.get_model_outputs(rm_bkg_img, viz_distn=False)

            # Extract logits
            gaus_ax_logits = outs['gaus_ax_logits']  # 360D
            gaus_pl_logits = outs['gaus_pl_logits']  # 180D
            gaus_ro_logits = outs['gaus_ro_logits']  # 360D
            conf_logits = outs['conf_logits']        # 2D

            # Store instance features: 902D vector
            instance_feat = np.concatenate([
                gaus_ax_logits,  # 360D
                gaus_pl_logits,  # 180D
                gaus_ro_logits,  # 360D
                conf_logits      # 2D
            ]).astype(np.float32)

            instance_features[next_instance_id] = instance_feat.tolist()  # Convert to list for JSON serialization

            # Assign instance ID to foreground pixels
            pixel_data[seg, 2] = float(next_instance_id)
            next_instance_id += 1

            # Clean up instance-specific memory
            del instance_img, rm_bkg_img, outs, gaus_ax_logits, gaus_pl_logits, gaus_ro_logits, conf_logits

        return {
            'pixel_data': pixel_data,  # (H, W, 3) - [fg_one_hot, instance_id]
            'instance_features': instance_features  # {instance_id: 902D_features}
        }


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

    async def extract_batch_async(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting ORIENTANY features", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_orientany_async(path, self.client) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="ORIENTANY", leave=False)
            results.extend(batch_results)
            gc.collect()
        return results


def make_orientany_extractor(device: torch.device, verbose: bool = False, data_dir: Optional[Path] = None, text_prompts: Optional[List[str]] = None) -> "ORIENTANYExtractor":
    if data_dir is None:
        raise ValueError("make_orientany_extractor requires data_dir")
    return ORIENTANYExtractor(device=device, data_dir=data_dir, text_prompts=text_prompts, verbose=verbose)


async def extract_orientany_batch(image_paths: List[str], device: torch.device, data_dir: Path, verbose: bool = False, text_prompts: Optional[List[str]] = None):
    extractor = make_orientany_extractor(device=device, verbose=verbose, data_dir=data_dir, text_prompts=text_prompts)
    return await extractor.extract_batch_async(image_paths)


async def process_single_image_orientany_async(image_path: str, orientany_client: AsyncMultiWrapper) -> Dict[str, Any]:
    return await orientany_client.compute_orientany_for_image_async(image_path)


if __name__ == "__main__":
    data_root = Path("datasets/f3rm/custom/betabook/small")
    image_dir = data_root / "images"
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    image_paths = [str(p) for p in image_paths[:3]]  # Just 3 for demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract orientation features using global prompts
    extractor = make_orientany_extractor(device=device, data_dir=data_root, text_prompts=["book"], verbose=True)
    features_data = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))
    print(f"Extracted {len(features_data)} feature maps")

    # Convert to full feature arrays for visualization
    features = []
    for data in features_data:
        pixel_data = data['pixel_data']  # (H, W, 3)
        instance_features = data['instance_features']  # {instance_id: 902D_features}

        # Reconstruct full feature array (H, W, 904)
        h, w, _ = pixel_data.shape
        full_features = np.zeros((h, w, 904), dtype=np.float32)

        # Set foreground one-hot at the end
        full_features[..., 902:904] = pixel_data[..., :2]  # foreground one-hot

        # For each foreground pixel, get its instance features
        for instance_id, instance_feat in instance_features.items():
            instance_id = int(instance_id)
            mask = (pixel_data[..., 2] == instance_id)
            if np.any(mask):
                # Convert list back to numpy array if needed
                if isinstance(instance_feat, list):
                    instance_feat = np.array(instance_feat, dtype=np.float32)
                full_features[mask, :902] = instance_feat

        features.append(full_features)

    print(f"Sample shape: {features[0].shape if features else None}")

    # Visualize results
    vis_count = min(2, len(image_paths))
    fig, axes = plt.subplots(3, vis_count, figsize=(6 * vis_count, 12))
    if vis_count == 1:
        axes = axes.reshape(3, 1)

    for i in range(vis_count):
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
            # Get argmax predictions for the three angles
            gaus_ax_pred = np.argmax(features[i][..., 0:360], axis=-1)  # 0-359
            gaus_pl_pred = np.argmax(features[i][..., 360:540], axis=-1)  # 0-179
            gaus_ro_pred = np.argmax(features[i][..., 540:900], axis=-1)  # 0-359

            # Convert to RGB: azimuth->R, polar->G, roll->B
            orient_rgb = np.zeros((*fg.shape, 3), dtype=np.uint8)
            orient_rgb[..., 0] = (gaus_ax_pred / 359.0 * 255).astype(np.uint8)  # R: azimuth
            orient_rgb[..., 1] = (gaus_pl_pred / 179.0 * 255).astype(np.uint8)  # G: polar
            orient_rgb[..., 2] = (gaus_ro_pred / 359.0 * 255).astype(np.uint8)  # B: roll

            # Only show colors for foreground pixels
            orient_rgb[~fg.astype(bool)] = 0
            axes[2, i].imshow(orient_rgb)
            axes[2, i].set_title("Orientation RGB")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
