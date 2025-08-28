import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sam2.features.utils import SAM2utils
from sam2.features.clip_main import CLIPfeatures
from f3rm.features.utils import LazyFeatures, SAM2LazyAutoMasks
from f3rm.features.sam2_extract import SAM2Args


class CLIPSAMInstanceSegmenter:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feat_root = self.data_dir / "features"
        clip_meta = torch.load(feat_root / "clip" / "meta.pt")
        self.feat_image_fnames = clip_meta["image_fnames"]
        self.clip_features = LazyFeatures(sorted(feat_root.glob("clip/chunk_*.npy")))
        self.sam2_masks = SAM2LazyAutoMasks(sorted(feat_root.glob("sam2/chunk_*.npz")))
        self.clip_model = CLIPfeatures(device=self.device)

    def _filter_sam2_inst_mask(self, inst_mask: np.ndarray, min_instance_percent: float) -> np.ndarray:
        unique_ids = np.unique(inst_mask)
        total_pixels = inst_mask.size
        for inst_id in unique_ids:
            if inst_id > 0:
                mask = (inst_mask == inst_id)
                percent = (np.sum(mask) / total_pixels) * 100
                if percent < min_instance_percent:
                    inst_mask[mask] = 0
        return inst_mask

    def prefilter_and_pack_auto_masks(self, auto_masks: list, min_instance_percent: float) -> list:
        if not auto_masks:
            return []
        inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
            auto_masks,
            min_iou=float(SAM2Args.pred_iou_thresh),
            min_area=float(SAM2Args.min_mask_region_area),
            assign_by="area",
            start_from="low",
        )
        if inst_mask is None:
            # No valid masks found, create empty instance mask
            if auto_masks:
                h, w = auto_masks[0]['segmentation'].shape
            else:
                # No image path available, use fallback dimensions
                h, w = 512, 512
            inst_mask = np.zeros((h, w), dtype=np.uint16)
        inst_mask = self._filter_sam2_inst_mask(inst_mask, min_instance_percent)

        packed_auto_masks = []
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
            packed_auto_masks.append({
                "segmentation": seg,
                "bbox": bbox,
                "predicted_iou": float("inf"),
                "area": float("inf"),
            })
        return packed_auto_masks

    def filter_auto_masks_by_similarity(self, image_path: str, auto_masks: list, text_prompts: list,
                                        negative_texts: list = ["object"], softmax_temp: float = 0.01,
                                        top_mean_percent: float = 5, sim_thresh: float = 0.7):
        feat_image_index = self.feat_image_fnames.index(image_path)
        clip_patch_feats = self.clip_features[feat_image_index].to(self.device)

        if not auto_masks:
            return []

        h, w = auto_masks[0]['segmentation'].shape

        segment_sim_maps = []
        for text_prompt in text_prompts:
            text_emb = self.clip_model.encode_text(text_prompt)
            neg_text_embs = torch.stack([self.clip_model.encode_text(neg_text) for neg_text in negative_texts], dim=0)
            sim_map = self.clip_model.compute_similarity(clip_patch_feats, text_emb, neg_text_embs=neg_text_embs,
                                                         softmax_temp=softmax_temp, normalize=True)
            sim_map_upscaled = np.array(Image.fromarray(sim_map.cpu().numpy()).resize((w, h), Image.BILINEAR))

            segment_sim_map = np.zeros_like(sim_map_upscaled)
            for mask_dict in auto_masks:
                mask = mask_dict['segmentation']
                if mask.shape != (h, w):
                    continue
                mask_sims = sim_map_upscaled[mask]
                n_top = max(1, int(len(mask_sims) * top_mean_percent / 100))
                top_sims = np.sort(mask_sims)[-n_top:]
                segment_sim_map[mask] = np.mean(top_sims)
            segment_sim_maps.append(segment_sim_map)

        combined_segment_sim_map = np.maximum.reduce(segment_sim_maps)
        filtered_auto_masks = []

        for mask_dict in auto_masks:
            mask = mask_dict['segmentation']
            if mask.shape != (h, w):
                continue
            if np.any(combined_segment_sim_map[mask] > sim_thresh):
                filtered_auto_masks.append({
                    "segmentation": mask_dict['segmentation'],
                    "bbox": mask_dict['bbox'],
                    "predicted_iou": mask_dict.get('predicted_iou', 0.0),
                    "area": mask_dict.get('area', 0)
                })

        return filtered_auto_masks


if __name__ == "__main__":
    DATA_DIR = "datasets/f3rm/custom/betaipad/small"
    IMAGE_PATH = "datasets/f3rm/custom/betaipad/small/images/frame_00043.png"
    TEXT_PROMPTS = ['ipad']
    NEGATIVE_TEXTS = ["object", "floor", "wall"]
    SOFTMAX_TEMP = 0.01
    MIN_INSTANCE_PERCENT = 1.0
    TOP_MEAN_PERCENT = 5
    SIM_THRESH = 0.4

    segmenter = CLIPSAMInstanceSegmenter(DATA_DIR)
    feat_image_index = segmenter.feat_image_fnames.index(IMAGE_PATH)
    raw_auto_masks = segmenter.sam2_masks[feat_image_index]
    prefiltered_masks = segmenter.prefilter_and_pack_auto_masks(raw_auto_masks, MIN_INSTANCE_PERCENT)
    filtered_auto_masks = segmenter.filter_auto_masks_by_similarity(
        IMAGE_PATH,
        prefiltered_masks,
        TEXT_PROMPTS,
        NEGATIVE_TEXTS,
        SOFTMAX_TEMP,
        TOP_MEAN_PERCENT,
        SIM_THRESH,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(Image.open(IMAGE_PATH).convert("RGB"))
    axes[0].set_title("Original RGB")
    axes[0].axis('off')

    # Generate and display mask
    filtered_inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(filtered_auto_masks,
                                                                  min_iou=float(SAM2Args.pred_iou_thresh),
                                                                  min_area=float(SAM2Args.min_mask_region_area),
                                                                  assign_by="area", start_from="low")
    if filtered_inst_mask is None:
        # No valid masks found, create empty instance mask
        if filtered_auto_masks:
            h, w = filtered_auto_masks[0]['segmentation'].shape
        else:
            # Use actual image dimensions as fallback
            img = Image.open(IMAGE_PATH)
            h, w = img.height, img.width
        filtered_inst_mask = np.zeros((h, w), dtype=np.uint16)
    viz_mask, cmap, norm = SAM2utils.make_viz_mask_and_cmap(filtered_inst_mask)
    axes[1].imshow(viz_mask, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title(f"Filtered Auto-masks: {len(np.unique(filtered_inst_mask)) - 1} instances")
    axes[1].axis('off')

    gt_rgb = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    overlay = gt_rgb.copy()
    # Create overlay with colored mask
    colored_mask = np.zeros((*viz_mask.shape, 3), dtype=np.uint8)
    for i in range(1, len(np.unique(viz_mask))):
        mask = (viz_mask == i)
        color = np.array(cmap(norm(i))[:3]) * 255
        colored_mask[mask] = color.astype(np.uint8)
    alpha = 0.6
    overlay = ((1 - alpha) * overlay + alpha * colored_mask).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("RGB + Viz-mask Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Kept {len(filtered_auto_masks)} masks above similarity threshold {SIM_THRESH}")
