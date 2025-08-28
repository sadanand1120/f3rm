import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps
from sam2.features.utils import SAM2utils, apply_pca_colormap
from sam2.features.clip_main import CLIPfeatures
from f3rm.features.utils import LazyFeatures, SAM2LazyAutoMasks
from f3rm.features.sam2_extract import SAM2Args
from depthany2.viz_utils import viz_pc, pcd_from_np, _save_pcd_via_open3d
from f3rm.shaders import CentroidShader


class CLIPSAMSegmenter:
    def __init__(self, data_dir: str, config_path: str, debug: bool = False):
        self.debug = debug
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)

        # Load pipeline first to get config
        _, self.pipeline, _, _ = eval_setup(config_path=self.config_path, test_mode="test")
        sam2_feature_type = self.pipeline.datamanager.config.sam2_feature_type

        self.feat_root = self.data_dir / "features"
        clip_meta = torch.load(self.feat_root / "clip" / "meta.pt")
        self.feat_image_fnames = clip_meta["image_fnames"]
        self.clip_features = LazyFeatures(sorted(self.feat_root.glob("clip/chunk_*.npy")))

        # Load SAM2 masks based on the feature type from config
        sam2_feat_dir = self.feat_root / sam2_feature_type.lower()
        # TODO: hijacked for now to maintain consistency with debug_instance.py
        sam2_feat_dir = self.feat_root / "sam2"
        # sam2_feat_dir = self.feat_root / "clipsam_book_table"
        if sam2_feat_dir.exists():
            self.sam2_masks = SAM2LazyAutoMasks(sorted(sam2_feat_dir.glob("chunk_*.npz")))
            print(f"****Loaded SAM2 lazy auto-masks for {sam2_feature_type} supervision cache")
        else:
            raise FileNotFoundError(f"SAM2 feature directory not found: {sam2_feat_dir}")

        self.pipeline.eval()
        self.centroid_shader = CentroidShader(self.pipeline.model.field.spatial_distortion)
        self.train_image_fnames = [str(p) for p in self.pipeline.datamanager.train_dataset.image_filenames]
        self.eval_image_fnames = [str(p) for p in self.pipeline.datamanager.eval_dataset.image_filenames]
        self.eval_offset = self.pipeline.datamanager.eval_offset

    def get_cam_info(self, image_path: str = None, feat_global_idx: int = None, split: str = None, local_cam_idx: int = None):
        if image_path is not None:
            feat_global_idx = self.feat_image_fnames.index(image_path)
        elif feat_global_idx is not None:
            image_path = self.feat_image_fnames[feat_global_idx]
        elif split is not None and local_cam_idx is not None:
            feat_global_idx = local_cam_idx if split == 'train' else local_cam_idx + self.eval_offset
            image_path = self.feat_image_fnames[feat_global_idx]
        else:
            raise ValueError("Must provide either image_path, feat_global_idx, or (split, local_cam_idx)")
        if feat_global_idx < self.eval_offset:
            split, local_cam_idx = 'train', feat_global_idx
        else:
            split, local_cam_idx = 'eval', feat_global_idx - self.eval_offset
        return feat_global_idx, image_path, split, local_cam_idx

    def filter_sam2_inst_mask(self, inst_mask, min_instance_percent):
        unique_ids = np.unique(inst_mask)
        total_pixels = inst_mask.size
        if self.debug:
            instance_sizes = [(inst_id, np.sum(inst_mask == inst_id)) for inst_id in unique_ids]
            instance_sizes.sort(key=lambda x: x[1], reverse=True)
            for inst_id, num_pixels in instance_sizes:
                if num_pixels > 0:
                    coords = np.where(inst_mask == inst_id)
                    y, x = coords[0][0], coords[1][0]
                    percent = (num_pixels / total_pixels) * 100
                    print(f"Instance {inst_id} ({percent:.2f}%): pixel at ({y}, {x}) = {inst_mask[y, x]}")
        for inst_id in unique_ids:
            if inst_id > 0:
                mask = (inst_mask == inst_id)
                percent = (np.sum(mask) / total_pixels) * 100
                if percent < min_instance_percent:
                    inst_mask[mask] = 0  # background
                    if self.debug:
                        print(f"Filtered out Instance {inst_id} ({percent:.2f}% < {min_instance_percent}%)")
        return inst_mask

    def compute_segment_similarity(self, sim_map_upscaled, inst_mask, top_percent):
        segment_sim_map = np.zeros_like(sim_map_upscaled)
        instance_sim_means = []
        for inst_id in np.unique(inst_mask):
            if inst_id > 0:
                mask = (inst_mask == inst_id)
                instance_sims = sim_map_upscaled[mask]
                n_pixels = len(instance_sims)
                n_top = max(1, int(n_pixels * top_percent / 100))
                top_sims = np.sort(instance_sims)[-n_top:]
                mean_sim = np.mean(top_sims)
                segment_sim_map[mask] = mean_sim
                instance_sim_means.append((inst_id, mean_sim, n_top, n_pixels))
        instance_sim_means.sort(key=lambda x: x[1], reverse=True)
        if self.debug:
            for inst_id, mean_sim, n_top, n_pixels in instance_sim_means:
                print(f"Instance {inst_id}: top {n_top}/{n_pixels} pixels, mean similarity = {mean_sim:.3f}")
        return segment_sim_map

    @torch.no_grad()
    def get_pipeline_outputs(self, split: str, local_cam_idx: int, render_features: bool = False, render_centroid: bool = True, render_spread: bool = True, render_foreground: bool = True):
        cams = self.pipeline.datamanager.eval_ray_generator.cameras if split == 'eval' else self.pipeline.datamanager.train_ray_generator.cameras
        c_tensor = torch.tensor([local_cam_idx], device=cams.device)
        camera_opt_to_camera = self.pipeline.datamanager.eval_camera_optimizer(c_tensor) if split == 'eval' else self.pipeline.datamanager.train_camera_optimizer(c_tensor)
        camera_ray_bundle = cams.generate_rays(camera_indices=local_cam_idx, camera_opt_to_camera=camera_opt_to_camera)
        outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(
            camera_ray_bundle,
            render_features=render_features,
            render_centroid=render_centroid,
            render_foreground=render_foreground,
        )
        pred_rgb = outputs["rgb"]
        depth_raw = outputs["depth"]
        depth_colored = colormaps.apply_depth_colormap(depth_raw, accumulation=outputs["accumulation"])
        result = {"pred_rgb": pred_rgb, "depth_raw": depth_raw.squeeze(-1), "depth_colored": depth_colored}
        if render_features:
            result["feature_pca"] = outputs["feature_pca"]
            result["feature"] = outputs["feature"]
        # Centroid predictions are provided independently of feature rendering
        if render_centroid and self.pipeline.model.config.centroid_enable:
            result["centroid_pred"] = outputs["centroid_pred"]
            result["centroid_pred_rgb"] = outputs["centroid_pred_rgb"]
            # Centroid spread predictions - now separate error and probability channels
            if render_spread:
                if "centroid_spread" in outputs:
                    result["centroid_spread"] = outputs["centroid_spread"]
                if "centroid_spread_error_rgb" in outputs:
                    result["centroid_spread_error_rgb"] = outputs["centroid_spread_error_rgb"]
                if "centroid_spread_prob_rgb" in outputs:
                    result["centroid_spread_prob_rgb"] = outputs["centroid_spread_prob_rgb"]
        # Foreground visualization
        if render_foreground and ("foreground_prob_rgb" in outputs):
            result["foreground_prob_rgb"] = outputs["foreground_prob_rgb"]
        # Model now returns aggregated logits; expose probs explicitly for raw viz
        if render_foreground and ("foreground_logits" in outputs):
            fg_logits = outputs["foreground_logits"]
            fg_probs = torch.softmax(fg_logits, dim=-1)
            result["foreground_logits"] = fg_logits
            result["foreground_probs"] = fg_probs
        original_c2w_34 = cams.camera_to_worlds[local_cam_idx].cpu().numpy()
        original_c2w = np.vstack([original_c2w_34, np.array([[0, 0, 0, 1]])])
        camera_opt_to_camera_4x4 = np.vstack([camera_opt_to_camera[0].cpu().numpy(), np.array([[0, 0, 0, 1]])])
        optimized_c2w = original_c2w @ camera_opt_to_camera_4x4
        return result, camera_ray_bundle, optimized_c2w

    @torch.no_grad()
    def generate_pointcloud_from_clipsam(self, segment_sim_map, depth_raw, camera_ray_bundle, pred_rgb, pc_thresh=0.7):
        torch.cuda.empty_cache()
        binary_mask = (segment_sim_map > pc_thresh).astype(bool)
        if not np.any(binary_mask):
            print("No pixels above threshold, returning empty pointcloud")
            return np.empty((0, 3)), np.empty((0, 3))

        ray_origins = camera_ray_bundle.origins.squeeze(0)
        ray_directions = camera_ray_bundle.directions.squeeze(0)
        H, W = segment_sim_map.shape
        if ray_origins.shape[:2] != (H, W):
            print(f"Resizing ray tensors from {ray_origins.shape[:2]} to {(H, W)}")
            ray_origins = torch.nn.functional.interpolate(ray_origins.permute(2, 0, 1).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
            ray_directions = torch.nn.functional.interpolate(ray_directions.permute(2, 0, 1).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

        points_3d = ray_origins + ray_directions * depth_raw.unsqueeze(-1)
        points_3d = points_3d[binary_mask]
        colors_3d = pred_rgb[binary_mask]
        return points_3d.cpu().numpy(), colors_3d.cpu().numpy()


if __name__ == "__main__":
    IMAGE_PATH = "datasets/f3rm/custom/betamulti1/small/images/frame_00001.png"
    CONFIG_PATH = "cent7_outputs/betam1_small_cstext_lang32_loss8e3_trunk0F_fg64x2/f3rm/2025-08-27_152754/config.yml"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEBUG = False

    TEXT_PROMPT = "notebook"
    NEGATIVE_TEXTS = ["object", "floor", "wall"]
    SOFTMAX_TEMP = 0.01
    MIN_INSTANCE_PERCENT = 1.0
    TOP_MEAN_PERCENT = 15

    RENDER_FEATURES = True
    RENDER_CENTROID = True
    RENDER_SPREAD = True
    RENDER_FOREGROUND = True

    print(f"IMAGE_PATH: {IMAGE_PATH}, CONFIG_PATH: {CONFIG_PATH}")
    segmenter = CLIPSAMSegmenter(data_dir=Path(IMAGE_PATH).parent.parent, config_path=CONFIG_PATH, debug=DEBUG)
    clip_model = CLIPfeatures(device=DEVICE)
    feat_image_index, image_path, split, local_cam_idx = segmenter.get_cam_info(image_path=IMAGE_PATH)
    clip_patch_feats = segmenter.clip_features[feat_image_index].to(DEVICE)
    text_emb = clip_model.encode_text(TEXT_PROMPT)
    neg_text_embs = torch.stack([clip_model.encode_text(neg_text) for neg_text in NEGATIVE_TEXTS], dim=0)
    sim_map = clip_model.compute_similarity(clip_patch_feats, text_emb, neg_text_embs=neg_text_embs, softmax_temp=SOFTMAX_TEMP, normalize=True)
    auto_masks = segmenter.sam2_masks[feat_image_index]
    inst_mask_result = SAM2utils.auto_masks_to_instance_mask(auto_masks, min_iou=float(SAM2Args.pred_iou_thresh), min_area=float(SAM2Args.min_mask_region_area), assign_by="area", start_from="low")
    if inst_mask_result[0] is None:
        # No valid masks found, create empty instance mask
        if auto_masks:
            h, w = auto_masks[0]['segmentation'].shape
        else:
            # Use actual image dimensions as fallback
            img = Image.open(image_path)
            h, w = img.height, img.width
        inst_mask = np.zeros((h, w), dtype=np.uint16)
    else:
        inst_mask = inst_mask_result[0]
    inst_mask = segmenter.filter_sam2_inst_mask(inst_mask, MIN_INSTANCE_PERCENT)
    sam2_viz_mask, cmap, norm = SAM2utils.make_viz_mask_and_cmap(inst_mask)
    sim_map_upscaled = np.array(Image.fromarray(sim_map.cpu().numpy()).resize((inst_mask.shape[1], inst_mask.shape[0]), Image.BILINEAR))
    segment_sim_map = segmenter.compute_segment_similarity(sim_map_upscaled, inst_mask, TOP_MEAN_PERCENT)
    pipeline_outputs, camera_ray_bundle, optimized_c2w = segmenter.get_pipeline_outputs(
        split,
        local_cam_idx,
        render_features=RENDER_FEATURES,
        render_centroid=RENDER_CENTROID,
        render_spread=RENDER_SPREAD,
        render_foreground=RENDER_FOREGROUND,
    )
    gt_clip_pca, proj_V, low_rank_min, low_rank_max = apply_pca_colormap(clip_patch_feats, return_proj=True, niter=5, q_min=0.01, q_max=0.99)
    if RENDER_FEATURES:
        pred_clip_pca = apply_pca_colormap(pipeline_outputs["feature"].to(DEVICE), proj_V=proj_V, low_rank_min=low_rank_min, low_rank_max=low_rank_max)

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes[0, 0].imshow(Image.open(image_path).convert("RGB"))
    axes[0, 0].set_title("GT RGB")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(sam2_viz_mask, cmap=cmap, norm=norm, interpolation='nearest')
    axes[0, 1].set_title(f"GT SAM2 Instance Mask: ({len(np.unique(inst_mask)) - 1} instances)")
    axes[0, 1].axis('off')
    axes[0, 2].imshow(gt_clip_pca.cpu().numpy())
    axes[0, 2].set_title("GT CLIP PCA Visualization")
    axes[0, 2].axis('off')
    axes[0, 3].imshow(sim_map_upscaled, cmap="gray")
    axes[0, 3].set_title(f"GT CLIP Similarity: {TEXT_PROMPT} vs {', '.join(NEGATIVE_TEXTS)}")
    axes[0, 3].axis('off')

    axes[1, 0].imshow(pipeline_outputs["pred_rgb"].cpu().numpy())
    axes[1, 0].set_title("Pred RGB")
    axes[1, 0].axis('off')
    axes[1, 1].imshow(pipeline_outputs["depth_colored"].cpu().numpy())
    axes[1, 1].set_title("Pred Depth (Colored)")
    axes[1, 1].axis('off')
    axes[1, 2].imshow(pipeline_outputs["depth_raw"].cpu().numpy(), cmap='gray')
    axes[1, 2].set_title("Pred Depth Raw (Grayscale)")
    axes[1, 2].axis('off')
    axes[1, 3].imshow(segment_sim_map, cmap="gray")
    axes[1, 3].set_title("GT Segment-Refined CLIP Sim")
    axes[1, 3].axis('off')

    # Feature viz (independent)
    if RENDER_FEATURES and ("feature_pca" in pipeline_outputs):
        axes[2, 0].imshow(pipeline_outputs["feature_pca"].numpy())
        axes[2, 0].set_title("Pred CLIP PCA (Pipeline)")
        axes[2, 0].axis('off')
    else:
        axes[2, 0].axis('off')

    # Centroid viz (independent)
    if RENDER_CENTROID and ("centroid_pred_rgb" in pipeline_outputs) and ("centroid_pred" in pipeline_outputs):
        cent_rgb = pipeline_outputs["centroid_pred_rgb"].cpu().numpy()
        axes[2, 1].imshow(cent_rgb)
        axes[2, 1].set_title("Pred Centroid RGB")
        axes[2, 1].axis('off')

        # Centroid spread error visualization (raw values)
        if "centroid_spread" in pipeline_outputs:
            spread_raw = pipeline_outputs["centroid_spread"].cpu().numpy()
            # Use only the first channel (error)
            spread_error = spread_raw[..., 0]  # HxWx1 -> HxW
            # Use a colormap that handles negative values well
            axes[2, 2].imshow(spread_error, cmap='RdYlBu_r')
            axes[2, 2].set_title("Pred Centroid Spread Error (Raw)")
            axes[2, 2].axis('off')
        else:
            axes[2, 2].axis('off')

        # Centroid spread probability visualization (raw logits)
        if "centroid_spread" in pipeline_outputs:
            spread_raw = pipeline_outputs["centroid_spread"].cpu().numpy()
            # Use only the second channel (probability)
            spread_prob = spread_raw[..., 1]  # HxWx1 -> HxW
            # Use grayscale for probability values
            axes[2, 3].imshow(spread_prob, cmap='gray')
            axes[2, 3].set_title("Pred Centroid Spread Logit (Raw)")
            axes[2, 3].axis('off')
        else:
            axes[2, 3].axis('off')
    else:
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')
        axes[2, 3].axis('off')

    # 4th row: Centroid spread shader outputs
    if RENDER_CENTROID and ("centroid_spread_error_rgb" in pipeline_outputs):
        spread_error_rgb = pipeline_outputs["centroid_spread_error_rgb"].cpu().numpy()
        axes[3, 0].imshow(spread_error_rgb)
        axes[3, 0].set_title("Centroid Spread Error (Shader)")
        axes[3, 0].axis('off')
    else:
        axes[3, 0].axis('off')

    if RENDER_CENTROID and ("centroid_spread_prob_rgb" in pipeline_outputs):
        spread_prob_rgb = pipeline_outputs["centroid_spread_prob_rgb"].cpu().numpy()
        axes[3, 1].imshow(spread_prob_rgb)
        axes[3, 1].set_title("Centroid Spread Prob (Shader)")
        axes[3, 1].axis('off')
    else:
        axes[3, 1].axis('off')

    # Foreground prob (shader) if present
    if RENDER_FOREGROUND and ("foreground_prob_rgb" in pipeline_outputs):
        fg_prob_rgb = pipeline_outputs["foreground_prob_rgb"].cpu().numpy()
        axes[3, 2].imshow(fg_prob_rgb)
        axes[3, 2].set_title("Foreground Prob (Shader)")
        axes[3, 2].axis('off')
    else:
        axes[3, 2].axis('off')
    # Foreground raw prob (channel 1) if present
    if RENDER_FOREGROUND and ("foreground_probs" in pipeline_outputs):
        fg_probs = pipeline_outputs["foreground_probs"].cpu().numpy()
        fg_prob_ch1 = fg_probs[..., 1]
        axes[3, 3].imshow(fg_prob_ch1, cmap='gray')
        axes[3, 3].set_title("Foreground Prob ch1 (Raw)")
        axes[3, 3].axis('off')
    else:
        axes[3, 3].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.show()
    print(f"Depth range: [{pipeline_outputs['depth_raw'].cpu().numpy().min():.3f}, {pipeline_outputs['depth_raw'].cpu().numpy().max():.3f}]")

    gt_rgb = torch.tensor(np.array(Image.open(image_path).convert("RGB")), dtype=torch.float32) / 255.0
    pred_rgb = pipeline_outputs["pred_rgb"]
    gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
    pred_rgb = torch.moveaxis(pred_rgb, -1, 0)[None, ...]
    psnr_value = segmenter.pipeline.model.psnr(pred_rgb.to(DEVICE), gt_rgb.to(DEVICE))
    print(f"PSNR: {psnr_value.item():.2f}")
