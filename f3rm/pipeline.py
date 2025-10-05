from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Type, Tuple
from pathlib import Path
from time import time

import torch
import numpy as np

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.utils import colormaps, writer
from nerfstudio.utils import profiler
from nerfstudio.utils.misc import step_check
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
from PIL import Image

from sam2.features.utils import SAM2utils
from f3rm.features.sam2_extract import SAM2Args


@dataclass
class FeaturePipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: FeaturePipeline)
    # Update frequency in steps. 0 => update only when the set of cameras in the current batch changes
    steps_per_train_cache_update: int = 0
    # Frequency (in steps) to visualize a Train Image (full-image render like Eval Images). 0 disables.
    steps_per_train_image_viz: int = 0
    # Cold start skip for cache updates to avoid initial compile/alloc stalls
    train_cache_cold_start_skip_steps: int = 0


class FeaturePipeline(VanillaPipeline):
    def __init__(
        self,
        config: FeaturePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler=None,
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )
        self._local_rank = local_rank
        self._train_depth_cache: Dict[int, torch.Tensor] = {}
        self._train_centroid_cache: Dict[int, torch.Tensor] = {}
        self._train_centroid_valid: Dict[int, torch.Tensor] = {}
        self._train_centroid_spread_cache: Dict[int, torch.Tensor] = {}
        self._train_centroid_spread_valid: Dict[int, torch.Tensor] = {}
        self._train_orientany_cache: Dict[int, torch.Tensor] = {}
        self._train_orientany_valid: Dict[int, torch.Tensor] = {}
        self._train_cache_current_set: Optional[frozenset] = None
        # Provide the model access to centroid cache
        # self.model unwraps DDP if present
        self.model._get_train_centroid_cache = self.get_last_train_centroid_cache  # type: ignore[attr-defined]
        self.model._get_train_centroid_spread_cache = self.get_last_train_centroid_spread_cache  # type: ignore[attr-defined]
        self.model._get_train_orientany_cache = self.get_last_train_orientany_cache  # type: ignore[attr-defined]
        self.model._train_centroid_cache_enabled = False  # Will be enabled after cold start

    @property
    def cfg(self) -> FeaturePipelineConfig:
        return self.config  # type: ignore[return-value]

    def _compute_blend_value(self, step: int) -> float:
        """Compute blend value using exponential schedule.

        Schedule:
        - centroid_blend_start_value until centroid_blend_after_steps
        - Exponential growth from centroid_blend_start_value to centroid_blend_end_value from centroid_blend_after_steps to centroid_blend_until_steps
        - centroid_blend_end_value after centroid_blend_until_steps
        """
        blend_after = int(getattr(self.model.config, "centroid_blend_after_steps", 0))
        blend_until = int(getattr(self.model.config, "centroid_blend_until_steps", 0))
        blend_start = float(getattr(self.model.config, "centroid_blend_start_value", 0.0))
        blend_end = float(getattr(self.model.config, "centroid_blend_end_value", 1.0))

        if step < blend_after:
            return blend_start
        elif step >= blend_until:
            return blend_end
        else:
            # Exponential growth from blend_start to blend_end
            progress = (step - blend_after) / (blend_until - blend_after)
            return blend_start + (blend_end - blend_start) * (1.0 - (1.0 - progress) ** 3)  # Cubic growth for smooth transition

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        # Same as VanillaPipeline.get_train_loss_dict, but update centroid cache BEFORE model forward
        ray_bundle, batch = self.datamanager.next_train(step)
        # Ensure centroid cache is up-to-date for this step/camera set
        self._maybe_update_train_centroid_cache(batch, step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        # Compute loss (centroid loss only applies after cold start and when cache exists)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        # Optionally visualize a Train Image (similar to Eval Images flow), decoupled from cache
        if self.cfg.steps_per_train_image_viz and step_check(step, self.cfg.steps_per_train_image_viz):
            self._log_train_images_for_step(batch, step)

        return model_outputs, loss_dict, metrics_dict

    def _maybe_update_train_centroid_cache(self, batch: Dict, step: int) -> None:
        if "indices" not in batch:
            return
        cam_idxs = batch["indices"][:, 0]
        if not isinstance(cam_idxs, torch.Tensor):
            cam_idxs = torch.as_tensor(cam_idxs, device=self.device)
        unique_cams: List[int] = torch.unique(cam_idxs).tolist()
        current_set = frozenset(int(ci) for ci in unique_cams)

        # Cold start skip; only enable cache after skip window
        if step < self.cfg.train_cache_cold_start_skip_steps:
            return
        # Enable cache assertions in model after cold start
        self.model._train_centroid_cache_enabled = True  # type: ignore[attr-defined]

        should_update = False
        # Update once immediately when the set of cams changes
        if current_set != self._train_cache_current_set:
            self._train_cache_current_set = current_set
            self._train_depth_cache = {}
            self._train_centroid_cache = {}
            self._train_centroid_valid = {}
            self._train_centroid_spread_cache = {}
            self._train_centroid_spread_valid = {}
            self._train_orientany_cache = {}
            self._train_orientany_valid = {}
            should_update = True
        # Or update periodically by steps if requested
        elif self.cfg.steps_per_train_cache_update and step_check(step, self.cfg.steps_per_train_cache_update):
            should_update = True

        if not should_update:
            return

        # Render and cache depths + centroid GT for ALL cameras present in this batch
        rep_depth = None
        rep_centroid_rgb = None
        rep_spread_err_rgb = None
        rep_spread_prob_rgb = None
        # Compute blend value using exponential schedule
        blend = self._compute_blend_value(step)
        blend_after = int(getattr(self.model.config, "centroid_blend_after_steps", 0))
        allow_blend = (step >= blend_after) and getattr(self.model, "_train_centroid_cache_enabled", False) and (blend > 0.0)
        # Centroid predictions are always available since centroid is always rendered
        if self._local_rank == 0:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Caching train centroid/depth", total=len(unique_cams))
                for ci in unique_cams:
                    # Skip feature rendering but include OrientAny for cache (centroid and foreground always rendered)
                    images = self._render_full_image_images_for_camera(ci, render_features=False, render_orientany=True)
                    if images is not None:
                        # Keep depth cache on CPU to reduce VRAM pressure
                        self._train_depth_cache[ci] = images["depth_raw"].detach().cpu()
                        c_img, v_img, c_rgb, s_img, s_err_rgb, s_prob_rgb, s_valid = self._compute_centroid_and_spread_gt_for_camera(ci, images, allow_blend=allow_blend, blend=blend)
                        self._train_centroid_cache[ci] = c_img.cpu()
                        self._train_centroid_valid[ci] = v_img.cpu()
                        self._train_centroid_spread_cache[ci] = s_img.cpu()
                        self._train_centroid_spread_valid[ci] = s_valid.cpu()
                        o_img, o_valid = self._compute_orientany_gt_for_camera(ci, images, allow_blend=allow_blend, blend=blend)
                        self._train_orientany_cache[ci] = o_img.cpu()
                        self._train_orientany_valid[ci] = o_valid.cpu()
                        images["centroid_spread_gt_full"] = s_img
                        images["centroid_spread_prob_soft_gt_full"] = torch.cat([1.0 - s_img[..., 1:2], s_img[..., 1:2]], dim=-1)
                        if rep_centroid_rgb is None:
                            rep_centroid_rgb = c_rgb
                        if rep_spread_err_rgb is None:
                            rep_spread_err_rgb = s_err_rgb
                        if rep_spread_prob_rgb is None:
                            rep_spread_prob_rgb = s_prob_rgb
                        if rep_depth is None:
                            rep_depth = images["depth"].cpu()
                    progress.advance(task)
        else:
            for ci in unique_cams:
                images = self._render_full_image_images_for_camera(ci, render_features=False, render_orientany=True)
                if images is not None:
                    self._train_depth_cache[ci] = images["depth_raw"].detach().cpu()
                    c_img, v_img, _, s_img, _, _, s_valid = self._compute_centroid_and_spread_gt_for_camera(ci, images, allow_blend=allow_blend, blend=blend)
                    self._train_centroid_cache[ci] = c_img.cpu()
                    self._train_centroid_valid[ci] = v_img.cpu()
                    self._train_centroid_spread_cache[ci] = s_img.cpu()
                    self._train_centroid_spread_valid[ci] = s_valid.cpu()
                    o_img, o_valid = self._compute_orientany_gt_for_camera(ci, images, allow_blend=allow_blend, blend=blend)
                    self._train_orientany_cache[ci] = o_img.cpu()
                    self._train_orientany_valid[ci] = o_valid.cpu()
                    images["centroid_spread_gt_full"] = s_img
                    images["centroid_spread_prob_soft_gt_full"] = torch.cat([1.0 - s_img[..., 1:2], s_img[..., 1:2]], dim=-1)
                    if rep_depth is None:
                        rep_depth = images["depth"].cpu()

        # Log one representative image to keep overhead minimal (post cold-start only)
        if getattr(self.model, "_train_centroid_cache_enabled", False):
            if rep_depth is not None:
                writer.put_image(name="Train Cache Images/depth", image=rep_depth, step=step)
            if rep_centroid_rgb is not None:
                writer.put_image(name="Train Cache Images/centroid", image=rep_centroid_rgb, step=step)
            if rep_spread_err_rgb is not None:
                writer.put_image(name="Train Cache Images/centroid_spread_error", image=rep_spread_err_rgb, step=step)
            if rep_spread_prob_rgb is not None:
                writer.put_image(name="Train Cache Images/centroid_spread_prob", image=rep_spread_prob_rgb, step=step)

    def _render_full_image_images_for_camera(self, camera_index: int, render_features: bool = True, render_orientany: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        cams = self.datamanager.train_ray_generator.cameras
        # camera_opt_to_camera transform for this camera; broadcasted inside generate_rays
        c_tensor = torch.tensor([camera_index], device=cams.device)
        camera_opt_to_camera = self.datamanager.train_camera_optimizer(c_tensor)
        camera_ray_bundle = cams.generate_rays(camera_indices=int(camera_index), camera_opt_to_camera=camera_opt_to_camera)
        # Progress for single image render (cache rendering doesn't need gradients)
        with torch.no_grad():
            if self._local_rank == 0:
                with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
                    task = progress.add_task("Rendering full image", total=1)
                    outputs = self.model.get_outputs_for_camera_ray_bundle(
                        camera_ray_bundle,
                        render_features=render_features,
                        render_orientany=render_orientany,
                    )
                    progress.advance(task)
            else:
                outputs = self.model.get_outputs_for_camera_ray_bundle(
                    camera_ray_bundle,
                    render_features=render_features,
                    render_orientany=render_orientany,
                )
        rgb = outputs["rgb"]
        acc_raw = outputs["accumulation"]
        acc = colormaps.apply_colormap(acc_raw)  # HWC in [0,1]
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])  # HWC
        images: Dict[str, torch.Tensor] = {"rgb": rgb, "accumulation": acc, "accumulation_raw": acc_raw, "depth": depth, "depth_raw": outputs["depth"]}
        # Add ray geometry for centroid projection
        images["ray_origins"] = camera_ray_bundle.origins
        images["ray_directions"] = camera_ray_bundle.directions
        if "feature_pca" in outputs:
            images["feature_pca"] = outputs["feature_pca"]
        # Always include centroid and spread for EMA blending
        images["centroid_pred_full"] = outputs["centroid"]
        images["centroid_spread_pred_full"] = outputs["centroid_spread"]
        # Append foreground viz
        images["foreground_prob"] = outputs["foreground_prob_rgb"]
        # Append OrientAny viz
        if "orientany_rgb" in outputs:
            images["orientany_rgb"] = outputs["orientany_rgb"]
        return images

    def _compute_centroid_and_spread_gt_for_camera(self, camera_index: int, images: Dict[str, torch.Tensor], is_eval: bool = False, allow_blend: bool = False, blend: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Require SAM2 masks (must be present)
        sam2 = self.datamanager.sam2_loader
        # Determine split and global index
        eval_offset = getattr(self.datamanager, "eval_offset", 0)
        global_idx = camera_index + (eval_offset if is_eval else 0)
        auto_masks = sam2[global_idx]

        # Early return for empty auto_masks (performance optimization)
        if len(auto_masks) == 0:
            # No masks; construct empty centroid outputs based on full-image size
            h = images["depth_raw"].shape[0]
            w = images["depth_raw"].shape[1]
            centroid_img = torch.zeros((h, w, 3), device=self.device, dtype=images["depth_raw"].dtype)
            valid_mask = torch.zeros((h, w, 1), device=self.device, dtype=torch.bool)
            centroid_rgb = self.model.centroid_shader(centroid_img, valid_mask)
            # Spread GT: channel 0 error (0), channel 1 prob=0 for background; supervise everywhere
            spread_gt = torch.zeros((h, w, 2), device=centroid_img.device, dtype=centroid_img.dtype)
            spread_gt[..., 1] = 0.0
            spread_valid = torch.ones((h, w, 1), dtype=torch.bool, device=centroid_img.device)
            spread_err_rgb = self.model.spread_shader(spread_gt[..., :1], spread_valid)
            spread_prob_rgb = self.model.prob_shader(spread_gt[..., 1:2], spread_valid)
            return centroid_img, valid_mask, centroid_rgb, spread_gt, spread_err_rgb, spread_prob_rgb, spread_valid

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
                # Use actual image dimensions from camera info
                cams = self.datamanager.eval_ray_generator.cameras if is_eval else self.datamanager.train_ray_generator.cameras
                h, w = cams.height[camera_index].item(), cams.width[camera_index].item()
            inst_mask = np.zeros((h, w), dtype=np.uint16)
        # Filter small instances by percent of image area
        min_percent = getattr(self.model.config, "centroid_min_instance_percent", 1.0)
        h, w = inst_mask.shape
        total = float(h * w)
        ids = torch.from_numpy(inst_mask.copy()).to(images["depth_raw"].device)
        # Compute depth/world points
        depth_raw = images["depth_raw"][..., 0]  # HxW
        origins = images["ray_origins"]
        directions = images["ray_directions"]
        world = origins + directions * depth_raw.unsqueeze(-1)
        # Compute centroids per instance
        unique_ids = torch.unique(ids)
        centroid_img = torch.zeros_like(world)
        valid_mask = torch.zeros((h, w), dtype=torch.bool, device=world.device)
        for inst_id in unique_ids:
            iid = int(inst_id.item())
            if iid <= 0:
                continue
            mask = ids == inst_id
            count = int(mask.sum().item())
            if count <= 0:
                continue
            percent = (100.0 * count) / total
            if percent < min_percent:
                continue
            pts = world[mask]
            if pts.numel() == 0:
                continue
            centroid = pts.mean(dim=0)
            centroid_img[mask] = centroid
            valid_mask[mask] = True
        # Optionally gate by accumulation
        min_acc = getattr(self.model.config, "centroid_min_accum", 0.0)
        if min_acc > 0.0 and "accumulation_raw" in images:
            acc = images["accumulation_raw"][..., 0]
            valid_mask &= (acc >= min_acc)
        # Blend with current model prediction to stabilize (EMA)
        # IMPORTANT: Use per-segment mean of current predictions, not per-pixel, for robustness
        # Note: blend value is passed from caller (computed using schedule)
        # Blend only after allowed and only if blend > 0
        if allow_blend and blend > 0.0:
            # Expect centroid prediction to be present in images (always rendered)
            assert "centroid_pred_full" in images, "Centroid prediction missing for EMA blending"
            pred_full = images["centroid_pred_full"].to(centroid_img)
            # centroid_img = (blend * pred_full + (1.0 - blend) * centroid_img)  # TODO: per-pixel blending (legacy, remove later if new blending working better)
            # Build an image where each pixel in a segment holds that segment's mean predicted centroid
            pred_seg_mean_img = torch.zeros_like(centroid_img)
            for inst_id in unique_ids:
                iid = int(inst_id.item())
                if iid <= 0:
                    continue
                mask = (ids == inst_id)
                count = int(mask.sum().item())
                if count <= 0:
                    continue
                percent = (100.0 * count) / total
                if percent < min_percent:
                    continue
                seg_pred = pred_full[mask]
                if seg_pred.numel() == 0:
                    continue
                seg_mean = seg_pred.mean(dim=0)
                pred_seg_mean_img[mask] = seg_mean
            # Apply EMA blending only on valid pixels
            centroid_img[valid_mask] = blend * pred_seg_mean_img[valid_mask] + (1.0 - blend) * centroid_img[valid_mask]
        # Colorize centroid using shader
        centroid_rgb = self.model.centroid_shader(centroid_img, valid_mask.unsqueeze(-1))

        # Build centroid-spread GT (two channels per pixel)
        # ch0: per-pixel centroid error; ch1: foreground probability label (1 for foreground, 0 for background)
        spread_gt = torch.zeros((h, w, 2), device=centroid_img.device, dtype=centroid_img.dtype)
        # Foreground label
        spread_gt[..., 1] = valid_mask.float()
        spread_valid = torch.ones((h, w, 1), dtype=torch.bool, device=centroid_img.device)
        if getattr(self.model, "_train_centroid_cache_enabled", False) and ("centroid_pred_full" in images):
            pred_full = images["centroid_pred_full"].to(centroid_img)
            l2 = torch.linalg.norm(pred_full - centroid_img, dim=-1, keepdim=True)
            if allow_blend and (blend > 0.0) and ("centroid_spread_pred_full" in images):
                # Use channel 0 (error) from predicted spread for EMA
                spread_pred_full = images["centroid_spread_pred_full"][..., :1].to(l2)
                spread_gt[..., :1] = blend * spread_pred_full + (1.0 - blend) * l2
            else:
                spread_gt[..., :1] = l2
        # Visualizations
        spread_err_rgb = self.model.spread_shader(spread_gt[..., :1], spread_valid)
        spread_prob_rgb = self.model.prob_shader(spread_gt[..., 1:2], spread_valid)
        return centroid_img, valid_mask.unsqueeze(-1), centroid_rgb, spread_gt, spread_err_rgb, spread_prob_rgb, spread_valid

    def _compute_orientany_gt_for_camera(self, camera_index: int, images: Dict[str, torch.Tensor], is_eval: bool = False, allow_blend: bool = False, blend: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get OrientAny GT from datamanager
        eval_offset = getattr(self.datamanager, "eval_offset", 0)
        global_idx = camera_index + (eval_offset if is_eval else 0)
        orientany_gt = self.datamanager.orientany_loader[global_idx]  # (H, W, 8) - compact GT

        h, w = orientany_gt.shape[:2]
        orientany_img = orientany_gt.to(self.device).float()  # Convert to float32 for consistency
        valid_mask = orientany_gt[..., 7] > 0.5  # OrientAny foreground mask

        # EMA blending with current predictions if allowed
        if allow_blend and blend > 0.0 and "orientany_logits" in images:
            # Get current OrientAny predictions (902D expanded logits)
            pred_logits = images["orientany_logits"]  # (H, W, 902)

            # Convert GT to expanded 902D form for blending
            gt_expanded = self._convert_orientany_compact_to_expanded(orientany_img)

            # Apply EMA blending in expanded form
            blended_logits = blend * pred_logits + (1.0 - blend) * gt_expanded

            # Convert back to compact 8D form for caching
            orientany_img = self._convert_orientany_expanded_to_compact(blended_logits)

        return orientany_img, valid_mask.unsqueeze(-1)

    def _convert_orientany_compact_to_expanded(self, orientany_compact: torch.Tensor) -> torch.Tensor:
        """Convert 8D compact OrientAny format to 902D expanded logits format."""
        h, w = orientany_compact.shape[:2]
        device = orientany_compact.device
        dtype = orientany_compact.dtype

        # Initialize expanded logits (use float32 for numerical stability)
        expanded_logits = torch.zeros((h, w, 902), device=device, dtype=torch.float32)

        # Get foreground mask
        fg_mask = orientany_compact[..., 7] > 0.5

        if fg_mask.any():
            # Extract compact parameters for foreground pixels
            ax_mean = orientany_compact[..., 0]  # azimuth mean
            ax_kappa = orientany_compact[..., 1]  # azimuth kappa
            pl_mean = orientany_compact[..., 2]  # polar mean
            pl_std = orientany_compact[..., 3]   # polar std
            ro_mean = orientany_compact[..., 4]  # roll mean
            ro_kappa = orientany_compact[..., 5]  # roll kappa

            # Vectorized conversion to expanded form
            # Azimuth: von Mises distribution (0-359)
            ax_logits = self._von_mises_to_logits_vectorized(ax_mean, ax_kappa, 360, fg_mask)
            expanded_logits[fg_mask, 0:360] = ax_logits[fg_mask]

            # Polar: normal distribution (0-179)
            pl_logits = self._normal_to_logits_vectorized(pl_mean, pl_std, 180, fg_mask)
            expanded_logits[fg_mask, 360:540] = pl_logits[fg_mask]

            # Roll: von Mises distribution (0-359)
            ro_logits = self._von_mises_to_logits_vectorized(ro_mean, ro_kappa, 360, fg_mask)
            expanded_logits[fg_mask, 540:900] = ro_logits[fg_mask]

            # Foreground logits: convert probabilities to logits
            fg_probs = orientany_compact[..., 6:8]
            fg_logits = torch.log(fg_probs + 1e-8)  # Add small epsilon for numerical stability
            expanded_logits[..., 900:902] = fg_logits

        return expanded_logits

    def _convert_orientany_expanded_to_compact(self, expanded_logits: torch.Tensor) -> torch.Tensor:
        """Convert 902D expanded logits format back to 8D compact format."""
        h, w = expanded_logits.shape[:2]
        device = expanded_logits.device
        dtype = expanded_logits.dtype

        # Initialize compact format (use float32 for consistency)
        compact = torch.zeros((h, w, 8), device=device, dtype=torch.float32)

        # Extract foreground predictions
        fg_logits = expanded_logits[..., 900:902]
        fg_probs = torch.softmax(fg_logits, dim=-1)
        compact[..., 6:8] = fg_probs

        # Get foreground mask
        fg_mask = fg_probs[..., 1] > 0.5

        if fg_mask.any():
            # Extract orientation predictions for foreground pixels only
            ax_logits = expanded_logits[..., 0:360]
            pl_logits = expanded_logits[..., 360:540]
            ro_logits = expanded_logits[..., 540:900]

            # Get argmax predictions for the three angles
            ax_pred = ax_logits.argmax(dim=-1).float()      # 0-359 (azimuth)
            pl_pred = pl_logits.argmax(dim=-1).float()      # 0-179 (polar)
            ro_pred = ro_logits.argmax(dim=-1).float()      # 0-359 (roll)

            # Convert to compact format (mean, variance approximation)
            compact[fg_mask, 0] = ax_pred[fg_mask]  # azimuth mean
            compact[fg_mask, 1] = 1.0  # azimuth kappa (fixed variance)
            compact[fg_mask, 2] = pl_pred[fg_mask]  # polar mean
            compact[fg_mask, 3] = 1.0  # polar std (fixed variance)
            compact[fg_mask, 4] = ro_pred[fg_mask]  # roll mean
            compact[fg_mask, 5] = 1.0  # roll kappa (fixed variance)

        return compact

    def _von_mises_to_logits_vectorized(self, means: torch.Tensor, kappas: torch.Tensor, num_classes: int, fg_mask: torch.Tensor) -> torch.Tensor:
        """Convert von Mises distribution parameters to logits (vectorized)."""
        h, w = means.shape
        device = means.device
        angles = torch.arange(num_classes, device=device, dtype=torch.float32)

        # Initialize output tensor
        logits = torch.zeros((h, w, num_classes), device=device, dtype=torch.float32)

        if fg_mask.any():
            # Vectorized computation for foreground pixels
            fg_means = means[fg_mask]  # (N,)
            fg_kappas = kappas[fg_mask]  # (N,)

            # Broadcast angles to match foreground pixels: (N, num_classes)
            angles_broadcast = angles.unsqueeze(0).expand(fg_means.shape[0], -1)
            means_broadcast = fg_means.unsqueeze(-1)  # (N, 1)
            kappas_broadcast = fg_kappas.unsqueeze(-1)  # (N, 1)

            # Von Mises log probability
            log_probs = kappas_broadcast * torch.cos(angles_broadcast * 2 * torch.pi / num_classes - means_broadcast * 2 * torch.pi / 360)

            # Normalize to logits
            logits_fg = log_probs - log_probs.max(dim=-1, keepdim=True)[0]

            # Store back in full tensor
            logits[fg_mask] = logits_fg

        return logits

    def _normal_to_logits_vectorized(self, means: torch.Tensor, stds: torch.Tensor, num_classes: int, fg_mask: torch.Tensor) -> torch.Tensor:
        """Convert normal distribution parameters to logits (vectorized)."""
        h, w = means.shape
        device = means.device
        values = torch.arange(num_classes, device=device, dtype=torch.float32)

        # Initialize output tensor
        logits = torch.zeros((h, w, num_classes), device=device, dtype=torch.float32)

        if fg_mask.any():
            # Vectorized computation for foreground pixels
            fg_means = means[fg_mask]  # (N,)
            fg_stds = stds[fg_mask]  # (N,)

            # Broadcast values to match foreground pixels: (N, num_classes)
            values_broadcast = values.unsqueeze(0).expand(fg_means.shape[0], -1)
            means_broadcast = fg_means.unsqueeze(-1)  # (N, 1)
            stds_broadcast = fg_stds.unsqueeze(-1)  # (N, 1)

            # Normal log probability
            log_probs = -0.5 * ((values_broadcast - means_broadcast) / stds_broadcast) ** 2

            # Normalize to logits
            logits_fg = log_probs - log_probs.max(dim=-1, keepdim=True)[0]

            # Store back in full tensor
            logits[fg_mask] = logits_fg

        return logits

    def _log_train_images_for_step(self, batch: Dict, step: int) -> None:
        # Choose one camera from current train batch to render full image
        if "indices" not in batch:
            return
        cam_idxs = batch["indices"][:, 0]
        if not isinstance(cam_idxs, torch.Tensor):
            cam_idxs = torch.as_tensor(cam_idxs, device=self.device)
        unique_cams: List[int] = torch.unique(cam_idxs).tolist()
        if not unique_cams:
            return
        ci = int(unique_cams[0])
        # Build full-image ray bundle for the selected train camera
        cams = self.datamanager.train_ray_generator.cameras
        c_tensor = torch.tensor([ci], device=cams.device)
        camera_opt_to_camera = self.datamanager.train_camera_optimizer(c_tensor)
        camera_ray_bundle = cams.generate_rays(camera_indices=ci, camera_opt_to_camera=camera_opt_to_camera)
        # Render outputs with a small progress bar (use no_grad for memory efficiency)
        with torch.no_grad():
            if self._local_rank == 0:
                with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
                    task = progress.add_task("Rendering train image", total=1)
                    outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True, render_orientany=self.model.config.orientany_enable)
                    progress.advance(task)
            else:
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True, render_orientany=self.model.config.orientany_enable)
        # Construct a batch with the full GT image to mirror eval flow
        full_batch = self.datamanager.train_dataset.get_data(ci)
        # Reuse model's image/metrics helper for identical formatting (GT|Pred concat)
        _, images_dict = self.model.get_image_metrics_and_images(outputs, full_batch)
        # Add foreground pred vs GT side-by-side using full-image GT from datamanager
        ci_global = ci  # train split uses train indices directly
        fg_map = self.datamanager.fg_loader[ci_global]
        fg_gt_prob = fg_map[..., 1:2]
        fg_gt_rgb = self.model.prob_from_probs_shader(fg_gt_prob)

        images_dict["foreground_prob_gt"] = fg_gt_rgb
        images_dict["foreground_prob_vs_gt"] = torch.cat([outputs["foreground_prob_rgb"], fg_gt_rgb], dim=1)

        # Add OrientAny pred vs GT side-by-side using full-image GT from datamanager
        if self.model.config.orientany_enable and ("orientany_rgb" in outputs):
            # Get full OrientAny GT tensor (memory efficient, no gradients needed)
            with torch.no_grad():
                ci_global = ci  # train split uses train indices directly
                orientany_gt = self.datamanager.orientany_loader[ci_global].cpu()

                # Convert GT distribution means to RGB (vectorized operations on CPU)
                ax_mean_gt = orientany_gt[..., 0]  # azimuth mean
                pl_mean_gt = orientany_gt[..., 2]  # polar mean
                ro_mean_gt = orientany_gt[..., 4]  # roll mean
                fg_gt = orientany_gt[..., 7] > 0.5  # foreground mask from GT

                orientany_gt_rgb = torch.stack([
                    torch.clamp(ax_mean_gt / 359.0, 0, 1),
                    torch.clamp(pl_mean_gt / 179.0, 0, 1),
                    torch.clamp(ro_mean_gt / 359.0, 0, 1)
                ], dim=-1)

                # Only show colors for foreground pixels (set background to black)
                orientany_gt_rgb[~fg_gt] = 0.0

                images_dict["orientany_gt_rgb"] = orientany_gt_rgb
                images_dict["orientany_vs_gt"] = torch.cat([outputs["orientany_rgb"], orientany_gt_rgb], dim=1)
        for key, img in images_dict.items():
            writer.put_image(name=f"Train Images/{key}", image=img, step=step)
        # Also log centroid cache if present (after cold start)
        if getattr(self.model, "_train_centroid_cache_enabled", False):
            c_img = self._train_centroid_cache[ci].to(self.device)
            v_img = self._train_centroid_valid[ci].to(self.device)
            centroid_rgb = self.model.centroid_shader(c_img, v_img)
            writer.put_image(name="Train Images/centroid_cache", image=centroid_rgb, step=step)

    def get_last_train_centroid_cache(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        # Returns mapping to (centroid_img HxWx3, valid_mask HxWx1)
        return {k: (self._train_centroid_cache[k], self._train_centroid_valid[k]) for k in self._train_centroid_cache}

    def get_last_train_centroid_spread_cache(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        # Returns mapping to (spread_img HxWx1, valid_mask HxWx1)
        return {k: (self._train_centroid_spread_cache[k], self._train_centroid_spread_valid[k]) for k in self._train_centroid_spread_cache}

    def get_last_train_orientany_cache(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        # Returns mapping to (orientany_img HxWx8, valid_mask HxWx1)
        return {k: (self._train_orientany_cache[k], self._train_orientany_valid[k]) for k in self._train_orientany_cache}

    # Eval image metrics/images with a progress bar around rendering
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        with torch.no_grad():
            if self._local_rank == 0:
                with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
                    task = progress.add_task("Rendering eval image", total=1)
                    outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True, render_orientany=self.model.config.orientany_enable)
                    progress.advance(task)
            else:
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True, render_orientany=self.model.config.orientany_enable)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        # Append centroid GT visualization for eval only after cold-start (no blending in eval)
        if getattr(self.model, "_train_centroid_cache_enabled", False):
            images = {"rgb": outputs.get("rgb"), "accumulation": outputs.get("accumulation"), "depth_raw": outputs.get("depth"), "ray_origins": camera_ray_bundle.origins, "ray_directions": camera_ray_bundle.directions}
            centroid_img, valid_img, centroid_rgb, spread_img, spread_err_rgb, spread_prob_rgb, _ = self._compute_centroid_and_spread_gt_for_camera(int(image_idx), images, is_eval=True, allow_blend=False)
            images_dict["centroid_cache"] = centroid_rgb
            images_dict["centroid_spread_error_cache"] = spread_err_rgb
            images_dict["centroid_spread_prob_cache"] = spread_prob_rgb
            # Side-by-side pred vs GT for centroid spread prob and prob_soft
            if ("centroid_spread_prob_rgb" in outputs) and (spread_prob_rgb is not None):
                images_dict["centroid_spread_prob_vs_gt"] = torch.cat([outputs["centroid_spread_prob_rgb"], spread_prob_rgb], dim=1)
            if ("centroid_spread_prob_soft_rgb" in outputs) and (spread_prob_rgb is not None):
                images_dict["centroid_spread_prob_soft_vs_gt"] = torch.cat([outputs["centroid_spread_prob_soft_rgb"], spread_prob_rgb], dim=1)
        # Add foreground pred vs GT side-by-side if available in eval batch
        ci_global = int(image_idx) + getattr(self.datamanager, "eval_offset", 0)
        fg_map = self.datamanager.fg_loader[ci_global]
        fg_gt_prob = fg_map[..., 1:2]
        fg_gt_rgb = self.model.prob_from_probs_shader(fg_gt_prob)

        images_dict["foreground_prob_gt"] = fg_gt_rgb
        images_dict["foreground_prob_vs_gt"] = torch.cat([outputs["foreground_prob_rgb"], fg_gt_rgb], dim=1)

        # Add OrientAny pred vs GT side-by-side if available in eval batch
        if self.model.config.orientany_enable and ("orientany_rgb" in outputs):
            # Get full OrientAny GT tensor (memory efficient, no gradients needed)
            with torch.no_grad():
                ci_global = int(image_idx) + getattr(self.datamanager, "eval_offset", 0)
                orientany_gt = self.datamanager.orientany_loader[ci_global].cpu()

                # Convert GT distribution means to RGB (vectorized operations on CPU)
                ax_mean_gt = orientany_gt[..., 0]  # azimuth mean
                pl_mean_gt = orientany_gt[..., 2]  # polar mean
                ro_mean_gt = orientany_gt[..., 4]  # roll mean
                fg_gt = orientany_gt[..., 7] > 0.5  # foreground mask from GT

                orientany_gt_rgb = torch.stack([
                    torch.clamp(ax_mean_gt / 359.0, 0, 1),
                    torch.clamp(pl_mean_gt / 179.0, 0, 1),
                    torch.clamp(ro_mean_gt / 359.0, 0, 1)
                ], dim=-1)

                # Only show colors for foreground pixels (set background to black)
                orientany_gt_rgb[~fg_gt] = 0.0

                images_dict["orientany_gt_rgb"] = orientany_gt_rgb
                images_dict["orientany_vs_gt"] = torch.cat([outputs["orientany_rgb"], orientany_gt_rgb], dim=1)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Memory-efficient override: iterate eval images, render per image with no_grad, free tensors between images."""
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for i, (camera_ray_bundle, batch) in enumerate(self.datamanager.fixed_indices_eval_dataloader):
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                with torch.no_grad():
                    outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True, render_orientany=self.model.config.orientany_enable)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                        )
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                metrics_dict["fps"] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)

                # Aggressive cleanup between images
                del outputs, images_dict
                if i % 10 == 0:  # Only clear cache every 10 images
                    torch.cuda.empty_cache()
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict
