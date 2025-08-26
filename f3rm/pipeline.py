from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Type, Tuple

import torch
import numpy as np

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import colormaps, writer
from nerfstudio.utils import profiler
from nerfstudio.utils.misc import step_check
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn

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
        self._train_cache_current_set: Optional[frozenset] = None
        # Provide the model access to centroid cache
        # self.model unwraps DDP if present
        self.model._get_train_centroid_cache = self.get_last_train_centroid_cache  # type: ignore[attr-defined]
        self.model._get_train_centroid_spread_cache = self.get_last_train_centroid_spread_cache  # type: ignore[attr-defined]
        self.model._train_centroid_cache_enabled = False  # Will be enabled after cold start

    @property
    def cfg(self) -> FeaturePipelineConfig:
        return self.config  # type: ignore[return-value]

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
        # Single knob: cache is active when centroid training is enabled
        if not getattr(self.model.config, "centroid_enable", False):
            return

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
        # Decide once whether we need centroid preds for EMA blending at this step
        blend = float(getattr(self.model.config, "centroid_gt_blend", 0.5))
        blend_after = int(getattr(self.model.config, "centroid_blend_after_steps", 0))
        allow_blend = (step >= blend_after) and getattr(self.model, "_train_centroid_cache_enabled", False) and (blend > 0.0)
        # We want centroid predictions:
        # - if EMA is enabled (allow_blend True) for centroid & spread GT EMA
        # - or after cold start to apply background-skip logic for spread even when EMA is off
        need_centroid_preds = allow_blend or getattr(self.model, "_train_centroid_cache_enabled", False)
        if self._local_rank == 0:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Caching train centroid/depth", total=len(unique_cams))
                for ci in unique_cams:
                    # After cold start, render centroid preds to enable EMA (if enabled) and spread logic; skip feature rendering
                    images = self._render_full_image_images_for_camera(ci, render_features=False, render_centroid=need_centroid_preds, render_foreground=self.model.config.foreground_enable)
                    if images is not None:
                        self._train_depth_cache[ci] = images["depth_raw"].detach()
                        c_img, v_img, c_rgb, s_img, s_err_rgb, s_prob_rgb, s_valid = self._compute_centroid_and_spread_gt_for_camera(ci, images, allow_blend=allow_blend)
                        self._train_centroid_cache[ci] = c_img.cpu()
                        self._train_centroid_valid[ci] = v_img.cpu()
                        self._train_centroid_spread_cache[ci] = s_img.cpu()
                        self._train_centroid_spread_valid[ci] = s_valid.cpu()
                        images["centroid_spread_gt_full"] = s_img
                        if rep_centroid_rgb is None:
                            rep_centroid_rgb = c_rgb
                        if rep_spread_err_rgb is None:
                            rep_spread_err_rgb = s_err_rgb
                        if rep_spread_prob_rgb is None:
                            rep_spread_prob_rgb = s_prob_rgb
                        if rep_depth is None:
                            rep_depth = images["depth"]
                    progress.advance(task)
        else:
            for ci in unique_cams:
                images = self._render_full_image_images_for_camera(ci, render_features=False, render_centroid=need_centroid_preds, render_foreground=self.model.config.foreground_enable)
                if images is not None:
                    self._train_depth_cache[ci] = images["depth_raw"].detach()
                    c_img, v_img, _, s_img, _, _, s_valid = self._compute_centroid_and_spread_gt_for_camera(ci, images, allow_blend=allow_blend)
                    self._train_centroid_cache[ci] = c_img.cpu()
                    self._train_centroid_valid[ci] = v_img.cpu()
                    self._train_centroid_spread_cache[ci] = s_img.cpu()
                    self._train_centroid_spread_valid[ci] = s_valid.cpu()
                    images["centroid_spread_gt_full"] = s_img
                    if rep_depth is None:
                        rep_depth = images["depth"]

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

    def _render_full_image_images_for_camera(self, camera_index: int, render_features: bool = True, render_centroid: bool = True, render_foreground: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        cams = self.datamanager.train_ray_generator.cameras
        # camera_opt_to_camera transform for this camera; broadcasted inside generate_rays
        c_tensor = torch.tensor([camera_index], device=cams.device)
        camera_opt_to_camera = self.datamanager.train_camera_optimizer(c_tensor)
        camera_ray_bundle = cams.generate_rays(camera_indices=int(camera_index), camera_opt_to_camera=camera_opt_to_camera)
        # Progress for single image render
        if self._local_rank == 0:
            with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
                task = progress.add_task("Rendering full image", total=1)
                outputs = self.model.get_outputs_for_camera_ray_bundle(
                    camera_ray_bundle,
                    render_features=render_features,
                    render_centroid=render_centroid,
                    render_foreground=render_foreground,
                )
                progress.advance(task)
        else:
            outputs = self.model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle,
                render_features=render_features,
                render_centroid=render_centroid,
                render_foreground=render_foreground,
            )
        if "rgb" not in outputs or "accumulation" not in outputs or "depth" not in outputs:
            return None
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
        # If features were rendered and centroid is available, include it for EMA blending
        if "centroid" in outputs:
            images["centroid_pred_full"] = outputs["centroid"]
        if "centroid_spread" in outputs:
            images["centroid_spread_pred_full"] = outputs["centroid_spread"]
        # Append foreground viz
        if "foreground_prob_rgb" in outputs:
            images["foreground_prob"] = outputs["foreground_prob_rgb"]
        return images

    def _compute_centroid_and_spread_gt_for_camera(self, camera_index: int, images: Dict[str, torch.Tensor], is_eval: bool = False, allow_blend: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Require SAM2 masks (must be present)
        sam2 = self.datamanager.sam2_masks
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
        blend = float(getattr(self.model.config, "centroid_gt_blend", 0.5))
        # Blend only after allowed and only if blend > 0
        if allow_blend and blend > 0.0:
            # Expect centroid prediction to be present in images if we requested features earlier
            assert "centroid_pred_full" in images, "Centroid prediction missing for EMA blending; expected render_features=True"
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
        # Render outputs with a small progress bar
        if self._local_rank == 0:
            with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
                task = progress.add_task("Rendering train image", total=1)
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True, render_centroid=True, render_foreground=self.model.config.foreground_enable)
                progress.advance(task)
        else:
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True, render_centroid=True, render_foreground=self.model.config.foreground_enable)
        # Construct a batch with the full GT image to mirror eval flow
        full_batch = self.datamanager.train_dataset.get_data(ci)
        # Reuse model's image/metrics helper for identical formatting (GT|Pred concat)
        _, images_dict = self.model.get_image_metrics_and_images(outputs, full_batch)
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

    # Eval image metrics/images with a progress bar around rendering
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        if self._local_rank == 0:
            with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
                task = progress.add_task("Rendering eval image", total=1)
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True, render_centroid=True, render_foreground=self.model.config.foreground_enable)
                progress.advance(task)
        else:
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True, render_centroid=True, render_foreground=self.model.config.foreground_enable)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        # Append centroid GT visualization for eval only after cold-start (no blending in eval)
        if getattr(self.model, "_train_centroid_cache_enabled", False):
            images = {"rgb": outputs.get("rgb"), "accumulation": outputs.get("accumulation"), "depth_raw": outputs.get("depth"), "ray_origins": camera_ray_bundle.origins, "ray_directions": camera_ray_bundle.directions}
            centroid_img, valid_img, centroid_rgb, spread_img, spread_err_rgb, spread_prob_rgb, _ = self._compute_centroid_and_spread_gt_for_camera(int(image_idx), images, is_eval=True, allow_blend=False)
            images_dict["centroid_cache"] = centroid_rgb
            images_dict["centroid_spread_error_cache"] = spread_err_rgb
            images_dict["centroid_spread_prob_cache"] = spread_prob_rgb
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict
