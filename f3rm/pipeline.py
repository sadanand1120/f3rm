from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Type, Tuple

import torch

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
        self._train_cache_current_set: Optional[frozenset] = None
        # Provide the model access to centroid cache
        # self.model unwraps DDP if present
        self.model._get_train_centroid_cache = self.get_last_train_centroid_cache  # type: ignore[attr-defined]
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
            should_update = True
        # Or update periodically by steps if requested
        elif self.cfg.steps_per_train_cache_update and step_check(step, self.cfg.steps_per_train_cache_update):
            should_update = True

        if not should_update:
            return

        # Render and cache depths + centroid GT for ALL cameras present in this batch
        rep_depth = None
        rep_centroid_rgb = None
        # Disable feature rendering to make cache render lighter
        if self._local_rank == 0:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Caching train centroid/depth", total=len(unique_cams))
                for ci in unique_cams:
                    images = self._render_full_image_images_for_camera(ci, render_features=False)
                    if images is not None:
                        self._train_depth_cache[ci] = images["depth_raw"].detach()
                        c_img, v_img, c_rgb = self._compute_centroid_gt_for_camera(ci, images)
                        self._train_centroid_cache[ci] = c_img.cpu()
                        self._train_centroid_valid[ci] = v_img.cpu()
                        if rep_centroid_rgb is None:
                            rep_centroid_rgb = c_rgb
                        if rep_depth is None:
                            rep_depth = images["depth"]
                    progress.advance(task)
        else:
            for ci in unique_cams:
                images = self._render_full_image_images_for_camera(ci, render_features=False)
                if images is not None:
                    self._train_depth_cache[ci] = images["depth_raw"].detach()
                    c_img, v_img, _ = self._compute_centroid_gt_for_camera(ci, images)
                    self._train_centroid_cache[ci] = c_img.cpu()
                    self._train_centroid_valid[ci] = v_img.cpu()
                    if rep_depth is None:
                        rep_depth = images["depth"]

        # Log one representative image to keep overhead minimal (post cold-start only)
        if getattr(self.model, "_train_centroid_cache_enabled", False):
            if rep_depth is not None:
                writer.put_image(name="Train Cache Images/depth", image=rep_depth, step=step)
            if rep_centroid_rgb is not None:
                writer.put_image(name="Train Cache Images/centroid", image=rep_centroid_rgb, step=step)

    def _render_full_image_images_for_camera(self, camera_index: int, render_features: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        cams = self.datamanager.train_ray_generator.cameras
        # camera_opt_to_camera transform for this camera; broadcasted inside generate_rays
        c_tensor = torch.tensor([camera_index], device=cams.device)
        camera_opt_to_camera = self.datamanager.train_camera_optimizer(c_tensor)
        camera_ray_bundle = cams.generate_rays(camera_indices=int(camera_index), camera_opt_to_camera=camera_opt_to_camera)
        # Progress for single image render
        if self._local_rank == 0:
            with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
                task = progress.add_task("Rendering full image", total=1)
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=render_features)
                progress.advance(task)
        else:
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=render_features)
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
        return images

    def _compute_centroid_gt_for_camera(self, camera_index: int, images: Dict[str, torch.Tensor], is_eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Require SAM2 masks (must be present)
        sam2 = self.datamanager.sam2_masks
        # Determine split and global index
        eval_offset = getattr(self.datamanager, "eval_offset", 0)
        global_idx = camera_index + (eval_offset if is_eval else 0)
        auto_masks = sam2[global_idx]
        inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
            auto_masks,
            min_iou=float(SAM2Args.pred_iou_thresh),
            min_area=float(SAM2Args.min_mask_region_area),
            assign_by="area",
            start_from="low",
        )
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
        # Colorize using shader
        centroid_rgb = self.model.centroid_shader(centroid_img, valid_mask.unsqueeze(-1))
        return centroid_img, valid_mask.unsqueeze(-1), centroid_rgb

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
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True)
                progress.advance(task)
        else:
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True)
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

    # Eval image metrics/images with a progress bar around rendering
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        if self._local_rank == 0:
            with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
                task = progress.add_task("Rendering eval image", total=1)
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True)
                progress.advance(task)
        else:
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        # Append centroid GT visualization for eval only after cold-start
        if getattr(self.model, "_train_centroid_cache_enabled", False):
            images = {"rgb": outputs.get("rgb"), "accumulation": outputs.get("accumulation"), "depth_raw": outputs.get("depth"), "ray_origins": camera_ray_bundle.origins, "ray_directions": camera_ray_bundle.directions}
            centroid_img, valid_img, centroid_rgb = self._compute_centroid_gt_for_camera(int(image_idx), images, is_eval=True)
            images_dict["centroid_cache"] = centroid_rgb
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict
