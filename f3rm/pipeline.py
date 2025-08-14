from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Type

import torch

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import colormaps, writer
from nerfstudio.utils import profiler
from nerfstudio.utils.misc import step_check
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn


@dataclass
class FeaturePipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: FeaturePipeline)
    # Enable/disable train depth cache updates
    train_depth_cache_enable: bool = False
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
        self._train_cache_current_set: Optional[frozenset] = None
        # Provide the model a lightweight accessor to the full-image train depth cache
        try:
            # self.model unwraps DDP if present
            self.model._get_train_depth_cache = self.get_last_train_depth_cache  # type: ignore[attr-defined]
            self.model._train_depth_cache_enabled = False  # Will be enabled after cold start
        except Exception:
            pass

    @property
    def cfg(self) -> FeaturePipelineConfig:
        return self.config  # type: ignore[return-value]

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        # Same as VanillaPipeline.get_train_loss_dict, then optionally update train depth cache
        ray_bundle, batch = self.datamanager.next_train(step)
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

        # Update train depth cache before loss so model can access current batch cache
        self._maybe_update_train_depth_cache(batch, step)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        # Optionally visualize a Train Image (similar to Eval Images flow), decoupled from cache
        if self.cfg.steps_per_train_image_viz and step_check(step, self.cfg.steps_per_train_image_viz):
            self._log_train_images_for_step(batch, step)

        return model_outputs, loss_dict, metrics_dict

    def _maybe_update_train_depth_cache(self, batch: Dict, step: int) -> None:
        if not self.cfg.train_depth_cache_enable:
            return

        if "indices" not in batch:
            return
        cam_idxs = batch["indices"][:, 0]
        if not isinstance(cam_idxs, torch.Tensor):
            cam_idxs = torch.as_tensor(cam_idxs, device=self.device)
        unique_cams: List[int] = torch.unique(cam_idxs).tolist()
        current_set = frozenset(int(ci) for ci in unique_cams)

        # Cold start skip
        if step < self.cfg.train_cache_cold_start_skip_steps:
            return
        # Enable cache assertions in model after cold start
        self.model._train_depth_cache_enabled = bool(self.config.train_depth_cache_enable)  # type: ignore[attr-defined]

        should_update = False
        # Update once immediately when the set of cams changes
        if current_set != self._train_cache_current_set:
            self._train_cache_current_set = current_set
            self._train_depth_cache = {}
            should_update = True
        # Or update periodically by steps if requested
        elif self.cfg.steps_per_train_cache_update and step_check(step, self.cfg.steps_per_train_cache_update):
            should_update = True

        if not should_update:
            return

        # Render and cache depths for ALL cameras present in this batch
        rep_depth = None
        # Disable feature rendering to make cache render lighter
        if self._local_rank == 0:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Caching train depth", total=len(unique_cams))
                for ci in unique_cams:
                    images = self._render_full_image_images_for_camera(ci, render_features=False)
                    if images is not None:
                        self._train_depth_cache[ci] = images["depth_raw"].detach()
                        if rep_depth is None:
                            rep_depth = images["depth"]
                    progress.advance(task)
        else:
            for ci in unique_cams:
                images = self._render_full_image_images_for_camera(ci, render_features=False)
                if images is not None:
                    self._train_depth_cache[ci] = images["depth_raw"].detach()
                    if rep_depth is None:
                        rep_depth = images["depth"]

        # Log one representative image to keep overhead minimal
        if rep_depth is not None:
            writer.put_image(name="Train Cache Images/depth", image=rep_depth, step=step)

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
        acc = colormaps.apply_colormap(outputs["accumulation"])  # HWC in [0,1]
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])  # HWC
        images: Dict[str, torch.Tensor] = {"rgb": rgb, "accumulation": acc, "depth": depth, "depth_raw": outputs["depth"]}
        if "feature_pca" in outputs:
            images["feature_pca"] = outputs["feature_pca"]
        return images

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

    # Getter for external access
    def get_last_train_depth_cache(self) -> Dict[int, torch.Tensor]:
        return self._train_depth_cache

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
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict
