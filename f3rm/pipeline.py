from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Type
from pathlib import Path
from time import time

import torch

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.utils import writer
from nerfstudio.utils import profiler
from nerfstudio.utils.misc import step_check
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
from PIL import Image


@dataclass
class FeaturePipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: FeaturePipeline)
    steps_per_train_image_viz: int = 0


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

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        if self.config.steps_per_train_image_viz and step_check(step, self.config.steps_per_train_image_viz):
            self._log_train_images_for_step(batch, step)
        return model_outputs, loss_dict, metrics_dict

    def _render_outputs_with_progress(
        self, camera_ray_bundle, description: str, render_features: bool = True
    ) -> Dict[str, torch.Tensor]:
        if self._local_rank == 0:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(description, total=1)
                outputs = self.model.get_outputs_for_camera_ray_bundle(
                    camera_ray_bundle, render_features=render_features
                )
                progress.advance(task)
            return outputs
        return self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=render_features)

    @staticmethod
    def _image_idx_from_batch(batch: Dict) -> int:
        image_idx = batch["image_idx"]
        if torch.is_tensor(image_idx):
            return int(image_idx.item())
        return int(image_idx)

    def _log_train_images_for_step(self, batch: Dict, step: int) -> None:
        # Select a representative train camera from the current sampled batch.
        if "indices" not in batch:
            return
        cam_idxs = batch["indices"][:, 0]
        if not isinstance(cam_idxs, torch.Tensor):
            cam_idxs = torch.as_tensor(cam_idxs, device=self.device)
        unique_cams: List[int] = torch.unique(cam_idxs).tolist()
        if not unique_cams:
            return
        ci = int(unique_cams[0])
        # Build and render full-image outputs for that train camera.
        cams = self.datamanager.train_ray_generator.cameras
        c_tensor = torch.tensor([ci], device=cams.device)
        camera_opt_to_camera = self.model.camera_optimizer(c_tensor)
        camera_ray_bundle = cams.generate_rays(camera_indices=ci, camera_opt_to_camera=camera_opt_to_camera)
        outputs = self._render_outputs_with_progress(
            camera_ray_bundle, description="Rendering train image", render_features=True
        )
        # Convert model outputs into standard image artifacts for logging.
        full_batch = self.datamanager.train_dataset.get_data(ci)
        _, images_dict = self.model.get_image_metrics_and_images(outputs, full_batch)
        # Append foreground prediction-vs-GT visualization for train diagnostics.
        ci_global = ci
        fg_map = self.datamanager.fg_loader[ci_global]
        fg_gt_prob = fg_map[..., 1:2]
        fg_gt_rgb = self.model.prob_from_probs_shader(fg_gt_prob)

        images_dict["foreground_prob_gt"] = fg_gt_rgb
        images_dict["foreground_prob_vs_gt"] = torch.cat([images_dict["foreground_prob_rgb"], fg_gt_rgb], dim=1)

        for key, img in images_dict.items():
            writer.put_image(name=f"Train Images/{key}", image=img, step=step)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        # Retrieve one eval camera + batch using the current Nerfstudio datamanager API.
        camera, batch = self.datamanager.next_eval_image(step)
        camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
        # Render full-image outputs and compute base image metrics/artifacts.
        outputs = self._render_outputs_with_progress(
            camera_ray_bundle, description="Rendering eval image", render_features=True
        )
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        # Append foreground prediction-vs-GT visualization for eval diagnostics.
        image_idx = self._image_idx_from_batch(batch)
        ci_global = image_idx + getattr(self.datamanager, "eval_offset", 0)
        fg_map = self.datamanager.fg_loader[ci_global]
        fg_gt_prob = fg_map[..., 1:2]
        fg_gt_rgb = self.model.prob_from_probs_shader(fg_gt_prob)

        images_dict["foreground_prob_gt"] = fg_gt_rgb
        images_dict["foreground_prob_vs_gt"] = torch.cat([images_dict["foreground_prob_rgb"], fg_gt_rgb], dim=1)

        # Add metadata expected by trainer-side eval logging.
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
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
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for i, (camera, batch) in enumerate(self.datamanager.fixed_indices_eval_dataloader):
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                if output_path is not None:
                    image_idx = self._image_idx_from_batch(batch)
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            output_path / "{0:06d}-{1}.jpg".format(image_idx, key)
                        )
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                metrics_dict["fps"] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)

                # Aggressive cleanup between images
                del outputs, images_dict
                if i > 0 and i % 10 == 0:  # Only clear cache every 10 images
                    torch.cuda.empty_cache()
                progress.advance(task)
        metrics_dict = {}
        for key in metrics_dict_list[0]:
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([m[key] for m in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([m[key] for m in metrics_dict_list]))
                )
        self.train()
        return metrics_dict
