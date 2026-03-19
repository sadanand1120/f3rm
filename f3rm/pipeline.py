from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Type

import torch
from PIL import Image
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.misc import step_check


@dataclass
class FeaturePipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: FeaturePipeline)
    steps_per_train_image_viz: int = 0


class FeaturePipeline(VanillaPipeline):
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        if self.config.steps_per_train_image_viz and step_check(step, self.config.steps_per_train_image_viz):
            self._log_train_images_for_step(batch, step)
        return model_outputs, loss_dict, metrics_dict

    def _log_train_images_for_step(self, batch: Dict, step: int) -> None:
        if "indices" not in batch:
            return
        cam_idxs = batch["indices"][:, 0]
        if not isinstance(cam_idxs, torch.Tensor):
            cam_idxs = torch.as_tensor(cam_idxs, device=self.device)
        unique_cams: List[int] = torch.unique(cam_idxs).tolist()
        if not unique_cams:
            return
        ci = int(unique_cams[0])
        cams = self.datamanager.train_ray_generator.cameras
        camera_ray_bundle = cams.generate_rays(camera_indices=ci, keep_shape=True)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=True)
        full_batch = self.datamanager.train_dataset.get_data(ci)
        _, images_dict = self.model.get_image_metrics_and_images(outputs, full_batch)

        for key, img in images_dict.items():
            writer.put_image(name=f"Train Images/{key}", image=img, step=step)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        if hasattr(self.datamanager, "fixed_indices_eval_dataloader") and self.datamanager.eval_dataset is not None:
            camera, batch = self.datamanager.fixed_indices_eval_dataloader.get_camera(step % max(len(self.datamanager.eval_dataset), 1))
        else:
            camera, batch = self.datamanager.next_eval_image(step)
        camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=False)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        image_idx = batch["image_idx"]
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = int(image_idx.item()) if torch.is_tensor(image_idx) else int(image_idx)
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
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
            inner_start = time()
            outputs = self.model.get_outputs_for_camera_ray_bundle(
                camera.generate_rays(camera_indices=0, keep_shape=True),
                render_features=False,
            )
            height, width = camera.height, camera.width
            num_rays = height * width
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

            if output_path is not None:
                image_idx = batch["image_idx"]
                image_idx = int(image_idx.item()) if torch.is_tensor(image_idx) else int(image_idx)
                for key, val in images_dict.items():
                    Image.fromarray((val * 255).byte().cpu().numpy()).save(output_path / f"{image_idx:06d}-{key}.jpg")
            metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
            metrics_dict["fps"] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
            metrics_dict_list.append(metrics_dict)

            del outputs, images_dict
        metrics_dict = {}
        for key in metrics_dict_list[0]:
            if get_std:
                key_std, key_mean = torch.std_mean(torch.tensor([m[key] for m in metrics_dict_list]))
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(torch.mean(torch.tensor([m[key] for m in metrics_dict_list])))
        self.train()
        return metrics_dict
