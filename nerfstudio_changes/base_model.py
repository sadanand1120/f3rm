# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base Model implementation which takes in RayBundles
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.scene_colliders import NearFarCollider


# Model related configs
@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: Model)
    """target class to instantiate"""
    enable_collider: bool = True
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    """parameters to instantiate scene collider with"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    """parameters to instantiate density field with"""
    eval_num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""
    prompt: Optional[str] = None
    """A prompt to be used in text to NeRF models"""


class Model(nn.Module):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        self.render_aabb: Optional[SceneBox] = None  # the box that we want to render - should be a subset of scene_box
        self.num_train_data = num_train_data
        self.kwargs = kwargs
        self.collider = None

        self.populate_modules()  # populate the modules
        self.callbacks = None
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        return []

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        # default instantiates optional modules that are common among many networks
        # NOTE: call `super().populate_modules()` in subclasses

        if self.config.enable_collider:
            assert self.config.collider_params is not None
            self.collider = NearFarCollider(
                near_plane=self.config.collider_params["near_plane"], far_plane=self.config.collider_params["far_plane"]
            )

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

    @abstractmethod
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    def forward(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {}

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                # outputs_lists[output_name].append(output)
                if output_name.startswith("feature"):   # mymodif
                    outputs_lists[output_name].append(output.cpu())  # mymodif
                else:   # mymodif
                    outputs_lists[output_name].append(output)  # mymodif
                del output   # mymodif
            if (i // num_rays_per_chunk) % 20 == 0:   # mymodif
                torch.cuda.empty_cache()   # mymodif
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_rgba_image(self, outputs: Dict[str, torch.Tensor], output_name: str = "rgb") -> torch.Tensor:
        """Returns the RGBA image from the outputs of the model.

        Args:
            outputs: Outputs of the model.

        Returns:
            RGBA image.
        """
        accumulation_name = output_name.replace("rgb", "accumulation")
        if (
            not hasattr(self, "renderer_rgb")
            or not hasattr(self.renderer_rgb, "background_color")
            or accumulation_name not in outputs
        ):
            raise NotImplementedError(f"get_rgba_image is not implemented for model {self.__class__.__name__}")
        rgb = outputs[output_name]
        if self.renderer_rgb.background_color == "random":  # type: ignore
            acc = outputs[accumulation_name]
            if acc.dim() < rgb.dim():
                acc = acc.unsqueeze(-1)
            return torch.cat((rgb / acc.clamp(min=1e-10), acc), dim=-1)
        return torch.cat((rgb, torch.ones_like(rgb[..., :1])), dim=-1)

    @abstractmethod
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        TODO: This shouldn't return a loss

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: dictionary of pre-trained model states
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state["model"].items()}
        self.load_state_dict(state)  # type: ignore

    def update_to_step(self, step: int) -> None:
        """Called when loading a model from a checkpoint. Sets any model parameters that change over
        training to the correct value, based on the training step of the checkpoint.

        Args:
            step: training step of the loaded checkpoint
        """
