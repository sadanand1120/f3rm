from dataclasses import dataclass, field
from collections import defaultdict
from functools import cached_property
from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerNumber,
    ViewerText,
)
from torch.nn import Parameter

from f3rm.feature_field import FeatureField
from f3rm.features.utils import apply_pca_colormap, compute_similarity_scores
from f3rm.renderer import FeatureRenderer


@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    """Note: make sure to use naming that doesn't conflict with NerfactoModelConfig"""
    _target: Type = field(default_factory=lambda: FeatureFieldModel)
    feat_loss_weight: float = 1e-3
    feat_train_ray_ratio: float = 1.0
    feat_use_pe: bool = True
    feat_pe_n_freq: int = 6
    feat_num_levels: int = 12
    feat_log2_hashmap_size: int = 19
    feat_start_res: int = 16
    feat_max_res: int = 128
    feat_features_per_level: int = 8
    feat_hidden_dim: int = 64
    feat_num_layers: int = 2
    inst_feature_dim: int = 8
    inst2d_lambda: float = 0.1
    inst_gamma: float = 1.0
    inst_pos_weight: float = 1.0
    inst_neg_weight: float = 1.0
    inst_min_mask_pixels: int = 8
    inst_use_pe: bool = True
    inst_pe_n_freq: int = 6
    inst_num_levels: int = 12
    inst_log2_hashmap_size: int = 19
    inst_start_res: int = 16
    inst_max_res: int = 128
    inst_features_per_level: int = 8
    inst_hidden_dim: int = 64
    inst_num_layers: int = 2


@dataclass
class ViewerUtils:
    pca_proj: Optional[torch.Tensor] = None
    instance_pca_proj: Optional[torch.Tensor] = None
    positives: List[str] = field(default_factory=list)
    pos_embed: Optional[torch.Tensor] = None
    negatives: List[str] = field(default_factory=list)
    neg_embed: Optional[torch.Tensor] = None
    softmax_temp: float = 0.1
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_device(self, device: torch.device):
        """Update device and invalidate cached CLIP model if needed."""
        if self.device == device:
            return
        self.device = device
        self.__dict__.pop("clip", None)

    @cached_property
    def clip(self):
        import open_clip
        from f3rm.features.clip_extract import CLIPArgs

        CONSOLE.print(f"Loading CLIP {CLIPArgs.model_name} for viewer")
        model, _, _ = open_clip.create_model_and_transforms(CLIPArgs.model_name, pretrained=CLIPArgs.model_pretrained, device=self.device)
        model.eval()
        return model

    @torch.no_grad()
    def handle_language_queries(self, raw_text: str, is_positive: bool):
        """Compute CLIP embeddings based on queries and update state"""
        import open_clip
        from f3rm.features.clip_extract import CLIPArgs

        texts = [text.strip() for text in raw_text.split(",") if text.strip()]
        if not texts:
            if is_positive:
                self.positives.clear()
                self.pos_embed = None
            else:
                self.negatives.clear()
                self.neg_embed = None
            return
        tokenize = open_clip.get_tokenizer(CLIPArgs.model_name)
        tokens = tokenize(texts).to(self.device)
        embed = self.clip.encode_text(tokens).float()
        embed = embed / embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if is_positive:
            self.positives = texts
            # Average embedding if we have multiple positives
            embed = embed.mean(dim=0, keepdim=True)
            embed = embed / embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.pos_embed = embed
        else:
            self.negatives = texts
            self.neg_embed = embed

    def update_softmax_temp(self, temp: float):
        self.softmax_temp = max(float(temp), 1e-6)

    def reset_pca_proj(self):
        self.pca_proj = None
        self.instance_pca_proj = None
        CONSOLE.print("Reset PCA projection")


viewer_utils = ViewerUtils(device=torch.device("cpu"))

FEATURE_OUTPUT_KEY = "feature"
INSTANCE_FEATURE_OUTPUT_KEY = "instance-feature"


class FeatureFieldModel(NerfactoModel):
    config: FeatureFieldModelConfig

    feature_field: FeatureField
    instance_field: FeatureField

    def _build_aux_field(self, prefix: str, feature_dim: int) -> FeatureField:
        return FeatureField(
            feature_dim=feature_dim,
            spatial_distortion=self.field.spatial_distortion,
            use_pe=getattr(self.config, f"{prefix}_use_pe"),
            pe_n_freq=getattr(self.config, f"{prefix}_pe_n_freq"),
            num_levels=getattr(self.config, f"{prefix}_num_levels"),
            log2_hashmap_size=getattr(self.config, f"{prefix}_log2_hashmap_size"),
            start_res=getattr(self.config, f"{prefix}_start_res"),
            max_res=getattr(self.config, f"{prefix}_max_res"),
            features_per_level=getattr(self.config, f"{prefix}_features_per_level"),
            hidden_dim=getattr(self.config, f"{prefix}_hidden_dim"),
            num_layers=getattr(self.config, f"{prefix}_num_layers"),
            implementation=self.config.implementation,
        )

    def populate_modules(self):
        super().populate_modules()

        feature_dim = self.kwargs["metadata"]["feature_dim"]
        if feature_dim <= 0:
            raise ValueError("Feature dimensionality must be positive.")

        self.feature_field = self._build_aux_field("feat", feature_dim)
        self.instance_field = self._build_aux_field("inst", self.config.inst_feature_dim)

        self.renderer_feature = FeatureRenderer()
        self.setup_gui()

    def setup_gui(self):
        viewer_utils.set_device(self.kwargs["device"])
        # Note: the GUI elements are shown based on alphabetical variable names
        self.btn_refresh_pca = ViewerButton("Refresh PCA Projection", cb_hook=lambda _: viewer_utils.reset_pca_proj())

        # Only setup GUI for language features if we're using CLIP
        if self.kwargs["metadata"]["feature_type"] != "CLIP":
            return
        self.hint_text = ViewerText(name="Note:", disabled=True, default_value="Use , to separate labels")
        self.lang_1_pos_text = ViewerText(
            name="Language (Positives)",
            default_value="",
            cb_hook=lambda elem: viewer_utils.handle_language_queries(elem.value, is_positive=True),
        )
        self.lang_2_neg_text = ViewerText(
            name="Language (Negatives)",
            default_value="",
            cb_hook=lambda elem: viewer_utils.handle_language_queries(elem.value, is_positive=False),
        )
        self.softmax_temp = ViewerNumber(
            name="Softmax temperature",
            default_value=viewer_utils.softmax_temp,
            cb_hook=lambda elem: viewer_utils.update_softmax_temp(elem.value),
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["feature_field"] = list(self.feature_field.parameters())
        param_groups["instance_field"] = list(self.instance_field.parameters())
        return param_groups

    @staticmethod
    def _select_training_ray_indices(weights: torch.Tensor, ratio: float) -> Optional[torch.Tensor]:
        if ratio >= 1.0 or weights.ndim != 3:
            return None
        num_rays = weights.shape[0]
        keep = max(1, int(num_rays * ratio))
        if keep >= num_rays:
            return None
        return torch.linspace(0, num_rays - 1, steps=keep, device=weights.device).round().long().unique(sorted=True)

    def _get_outputs_internal(
        self,
        ray_bundle: RayBundle,
        render_features: bool,
        render_instance_features: bool = False,
        subsample_supervision_rays: bool = True,
    ):
        """Core rendering that can optionally skip feature-field computation."""
        # Match Nerfacto behavior: apply learned camera pose deltas during training.
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # Always use detached weights to avoid gradients flowing to NeRF/cameras
        custom_weights = weights.detach()

        feat_indices = None
        feat_ray_samples = ray_samples
        feat_weights = custom_weights
        if render_features and self.training and subsample_supervision_rays:
            feat_indices = self._select_training_ray_indices(custom_weights, self.config.feat_train_ray_ratio)
            if feat_indices is not None:
                feat_ray_samples = ray_samples[feat_indices]
                feat_weights = custom_weights[feat_indices]

        if render_features:
            feat_vals = self.feature_field.get_feature(feat_ray_samples)
            features = self.renderer_feature(features=feat_vals, weights=feat_weights)

        if render_instance_features:
            instance_vals = self.instance_field.get_feature(ray_samples)
            instance_features = self.renderer_feature(features=instance_vals, weights=custom_weights)

        outputs_feature_indices = feat_indices if render_features and feat_indices is not None else None

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }
        if render_features:
            outputs[FEATURE_OUTPUT_KEY] = features
        if render_instance_features:
            outputs[INSTANCE_FEATURE_OUTPUT_KEY] = instance_features
        if outputs_feature_indices is not None:
            outputs["feature_ray_indices"] = outputs_feature_indices

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                custom_weights, field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                custom_weights,
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_outputs(self, ray_bundle: RayBundle):
        """Modified from nerfacto.get_outputs to include feature field outputs."""
        return self._get_outputs_internal(ray_bundle, render_features=True)

    def forward(
        self, ray_bundle: Union[RayBundle, Cameras], instance_ray_bundle: Optional[RayBundle] = None
    ) -> Dict[str, torch.Tensor | List]:
        if isinstance(ray_bundle, Cameras):
            raise TypeError("FeatureFieldModel.forward expects a RayBundle during training.")
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        outputs = self.get_outputs(ray_bundle)
        if instance_ray_bundle is not None:
            outputs.update(self.get_instance_outputs(instance_ray_bundle))
        return outputs

    def get_instance_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        # Match the CLIP feature branch: instance supervision consumes detached rendering weights.
        with torch.no_grad():
            if self.training:
                self.camera_optimizer.apply_to_raybundle(ray_bundle)
            if self.collider is not None:
                ray_bundle = self.collider(ray_bundle)
            ray_samples, _, _ = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
            density, _ = self.field.get_density(ray_samples)
            weights = ray_samples.get_weights(density).detach()

        instance_values = self.instance_field.get_feature(ray_samples)
        instance_features = self.renderer_feature(features=instance_values, weights=weights)
        return {
            INSTANCE_FEATURE_OUTPUT_KEY: instance_features,
        }

    @staticmethod
    def _instance_image_height(batch: Dict[str, torch.Tensor | int]) -> int:
        image_height = batch["instance_image_height"]
        if torch.is_tensor(image_height):
            return int(image_height.item())
        return int(image_height)

    @staticmethod
    def _instance_image_width(batch: Dict[str, torch.Tensor | int]) -> int:
        image_width = batch["instance_image_width"]
        if torch.is_tensor(image_width):
            return int(image_width.item())
        return int(image_width)

    @staticmethod
    def _instance_image_count(batch: Dict[str, torch.Tensor | int]) -> int:
        num_images = batch["num_instance_images"]
        if torch.is_tensor(num_images):
            return int(num_images.item())
        return int(num_images)

    def _reshape_instance_images(self, values: torch.Tensor, batch: Dict[str, torch.Tensor | int]) -> torch.Tensor:
        image_height = self._instance_image_height(batch)
        image_width = self._instance_image_width(batch)
        num_images = self._instance_image_count(batch)
        expected_rays = num_images * image_height * image_width
        if values.shape[0] != expected_rays:
            raise ValueError(
                f"Expected {expected_rays} rays for {num_images} full instance images of size {image_height}x{image_width}, got {values.shape[0]}."
            )
        if values.ndim == 1:
            return values.view(num_images, image_height, image_width)
        return values.view(num_images, image_height, image_width, -1)

    def _compute_instance_image_terms(self, feat_map: torch.Tensor, mask_map: torch.Tensor):
        valid_labels = torch.unique(mask_map)
        valid_labels = valid_labels[valid_labels >= 0]

        prototypes = []
        positive_terms = []
        valid_mask_count = 0
        valid_pixel_count = 0
        for label in valid_labels:
            label_mask = mask_map == label
            pixel_count = int(label_mask.sum().item())
            if pixel_count < self.config.inst_min_mask_pixels:
                continue
            label_features = feat_map[label_mask]
            prototype = label_features.mean(dim=0)
            prototypes.append(prototype)
            positive_terms.append((label_features - prototype).square().sum(dim=-1).mean())
            valid_mask_count += 1
            valid_pixel_count += pixel_count

        pos_loss = torch.stack(positive_terms).mean() if positive_terms else feat_map.new_zeros(())
        if len(prototypes) >= 2:
            proto_stack = torch.stack(prototypes, dim=0)
            neg_loss = F.relu(self.config.inst_gamma - torch.pdist(proto_stack, p=2)).mean()
        else:
            neg_loss = feat_map.new_zeros(())
        return pos_loss, neg_loss, valid_mask_count, valid_pixel_count

    def _compute_instance_loss_terms(self, outputs, batch):
        feat_maps = self._reshape_instance_images(outputs[INSTANCE_FEATURE_OUTPUT_KEY].to(dtype=torch.float32), batch)
        mask_maps = self._reshape_instance_images(batch["instance_mask"].to(device=self.device, dtype=torch.long), batch)

        image_pos_losses = []
        image_neg_losses = []
        valid_mask_count = 0
        valid_pixel_count = 0
        for feat_map, mask_map in zip(feat_maps, mask_maps):
            pos_loss, neg_loss, image_valid_mask_count, image_valid_pixel_count = self._compute_instance_image_terms(feat_map, mask_map)
            if image_valid_mask_count > 0:
                image_pos_losses.append(pos_loss)
            if image_valid_mask_count > 1:
                image_neg_losses.append(neg_loss)
            valid_mask_count += image_valid_mask_count
            valid_pixel_count += image_valid_pixel_count

        pos_loss = torch.stack(image_pos_losses).mean() if image_pos_losses else feat_maps.new_zeros(())
        neg_loss = torch.stack(image_neg_losses).mean() if image_neg_losses else feat_maps.new_zeros(())
        inst2d_unscaled = self.config.inst_pos_weight * pos_loss + self.config.inst_neg_weight * neg_loss

        return {
            "instance_pos_loss": pos_loss,
            "instance_neg_loss": neg_loss,
            "instance_2d_loss_unscaled": inst2d_unscaled,
            "instance_valid_mask_count": float(valid_mask_count),
            "instance_valid_pixel_count": float(valid_pixel_count),
        }

    def _feature_targets(self, outputs, batch) -> torch.Tensor:
        feature_ray_indices = outputs.get("feature_ray_indices")
        target_feats = batch["feature"].to(device=self.device, dtype=torch.float32)
        if feature_ray_indices is not None:
            target_feats = target_feats[feature_ray_indices]
        return target_feats

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Feature metrics
        target_feats = self._feature_targets(outputs, batch)
        pred_feats = outputs[FEATURE_OUTPUT_KEY].to(dtype=torch.float32)
        metrics_dict["feature_error"] = F.mse_loss(pred_feats, target_feats)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        # Feature loss
        feature_error = metrics_dict.get("feature_error") if metrics_dict is not None else None
        if feature_error is None:
            pred_feats = outputs[FEATURE_OUTPUT_KEY].to(dtype=torch.float32)
            feature_error = F.mse_loss(pred_feats, self._feature_targets(outputs, batch))
        loss_dict["feature_loss"] = self.config.feat_loss_weight * feature_error
        return loss_dict

    def get_instance_metrics_dict(self, outputs, batch):
        return self._compute_instance_loss_terms(outputs, batch)

    def get_instance_loss_dict(self, outputs, batch, metrics_dict=None):
        metrics_dict = metrics_dict or self._compute_instance_loss_terms(outputs, batch)
        return {
            "instance_2d_loss": self.config.inst2d_lambda * metrics_dict["instance_2d_loss_unscaled"],
        }

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, render_features: bool = True) -> Dict[str, torch.Tensor]:
        """Full-image render with optional auxiliary feature computation. Features are kept on CPU."""
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            ray_bundle = ray_bundle.to(self.device)
            if self.collider is not None:
                ray_bundle = self.collider(ray_bundle)
            outputs_chunk = self._get_outputs_internal(
                ray_bundle,
                render_features=render_features,
                render_instance_features=render_features,
                subsample_supervision_rays=False,
            )
            for output_name, output in outputs_chunk.items():
                if not torch.is_tensor(output):
                    continue
                if output_name in {FEATURE_OUTPUT_KEY, INSTANCE_FEATURE_OUTPUT_KEY}:
                    outputs_lists[output_name].append(output.cpu())
                else:
                    outputs_lists[output_name].append(output.to(input_device))
                del output
        outputs: Dict[str, torch.Tensor] = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)

        # If requested depth-only render, exit early.
        if not render_features:
            return outputs

        if self.kwargs["metadata"]["feature_type"] != "CLIP" or not viewer_utils.positives or viewer_utils.pos_embed is None:
            return outputs

        clip_features = outputs[FEATURE_OUTPUT_KEY].to(viewer_utils.device)
        outputs["similarity"] = compute_similarity_scores(
            clip_features=clip_features,
            pos_embed=viewer_utils.pos_embed,
            neg_embed=viewer_utils.neg_embed if viewer_utils.negatives and viewer_utils.neg_embed is not None else None,
            softmax_temp=viewer_utils.softmax_temp,
        )
        return outputs

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        # Torchmetrics LPIPS is strict about [0,1] bounds; clamp tiny numeric drift for metric inputs only.
        outputs_for_metrics = dict(outputs)
        batch_for_metrics = dict(batch)
        if "rgb" in outputs_for_metrics:
            outputs_for_metrics["rgb"] = torch.clamp(
                torch.nan_to_num(outputs_for_metrics["rgb"], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
            )
        if "image" in batch_for_metrics:
            batch_for_metrics["image"] = torch.clamp(
                torch.nan_to_num(batch_for_metrics["image"], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
            )

        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs_for_metrics, batch_for_metrics)

        # pred normals
        if "pred_normals" in outputs:
            images_dict["pred_normals"] = outputs["pred_normals"]

        # feature PCA
        if FEATURE_OUTPUT_KEY in outputs:
            images_dict["feature_pca"], viewer_utils.pca_proj, *_ = apply_pca_colormap(
                outputs[FEATURE_OUTPUT_KEY],
                proj_V=viewer_utils.pca_proj,
                return_proj=True,
            )
            images_dict["feature_pca"] = images_dict["feature_pca"].to(torch.float16)
        if INSTANCE_FEATURE_OUTPUT_KEY in outputs:
            images_dict["instance_feature_pca"], viewer_utils.instance_pca_proj, *_ = apply_pca_colormap(
                outputs[INSTANCE_FEATURE_OUTPUT_KEY],
                proj_V=viewer_utils.instance_pca_proj,
                return_proj=True,
            )
            images_dict["instance_feature_pca"] = images_dict["instance_feature_pca"].to(torch.float16)

        return metrics_dict, images_dict
