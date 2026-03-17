from dataclasses import dataclass, field
from collections import defaultdict
from functools import cached_property
from typing import Dict, List, Optional, Type

import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle, RaySamples
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
from f3rm.pca_colormap import apply_pca_colormap_return_proj
from f3rm.renderer import FeatureRenderer, ScalarRenderer
from f3rm.features.utils import compute_similarity_scores, parse_comma_separated_labels
from f3rm.shaders import ProbFromProbsShader


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
    foreground_loss_weight: float = 1e-3
    foreground_train_ray_ratio: float = 1.0
    foreground_hidden_dim: int = 64
    foreground_num_layers: int = 1


@dataclass
class ViewerUtils:
    pca_proj: Optional[torch.Tensor] = None
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

        texts = parse_comma_separated_labels(raw_text)
        # Clear the GUI state if there are no texts
        if not texts:
            self.clear_positives() if is_positive else self.clear_negatives()
            return
        # Embed text queries
        tokenize = open_clip.get_tokenizer(CLIPArgs.model_name)
        tokens = tokenize(texts).to(self.device)
        # Keep viewer embeddings in fp32 to avoid dtype mismatch during similarity matmul.
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

    @property
    def has_positives(self) -> bool:
        return bool(self.positives) and self.pos_embed is not None

    def clear_positives(self):
        self.positives.clear()
        self.pos_embed = None

    @property
    def has_negatives(self) -> bool:
        return bool(self.negatives) and self.neg_embed is not None

    def clear_negatives(self):
        self.negatives.clear()
        self.neg_embed = None

    def update_softmax_temp(self, temp: float):
        self.softmax_temp = max(float(temp), 1e-6)

    def reset_pca_proj(self):
        self.pca_proj = None
        CONSOLE.print("Reset PCA projection")


viewer_utils = ViewerUtils(device=torch.device("cpu"))


class FeatureFieldModel(NerfactoModel):
    config: FeatureFieldModelConfig

    feature_field: FeatureField

    def populate_modules(self):
        super().populate_modules()

        feature_dim = self.kwargs["metadata"]["feature_dim"]
        if feature_dim <= 0:
            raise ValueError("Feature dimensionality must be positive.")

        self.feature_field = FeatureField(
            feature_dim=feature_dim,
            spatial_distortion=self.field.spatial_distortion,
            use_pe=self.config.feat_use_pe,
            pe_n_freq=self.config.feat_pe_n_freq,
            num_levels=self.config.feat_num_levels,
            log2_hashmap_size=self.config.feat_log2_hashmap_size,
            start_res=self.config.feat_start_res,
            max_res=self.config.feat_max_res,
            features_per_level=self.config.feat_features_per_level,
            hidden_dim=self.config.feat_hidden_dim,
            num_layers=self.config.feat_num_layers,
            foreground_hidden_dim=self.config.foreground_hidden_dim,
            foreground_num_layers=self.config.foreground_num_layers,
            implementation=self.config.implementation,
        )

        self.renderer_feature = FeatureRenderer()
        self.renderer_spread = ScalarRenderer()
        self.prob_from_probs_shader = ProbFromProbsShader()
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

    def _get_outputs_internal(self, ray_bundle: RayBundle, render_features: bool):
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

        fg_indices = None
        fg_ray_samples = ray_samples
        fg_weights = custom_weights
        if self.training:
            fg_indices = self._select_training_ray_indices(custom_weights, self.config.foreground_train_ray_ratio)
            if fg_indices is not None:
                fg_ray_samples = ray_samples[fg_indices]
                fg_weights = custom_weights[fg_indices]

        feat_indices = None
        feat_ray_samples = ray_samples
        feat_weights = custom_weights
        if render_features and self.training:
            if fg_indices is not None and self.config.feat_train_ray_ratio == self.config.foreground_train_ray_ratio:
                feat_indices = fg_indices
            else:
                feat_indices = self._select_training_ray_indices(custom_weights, self.config.feat_train_ray_ratio)
            if feat_indices is not None:
                feat_ray_samples = ray_samples[feat_indices]
                feat_weights = custom_weights[feat_indices]

        fg_vals = self.feature_field.get_foreground(fg_ray_samples)
        foreground_logits = self.renderer_spread(values=fg_vals, weights=fg_weights)
        del fg_vals

        if render_features:
            feat_vals = self.feature_field.get_feature(feat_ray_samples)
            features = self.renderer_feature(features=feat_vals, weights=feat_weights)
            del feat_vals

        outputs_feature_indices = feat_indices if render_features and feat_indices is not None else None
        outputs_foreground_indices = fg_indices

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "foreground_logits": foreground_logits,
        }
        if render_features:
            outputs["feature"] = features
        if outputs_feature_indices is not None:
            outputs["feature_ray_indices"] = outputs_feature_indices
        if outputs_foreground_indices is not None:
            outputs["foreground_ray_indices"] = outputs_foreground_indices

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

    @staticmethod
    def _foreground_target_index(fg_target: torch.Tensor) -> torch.Tensor:
        if fg_target.ndim == 1:
            return fg_target.to(dtype=torch.long)
        return fg_target.argmax(dim=-1)

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Feature metrics
        feature_ray_indices = outputs.get("feature_ray_indices")
        target_feats = batch["feature"].to(device=self.device, dtype=torch.float32)
        if feature_ray_indices is not None:
            target_feats = target_feats[feature_ray_indices]
        pred_feats = outputs["feature"].to(dtype=torch.float32)
        metrics_dict["feature_error"] = F.mse_loss(pred_feats, target_feats)
        # Foreground metrics
        probs = torch.softmax(outputs["foreground_logits"].to(dtype=torch.float32), dim=-1)
        foreground_ray_indices = outputs.get("foreground_ray_indices")
        fg_target = batch["foreground"].to(self.device)
        if foreground_ray_indices is not None:
            fg_target = fg_target[foreground_ray_indices]
        pred = probs.argmax(dim=-1)
        targ = self._foreground_target_index(fg_target)
        metrics_dict["foreground_acc"] = (pred == targ).float().mean()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        # Feature loss
        feature_ray_indices = outputs.get("feature_ray_indices")
        target_feats = batch["feature"].to(device=self.device, dtype=torch.float32)
        if feature_ray_indices is not None:
            target_feats = target_feats[feature_ray_indices]
        pred_feats = outputs["feature"].to(dtype=torch.float32)
        loss_dict["feature_loss"] = self.config.feat_loss_weight * F.mse_loss(pred_feats, target_feats)
        # Foreground loss
        fg_logits = outputs["foreground_logits"].to(dtype=torch.float32).view(-1, 2)
        foreground_ray_indices = outputs.get("foreground_ray_indices")
        fg_target = batch["foreground"].to(self.device)
        if foreground_ray_indices is not None:
            fg_target = fg_target[foreground_ray_indices]
        fg_target_idx = self._foreground_target_index(fg_target).view(-1)
        ce = F.cross_entropy(fg_logits, fg_target_idx)
        loss_dict["foreground_loss"] = self.config.foreground_loss_weight * ce
        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, render_features: bool = True) -> Dict[str, torch.Tensor]:
        """Full-image render with optional feature computation. Features are kept on CPU."""
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
            outputs_chunk = self._get_outputs_internal(ray_bundle, render_features=render_features)
            for output_name, output in outputs_chunk.items():
                if not torch.is_tensor(output):
                    continue
                if output_name.startswith("feature"):
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

        # Nothing else to do if not CLIP features or no positives
        if self.kwargs["metadata"]["feature_type"] != "CLIP" or not viewer_utils.has_positives:
            return outputs

        # Normalize CLIP features rendered by feature field
        clip_features = outputs["feature"].to(viewer_utils.device)
        outputs["similarity"] = compute_similarity_scores(
            clip_features=clip_features,
            pos_embed=viewer_utils.pos_embed,
            neg_embed=viewer_utils.neg_embed if viewer_utils.has_negatives else None,
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

        # foreground probability
        probs = torch.softmax(outputs["foreground_logits"], dim=-1)[..., 1:2]
        images_dict["foreground_prob_rgb"] = self.prob_from_probs_shader(probs)

        # feature PCA
        if "feature" in outputs:
            images_dict["feature_pca"], viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
                outputs["feature"], viewer_utils.pca_proj
            )

        return metrics_dict, images_dict
