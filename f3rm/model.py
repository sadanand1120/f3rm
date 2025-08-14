from dataclasses import dataclass, field
from collections import defaultdict
from functools import cached_property
from typing import Dict, List, Optional, Type

import open_clip
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

from f3rm.feature_field import FeatureField, FeatureFieldHeadNames
from f3rm.pca_colormap import apply_pca_colormap_return_proj
from f3rm.renderer import FeatureRenderer
from f3rm.features.clip_extract import CLIPArgs


@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    """Note: make sure to use naming that doesn't conflict with NerfactoModelConfig"""

    _target: Type = field(default_factory=lambda: FeatureFieldModel)
    # Weighing for the feature loss
    feat_loss_weight: float = 1e-3
    # Condition Feature Field on NeRF density embedding (geo features)
    feat_condition_on_density: bool = True
    # Allow gradients from feature field to flow back into NeRF via density embedding
    feat_condition_density_grad_to_nerf: bool = False
    # Feature Field Positional Encoding
    feat_use_pe: bool = True
    feat_pe_n_freq: int = 6
    # Feature Field Hash Grid
    feat_num_levels: int = 12
    feat_log2_hashmap_size: int = 19
    feat_start_res: int = 16
    feat_max_res: int = 128
    feat_features_per_level: int = 8
    # Feature Field MLP Head
    feat_hidden_dim: int = 64
    feat_num_layers: int = 2


@dataclass
class ViewerUtils:
    pca_proj: Optional[torch.Tensor] = None
    positives: List[str] = field(default_factory=list)
    pos_embed: Optional[torch.Tensor] = None
    negatives: List[str] = field(default_factory=list)
    neg_embed: Optional[torch.Tensor] = None
    softmax_temp: float = 0.1
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @cached_property
    def clip(self):
        CONSOLE.print(f"Loading CLIP {CLIPArgs.model_name} for viewer")
        model, _, _ = open_clip.create_model_and_transforms(CLIPArgs.model_name, pretrained=CLIPArgs.model_pretrained, device=self.device)
        model.eval()
        return model

    @torch.no_grad()
    def handle_language_queries(self, raw_text: str, is_positive: bool):
        """Compute CLIP embeddings based on queries and update state"""
        texts = [x.strip() for x in raw_text.split(",") if x.strip()]
        # Clear the GUI state if there are no texts
        if not texts:
            self.clear_positives() if is_positive else self.clear_negatives()
            return
        # Embed text queries
        tokenize = open_clip.get_tokenizer(CLIPArgs.model_name)
        tokens = tokenize(texts).to(self.device)
        embed = self.clip.encode_text(tokens).float()
        if is_positive:
            self.positives = texts
            # Average embedding if we have multiple positives
            embed = embed.mean(dim=0, keepdim=True)
            embed /= embed.norm(dim=-1, keepdim=True)
            self.pos_embed = embed
        else:
            self.negatives = texts
            # We don't average the negatives as we compute pair-wise softmax
            embed /= embed.norm(dim=-1, keepdim=True)
            self.neg_embed = embed

    @property
    def has_positives(self) -> bool:
        return self.positives and self.pos_embed is not None

    def clear_positives(self):
        self.positives.clear()
        self.pos_embed = None

    @property
    def has_negatives(self) -> bool:
        return self.negatives and self.neg_embed is not None

    def clear_negatives(self):
        self.negatives.clear()
        self.neg_embed = None

    def update_softmax_temp(self, temp: float):
        self.softmax_temp = temp

    def reset_pca_proj(self):
        self.pca_proj = None
        CONSOLE.print("Reset PCA projection")


viewer_utils = ViewerUtils(device=torch.device("cpu"))


class FeatureFieldModel(NerfactoModel):
    config: FeatureFieldModelConfig

    feature_field: FeatureField
    renderer_feature: FeatureRenderer
    _train_depth_cache_enabled: bool = False

    # Default no-op cache accessor; pipeline overrides this with a real function
    def _get_train_depth_cache(self):
        return {}

    def populate_modules(self):
        super().populate_modules()

        feature_dim = self.kwargs["metadata"]["feature_dim"]
        if feature_dim <= 0:
            raise ValueError("Feature dimensionality must be positive.")

        self.feature_field = FeatureField(
            feature_dim=feature_dim,
            spatial_distortion=self.field.spatial_distortion,
            cond_on_density=self.config.feat_condition_on_density,
            density_embedding_dim=getattr(self.field, "geo_feat_dim", 15),
            use_pe=self.config.feat_use_pe,
            pe_n_freq=self.config.feat_pe_n_freq,
            num_levels=self.config.feat_num_levels,
            log2_hashmap_size=self.config.feat_log2_hashmap_size,
            start_res=self.config.feat_start_res,
            max_res=self.config.feat_max_res,
            features_per_level=self.config.feat_features_per_level,
            hidden_dim=self.config.feat_hidden_dim,
            num_layers=self.config.feat_num_layers,
        )

        self.renderer_feature = FeatureRenderer()
        self.setup_gui()

    def setup_gui(self):
        viewer_utils.device = self.kwargs["device"]
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

    def _get_outputs_internal(self, ray_bundle: RayBundle, render_features: bool):
        """Core rendering that can optionally skip feature-field computation."""
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

        # Feature outputs (optionally conditioned on density embedding from NeRF)
        if render_features:
            density_embedding = None
            if self.config.feat_condition_on_density:
                _, density_embedding = self.field.get_density(ray_samples)
                if not self.config.feat_condition_density_grad_to_nerf:
                    density_embedding = density_embedding.detach()
            ff_outputs = self.feature_field(ray_samples, density_embedding=density_embedding)
            features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }
        if render_features:
            outputs["feature"] = features

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
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_outputs(self, ray_bundle: RayBundle):
        """Modified from nerfacto.get_outputs to include feature field outputs."""
        return self._get_outputs_internal(ray_bundle, render_features=True)

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        target_feats = batch["feature"].to(self.device)
        metrics_dict["feature_error"] = F.mse_loss(outputs["feature"], target_feats)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        target_feats = batch["feature"].to(self.device)
        loss_dict["feature_loss"] = self.config.feat_loss_weight * F.mse_loss(outputs["feature"], target_feats)
        # Dummy check: only during training, assert cached depth exists and is HxWx1
        if self.training and self._train_depth_cache_enabled:
            full_cache = self._get_train_depth_cache()  # type: ignore[attr-defined]
            cam_idx = int(torch.mode(batch["indices"][:, 0].to(self.device)).values.item())
            assert cam_idx in full_cache, f"train depth cache missing camera {cam_idx}"
            depth_img = full_cache[cam_idx]
            assert depth_img.ndim == 3 and depth_img.shape[-1] == 1, "cached depth must be HxWx1"
            # Also verify cache covers all cameras present in this batch
            unique_cams = torch.unique(batch["indices"][:, 0]).tolist()
            missing = [int(ci) for ci in unique_cams if int(ci) not in full_cache]
            assert not missing, f"train depth cache missing cameras {missing}; expected all of {list(map(int, unique_cams))}"
        return loss_dict

    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, render_features: bool = True) -> Dict[str, torch.Tensor]:
        """Full-image render with optional feature computation.

        render_features=False will skip feature-field computation for speed (used by cache renders).
        """
        with torch.no_grad():
            num_rays_per_chunk = self.config.eval_num_rays_per_chunk
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
            num_rays = len(camera_ray_bundle)
            outputs_lists = defaultdict(list)
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                if self.collider is not None:
                    ray_bundle = self.collider(ray_bundle)
                outputs_chunk = self._get_outputs_internal(ray_bundle, render_features=render_features)
                for output_name, output in outputs_chunk.items():
                    if not torch.is_tensor(output):
                        continue
                    if output_name.startswith("feature"):
                        outputs_lists[output_name].append(output.cpu())
                    else:
                        outputs_lists[output_name].append(output)
                    del output
                if (i // num_rays_per_chunk) % 20 == 0:
                    torch.cuda.empty_cache()
            outputs: Dict[str, torch.Tensor] = {}
            for output_name, outputs_list in outputs_lists.items():
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)

        # If requested depth-only render, exit early
        if not render_features:
            return outputs

        # Compute PCA of features separately, so we can reuse the same projection matrix
        outputs["feature_pca"], viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
            outputs["feature"], viewer_utils.pca_proj
        )

        # Nothing else to do if not CLIP features or no positives
        if self.kwargs["metadata"]["feature_type"] != "CLIP" or not viewer_utils.has_positives:
            return outputs

        # Normalize CLIP features rendered by feature field
        clip_features = outputs["feature"]   # is on cpu()
        clip_features = clip_features.to(viewer_utils.device)
        clip_features /= clip_features.norm(dim=-1, keepdim=True)

        # If there are no negatives, just show the cosine similarity with the positives
        if not viewer_utils.has_negatives:
            sims = clip_features @ viewer_utils.pos_embed.T
            # Show the mean similarity if there are multiple positives
            if sims.shape[-1] > 1:
                sims = sims.mean(dim=-1, keepdim=True)
            outputs["similarity"] = sims
            return outputs

        # Use paired softmax method as described in the paper with positive and negative texts
        text_embs = torch.cat([viewer_utils.pos_embed, viewer_utils.neg_embed], dim=0)
        raw_sims = clip_features @ text_embs.T

        # Broadcast positive label similarities to all negative labels
        pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
        pos_sims = pos_sims.broadcast_to(neg_sims.shape)
        paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)

        # Compute paired softmax
        probs = (paired_sims / viewer_utils.softmax_temp).softmax(dim=-1)[..., :1]
        torch.nan_to_num_(probs, nan=0.0)
        sims, _ = probs.min(dim=-1, keepdim=True)
        outputs["similarity"] = sims
        return outputs

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        if "feature_pca" in outputs:
            images_dict["feature_pca"] = outputs["feature_pca"]
        return metrics_dict, images_dict
