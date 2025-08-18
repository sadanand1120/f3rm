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
from f3rm.renderer import FeatureRenderer, CentroidRenderer
from f3rm.features.clip_extract import CLIPArgs
from f3rm.shaders import CentroidShader


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

    # Centroid-offset head controls (single knob)
    centroid_enable: bool = True  # Enable centroid-offset head (train + render + cache)
    centroid_loss_weight: float = 1e-3  # Loss weight for centroid-offset regression
    centroid_condition_on_density: bool = True  # Condition centroid head on NeRF density embedding
    centroid_condition_density_grad_to_nerf: bool = False  # Allow centroid head gradients into NeRF density embedding
    centroid_hidden_dim: int = 64  # Centroid head hidden dim
    centroid_num_layers: int = 2  # Centroid head num layers
    # Centroid cache helpers (used in pipeline)
    centroid_min_instance_percent: float = 1.0  # Filter small instances when building centroid GT
    centroid_min_accum: float = 0.0  # Min accumulation to consider a pixel valid in centroid GT


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

    def populate_modules(self):
        super().populate_modules()

        feature_dim = self.kwargs["metadata"]["feature_dim"]
        if feature_dim <= 0:
            raise ValueError("Feature dimensionality must be positive.")

        self.feature_field = FeatureField(
            feature_dim=feature_dim,
            spatial_distortion=self.field.spatial_distortion,
            cond_on_density_feature=self.config.feat_condition_on_density,
            cond_on_density_centroid=self.config.centroid_condition_on_density,
            density_embedding_dim=getattr(self.field, "geo_feat_dim", 15),
            feat_grad_to_density=self.config.feat_condition_density_grad_to_nerf,
            centroid_grad_to_density=self.config.centroid_condition_density_grad_to_nerf,
            use_pe=self.config.feat_use_pe,
            pe_n_freq=self.config.feat_pe_n_freq,
            num_levels=self.config.feat_num_levels,
            log2_hashmap_size=self.config.feat_log2_hashmap_size,
            start_res=self.config.feat_start_res,
            max_res=self.config.feat_max_res,
            features_per_level=self.config.feat_features_per_level,
            hidden_dim=self.config.feat_hidden_dim,
            num_layers=self.config.feat_num_layers,
            centroid_hidden_dim=self.config.centroid_hidden_dim,
            centroid_num_layers=self.config.centroid_num_layers,
        )

        self.renderer_feature = FeatureRenderer()
        self.renderer_centroid = CentroidRenderer()
        self.centroid_shader = CentroidShader(self.field.spatial_distortion)
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

        # Feature/Centroid outputs (optionally conditioned on density embedding from NeRF)
        if render_features:
            need_density = self.config.feat_condition_on_density or (self.config.centroid_enable and self.config.centroid_condition_on_density)
            density_embedding = None
            if need_density:
                _, density_embed_raw = self.field.get_density(ray_samples)
                allow_grad = False
                if self.config.feat_condition_on_density and self.config.feat_condition_density_grad_to_nerf:
                    allow_grad = True
                if self.config.centroid_enable and self.config.centroid_condition_on_density and self.config.centroid_condition_density_grad_to_nerf:
                    allow_grad = True
                density_embedding = density_embed_raw if allow_grad else density_embed_raw.detach()
            ff_outputs = self.feature_field(ray_samples, density_embedding=density_embedding)
            features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)
            # Allow centroid forward anytime during eval/inference; gate during training until post cold-start
            if self.config.centroid_enable and (not self.training or getattr(self, "_train_centroid_cache_enabled", False)):
                centroid_offset = self.renderer_centroid(values=ff_outputs[FeatureFieldHeadNames.CENTROID_OFFSET], weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }
        if render_features:
            outputs["feature"] = features
            if self.config.centroid_enable and (not self.training or getattr(self, "_train_centroid_cache_enabled", False)):
                outputs["centroid_offset"] = centroid_offset

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
            # Save ray geometry for centroid supervision
            outputs["ray_origins"] = ray_bundle.origins
            outputs["ray_directions"] = ray_bundle.directions

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
        # Centroid-offset loss (supervision via cache).
        if self.training and self.config.centroid_enable and getattr(self, "_train_centroid_cache_enabled", False):
            full_cache = self._get_train_centroid_cache()  # type: ignore[attr-defined]
            ray_indices = batch["indices"].to(self.device)
            cam_idx = ray_indices[:, 0].long()
            yi = ray_indices[:, 1].long()
            xi = ray_indices[:, 2].long()

            # Strict: all cameras in batch must be present in cache
            unique_cams = torch.unique(cam_idx).tolist()
            for ci in unique_cams:
                assert int(ci) in full_cache, f"Centroid cache missing camera {int(ci)}"

            # Assemble GT on CPU per camera, then move picks to GPU (avoids moving full images to GPU)
            num_rays = cam_idx.shape[0]
            gt_centroids = torch.zeros((num_rays, 3), device=self.device, dtype=outputs["depth"].dtype)
            valid_mask_t = torch.zeros((num_rays,), device=self.device, dtype=torch.bool)
            with torch.no_grad():
                for ci in unique_cams:
                    ci_int = int(ci)
                    sel = (cam_idx == ci)
                    idxs = torch.nonzero(sel, as_tuple=True)[0]
                    if idxs.numel() == 0:
                        continue
                    y_sel = yi[idxs].cpu().long()
                    x_sel = xi[idxs].cpu().long()
                    c_img_cpu, v_img_cpu = full_cache[ci_int]
                    # c_img_cpu: HxWx3 float32 (CPU), v_img_cpu: HxWx1 bool (CPU)
                    c_pick = c_img_cpu[y_sel, x_sel]              # Nx3 (CPU)
                    v_pick = v_img_cpu[y_sel, x_sel].squeeze(-1)  # N (CPU) bool
                    gt_centroids[idxs] = c_pick.to(self.device, non_blocking=True)
                    valid_mask_t[idxs] = v_pick.to(self.device, non_blocking=True)

            # World point via median depth
            w = outputs["ray_origins"].to(self.device) + outputs["ray_directions"].to(self.device) * outputs["depth"].to(self.device)
            pred_offset = outputs["centroid_offset"].to(self.device)
            gt_offset = w - gt_centroids

            if valid_mask_t.any():
                sl1 = F.smooth_l1_loss(pred_offset[valid_mask_t], gt_offset[valid_mask_t])
                loss_dict["centroid_offset_loss"] = self.config.centroid_loss_weight * sl1
                if metrics_dict is not None:
                    l2 = torch.linalg.norm(pred_offset[valid_mask_t] - gt_offset[valid_mask_t], dim=-1).mean()
                    metrics_dict["centroid_offset_l2"] = l2
                    metrics_dict["centroid_valid_fraction"] = valid_mask_t.float().mean()
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

        # Append centroid prediction visualization if enabled
        # Allow during eval/inference regardless of cold-start; gate during training until post cold-start
        if self.config.centroid_enable and (not self.training or getattr(self, "_train_centroid_cache_enabled", False)) and "centroid_offset" in outputs:
            w = camera_ray_bundle.origins + camera_ray_bundle.directions * outputs["depth"]
            centroid_pred = w - outputs["centroid_offset"]
            outputs["centroid_pred"] = centroid_pred
            outputs["centroid_pred_rgb"] = self.centroid_shader(centroid_pred)

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
        if "centroid_pred_rgb" in outputs:
            images_dict["centroid_pred_rgb"] = outputs["centroid_pred_rgb"]
        return metrics_dict, images_dict
