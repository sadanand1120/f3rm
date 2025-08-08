from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Type, Tuple

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
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TextColumn
import time
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
    # Centroid regression
    centroid_loss_weight: float = 0.5
    """Weight for centroid regression loss."""
    do_sam2_after_steps: int = 5000
    """Start centroid regression after this many training steps (cold-start)."""


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

        feat_type = self.kwargs["metadata"]["feature_type"]
        feature_dim = self.kwargs["metadata"]["feature_dim"]
        self._do_sam2 = self.kwargs["metadata"].get("do_sam2", False)
        if feature_dim <= 0:
            raise ValueError("Feature dimensionality must be positive.")

        # Always create centroid head if SAM2 is configured (even during cold-start)
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
            enable_centroid=self._do_sam2,  # Create head if configured, regardless of step
        )

        self.renderer_feature = FeatureRenderer()

        # Centroid cache (per-batch of images). Keys: (image_name, instance_id) -> centroid (3,)
        self._centroid_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self._cached_images: set[str] = set()
        self._index_maps_built: bool = False
        self._train_name_to_idx: Dict[str, int] = {}
        self._eval_name_to_idx: Dict[str, int] = {}
        self._centroid_started: bool = False  # Track if we've announced centroid start
        self._current_step: int = 0  # Track training step ourselves

        # Announce cold-start configuration
        if self._do_sam2:
            CONSOLE.print(f"[SAM2] Centroid regression configured with cold-start: will begin at step {self.config.do_sam2_after_steps}")

        self.setup_gui()

    @property
    def enable_centroid(self) -> bool:
        """Dynamic property that enables centroid regression after cold-start period."""
        if not self._do_sam2:
            return False

        should_enable = self._current_step >= self.config.do_sam2_after_steps

        # One-time announcement when centroid regression starts
        if should_enable and not self._centroid_started:
            CONSOLE.print(f"[SAM2] Starting centroid regression at step {self._current_step} (cold-start period complete)")
            self._centroid_started = True

        return should_enable

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

    def _compute_centroid_targets(self, outputs: Dict, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute centroid-offset targets using cached per-image centroids under no_grad.

        The cache is refreshed once whenever the set of image_names in the batch changes.
        Returns (targets, mask). All computations here are detached to avoid gradient circularity.
        """
        if not ("depth" in outputs and "ray_origins" in outputs and "ray_directions" in outputs
                and "image_name" in batch and "sam2_mask" in batch):
            # Skip this step if inputs aren't ready (timing issue during cold-start activation)
            n = len(batch.get("image_name", []))
            return torch.zeros((n, 3), device=self.device), torch.zeros(n, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # Refresh cache if needed for current batch images
            self._maybe_refresh_centroid_cache(batch)

            depths = outputs["depth"].squeeze(-1)
            ray_origins = outputs["ray_origins"]
            ray_directions = outputs["ray_directions"]
            image_names = batch["image_name"]
            sam2_masks = batch["sam2_mask"]

            # Compute 3D positions once
            positions = ray_origins + depths.unsqueeze(-1) * ray_directions  # (N,3)
            n = positions.shape[0]

            targets = torch.zeros((n, 3), device=self.device)
            mask = torch.zeros(n, dtype=torch.bool, device=self.device)

            for i in range(n):
                instance_id = int(sam2_masks[i].item()) if torch.is_tensor(sam2_masks) else int(sam2_masks[i])
                if instance_id <= 0:
                    continue
                key = (image_names[i], instance_id)
                centroid = self._centroid_cache.get(key)
                if centroid is None:
                    continue
                targets[i] = centroid - positions[i]
                mask[i] = True

        return targets, mask

    def _build_index_maps_if_needed(self) -> None:
        if getattr(self, '_index_maps_built', False):
            return
        dm = getattr(getattr(self, "_trainer", None), "pipeline", None)
        if dm is None:
            return
        dm = dm.datamanager
        train_files = [str(p) for p in dm.train_dataset.image_filenames]
        eval_files = [str(p) for p in dm.eval_dataset.image_filenames]
        self._train_name_to_idx = {fn: i for i, fn in enumerate(train_files)}
        self._eval_name_to_idx = {fn: i for i, fn in enumerate(eval_files)}
        self._index_maps_built = True

    def _lookup_image_indices(self, image_name: str) -> Tuple[str, int, int]:
        """Return (split, local_idx, global_idx) for a given image name."""
        dm = getattr(getattr(self, "_trainer", None), "pipeline", None)
        if dm is None:
            raise RuntimeError("Trainer/datamanager not available")
        dm = dm.datamanager
        self._build_index_maps_if_needed()

        if image_name in self._train_name_to_idx:
            li = self._train_name_to_idx[image_name]
            return ("train", int(li), int(li))
        if image_name in self._eval_name_to_idx:
            li = self._eval_name_to_idx[image_name]
            gi = li + len(dm.train_dataset)
            return ("eval", int(li), int(gi))

        raise KeyError(f"Image name not found in datasets: {image_name}")

    def _maybe_refresh_centroid_cache(self, batch: Dict) -> None:
        if not self.enable_centroid:
            return
        image_names: List[str] = batch.get("image_name", [])
        if not image_names:
            return
        current = set(image_names)
        if current == self._cached_images:
            return

        # New batch encountered: recompute cache for these images
        self._centroid_cache.clear()
        self._cached_images = current
        if not hasattr(self, "_trainer"):
            return
        self._build_index_maps_if_needed()
        num_images = len(current)
        CONSOLE.print(f"[SAM2] Recomputing per-image centroids for {num_images} images")
        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Computing centroids", total=num_images)
            start_all = time.perf_counter()
            for name in current:
                self._compute_and_store_image_centroids(name)
                progress.update(task, advance=1)
            CONSOLE.print(f"[SAM2] Centroid cache refresh complete in {time.perf_counter()-start_all:.2f}s")

    @torch.no_grad()
    def _compute_and_store_image_centroids(self, image_name: str) -> None:
        dm = getattr(getattr(self, "_trainer", None), "pipeline", None)
        if dm is None:
            return
        dm = dm.datamanager
        split, local_idx, global_idx = self._lookup_image_indices(image_name)

        cameras = dm.train_dataset.cameras if split == "train" else dm.eval_dataset.cameras
        cameras = cameras.to(self.device)

        # Full-image rays and depth render
        rb = cameras.generate_rays(camera_indices=int(local_idx)).to(self.device)
        # Use base implementation to avoid viewer PCA and extra work
        outputs = super().get_outputs_for_camera_ray_bundle(rb)
        depth = outputs.get("depth", None)
        if depth is None:
            return
        depth = depth.squeeze(-1)  # (H,W)

        origins = rb.origins.squeeze(0)  # (H,W,3)
        directions = rb.directions.squeeze(0)  # (H,W,3)
        positions = origins + directions * depth.unsqueeze(-1)  # (H,W,3)

        # Load SAM2 mask for this image (global index over train+eval)
        sam2 = getattr(dm, "sam2_masks", None)
        if sam2 is None:
            return

        # LazyFeatures expects (image_idx, y, x) triple, but we need full image
        # Access the underlying shard data directly
        sid, shard_local_idx = sam2._loc(int(global_idx))
        shard = sam2._get_shard(sid)
        mask = shard[shard_local_idx]  # (H, W, 1) numpy array

        if isinstance(mask, torch.Tensor):
            mask = mask.to(self.device)
        else:
            mask = torch.as_tensor(mask, device=self.device)

        # Ensure mask is (H,W)
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask[0]

        # Resize mask to match depth resolution if needed (nearest)
        H_img, W_img = depth.shape[-2], depth.shape[-1]
        if mask.shape != depth.shape:
            mask_ = mask.unsqueeze(0).unsqueeze(0).float()
            mask = F.interpolate(mask_, size=(H_img, W_img), mode="nearest").squeeze().long()
        else:
            mask = mask.long()

        # Compute per-instance centroids
        pos_flat = positions.view(-1, 3)
        ids_flat = mask.view(-1)
        unique_ids = torch.unique(ids_flat)
        cached = 0
        for iid in unique_ids.tolist():
            if iid <= 0:
                continue
            sel = ids_flat == iid
            if not torch.any(sel):
                continue
            centroid = pos_flat[sel].mean(dim=0)
            self._centroid_cache[(image_name, int(iid))] = centroid.detach()
            cached += 1

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["feature_field"] = list(self.feature_field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        """Modified from nerfacto.get_outputs to include feature field outputs."""
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

        _, density_embedding = self.field.get_density(ray_samples)
        ff_outputs = self.feature_field(ray_samples, density_embedding=density_embedding)
        features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)

        outputs: Dict[str, torch.Tensor] = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "feature": features,
        }

        # Expose per-ray positions ingredients for centroid supervision if enabled
        if self.enable_centroid:
            # Use the input ray bundle per-ray origins and directions
            outputs["ray_origins"] = ray_bundle.origins.view(-1, 3)
            outputs["ray_directions"] = ray_bundle.directions.view(-1, 3)

        # Add centroid outputs if enabled
        if self.enable_centroid:
            centroids = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.CENTROID], weights=weights)
            outputs["centroid"] = centroids

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

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        target_feats = batch["feature"].to(self.device)
        metrics_dict["feature_error"] = F.mse_loss(outputs["feature"], target_feats)

        # Add centroid metrics with on-the-fly batch centroids
        if self.enable_centroid:
            centroid_targets, centroid_mask = self._compute_centroid_targets(outputs, batch)

            if centroid_mask.any():
                pred_centroids = outputs["centroid"][centroid_mask]
                target_centroids = centroid_targets[centroid_mask]
                metrics_dict["centroid_error"] = F.mse_loss(pred_centroids, target_centroids)
                metrics_dict["num_centroid_targets"] = centroid_mask.sum().float()
            else:
                metrics_dict["centroid_error"] = torch.tensor(0.0, device=self.device)
                metrics_dict["num_centroid_targets"] = torch.tensor(0.0, device=self.device)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        # Increment our step counter during training
        if self.training:
            self._current_step += 1

        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        target_feats = batch["feature"].to(self.device)
        loss_dict["feature_loss"] = self.config.feat_loss_weight * F.mse_loss(outputs["feature"], target_feats)

        # Handle centroid loss with on-the-fly batch centroids
        if self.enable_centroid:
            centroid_targets, centroid_mask = self._compute_centroid_targets(outputs, batch)

            # Compute loss only on valid centroid targets
            if centroid_mask.any():
                pred_centroids = outputs["centroid"][centroid_mask]
                target_centroids = centroid_targets[centroid_mask]
                loss_dict["centroid_loss"] = self.config.centroid_loss_weight * F.smooth_l1_loss(
                    pred_centroids, target_centroids
                )
            else:
                loss_dict["centroid_loss"] = torch.tensor(0.0, device=self.device)

        return loss_dict

    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        # Compute PCA of features separately, so we can reuse the same projection matrix
        outputs["feature_pca"], viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
            outputs["feature"], viewer_utils.pca_proj
        )

        # Nothing else to do if not CLIP features or no positives
        if self.kwargs["metadata"]["feature_type"] != "CLIP" or not viewer_utils.has_positives:
            return outputs

        # Normalize CLIP features rendered by feature field
        clip_features = outputs["feature"]   # is on cpu() because of your base_model.py modification to nerfstudio code
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
