from typing import Dict, Optional, Tuple

import numpy as np
import tinycudann as tcnn
from jaxtyping import Float, Shaped
import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from torch import Tensor


class FeatureFieldHeadNames:
    FEATURE: str = "feature"
    CENTROID: str = "centroid"
    CENTROID_SPREAD: str = "centroid_spread"
    FOREGROUND: str = "foreground"


class FeatureField(Field):
    def __init__(
        self,
        feature_dim: int,
        spatial_distortion: SpatialDistortion,
        # Density embedding conditioning
        cond_on_density_feature: bool = True,
        cond_on_density_centroid: bool = True,
        cond_on_density_foreground: bool = True,
        density_embedding_dim: int = 15,
        # Per-head grad flow controls for density embedding
        feat_grad_to_density: bool = False,
        centroid_grad_to_density: bool = False,
        foreground_grad_to_density: bool = False,
        # Positional encoding
        use_pe: bool = True,
        pe_n_freq: int = 6,
        # Hash grid
        num_levels: int = 12,
        log2_hashmap_size: int = 19,
        start_res: int = 16,
        max_res: int = 128,
        features_per_level: int = 8,
        # Feature head MLP
        hidden_dim: int = 64,
        num_layers: int = 2,
        # Centroid head MLP (predicts world-space centroids directly)
        centroid_hidden_dim: int = 64,
        centroid_num_layers: int = 2,
        # Foreground head MLP (binary classification with 2 logits)
        foreground_hidden_dim: int = 64,
        foreground_num_layers: int = 2,
        # Optional trunk tap from centroid-spread encoder
        centroid_spread_trunk_fg: int = 0,
        foreground_trunk_grad_to_spread: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_distortion = spatial_distortion
        self.cond_on_density_feature = cond_on_density_feature
        self.cond_on_density_centroid = cond_on_density_centroid
        self.cond_on_density_foreground = cond_on_density_foreground
        self.feat_grad_to_density = feat_grad_to_density
        self.centroid_grad_to_density = centroid_grad_to_density
        self.foreground_grad_to_density = foreground_grad_to_density
        # Foreground head shares the same positional encoding as other heads (no direction encoding)
        self.centroid_spread_trunk_fg = int(centroid_spread_trunk_fg)
        self.foreground_trunk_grad_to_spread = bool(foreground_trunk_grad_to_spread)

        # Feature field has its own hash grid
        growth_factor = np.exp((np.log(max_res) - np.log(start_res)) / (num_levels - 1))
        encoding_config = {
            "otype": "Composite",
            "nested": [
                {
                    "otype": "HashGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": start_res,
                    "per_level_scale": growth_factor,
                }
            ],
        }

        if use_pe:
            encoding_config["nested"].append(
                {
                    "otype": "Frequency",
                    "n_frequencies": pe_n_freq,
                    "n_dims_to_encode": 3,
                }
            )

        # Separate encoding and MLPs so we can concatenate density embeddings post-encoding per head
        self.encoding = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        mlp_in_dims_feature = self.encoding.n_output_dims + (density_embedding_dim if self.cond_on_density_feature else 0)
        self.mlp_feature = tcnn.Network(
            n_input_dims=mlp_in_dims_feature,
            n_output_dims=self.feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers,
            },
        )

        mlp_in_dims_centroid = self.encoding.n_output_dims + (density_embedding_dim if self.cond_on_density_centroid else 0)
        self.mlp_centroid = tcnn.Network(
            n_input_dims=mlp_in_dims_centroid,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": centroid_hidden_dim,
                "n_hidden_layers": centroid_num_layers,
            },
        )

        # Centroid spread head (scalar) - outputs both spread predictions and trunk features for foreground
        mlp_in_dims_spread = self.encoding.n_output_dims + (density_embedding_dim if self.cond_on_density_centroid else 0)
        self.mlp_centroid_spread = tcnn.Network(
            n_input_dims=mlp_in_dims_spread,
            n_output_dims=2 + (self.centroid_spread_trunk_fg if self.centroid_spread_trunk_fg > 0 else 0),
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": centroid_hidden_dim,
                "n_hidden_layers": centroid_num_layers,
            },
        )

        # Foreground classification head (2 logits with optional direction + density conditioning)
        mlp_in_dims_foreground = self.encoding.n_output_dims + (density_embedding_dim if self.cond_on_density_foreground else 0) + (self.centroid_spread_trunk_fg if self.centroid_spread_trunk_fg > 0 else 0)
        self.mlp_foreground = tcnn.Network(
            n_input_dims=mlp_in_dims_foreground,
            n_output_dims=2,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": foreground_hidden_dim,
                "n_hidden_layers": foreground_num_layers,
            },
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        raise NotImplementedError("get_density not supported for FeatureField")

    def _encode_positions(self, ray_samples: RaySamples) -> Tensor:
        """Apply scene contraction and encode positions once."""
        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)   # Apply scene contraction (same as nerfacto field) with L_inf, maps to a cube [-2,2]^3
        positions = (positions + 2.0) / 4.0    # Remaps from [-2, 2] → [0, 1], Required for HashGrid encoding, which expects input coordinates in [0, 1]
        positions_flat = positions.view(-1, 3)
        # Encode positions and concatenate density embedding if enabled
        encoded_base = self.encoding(positions_flat)
        return encoded_base

    def get_feature(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Tensor:
        encoded_base = self._encode_positions(ray_samples)
        encoded_feat = encoded_base
        if self.cond_on_density_feature and density_embedding is not None:
            cond = density_embedding.view(-1, density_embedding.shape[-1]).to(encoded_base)
            if not self.feat_grad_to_density:
                cond = cond.detach()
            encoded_feat = torch.cat([encoded_feat, cond], dim=-1)
        features = self.mlp_feature(encoded_feat).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return features

    def get_centroid(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Tensor:
        encoded_base = self._encode_positions(ray_samples)
        encoded_cent = encoded_base
        if self.cond_on_density_centroid and density_embedding is not None:
            cond = density_embedding.view(-1, density_embedding.shape[-1]).to(encoded_base)
            if not self.centroid_grad_to_density:
                cond = cond.detach()
            encoded_cent = torch.cat([encoded_cent, cond], dim=-1)
        centroid = self.mlp_centroid(encoded_cent).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return centroid

    def get_centroid_spread(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None, return_full: bool = False) -> Tensor:
        encoded_base = self._encode_positions(ray_samples)
        encoded_spread = encoded_base
        if self.cond_on_density_centroid and density_embedding is not None:
            cond = density_embedding.view(-1, density_embedding.shape[-1]).to(encoded_base)
            if not self.centroid_grad_to_density:
                cond = cond.detach()
            encoded_spread = torch.cat([encoded_spread, cond], dim=-1)
        spread_full = self.mlp_centroid_spread(encoded_spread).view(*ray_samples.frustums.directions.shape[:-1], -1)
        # Return full output if requested, otherwise only first 2 channels for backward compatibility
        return spread_full if return_full else spread_full[..., :2]

    def get_foreground(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Tensor:
        encoded_base = self._encode_positions(ray_samples)
        parts = [encoded_base]
        if self.cond_on_density_foreground and density_embedding is not None:
            cond = density_embedding.view(-1, density_embedding.shape[-1]).to(encoded_base)
            if not self.foreground_grad_to_density:
                cond = cond.detach()
            parts.append(cond)
        # Append optional centroid-spread trunk features if enabled
        if self.centroid_spread_trunk_fg > 0:
            # Get the full centroid spread output and extract trunk features
            spread_full = self.get_centroid_spread(ray_samples, density_embedding=density_embedding, return_full=True)
            trunk_feat = spread_full[..., 2:2 + self.centroid_spread_trunk_fg]
            # Flatten trunk features to match encoded_base shape for concatenation
            trunk_feat = trunk_feat.view(-1, self.centroid_spread_trunk_fg)
            if not self.foreground_trunk_grad_to_spread:
                trunk_feat = trunk_feat.detach()
            parts.append(trunk_feat)
        encoded_fg = torch.cat(parts, dim=-1)
        logits = self.mlp_foreground(encoded_fg).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return logits

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Dict[FieldHeadNames, Tensor]:
        """Backward-compatible method that computes both heads."""
        features = self.get_feature(ray_samples, density_embedding=density_embedding)
        centroid = self.get_centroid(ray_samples, density_embedding=density_embedding)
        centroid_spread = self.get_centroid_spread(ray_samples, density_embedding=density_embedding)
        foreground = self.get_foreground(ray_samples, density_embedding=density_embedding)
        return {
            FeatureFieldHeadNames.FEATURE: features,
            FeatureFieldHeadNames.CENTROID: centroid,
            FeatureFieldHeadNames.CENTROID_SPREAD: centroid_spread,
            FeatureFieldHeadNames.FOREGROUND: foreground,
        }

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False, density_embedding: Optional[Tensor] = None) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals:
            raise ValueError("FeatureField does not support computing normals")
        return self.get_outputs(ray_samples, density_embedding=density_embedding)
