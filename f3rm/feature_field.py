from typing import Dict, Optional, Tuple

import numpy as np
import torch
import tinycudann as tcnn
from jaxtyping import Float, Shaped
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from torch import Tensor


class FeatureFieldHeadNames:
    FEATURE: str = "feature"
    CENTROID: str = "centroid"


class FeatureField(Field):
    def __init__(
        self,
        feature_dim: int,
        spatial_distortion: SpatialDistortion,
        # Positional encoding
        use_pe: bool = True,
        pe_n_freq: int = 6,
        # Hash grid
        num_levels: int = 12,
        log2_hashmap_size: int = 19,
        start_res: int = 16,
        max_res: int = 128,
        features_per_level: int = 8,
        # MLP head
        hidden_dim: int = 64,
        num_layers: int = 2,
        # Centroid regression
        enable_centroid: bool = False,
        geo_feat_dim: int = 15,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_distortion = spatial_distortion
        self.enable_centroid = enable_centroid
        # Store parameters for dimension calculation
        self.use_pe = use_pe
        self.pe_n_freq = pe_n_freq
        self.feat_num_levels = num_levels
        self.feat_features_per_level = features_per_level
        self.geo_feat_dim = geo_feat_dim

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

        # Create position encoder for feature head
        self.position_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=encoding_config,
        )

        # Main feature network - uses position encoding + density embedding
        feature_input_dim = self.position_encoder.n_output_dims + geo_feat_dim  # position encoding + density embedding
        self.field = tcnn.Network(
            n_input_dims=feature_input_dim,
            n_output_dims=self.feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers,
            },
        )

        # Optional centroid regression head (3D offset vector)
        if self.enable_centroid:
            # Create position encoder for centroid head
            self.centroid_position_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

            # Centroid head uses position encoding + density embedding
            # Get the actual output dimension from the encoder
            centroid_input_dim = self.centroid_position_encoder.n_output_dims + geo_feat_dim  # position encoding + density embedding

            self.centroid_head = tcnn.Network(
                n_input_dims=centroid_input_dim,
                n_output_dims=3,  # 3D offset vector
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim // 2,  # Smaller network for centroid
                    "n_hidden_layers": max(1, num_layers - 1),
                },
            )

    def get_density(
        self, ray_samples: RaySamples
    ) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        raise NotImplementedError("get_density not supported for FeatureField")

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None, "density_embedding is required for FeatureField"

        # Apply scene contraction
        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)   # Apply scene contraction (same as nerfacto field) with L_inf, maps to a cube [-2,2]^3
        positions = (positions + 2.0) / 4.0    # Remaps from [-2, 2] → [0, 1], Required for HashGrid encoding, which expects input coordinates in [0, 1]
        positions_flat = positions.view(-1, 3)
        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # Encode positions for feature head
        position_encoded = self.position_encoder(positions_flat)
        # Combine position encoding with density embedding for main features
        feature_input = torch.cat([position_encoded, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
        features = self.field(feature_input).view(*outputs_shape, -1)
        outputs = {FeatureFieldHeadNames.FEATURE: features}

        # Get centroid predictions if enabled (using density embedding for better geometry awareness)
        if self.enable_centroid:
            # Encode positions for centroid head
            centroid_position_encoded = self.centroid_position_encoder(positions_flat)
            # Combine position encoding with density embedding for centroid head
            centroid_input = torch.cat([centroid_position_encoded, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
            centroid_offset = self.centroid_head(centroid_input).view(*outputs_shape, 3)
            outputs[FeatureFieldHeadNames.CENTROID] = centroid_offset

        return outputs

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False, density_embedding: Optional[Tensor] = None) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals:
            raise ValueError("FeatureField does not support computing normals")
        return self.get_outputs(ray_samples, density_embedding=density_embedding)
