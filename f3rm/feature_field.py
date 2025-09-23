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
    ORIENTANY: str = "orientany"


class FeatureField(Field):
    def __init__(
        self,
        feature_dim: int,
        spatial_distortion: SpatialDistortion,
        use_pe: bool = True,
        pe_n_freq: int = 6,
        num_levels: int = 12,
        log2_hashmap_size: int = 19,
        start_res: int = 16,
        max_res: int = 128,
        features_per_level: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        centroid_hidden_dim: int = 64,
        centroid_num_layers: int = 2,
        foreground_hidden_dim: int = 64,
        foreground_num_layers: int = 1,
        orientany_hidden_dim: int = 64,
        orientany_num_layers: int = 1,
        orientany_use_xyz_encoding: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_distortion = spatial_distortion
        self.orientany_use_xyz_encoding = bool(orientany_use_xyz_encoding)
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

        self.encoding = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        self.mlp_feature = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=self.feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers,
            },
        )

        self.mlp_centroid = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": centroid_hidden_dim,
                "n_hidden_layers": centroid_num_layers,
            },
        )

        self.mlp_centroid_spread = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=4,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": centroid_hidden_dim,
                "n_hidden_layers": centroid_num_layers,
            },
        )

        self.mlp_foreground = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=2,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": foreground_hidden_dim,
                "n_hidden_layers": foreground_num_layers,
            },
        )

        if not self.orientany_use_xyz_encoding:
            self.orientany_centroid_encoding = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        else:
            self.orientany_centroid_encoding = None
        orientany_input_dims = self.orientany_centroid_encoding.n_output_dims if self.orientany_centroid_encoding is not None else self.encoding.n_output_dims
        self.mlp_orientany_azimuth = tcnn.Network(
            n_input_dims=orientany_input_dims,
            n_output_dims=360,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": orientany_hidden_dim,
                "n_hidden_layers": orientany_num_layers,
            },
        )

        self.mlp_orientany_polar = tcnn.Network(
            n_input_dims=orientany_input_dims,
            n_output_dims=180,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": orientany_hidden_dim,
                "n_hidden_layers": orientany_num_layers,
            },
        )

        self.mlp_orientany_roll = tcnn.Network(
            n_input_dims=orientany_input_dims,
            n_output_dims=360,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": orientany_hidden_dim,
                "n_hidden_layers": orientany_num_layers,
            },
        )

        self.mlp_orientany_foreground = tcnn.Network(
            n_input_dims=orientany_input_dims,
            n_output_dims=2,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": orientany_hidden_dim,
                "n_hidden_layers": orientany_num_layers,
            },
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        raise NotImplementedError("get_density not supported for FeatureField")

    def _encode_positions(self, ray_samples: RaySamples) -> Tensor:
        """Apply scene contraction and encode positions once."""
        positions = ray_samples.frustums.get_positions().detach()
        positions = self._preprocess_positions(positions)
        positions_flat = positions.view(-1, 3)
        # Encode positions
        encoded_base = self.encoding(positions_flat)
        return encoded_base

    def _preprocess_positions(self, positions: Tensor) -> Tensor:
        """Apply scene contraction and range normalization to positions."""
        positions = self.spatial_distortion(positions)   # Apply scene contraction (same as nerfacto field) with L_inf, maps to a cube [-2,2]^3
        positions = (positions + 2.0) / 4.0    # Remaps from [-2, 2] → [0, 1], Required for HashGrid encoding, which expects input coordinates in [0, 1]
        return positions

    def _encode_centroid_positions(self, centroid_positions: Tensor) -> Tensor:
        """Apply same preprocessing and encoding to centroid predictions."""
        centroid_positions = self._preprocess_positions(centroid_positions)
        centroid_positions_flat = centroid_positions.view(-1, 3)
        return self.orientany_centroid_encoding(centroid_positions_flat)

    def get_feature(self, ray_samples: RaySamples) -> Tensor:
        encoded_base = self._encode_positions(ray_samples)
        features = self.mlp_feature(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return features

    def get_centroid(self, ray_samples: RaySamples) -> Tensor:
        encoded_base = self._encode_positions(ray_samples)
        centroid = self.mlp_centroid(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return centroid

    def get_centroid_spread(self, ray_samples: RaySamples) -> Tensor:
        encoded_base = self._encode_positions(ray_samples)
        spread = self.mlp_centroid_spread(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return spread

    def get_foreground(self, ray_samples: RaySamples) -> Tensor:
        encoded_base = self._encode_positions(ray_samples)
        logits = self.mlp_foreground(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return logits

    def get_orientany_azimuth(self, ray_samples: RaySamples) -> Tensor:
        if self.orientany_use_xyz_encoding:
            encoded_base = self._encode_positions(ray_samples)
        else:
            # Use centroid predictions (detached) encoded via separate encoder
            with torch.no_grad():
                centroid_pred = self.get_centroid(ray_samples)
            encoded_base = self._encode_centroid_positions(centroid_pred)
        logits = self.mlp_orientany_azimuth(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return logits

    def get_orientany_polar(self, ray_samples: RaySamples) -> Tensor:
        if self.orientany_use_xyz_encoding:
            encoded_base = self._encode_positions(ray_samples)
        else:
            with torch.no_grad():
                centroid_pred = self.get_centroid(ray_samples)
            encoded_base = self._encode_centroid_positions(centroid_pred)
        logits = self.mlp_orientany_polar(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return logits

    def get_orientany_roll(self, ray_samples: RaySamples) -> Tensor:
        if self.orientany_use_xyz_encoding:
            encoded_base = self._encode_positions(ray_samples)
        else:
            with torch.no_grad():
                centroid_pred = self.get_centroid(ray_samples)
            encoded_base = self._encode_centroid_positions(centroid_pred)
        logits = self.mlp_orientany_roll(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return logits

    def get_orientany_foreground(self, ray_samples: RaySamples) -> Tensor:
        if self.orientany_use_xyz_encoding:
            encoded_base = self._encode_positions(ray_samples)
        else:
            with torch.no_grad():
                centroid_pred = self.get_centroid(ray_samples)
            encoded_base = self._encode_centroid_positions(centroid_pred)
        logits = self.mlp_orientany_foreground(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return logits

    def get_outputs(self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
        """Compute all field outputs."""
        features = self.get_feature(ray_samples)
        centroid = self.get_centroid(ray_samples)
        centroid_spread = self.get_centroid_spread(ray_samples)
        foreground = self.get_foreground(ray_samples)

        # Get OrientAny components and concatenate
        azimuth = self.get_orientany_azimuth(ray_samples)
        polar = self.get_orientany_polar(ray_samples)
        roll = self.get_orientany_roll(ray_samples)
        orientany_fg = self.get_orientany_foreground(ray_samples)
        orientany = torch.cat([azimuth, polar, roll, orientany_fg], dim=-1)

        return {
            FeatureFieldHeadNames.FEATURE: features,
            FeatureFieldHeadNames.CENTROID: centroid,
            FeatureFieldHeadNames.CENTROID_SPREAD: centroid_spread,
            FeatureFieldHeadNames.FOREGROUND: foreground,
            FeatureFieldHeadNames.ORIENTANY: orientany,
        }

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals:
            raise ValueError("FeatureField does not support computing normals")
        return self.get_outputs(ray_samples)
