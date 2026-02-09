from typing import Dict, Tuple

import numpy as np
import tinycudann as tcnn
from jaxtyping import Float, Shaped
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from torch import Tensor


class FeatureFieldHeadNames:
    FEATURE = "feature"
    FOREGROUND = "foreground"


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
        foreground_hidden_dim: int = 64,
        foreground_num_layers: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_distortion = spatial_distortion
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

    def get_density(self, ray_samples: RaySamples) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        raise NotImplementedError("get_density not supported for FeatureField")

    def _encode_positions(self, ray_samples: RaySamples) -> Tensor:
        """Apply scene contraction and encode positions once."""
        positions = ray_samples.frustums.get_positions().detach()
        positions = self._preprocess_positions(positions)
        return self.encoding(positions.view(-1, 3))

    def _preprocess_positions(self, positions: Tensor) -> Tensor:
        """Apply scene contraction and range normalization to positions."""
        positions = self.spatial_distortion(positions)   # Apply scene contraction (same as nerfacto field) with L_inf, maps to a cube [-2,2]^3
        positions = (positions + 2.0) / 4.0    # Remaps from [-2, 2] → [0, 1], Required for HashGrid encoding, which expects input coordinates in [0, 1]
        return positions

    def get_feature(self, ray_samples: RaySamples) -> Tensor:
        encoded_base = self._encode_positions(ray_samples)
        features = self.mlp_feature(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return features

    def get_foreground(self, ray_samples: RaySamples) -> Tensor:
        encoded_base = self._encode_positions(ray_samples)
        logits = self.mlp_foreground(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return logits

    def get_outputs(self, ray_samples: RaySamples) -> Dict[str, Tensor]:
        """Compute all field outputs."""
        features = self.get_feature(ray_samples)
        foreground = self.get_foreground(ray_samples)

        return {
            FeatureFieldHeadNames.FEATURE: features,
            FeatureFieldHeadNames.FOREGROUND: foreground,
        }

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[str, Tensor]:
        if compute_normals:
            raise ValueError("FeatureField does not support computing normals")
        return self.get_outputs(ray_samples)
