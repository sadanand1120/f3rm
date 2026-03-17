from typing import Dict, Literal, Optional, Tuple

import torch
from jaxtyping import Float, Shaped
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from torch import nn
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
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_distortion = spatial_distortion

        self.feature_hash_encoding = HashEncoding(
            num_levels=num_levels,
            min_res=start_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.feature_pe_encoding: Optional[NeRFEncoding] = None
        if use_pe:
            self.feature_pe_encoding = NeRFEncoding(
                in_dim=3,
                num_frequencies=pe_n_freq,
                min_freq_exp=0,
                max_freq_exp=pe_n_freq - 1,
                implementation=implementation,
            )

        feature_enc_out_dim = self.feature_hash_encoding.get_out_dim()
        if self.feature_pe_encoding is not None:
            feature_enc_out_dim += self.feature_pe_encoding.get_out_dim()

        self.foreground_hash_encoding = HashEncoding(
            num_levels=num_levels,
            min_res=start_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.foreground_pe_encoding: Optional[NeRFEncoding] = None
        if use_pe:
            self.foreground_pe_encoding = NeRFEncoding(
                in_dim=3,
                num_frequencies=pe_n_freq,
                min_freq_exp=0,
                max_freq_exp=pe_n_freq - 1,
                implementation=implementation,
            )
        foreground_enc_out_dim = self.foreground_hash_encoding.get_out_dim()
        if self.foreground_pe_encoding is not None:
            foreground_enc_out_dim += self.foreground_pe_encoding.get_out_dim()

        self.mlp_feature = MLP(
            in_dim=feature_enc_out_dim,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=self.feature_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.mlp_foreground = MLP(
            in_dim=foreground_enc_out_dim,
            num_layers=foreground_num_layers,
            layer_width=foreground_hidden_dim,
            out_dim=2,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        raise NotImplementedError("get_density not supported for FeatureField")

    def _encode_positions(
        self, ray_samples: RaySamples, hash_encoding: HashEncoding, pe_encoding: Optional[NeRFEncoding]
    ) -> Tensor:
        """Apply scene contraction and encode positions for a specific branch."""
        positions = ray_samples.frustums.get_positions().detach()
        positions = self._preprocess_positions(positions)
        positions_flat = positions.view(-1, 3)
        encoded = [hash_encoding(positions_flat)]
        if pe_encoding is not None:
            encoded.append(pe_encoding(positions_flat))
        return torch.cat(encoded, dim=-1) if len(encoded) > 1 else encoded[0]

    def _preprocess_positions(self, positions: Tensor) -> Tensor:
        """Apply scene contraction and range normalization to positions."""
        positions = self.spatial_distortion(positions)   # Apply scene contraction (same as nerfacto field) with L_inf, maps to a cube [-2,2]^3
        positions = (positions + 2.0) / 4.0    # Remaps from [-2, 2] → [0, 1], Required for HashGrid encoding, which expects input coordinates in [0, 1]
        return positions

    def get_feature(self, ray_samples: RaySamples) -> Tensor:
        encoded_base = self._encode_positions(
            ray_samples=ray_samples,
            hash_encoding=self.feature_hash_encoding,
            pe_encoding=self.feature_pe_encoding,
        )
        features = self.mlp_feature(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return features

    def get_foreground(self, ray_samples: RaySamples) -> Tensor:
        encoded_base = self._encode_positions(
            ray_samples=ray_samples,
            hash_encoding=self.foreground_hash_encoding,
            pe_encoding=self.foreground_pe_encoding,
        )
        logits = self.mlp_foreground(encoded_base).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return logits

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute all field outputs."""
        del density_embedding  # Unused for this field; kept for Field API compatibility.
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
