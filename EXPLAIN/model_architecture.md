# Model Architecture: Neural Networks Deep Dive

## Overview

F3RM consists of multiple neural networks working in parallel: RGB/density fields, feature fields, proposal networks, and auxiliary heads. All networks use modern hash grid encodings and MLP architectures optimized for speed and quality via the `NerfactoField` (`/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/fields/nerfacto_field.py`) and `FeatureField` (`f3rm/feature_field.py`).

## Architecture Definitions

### NerfactoField Architecture
**Location**: `/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/fields/nerfacto_field.py:73-99`

```python
class NerfactoField(Field):
    def __init__(self, aabb, num_images, num_layers=2, hidden_dim=64, geo_feat_dim=15,
                 num_levels=16, base_res=16, max_res=2048, log2_hashmap_size=19,
                 num_layers_color=3, features_per_level=2, hidden_dim_color=64,
                 appearance_embedding_dim=32, use_pred_normals=False, ...):
        
        # Core components
        self.mlp_base_grid = HashEncoding(num_levels=16, min_res=16, max_res=2048, 
                                         log2_hashmap_size=19, features_per_level=2)
        self.mlp_base_mlp = MLP(in_dim=32, num_layers=2, layer_width=64, out_dim=16)
        self.mlp_head = MLP(in_dim=72, num_layers=3, layer_width=64, out_dim=3)
        
        # Encodings
        self.direction_encoding = SHEncoding(levels=4)  # 25 dims
        self.position_encoding = NeRFEncoding(num_frequencies=2)  # 12 dims (optional)
        
        # Embeddings
        self.embedding_appearance = Embedding(num_images, 32)
        
        # Optional heads
        if use_pred_normals:
            self.mlp_pred_normals = MLP(in_dim=27, num_layers=3, layer_width=64, out_dim=64)
```

### F3RM FeatureField Architecture  
**Location**: `f3rm/model.py:30-48` (config) + `f3rm/feature_field.py:17-75` (implementation)

```python
class FeatureFieldModelConfig(NerfactoModelConfig):
    # Feature field parameters
    feat_loss_weight: float = 1e-3
    feat_use_pe: bool = True
    feat_pe_n_freq: int = 6
    feat_num_levels: int = 12
    feat_log2_hashmap_size: int = 19
    feat_start_res: int = 16
    feat_max_res: int = 128
    feat_features_per_level: int = 8
    feat_hidden_dim: int = 64
    feat_num_layers: int = 2

class FeatureField(Field):
    def __init__(self, feature_dim, spatial_distortion, use_pe=True, pe_n_freq=6,
                 num_levels=12, log2_hashmap_size=19, start_res=16, max_res=128,
                 features_per_level=8, hidden_dim=64, num_layers=2):
        
        # TCNN network with composite encoding
        self.field = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=feature_dim,  # 768 for CLIP
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {"otype": "HashGrid", "n_levels": 12, "n_features_per_level": 8, 
                     "log2_hashmap_size": 19, "base_resolution": 16, "max_resolution": 128},
                    {"otype": "Frequency", "n_frequencies": 6, "n_dims_to_encode": 3}  # if use_pe
                ]
            },
            network_config={
                "otype": "FullyFusedMLP",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )
```

## Core Architecture Components

### 1. Hash Grid Encodings

Hash grids provide multi-resolution position encodings that are both memory-efficient and fast to evaluate via `HashEncoding` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/encodings.py`).

#### Mathematical Foundation
```
For position x ∈ [0,1]³ and resolution level l:

1. Scale position: x_l = x * resolution_l
2. Hash function: h = hash(⌊x_l⌋) mod table_size_l  
3. Feature lookup: f_l = table_l[h]  # [features_per_level]
4. Trilinear interpolation within grid cell
5. Concatenate all levels: [f_0, f_1, ..., f_L]
```

#### Implementation Details
```python
# RGB field hash grid (high resolution) via NerfactoField
rgb_hash_config = {
    "num_levels": 16,                 # Multi-resolution levels
    "min_res": 16,                    # Base resolution
    "max_res": 2048,                  # Maximum resolution
    "log2_hashmap_size": 19,          # 2^19 = 512K entries per level
    "features_per_level": 2,          # Features per grid cell
    "implementation": "tcnn"          # TinyCUDA implementation
}

# Feature field hash grid (lower resolution, more features) via FeatureField
feature_hash_config = {
    "otype": "HashGrid", 
    "n_levels": 12,                   # Fewer levels
    "n_features_per_level": 8,        # More features per cell
    "log2_hashmap_size": 19,          # Same memory per level
    "base_resolution": 16,
    "per_level_scale": 1.20,          # Slower growth
    # Finest level: 16 * 1.20^11 ≈ 128³ grid
}
```

#### Memory Analysis
```python
# RGB field memory usage
rgb_memory = 16 * 2 * 2^19 * 4 bytes = 16 * 2 * 512K * 4B = 64MB

# Feature field memory usage  
feat_memory = 12 * 8 * 2^19 * 4 bytes = 12 * 8 * 512K * 4B = 192MB

# Total hash grid memory: ~256MB
```

### 2. Positional Encodings (Optional)

Classic sinusoidal encodings for fine details via `NeRFEncoding` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/encodings.py`):

#### Frequency Encoding
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/encodings.py
class NeRFEncoding(Encoding):
    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
        implementation: Literal["tcnn", "torch"] = "torch",
    ):
        # For F3RM feature field: num_frequencies=6, min_freq_exp=0, max_freq_exp=5
        # Output: 3 * 2 * 6 = 36 dimensions
```

#### Spherical Harmonics (Directions) via `SHEncoding`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/encodings.py
class SHEncoding(Encoding):
    def __init__(self, levels: int = 4, implementation: Literal["tcnn", "torch"] = "torch"):
        # For NerfactoField: levels=4
        # Output: (4+1)² = 25 coefficients
        # Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, ..., Y_4^4
```

### 3. Multi-Layer Perceptrons (MLPs)

F3RM uses TinyCUDA's optimized MLPs via `MLP` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/mlp.py`):

#### RGB Field Geometry MLP via `NerfactoField`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/fields/nerfacto_field.py
self.mlp_base_mlp = MLP(
    in_dim=self.mlp_base_grid.get_out_dim(),  # 32 (hash grid)
    num_layers=2,                              # Hidden layers
    layer_width=64,                            # Hidden dimension
    out_dim=1 + self.geo_feat_dim,            # 1 + 15 = 16 (density + features)
    activation=nn.ReLU(),
    out_activation=None,
    implementation="tcnn"
)

# Architecture: [32] → [64] → [64] → [16]
# Parameters: 32*64 + 64*64 + 64*16 = 2,048 + 4,096 + 1,024 = 7,168
```

#### RGB Field Color MLP via `NerfactoField`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/fields/nerfacto_field.py
self.mlp_head = MLP(
    in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
    # 25 (SH) + 15 (geo) + 32 (appearance) = 72 dimensions
    num_layers=3,                              # Hidden layers
    layer_width=64,                            # Hidden dimension
    out_dim=3,                                 # RGB values
    activation=nn.ReLU(),
    out_activation=nn.Sigmoid(),               # RGB ∈ [0,1]
    implementation="tcnn"
)

# Architecture: [72] → [64] → [64] → [64] → [3]
# Parameters: 72*64 + 64*64 + 64*64 + 64*3 = 4,608 + 4,096 + 4,096 + 192 = 12,992
```

#### Feature Field MLP via `FeatureField`
```python
# From f3rm/feature_field.py
self.field = tcnn.NetworkWithInputEncoding(
    n_input_dims=3,                           # Raw positions
    n_output_dims=self.feature_dim,           # 768 for CLIP
    encoding_config=encoding_config,          # Hash grid + PE
    network_config={
        "otype": "FullyFusedMLP",
        "activation": "ReLU", 
        "output_activation": "None",          # Raw features
        "n_neurons": 64,                      # Hidden dimension
        "n_hidden_layers": 2                  # Hidden layers
    }
)

# Architecture: [132] → [64] → [64] → [768] (132 = 96 hash + 36 PE)
# Parameters: 132*64 + 64*64 + 64*768 = 8,448 + 4,096 + 49,152 = 61,696
```

### 4. Proposal Networks

Lightweight networks for hierarchical sampling via `NerfactoModelConfig`:

#### Proposal Network Architecture
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/nerfacto.py
proposal_net_args_list: List[Dict] = field(
    default_factory=lambda: [
        {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
        {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
    ]
)

# Two proposal networks with different resolutions
proposal_net_1 = HashMLPDensityField(
    num_levels=5,
    max_res=128,         # Coarse
    hidden_dim=16,       # Much smaller than main network
    log2_hashmap_size=17 # 2^17 = 128K (vs 512K for main)
)

proposal_net_2 = HashMLPDensityField(
    num_levels=5, 
    max_res=256,         # Finer
    hidden_dim=16,
    log2_hashmap_size=17
)
```

### 5. Auxiliary Heads

#### Predicted Normals Head via `PredNormalsFieldHead`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/field_heads.py
class PredNormalsFieldHead(FieldHead):
    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Tanh()):
        super().__init__(in_dim=in_dim, out_dim=3, field_head_name=FieldHeadNames.PRED_NORMALS, activation=activation)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Needed to normalize the output into valid normals."""
        out_tensor = super().forward(in_tensor)
        out_tensor = torch.nn.functional.normalize(out_tensor, dim=-1)
        return out_tensor
```

#### Appearance Embeddings via `Embedding`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/embedding.py
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize with small random values
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
    
# For N training images via NerfactoField
self.embedding_appearance = Embedding(
    num_embeddings=self.num_images,      # One embedding per training image
    embedding_dim=self.appearance_embedding_dim  # 32 (from config)
)
```

## Complete Network Flow

### Forward Pass Data Flow via `NerfactoField.get_outputs()`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/fields/nerfacto_field.py
def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None):
    # 1. Extract and process positions
    positions = ray_samples.frustums.get_positions()  # [N_rays, N_samples, 3]
    positions = self.spatial_distortion(positions)    # Scene contraction
    positions = (positions + 2.0) / 4.0              # Normalize to [0,1]³
    
    # 2. Hash grid encoding via HashEncoding
    rgb_encoded = self.mlp_base_grid(positions)      # [N_rays, N_samples, 32]
    
    # 3. Geometry network (shared by RGB and normals)
    geo_output = self.mlp_base_mlp(rgb_encoded)      # [N_rays, N_samples, 16]
    density = trunc_exp(geo_output[..., :1])         # [N_rays, N_samples, 1]
    geo_features = geo_output[..., 1:]               # [N_rays, N_samples, 15]
    
    # 4. RGB color network
    directions = ray_samples.frustums.directions     # [N_rays, 1, 3]
    dir_encoded = self.direction_encoding(directions) # [N_rays, 1, 25]
    appearance = self.embedding_appearance(camera_idx) # [N_rays, 1, 32]
    
    color_input = torch.cat([
        geo_features,                                # [N_rays, N_samples, 15]
        dir_encoded.expand_as(geo_features[..., :25]), # [N_rays, N_samples, 25]
        appearance.expand_as(geo_features[..., :32])   # [N_rays, N_samples, 32]
    ], dim=-1)                                       # [N_rays, N_samples, 72]
    
    rgb = self.mlp_head(color_input)                 # [N_rays, N_samples, 3]
    
    # 5. Predicted normals (optional)
    if self.use_pred_normals:
        pos_encoded = self.position_encoding(positions)  # [N_rays, N_samples, 12]
        normals_input = torch.cat([geo_features, pos_encoded], dim=-1)  # [N_rays, N_samples, 27]
        pred_normals = self.field_head_pred_normals(self.mlp_pred_normals(normals_input))
    
    return {
        "density": density,
        "rgb": rgb,
        "pred_normals": pred_normals if self.use_pred_normals else None
    }
```

### Feature Field Forward Pass via `FeatureField.get_outputs()`
```python
# From f3rm/feature_field.py
def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None):
    # 1. Extract and process positions
    positions = ray_samples.frustums.get_positions().detach()
    positions = self.spatial_distortion(positions)   # Scene contraction
    positions = (positions + 2.0) / 4.0             # Normalize to [0,1]³
    positions_flat = positions.view(-1, 3)
    
    # 2. Feature network (parallel to RGB)
    features = self.field(positions_flat).view(*ray_samples.frustums.directions.shape[:-1], -1)
    # [N_rays, N_samples, 768] for CLIP
    
    return {FeatureFieldHeadNames.FEATURE: features}
```

## Memory and Compute Analysis

### Parameter Counts
```python
# Hash grid tables
rgb_hash_params = 16 * 2 * (2**19) = 16.8M parameters
feature_hash_params = 12 * 8 * (2**19) = 50.3M parameters

# MLP parameters  
geometry_mlp_params = 7.2K (actual: 32*64 + 64*64 + 64*16)
color_mlp_params = 13K (actual: 72*64 + 64*64 + 64*64 + 64*3)
feature_mlp_params = 61.7K (actual: 132*64 + 64*64 + 64*768)
normals_head_params = 5K (estimated)

# Embeddings
appearance_params = N_images * 32

# Proposal networks
proposal_params = 2 * 0.5M = 1M (estimated)

# Total: ~68M parameters (dominated by hash grids)
```

### Memory Usage
```python
# Hash grid storage (largest component)
hash_grid_memory = (16.8M + 50.3M) * 4 bytes = 268MB

# Activations during forward pass
batch_size = 8192  # rays per batch (from FeatureDataManagerConfig)
samples_per_ray = 48  # from NerfactoModelConfig

# Intermediate activations
positions_mem = 8192 * 48 * 3 * 4 = 4.7MB
rgb_encoded_mem = 8192 * 48 * 32 * 4 = 50MB  # Hash grid only
features_mem = 8192 * 48 * 768 * 4 = 1.2GB   # Largest activation

# Peak memory: ~1.5GB during training
```

### Computational Complexity
```python
# Hash grid lookups: O(L) where L = num_levels
rgb_hash_flops = 16 levels * trilinear_interpolation
feature_hash_flops = 12 levels * trilinear_interpolation

# MLP forward passes: O(layer_width²)
geometry_mlp_flops = 32*64 + 64*64 + 64*16 = 7.2K per sample
color_mlp_flops = 72*64 + 64*64 + 64*64 + 64*3 = 13K per sample
feature_mlp_flops = 132*64 + 64*64 + 64*768 = 61.7K per sample

# Total: ~82K FLOPs per sample
# For 8K rays * 48 samples = 31B FLOPs per batch
```

## Architecture Design Choices

### Hash Grid Trade-offs
```python
# RGB field: High resolution, few features per level
# - Pro: Fine geometric details, sharp textures
# - Con: Limited feature capacity per location
# - Config: 16 levels, 2 features/level, max_res=2048

# Feature field: Lower resolution, many features per level  
# - Pro: Rich semantic representations
# - Con: Coarser spatial resolution
# - Config: 12 levels, 8 features/level, max_res=128
# - Rationale: Semantic features need less spatial precision than RGB
```

### MLP Depth Trade-offs
```python
# Geometry MLP: Shallow (2 layers)
# - Pro: Fast inference, avoids overfitting
# - Con: Limited representational capacity
# - Rationale: Density is relatively simple to learn

# Color MLP: Medium (3 layers)  
# - Pro: Captures view-dependent effects, complex lighting
# - Con: Slower than geometry network
# - Rationale: Color depends on complex interactions

# Feature MLP: Shallow (2 layers)
# - Pro: Fast distillation from 2D features
# - Con: May underfit complex semantic relationships
# - Rationale: 2D features already contain high-level semantics
```

### Memory Optimizations
```python
# Hash grid collision handling
# - Multiple positions may hash to same table entry
# - Collision rate ~5-10% at log2_hashmap_size=19
# - Trade-off: memory vs quality

# Mixed precision training via F3RMTrainerConfig
mixed_precision=True  # FP16 forward pass, FP32 gradients

# Chunked processing via FeatureFieldModelConfig
eval_num_rays_per_chunk = 1 << 14  # 16384 rays per chunk
```

The architecture balances quality, speed, and memory through careful design of hash grids, MLP depths, and feature dimensions. Each component is optimized for its specific role in the overall F3RM system via the coordinated execution of RGB reconstruction, feature distillation, and geometric consistency networks. 