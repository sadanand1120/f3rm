# Feature Head: Semantic Feature Distillation

## Overview

The feature head distills 2D foundation models (CLIP, DINO) into a 3D neural field by learning to predict high-dimensional feature vectors at each 3D point. This enables semantic understanding and language-guided queries in 3D space via `FeatureField` (`f3rm/feature_field.py`).

## Mathematical Formulation

### Function Signature
```
f_feat: (x, y, z) → f ∈ R^d
```
Where d = 512 for CLIP ViT-L-14-336, d = 768 for DINO ViT-S, etc.

### Volume Rendering Role
Feature vectors are integrated along rays using shared density weights:
```
Ĉ_feat = Σᵢ wᵢ fᵢ
```
Where `wᵢ` are volume rendering weights computed from density (shared with RGB head).

### Key Properties
- **Position-only**: Features are view-independent (unlike RGB)
- **High-dimensional**: 768D CLIP features vs 3D RGB
- **Semantic**: Encode object categories, materials, functionality
- **Distilled**: Transfer 2D foundation model knowledge to 3D

## Pipeline Integration: Pre/Model_Forward/Post

### **Pre-Processing Stage** (Shared with all heads)
**Location**: `FeatureFieldModel.get_outputs()` → `FeatureField.forward()`

```python
# 1. Camera Ray Generation (Shared)
# Location: VanillaDataManager.next_train() / next_eval()
ray_bundle = cameras.generate_rays(camera_indices, coords)  # [N_rays, 3]

# 2. Proposal Sampling (Shared)
# Location: ProposalNetworkSampler.generate_ray_samples()
ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns)

# 3. Spatial Distortion (Shared)
# Location: SceneContraction.apply()
positions = self.spatial_distortion(ray_samples.frustums.get_positions())  # [-2,2]³
positions_normalized = (positions + 2.0) / 4.0  # [0,1]³
```

### **Model Forward Pass Stage** (Independent from RGB/Density/Normals)
**Location**: `FeatureField.forward()` → `FeatureFieldModel.get_outputs()`

```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
ff_outputs = self.feature_field(ray_samples)

# Feature output from FeatureField:
feature_samples = ff_outputs[FeatureFieldHeadNames.FEATURE]  # [N_rays, N_samples, d]
```

#### Neural Architecture

The feature head uses a completely separate neural field parallel to the RGB field via `FeatureField`:

##### Hash Grid Encoding (Separate from RGB)
```python
# Feature field has its own hash grid parameters from FeatureFieldModelConfig
feat_hash_config = {
    "otype": "HashGrid",
    "n_levels": 12,                   # Fewer levels than RGB (12 vs 16)
    "n_features_per_level": 8,        # More features per level (8 vs 2)
    "log2_hashmap_size": 19,          # Same hash table size
    "base_resolution": 16,
    "per_level_scale": 1.20,          # Slower growth
    # Finest level: 16 * 1.20^11 ≈ 128³ grid
}

# Multi-resolution encoding for features via TinyCUDA HashGrid
pos_encoded = HashGrid(xyz_normalized)  # R^3 → R^96 (12×8)
```

##### Positional Encoding (Optional)
```python  
# From f3rm/feature_field.py:__init__()
if use_pe:
    pos_pe = FrequencyEncoding(xyz_contracted, n_frequencies=6)  # R^3 → R^36 (6×2×3)
    encoding = concat([pos_encoded, pos_pe])                    # R^132 (96+36)
else:
    encoding = pos_encoded                                       # R^96
```

##### Feature MLP (Separate from RGB)
```python
# Dedicated TinyCUDA MLP for feature prediction
# From f3rm/feature_field.py:__init__()
self.field = tcnn.NetworkWithInputEncoding(
    n_input_dims=3,
    n_output_dims=self.feature_dim,  # 768 for CLIP
    encoding_config=encoding_config,  # HashGrid + Frequency
    network_config={
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": hidden_dim,      # 64
        "n_hidden_layers": num_layers, # 2
    },
)

features = self.field(positions_flat)  # R^132 → R^768
```

### **Post-Processing Stage** (Independent rendering, shared weights)
**Location**: `FeatureRenderer.forward()` → `FeatureFieldModel.get_outputs()`

```python
# Shared: Volume rendering weights (computed from density)
weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

# Independent: Feature-specific rendering
features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)
```

#### Feature Renderer Implementation
```python
# From f3rm/renderer.py
class FeatureRenderer(nn.Module):
    @classmethod
    def forward(cls, features: Float[Tensor, "*bs num_samples num_channels"], 
                weights: Float[Tensor, "*bs num_samples 1"]) -> Float[Tensor, "*bs num_channels"]:
        output = torch.sum(weights * features, dim=-2)  # Weighted sum along ray
        return output
```

## Training Data Pipeline

### Feature Extraction
**Location**: `extract_features_for_dataset()` → `FeatureDataManager.__init__()`

Ground truth features are extracted from training images using foundation models via `f3rm/features/extract_features_standalone.py`:

```python
# CLIP feature extraction from f3rm/features/clip_extract.py
import open_clip
model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336-quickgelu", pretrained="openai")

# Extract dense patch-level features per image
@torch.inference_mode()
def get_patch_encodings(model, image_batch):
    # Extract patch tokens (excluding class token)
    x = model.visual.conv1(image_batch)  # [B, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid²]
    x = x.permute(0, 2, 1)  # [B, grid², width]
    # Add class embedding and positional embedding
    x = torch.cat([class_embed, x], dim=1)  # [B, grid²+1, width]
    x = x + positional_embedding
    # Pass through transformer layers
    x = transformer_layers(x)
    x = x[:, 1:, :]  # Remove class token, keep patch tokens
    x = ln_post(x)
    if proj is not None:
        x = x @ proj
    return x  # [B, grid², 768]

# Reshape to spatial dimensions
embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
```

### Pixel-to-Feature Matching
**Location**: `FeatureDataManager.next_train()` (`f3rm/feature_datamanager.py` lines 147-161)

Features are sampled at pixel locations during training:
```python
# RGB ground truth
batch["image"] = _async_to_cuda(batch["image"], self.device)

# Feature ground truth (training only)
camera_idx, y_idx, x_idx = self._index_triplet(batch, self.scale_h, self.scale_w)
batch["feature"] = self._gather_feats(self.features, camera_idx, y_idx, x_idx)
```

## Cross-Head Dependencies

### **Dependencies on Density Head**
- **Critical**: Feature rendering uses density weights via `ray_samples.get_weights(density)`
- **Shared Volume Rendering**: Both use same volume rendering equation
- **Location**: `ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])`

### **Independence from RGB Head**
- **Separate Fields**: Features use `FeatureField`, RGB uses `NerfactoField`
- **Separate Hash Grids**: Different encoding parameters and MLPs
- **Separate Training Data**: Features from foundation models, RGB from images
- **Location**: `FeatureField.forward()` vs `NerfactoField.forward()`

### **Independence from Normals Head**
- **Separate Computation**: Features don't use normal outputs
- **Separate Architecture**: Features have dedicated hash grid + MLP
- **Location**: `FeatureField.forward()` (completely separate)

## Training vs Inference Differences

### **Training-Only Operations**
```python
# Location: FeatureFieldModel.get_outputs() (training=True)
if self.training:
    # Store intermediate results for loss computation
    outputs["weights_list"] = weights_list
    outputs["ray_samples_list"] = ray_samples_list

# Location: FeatureDataManager.next_train()
# Ground truth feature extraction and loading
camera_idx, y_idx, x_idx = self._index_triplet(batch, self.scale_h, self.scale_w)
batch["feature"] = self._gather_feats(self.features, camera_idx, y_idx, x_idx)
```

### **Inference-Only Operations**
```python
# Location: FeatureFieldModel.get_outputs_for_camera_ray_bundle() (f3rm/model.py lines 296-337)
# PCA projection for feature visualization
outputs["feature_pca"], viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
    outputs["feature"], viewer_utils.pca_proj
)

# Language similarity computation (CLIP only)
if self.kwargs["metadata"]["feature_type"] in ["CLIP", "DINOCLIP"]:
    # Normalize CLIP features
    clip_features = outputs["feature"].to(viewer_utils.device)
    clip_features /= clip_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity with positive/negative text embeddings
    if viewer_utils.has_positives:
        sims = clip_features @ viewer_utils.pos_embed.T
        outputs["similarity"] = sims
```

### **Shared Operations**
- Camera ray generation via `VanillaDataManager`
- Proposal sampling via `ProposalNetworkSampler`
- Spatial distortion via `SceneContraction`
- Field evaluation via `FeatureField.forward()`
- Volume rendering via `FeatureRenderer`

## Loss Function

### Feature Distillation Loss
**Location**: `FeatureFieldModel.get_loss_dict()` (`f3rm/model.py` lines 279-295)

```python  
# From f3rm/model.py
def get_loss_dict(self, outputs, batch, metrics_dict=None):
    loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
    ft = self.kwargs["metadata"]["feature_type"]
    if ft == "DINOCLIP":
        loss_dict["feature_dino_loss"] = (
            self.config.feat_loss_weight
            * F.mse_loss(outputs["feature_dino"], batch["feature_dino"].to(self.device))
        )
        loss_dict["feature_clip_loss"] = (
            self.config.feat_loss_weight
            * F.mse_loss(outputs["feature_clip"], batch["feature_clip"].to(self.device))
        )
    else:
        target_feats = batch["feature"].to(self.device)
        loss_dict["feature_loss"] = self.config.feat_loss_weight * F.mse_loss(outputs["feature"], target_feats)
    return loss_dict
```

#### Properties
- **Weight**: 1e-3 (from `FeatureFieldModelConfig.feat_loss_weight`)
- **Purpose**: Transfer semantic understanding from 2D to 3D
- **Rationale**: Features have 768D vs RGB's 3D, need lower weight
- **Range**: Varies by foundation model (CLIP features are L2-normalized to unit vectors)
- **Training data**: From `FeatureDataManager` via `batch["feature"]`

## Neural Network Architecture Details

### Hash Grid Encoding (Feature-Specific)
```python
# From FeatureFieldModelConfig (f3rm/model.py)
feat_hash_config = {
    "num_levels": 12,                 # Fewer levels than RGB (12 vs 16)
    "max_res": 128,                   # Lower resolution (128 vs 2048)
    "features_per_level": 8,          # More features per level (8 vs 2)
    "log2_hashmap_size": 19,          # Same hash table size
    "start_res": 16,                  # Base resolution
    "hidden_dim": 64,                 # MLP hidden dimension
    "num_layers": 2,                  # MLP layers
}

# Memory usage: 12 × 8 × 2^19 × 4 bytes = 192MB
```

### Positional Encoding (Optional)
```python
# From f3rm/feature_field.py
if use_pe:
    encoding_config["nested"].append({
        "otype": "Frequency",
        "n_frequencies": pe_n_freq,  # 6
        "n_dims_to_encode": 3,
    })
    # Output: 3 × 2 × 6 = 36 dimensions
```

### Feature MLP
```python
# From f3rm/feature_field.py
self.field = tcnn.NetworkWithInputEncoding(
    n_input_dims=3,
    n_output_dims=self.feature_dim,  # 768 for CLIP
    encoding_config=encoding_config,  # HashGrid + Frequency
    network_config={
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": hidden_dim,      # 64
        "n_hidden_layers": num_layers, # 2
    },
)
```

## Performance Characteristics

### Memory Usage
- **Hash Grid**: 12 × 8 × 2^19 × 4 bytes = 192MB
- **MLP**: ~0.5M parameters for feature network
- **Total**: ~192.5MB for feature field

### Computational Complexity
- **Hash Grid Lookup**: O(1) per sample via TinyCUDA-NN
- **MLP Evaluation**: O(1) per sample
- **Volume Rendering**: O(N_samples) weighted sum per ray

### Feature Dimensions
- **CLIP ViT-L-14-336**: 768-dimensional features
- **DINO ViT-S**: 384-dimensional features
- **ROBOPOINT**: Custom dimensions for robotics applications

### Quality Metrics
- **Feature Similarity**: Cosine similarity with ground truth features
- **Semantic Understanding**: Language-guided scene understanding
- **Cross-Modal Retrieval**: Text-to-3D feature matching

## Key Innovations

### 1. Parallel Feature Distillation
- **Implementation**: `FeatureField` trained alongside `NerfactoField` in `FeatureFieldModel`
- **Location**: `f3rm/model.py` lines 125-165
- **Key Insight**: Features use same volume rendering weights as RGB for consistency

### 2. Foundation Model Integration
- **CLIP Features**: 768D semantic vectors for language understanding
- **DINO Features**: 384D self-supervised features for visual understanding
- **ROBOPOINT**: Custom features for robotics applications

### 3. Memory-Efficient Feature Storage
- **LazyFeatures**: Memory-mapped feature shards for large datasets
- **Location**: `f3rm/features/extract_features_standalone.py`
- **Benefits**: Handle datasets with thousands of images efficiently

The feature head enables semantic understanding in 3D space by distilling 2D foundation model knowledge through a dedicated neural field that shares volume rendering weights with the RGB head for consistency. 