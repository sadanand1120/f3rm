# F3RM System Overview

## Architecture Summary

F3RM extends the Nerfacto model from Nerfstudio to enable distillation of 2D foundation models (CLIP, DINO) into 3D neural fields. The system consists of **neural network heads** (with learnable parameters) and **computed outputs** (mathematical computations) via the `FeatureFieldModel` (`f3rm/model.py`) which inherits from `NerfactoModel` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/nerfacto.py`):

### **Neural Network Heads** (Learnable Parameters)

1. **RGB Head**: `NerfactoField.mlp_head` - Standard NeRF color outputs `f_rgb: (x,y,z,θ,φ) → (r,g,b)` via learnable MLP
2. **Density Head**: `NerfactoField.mlp_base_mlp` - Volume density for ray marching `f_σ: (x,y,z) → σ` via learnable MLP  
3. **Feature Head**: `FeatureField.field` - High-dimensional feature vectors `f_feat: (x,y,z) → f^d` (d=768 for CLIP) via learnable MLP
4. **Predicted Normals Head**: `NerfactoField.mlp_pred_normals` + `PredNormalsFieldHead` - Surface normals `f_n: (x,y,z) → (n_x,n_y,n_z)` via learnable MLP

### **Computed Outputs** (Mathematical Computations)

1. **Ground Truth Normals**: Computed from density gradients via automatic differentiation
2. **Depth**: Computed from weights and ray samples via `DepthRenderer`
3. **Accumulation**: Computed from weights via `AccumulationRenderer` 
4. **Expected Depth**: Computed from weights and ray samples via `DepthRenderer`

## Unified Pipeline: Pre/Model_Forward/Post Stages

### Pipeline Overview
```
Camera Parameters → Ray Generation → Proposal Sampling → Field Evaluation → Volume Rendering → Loss Computation
     (Pre)              (Pre)            (Pre)            (Model_Fwd)        (Post)           (Post)
```

### Stage-by-Stage Analysis

#### **Pre-Processing Stage** (Shared across all heads)
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

#### **Model Forward Pass Stage** (Mixed shared/independent)
```python
# Location: FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)

# Shared: Base field evaluation (RGB + Density + Predicted Normals)
field_outputs = self.field.forward(ray_samples, compute_normals=True)
# - Density: field_outputs[FieldHeadNames.DENSITY] (from mlp_base_mlp)
# - RGB: field_outputs[FieldHeadNames.RGB] (from mlp_head)
# - Predicted Normals: field_outputs[FieldHeadNames.PRED_NORMALS] (from mlp_pred_normals)

# Independent: Feature field evaluation
ff_outputs = self.feature_field(ray_samples)  # Separate hash grid + MLP
# - Features: ff_outputs[FeatureFieldHeadNames.FEATURE] (from FeatureField.field)
```

#### **Post-Processing Stage** (Mixed shared/independent)
```python
# Shared: Volume rendering weights
weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

# Neural Network Head Rendering (using learned outputs)
rgb = self.renderer_rgb(field_outputs[FieldHeadNames.RGB], weights)
features = self.renderer_feature(ff_outputs[FeatureFieldHeadNames.FEATURE], weights)
pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights)

# Computed Outputs (mathematical computations)
gt_normals = self.renderer_normals(field_outputs[FieldHeadNames.NORMALS], weights)  # from density gradients
depth = self.renderer_depth(weights, ray_samples)  # computed from weights
accumulation = self.renderer_accumulation(weights)  # computed from weights
expected_depth = self.renderer_expected_depth(weights, ray_samples)  # computed from weights
```

### Training vs Inference Differences

#### **Training-Only Operations**
```python
# Location: FeatureFieldModel.get_outputs() (training=True)
if self.training:
    # Store intermediate results for loss computation
    outputs["weights_list"] = weights_list
    outputs["ray_samples_list"] = ray_samples_list
    
    # Normal-specific losses
    if self.config.predict_normals:
        outputs["rendered_orientation_loss"] = orientation_loss(...)
        outputs["rendered_pred_normal_loss"] = pred_normal_loss(...)

# Location: FeatureDataManager.next_train() (f3rm/feature_datamanager.py lines 147-161)
# Ground truth feature extraction and loading
batch["feature"] = self._gather_feats(self.features, camera_idx, y_idx, x_idx)
```

#### **Inference-Only Operations**
```python
# Location: FeatureFieldModel.get_outputs_for_camera_ray_bundle() (f3rm/model.py lines 296-337)
# PCA projection for feature visualization
outputs["feature_pca"], viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(...)

# Language similarity computation (CLIP only)
if self.kwargs["metadata"]["feature_type"] in ["CLIP", "DINOCLIP"]:
    clip_features = outputs["feature"] / outputs["feature"].norm(dim=-1, keepdim=True)
    sims = clip_features @ viewer_utils.pos_embed.T
    outputs["similarity"] = sims
```

#### **Shared Operations**
- Camera ray generation via `VanillaDataManager`
- Proposal sampling via `ProposalNetworkSampler`
- Spatial distortion via `SceneContraction`
- Field evaluation via `NerfactoField.forward()` and `FeatureField.forward()`
- Volume rendering via renderers (`RGBRenderer`, `FeatureRenderer`, etc.)

## Core Components

### Model Architecture

#### Base Field: `NerfactoField`
- **Location**: `/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/fields/nerfacto_field.py`
- **Function**: Handles RGB + density using hash grids + MLPs
- **Key Methods**: 
  - `get_density()`: Computes volume density and geometry features
  - `get_outputs()`: Computes RGB, normals, and other outputs
- **Architecture**: Hash grid encoding + geometry MLP + color MLP

#### Feature Field: `FeatureField`
- **Location**: `f3rm/feature_field.py`
- **Function**: Parallel hash grid + MLP for features
- **Key Methods**:
  - `get_outputs()`: Computes feature vectors at 3D positions
  - `forward()`: Main interface for feature computation
- **Architecture**: Hash grid encoding + feature MLP

#### Spatial Distortion: `SceneContraction`
- **Location**: `/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/spatial_distortions.py`
- **Function**: Scene contraction mapping unbounded space to unit sphere
- **Implementation**: L-infinity norm contraction to radius 2 sphere

#### Volume Renderers
- **RGB Renderer**: `RGBRenderer` from Nerfstudio
- **Feature Renderer**: `FeatureRenderer` (`f3rm/renderer.py`) - simple weighted sum
- **Depth Renderer**: `DepthRenderer` from Nerfstudio
- **Normals Renderer**: `NormalsRenderer` from Nerfstudio

### Training Pipeline

#### 1. Feature Extraction via `FeatureDataManager`
- **Location**: `f3rm/feature_datamanager.py`
- **Method**: `extract_features_sharded()` calls `extract_features_for_dataset()`
- **Process**: Extract CLIP/DINO features from training images with caching
- **Output**: `LazyFeatures` or tensors with feature maps

#### 2. Ray Sampling via `VanillaDataManager`
- **Location**: Inherited from Nerfstudio's base datamanager
- **Process**: Generate ray bundles from camera parameters
- **Output**: `RayBundle` objects with origins, directions, and metadata

#### 3. Proposal Sampling via `ProposalNetworkSampler`
- **Location**: `/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/ray_samplers.py`
- **Process**: Use coarse networks to guide fine sampling
- **Implementation**: Hierarchical sampling with proposal networks

#### 4. Field Evaluation via `FeatureFieldModel.get_outputs()`
- **Location**: `f3rm/model.py` lines 198-260
- **Process**: Query all heads at 3D sample points
- **Sequence**:
  ```python
  # Proposal sampling
  ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
  
  # Base field evaluation (RGB + density + normals)
  field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
  
  # Feature field evaluation
  ff_outputs = self.feature_field(ray_samples)
  ```

#### 5. Volume Rendering via Renderers
- **Process**: Integrate along rays with learned density weights
- **Implementation**: Weighted sum of field outputs using `weights = ray_samples.get_weights(density)`
- **Outputs**: RGB, features, depth, normals for each ray

#### 6. Loss Computation via `FeatureFieldModel.get_loss_dict()`
- **Location**: `f3rm/model.py` lines 279-295
- **Process**: MSE losses for RGB, features, normals + regularization
- **Feature Loss**: `F.mse_loss(outputs["feature"], target_feats)` weighted by `feat_loss_weight`

### Mathematical Foundation

The core volume rendering equation for any quantity Q:
```
Ĉ(r) = ∫ T(t)σ(r(t))Q(r(t))dt
```
Where:
- `T(t) = exp(-∫₀ᵗ σ(r(s))ds)` is transmittance
- `σ(r(t))` is volume density  
- `Q(r(t))` can be RGB, features, or normals

Discretized for neural networks:
```
Ĉ = Σᵢ wᵢ Qᵢ
wᵢ = αᵢ Πⱼ₌₁ᶦ⁻¹ (1-αⱼ)  
αᵢ = 1 - exp(-σᵢδᵢ)
```

**Implementation**: This is implemented in `ray_samples.get_weights()` method from Nerfstudio's ray sampling utilities.

## Key Innovations

### 1. Parallel Feature Distillation
- **Implementation**: `FeatureField` trained alongside `NerfactoField` in `FeatureFieldModel`
- **Location**: `f3rm/model.py` lines 125-165
- **Key Insight**: Features use same volume rendering weights as RGB for consistency

### 2. Hash Grid Encodings
- **RGB Field**: 16 levels, 2 features/level, max_res=2048 (`NerfactoModelConfig`)
- **Feature Field**: 12 levels, 8 features/level, max_res=128 (`FeatureFieldModelConfig`)
- **Implementation**: TinyCUDA-NN `HashGrid` encoding with multi-resolution levels

### 3. Proposal Networks
- **Location**: `/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/ray_samplers.py`
- **Function**: Hierarchical sampling for efficiency
- **Configuration**: 2 iterations, (256, 96) samples per ray

### 4. Scene Contraction
- **Location**: `/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/spatial_distortions.py`
- **Function**: Handle unbounded scenes via spatial warping
- **Implementation**: L-infinity norm contraction to unit sphere

### 5. Multi-Head Training
- **Implementation**: Joint optimization of appearance and semantic features
- **Loss Combination**: RGB loss + feature loss + normal losses + regularization
- **Location**: `f3rm/model.py` lines 279-295

## Implementation Details

The system is implemented in PyTorch using:
- **TinyCUDA-NN**: Fast CUDA kernels for hash grids and MLPs via `tcnn.NetworkWithInputEncoding`
- **Nerfstudio**: Base infrastructure for NeRF training (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/`)
- **Open-CLIP**: Feature extraction from foundation models via `open_clip.create_model_and_transforms()`
- **Custom Renderers**: Volume integration for different output types via `FeatureRenderer`

## Configuration and Hyperparameters

### Main Configuration Files
- **F3RM Config**: `f3rm/f3rm_config.py` - Main training configuration
- **Model Config**: `FeatureFieldModelConfig` in `f3rm/model.py` - Feature field parameters
- **Trainer Config**: `F3RMTrainerConfig` in `f3rm/trainer.py` - Training and seeding parameters
- **Data Config**: `FeatureDataManagerConfig` in `f3rm/feature_datamanager.py` - Data loading parameters

### Key Hyperparameters
- **Feature Loss Weight**: `feat_loss_weight=1e-3` - Controls feature vs RGB emphasis
- **Hash Grid Parameters**: Different resolutions for RGB (2048) vs features (128)
- **Training Steps**: `max_num_iterations=30000` - Total training iterations
- **Batch Size**: `train_num_rays_per_batch=8192` - Rays per training step

## Performance Characteristics

- **Memory Usage**: ~12GB GPU memory with default settings (`eval_num_rays_per_chunk=1<<14`)
- **Training Speed**: ~30K training iterations in 2-3 hours on RTX 3090
- **Feature Types**: CLIP (512D), DINO (768D), ROBOPOINT (custom dimensions)
- **Quality**: Photorealistic RGB + semantic understanding for language queries

The system achieves high-quality 3D scene understanding by combining traditional NeRF photometric supervision with semantic feature distillation from 2D foundation models, enabling both visual reconstruction and language-guided scene understanding. 