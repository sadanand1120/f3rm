# Ray Marching Pipeline: Pre/Model_Forward/Post Analysis

## Overview

The ray marching pipeline transforms camera parameters into rendered pixel values through hierarchical sampling, field evaluation, and volume rendering. This document breaks down the complete pipeline into **Pre-Processing**, **Model Forward Pass**, and **Post-Processing** stages, showing which components are shared vs independent across the different heads (RGB, Density, Features, Normals).

## Pipeline Stage Breakdown

### **Stage 1: Pre-Processing** (Shared across all heads)

#### 1.1 Camera Ray Generation
**Location**: `VanillaDataManager.next_train()` / `next_eval()` → `Cameras.generate_rays()`

```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/cameras/cameras.py
def generate_rays(
    self,
    camera_indices: Union[Int[Tensor, "*num_rays num_cameras_batch_dims"], int],
    coords: Optional[Float[Tensor, "*num_rays 2"]] = None,
    camera_opt_to_camera: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
    distortion_params_delta: Optional[Float[Tensor, "*num_rays 6"]] = None,
    keep_shape: Optional[bool] = None,
    disable_distortion: bool = False,
    aabb_box: Optional[SceneBox] = None,
    obb_box: Optional[OrientedBox] = None,
) -> RayBundle:
```

**Ray Bundle Creation**:
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/cameras/rays.py
ray_bundle = RayBundle(
    origins=origins,        # [H, W, 3] - ray origins
    directions=directions,  # [H, W, 3] - unit ray directions
    pixel_area=pixel_area, # [H, W, 1] - for cone tracing
    nears=nears,           # [H, W, 1] - near clipping  
    fars=fars,             # [H, W, 1] - far clipping
    camera_indices=cam_idx # [H, W, 1] - which camera
)
```

**Mathematical Foundation**:
```
r(t) = o + td
```
Where:
- `o ∈ R³` = ray origin (camera center)  
- `d ∈ R³` = ray direction (unit vector)
- `t ∈ R⁺` = distance along ray

#### 1.2 Proposal Sampling (Hierarchical)
**Location**: `ProposalNetworkSampler.generate_ray_samples()` → `FeatureFieldModel.get_outputs()`

F3RM uses hierarchical sampling with proposal networks for efficiency via `ProposalNetworkSampler` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/ray_samplers.py`):

```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

# Configuration from NerfactoModelConfig
num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)  # Coarse → fine
num_nerf_samples_per_ray: int = 48                        # Final samples
num_proposal_iterations: int = 2                          # Number of stages
```

**Proposal Network Sampler Implementation**:
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/ray_samplers.py
def generate_ray_samples(
    self,
    ray_bundle: Optional[RayBundle] = None,
    density_fns: Optional[List[Callable]] = None,
) -> Tuple[RaySamples, List, List]:
    """Hierarchical sampling with proposal networks."""
    weights_list = []
    ray_samples_list = []
    
    n = self.num_proposal_network_iterations
    weights = None
    ray_samples = None
    
    for i_level in range(n + 1):
        is_prop = i_level < n
        num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
        
        if i_level == 0:
            # Uniform sampling because we need to start with some samples
            ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
        else:
            # PDF sampling based on the last samples and their weights
            annealed_weights = torch.pow(weights, self._anneal)
            ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
        
        # Evaluate density at these samples
        if density_fns is not None:
            weights = self.get_weights_from_density_fn(ray_samples, density_fns[i_level])
        else:
            weights = None
            
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
    
    return ray_samples, weights_list, ray_samples_list
```

#### 1.3 Spatial Distortion (Scene Contraction)
**Location**: `SceneContraction.apply()` → `NerfactoField.get_density()` / `FeatureField.get_outputs()`

All 3D coordinates undergo scene contraction to handle unbounded scenes using `SceneContraction` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/spatial_distortions.py`):

```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/spatial_distortions.py
def apply(self, positions: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3"]:
    """Apply scene contraction to positions."""
    if self.order == float("inf"):
        # L-infinity norm contraction
        mag = torch.max(torch.abs(positions), dim=-1, keepdim=True)[0]
        scale = torch.where(mag <= 1.0, torch.ones_like(mag), (2.0 - 1.0 / mag) / mag)
        return positions * scale
    else:
        # L-p norm contraction
        mag = torch.norm(positions, dim=-1, keepdim=True, p=self.order)
        scale = torch.where(mag <= 1.0, torch.ones_like(mag), (2.0 - 1.0 / mag) / mag)
        return positions * scale
```

**Mathematical Formulation**:
```
For position x ∈ R³:
mag = ||x||_∞  # L-infinity norm
if mag ≤ 1:
    return x
else:
    return (2 - 1/mag) * (x/mag)  # Maps R³ → [-2,2]³
```

**Normalization for Hash Grids**:
```python
# From NerfactoField.get_density() and FeatureField.get_outputs()
positions_normalized = (positions_contracted + 2.0) / 4.0  # [-2,2]³ → [0,1]³
```

### **Stage 2: Model Forward Pass** (Mixed shared/independent)

#### 2.1 Shared Field Evaluation (RGB + Density + Normals)
**Location**: `NerfactoField.forward()` → `FeatureFieldModel.get_outputs()`

```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)

# Outputs from NerfactoField:
# - field_outputs[FieldHeadNames.DENSITY]: Volume density σ ∈ R⁺
# - field_outputs[FieldHeadNames.RGB]: RGB colors (r,g,b) ∈ [0,1]³  
# - field_outputs[FieldHeadNames.NORMALS]: Gradient-based normals ∈ R³
# - field_outputs[FieldHeadNames.PRED_NORMALS]: MLP-predicted normals ∈ R³
```

**NerfactoField Architecture**:
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/fields/nerfacto_field.py
def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
    # 1. Get density and geometry features
    density, geo_features = self.get_density(ray_samples)
    
    # 2. Get RGB colors
    rgb = self.get_outputs(ray_samples, geo_features)
    
    # 3. Get normals (if requested)
    if compute_normals:
        normals = self.get_normals()
        pred_normals = self.pred_normals_head(geo_features)
    
    return {
        FieldHeadNames.DENSITY: density,
        FieldHeadNames.RGB: rgb,
        FieldHeadNames.NORMALS: normals,
        FieldHeadNames.PRED_NORMALS: pred_normals,
    }
```

#### 2.2 Independent Feature Field Evaluation
**Location**: `FeatureField.forward()` → `FeatureFieldModel.get_outputs()`

```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
ff_outputs = self.feature_field(ray_samples)

# Outputs from FeatureField:
# - ff_outputs[FeatureFieldHeadNames.FEATURE]: Feature vectors f ∈ R^d (d=512 for CLIP)
```

**FeatureField Architecture**:
```python
# From f3rm/feature_field.py
def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
    # 1. Apply spatial distortion (same as NerfactoField)
    positions = ray_samples.frustums.get_positions().detach()
    positions = self.spatial_distortion(positions)   # Apply spatial distortion
    positions = (positions + 2.0) / 4.0    # Remaps from [-2, 2] → [0, 1]
    
    # 2. Hash grid encoding (separate from RGB field)
    positions_flat = positions.view(-1, 3)
    features = self.field(positions_flat)  # tcnn.NetworkWithInputEncoding
    
    # 3. Reshape to original dimensions
    features = features.view(*ray_samples.frustums.directions.shape[:-1], -1)
    
    return {FeatureFieldHeadNames.FEATURE: features}
```

### **Stage 3: Post-Processing** (Mixed shared/independent)

#### 3.1 Shared Volume Rendering Weights
**Location**: `RaySamples.get_weights()` → All renderers

```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
```

**Volume Rendering Equation**:
```
α_i = 1 - exp(-σ_i δ_i)                    # Alpha (opacity)
T_i = ∏_{j=1}^{i-1} (1 - α_j)              # Transmittance 
w_i = α_i T_i                              # Rendering weight
```

**Mathematical Foundation**:
```
C(r) = ∫[t_near, t_far] T(t) σ(r(t)) c(r(t), d) dt
T(t) = exp(-∫[t_near, t] σ(r(s)) ds)  # Transmittance
```

Discretized for neural networks:
```
Ĉ = Σᵢ wᵢ Qᵢ
wᵢ = αᵢ Πⱼ₌₁ᶦ⁻¹ (1-αⱼ)  
αᵢ = 1 - exp(-σᵢδᵢ)
```

#### 3.2 Independent Head-Specific Rendering

**RGB Rendering**:
```python
# From FeatureFieldModel.get_outputs()
rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
```

**Feature Rendering**:
```python
# From FeatureFieldModel.get_outputs()
features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)
```

**Feature Renderer Implementation**:
```python
# From f3rm/renderer.py
class FeatureRenderer(nn.Module):
    @classmethod
    def forward(cls, features: Float[Tensor, "*bs num_samples num_channels"], 
                weights: Float[Tensor, "*bs num_samples 1"]) -> Float[Tensor, "*bs num_channels"]:
        output = torch.sum(weights * features, dim=-2)  # Weighted sum along ray
        return output
```

**Normals Rendering**:
```python
# From FeatureFieldModel.get_outputs()
normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
```

**Depth Rendering**:
```python
# From FeatureFieldModel.get_outputs()
depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
```

## Training vs Inference Pipeline Differences

### **Training-Only Operations**

#### 3.3 Training-Specific Post-Processing
**Location**: `FeatureFieldModel.get_outputs()` (training=True)

```python
# Store intermediate results for loss computation
if self.training:
    outputs["weights_list"] = weights_list
    outputs["ray_samples_list"] = ray_samples_list

# Normal-specific losses (only during training)
if self.training and self.config.predict_normals:
    outputs["rendered_orientation_loss"] = orientation_loss(
        weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
    )
    outputs["rendered_pred_normal_loss"] = pred_normal_loss(
        weights.detach(),
        field_outputs[FieldHeadNames.NORMALS].detach(),
        field_outputs[FieldHeadNames.PRED_NORMALS],
    )
```

#### 3.4 Ground Truth Data Loading
**Location**: `FeatureDataManager.next_train()` (`f3rm/feature_datamanager.py` lines 147-161)

```python
# RGB ground truth
batch["image"] = _async_to_cuda(batch["image"], self.device)

# Feature ground truth (training only)
camera_idx, y_idx, x_idx = self._index_triplet(batch, self.scale_h, self.scale_w)
batch["feature"] = self._gather_feats(self.features, camera_idx, y_idx, x_idx)
```

### **Inference-Only Operations**

#### 3.5 Inference-Specific Post-Processing
**Location**: `FeatureFieldModel.get_outputs_for_camera_ray_bundle()` (`f3rm/model.py` lines 296-337)

```python
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

### **Shared Operations** (Training & Inference)
- Camera ray generation via `VanillaDataManager`
- Proposal sampling via `ProposalNetworkSampler`
- Spatial distortion via `SceneContraction`
- Field evaluation via `NerfactoField.forward()` and `FeatureField.forward()`
- Volume rendering via renderers (`RGBRenderer`, `FeatureRenderer`, etc.)

## Cross-Head Dependencies

### **Density Head Dependencies**
- **Shared by**: RGB, Features, Normals (all use same volume rendering weights)
- **Independent**: None (density affects all other heads)
- **Location**: `ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])`

### **RGB Head Dependencies**
- **Shared with**: Density (same geometry network), Normals (same geometry features)
- **Independent from**: Features (separate hash grid + MLP)
- **Location**: `NerfactoField.get_outputs()` vs `FeatureField.forward()`

### **Feature Head Dependencies**
- **Shared with**: Density (same volume rendering weights)
- **Independent from**: RGB (separate field), Normals (separate computation)
- **Location**: `FeatureField.forward()` (completely separate from `NerfactoField`)

### **Normals Head Dependencies**
- **Shared with**: RGB, Density (same geometry network)
- **Independent from**: Features (separate computation)
- **Location**: `NerfactoField.get_normals()` and `PredNormalsFieldHead`

## Performance Considerations

### **Memory Usage**
- **Proposal sampling**: Stores intermediate weights and samples
- **Field evaluation**: Parallel evaluation of RGB and feature fields
- **Volume rendering**: Weighted sums along ray dimensions

### **Computational Complexity**
- **Proposal networks**: O(N_proposal) for hierarchical sampling
- **Hash grid lookups**: O(1) per sample via TinyCUDA-NN
- **Volume rendering**: O(N_samples) weighted sum per ray

### **Optimization Strategies**
- **Chunked rendering**: Process rays in chunks to manage memory
- **Mixed precision**: Use FP16 for faster computation
- **TinyCUDA-NN**: Optimized CUDA kernels for hash grids and MLPs

The ray marching pipeline efficiently combines shared pre-processing and post-processing stages with head-specific model forward passes, enabling parallel training of multiple representations while maintaining computational efficiency. 