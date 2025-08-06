# Nerfacto Pipeline: Grounded Implementation Analysis

This document provides a **grounded** analysis of the Nerfacto pipeline by systematically cross-referencing the [official documentation](https://docs.nerf.studio/nerfology/methods/nerfacto.html) with actual code implementations in nerfstudio and F3RM.

## Overview

Nerfacto combines multiple published techniques into a unified model for real-world static scene reconstruction:

1. **Camera pose refinement** - Optimizes camera poses during training
2. **Per-image appearance conditioning** - Handles varying lighting conditions  
3. **Proposal sampling** - Hierarchical sampling for efficiency
4. **Scene contraction** - Handles unbounded scenes
5. **Hash encoding** - Fast spatial feature encoding (replaces vanilla NeRF's positional encoding)

**Key Difference from Vanilla NeRF**: While vanilla NeRF uses pure sinusoidal positional encoding (`NeRFEncoding` with 10 frequencies for positions, 4 for directions), Nerfacto uses hash grid encoding as the primary spatial representation, with positional encoding only used optionally for predicted normals.

## Pipeline Components

### 1. Camera Pose Refinement

**Documentation Claim**: "The NeRF framework allows us to backpropagate loss gradients to the input pose calculations"

**Implementation**: `/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/cameras/camera_optimizers.py`

```python
class CameraOptimizer(nn.Module):
    def __init__(self, config, num_cameras, device, ...):
        # Initialize learnable parameters
        if self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
    
    def forward(self, indices):
        # Apply learned transformation delta
        if self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
```

**F3RM Integration**: `f3rm_config.py:25-29`
```python
camera_optimizer=CameraOptimizerConfig(
    mode="SO3xR3",
    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
),
```

**Input**: Camera indices → **Output**: 3×4 transformation matrices

### 2. Piecewise Sampler

**Documentation Claim**: "Allocates half of the samples uniformly up to a distance of 1 from the camera. The remaining samples are distributed such that the step size increases with each sample"

**Implementation**: `/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/model_components/ray_samplers.py:225-250`

```python
class UniformLinDispPiecewiseSampler(SpacedSampler):
    def __init__(self, ...):
        super().__init__(
            spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
        )
```

**Nerfacto Integration**: `nerfacto.py:196-200`
```python
if self.config.proposal_initial_sampler == "uniform":
    initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)
else:
    initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=self.config.use_single_jitter)
```

**Input**: Ray bundle → **Output**: Ray samples with piecewise spacing

### 3. Proposal Sampling

**Documentation Claim**: "Consolidates sample locations to regions that contribute most to final render"

**Implementation**: `/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/model_components/ray_samplers.py:523-620`

```python
class ProposalNetworkSampler(Sampler):
    def generate_ray_samples(self, ray_bundle, density_fns):
        for i_level in range(n + 1):
            if i_level == 0:
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            
            if is_prop:
                density = density_fns[i_level](ray_samples.frustums.get_positions())
                weights = ray_samples.get_weights(density)
```

**Nerfacto Integration**: `nerfacto.py:280-285`
```python
def get_outputs(self, ray_bundle: RayBundle):
    ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
```

**Input**: Ray bundle + density functions → **Output**: Refined ray samples + weights

### 4. Density Field (Proposal Networks)

**Documentation Claim**: "Combining a hash encoding with a small fused MLP provides a fast way to query the scene"

**Implementation**: `/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/fields/density_fields.py`

**Nerfacto Integration**: `nerfacto.py:165-185`
```python
self.proposal_networks = torch.nn.ModuleList()
for i in range(num_prop_nets):
    network = HashMLPDensityField(
        self.scene_box.aabb,
        spatial_distortion=scene_contraction,
        **prop_net_args,
        implementation=self.config.implementation,
    )
    self.proposal_networks.append(network)
```

**Input**: 3D positions → **Output**: Density values

### 5. Scene Contraction

**Documentation Claim**: "Contract unbounded space using the contraction proposed in MipNeRF-360"

**Implementation**: `/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/field_components/spatial_distortions.py:40-91`

```python
class SceneContraction(SpatialDistortion):
    def forward(self, positions):
        def contract(x):
            mag = torch.linalg.norm(x, ord=self.order, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        return contract(positions)
```

**Nerfacto Integration**: `nerfacto.py:143-147`
```python
if self.config.disable_scene_contraction:
    scene_contraction = None
else:
    scene_contraction = SceneContraction(order=float("inf"))
```

**Input**: Unbounded 3D positions → **Output**: Contracted positions in [-2, 2]³

### 6. Hash Encoding

**Documentation Claim**: "Fast spatial feature encoding using multi-resolution hash grids"

**Implementation**: `/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/field_components/encodings.py:244-378`

```python
class HashEncoding(Encoding):
    def __init__(self, num_levels=16, min_res=16, max_res=1024, ...):
        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor**levels)
    
    def hash_fn(self, in_tensor):
        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861])
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        return x % self.hash_table_size
```

**Nerfacto Integration**: `nerfacto_field.py:125-135`
```python
self.mlp_base_grid = HashEncoding(
    num_levels=num_levels,
    min_res=base_res,
    max_res=max_res,
    log2_hashmap_size=log2_hashmap_size,
    features_per_level=features_per_level,
    implementation=implementation,
)
```

**Input**: 3D positions → **Output**: Multi-resolution feature vectors

### 7. Nerfacto Field

**Documentation Claim**: "Combines hash encoding, positional encoding, and appearance embeddings"

**Implementation**: `/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/fields/nerfacto_field.py`

**Key Components**:

#### Hash Grid Encoding (Primary)
```python
self.mlp_base_grid = HashEncoding(
    num_levels=num_levels,        # 16 levels
    min_res=base_res,             # 16 base resolution
    max_res=max_res,              # 2048 max resolution
    log2_hashmap_size=log2_hashmap_size,  # 2^19 = 512K entries
    features_per_level=features_per_level,  # 2 features per level
    implementation=implementation,
)
```

#### Positional Encoding (Optional, for predicted normals)
```python
self.position_encoding = NeRFEncoding(
    in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2 - 1
)
```

#### Direction Encoding (Spherical Harmonics)
```python
self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
```

#### Appearance Embedding
```python
self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
```

#### Density Network
```python
def get_density(self, ray_samples):
    positions = self.spatial_distortion(positions)  # Scene contraction
    positions = (positions + 2.0) / 4.0            # Normalize to [0,1]³
    h = self.mlp_base(positions_flat)              # Hash grid + MLP
    density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim])
    density = trunc_exp(density_before_activation)  # Ensure positivity
```

#### Color Network
```python
def get_outputs(self, ray_samples):
    h = torch.cat([
        d,  # direction encoding (spherical harmonics)
        density_embedding,  # geometric features from density network
        embedded_appearance,  # appearance embedding
    ], dim=-1)
    rgb = self.mlp_head(h)  # Final color MLP
```

**Input**: Ray samples → **Output**: Density + RGB + normals + other field outputs

### 8. F3RM Feature Field Extension

**Implementation**: `f3rm/feature_field.py`

**Standard Feature Field**:
```python
class FeatureField(Field):
    def __init__(self, feature_dim, spatial_distortion, use_pe=True, ...):
        encoding_config = {
            "otype": "Composite",
            "nested": [
                {
                    "otype": "HashGrid",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    # ... hash grid config
                }
            ],
        }
        if use_pe:
            encoding_config["nested"].append({
                "otype": "Frequency",
                "n_frequencies": pe_n_freq,
                "n_dims_to_encode": 3,
            })
        
        self.field = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=self.feature_dim,
            encoding_config=encoding_config,
            network_config={...},
        )
```

**Dual Feature Field (DINO+CLIP)**:
```python
class DualFeatureField(FeatureField):
    def __init__(self, dino_dim, clip_dim, ...):
        super().__init__(feature_dim=dino_dim, ...)
        self.projector = tcnn.Network(
            n_input_dims=dino_dim,
            n_output_dims=clip_dim,
            network_config={...},
        )
    
    def forward(self, ray_samples):
        feat_dino = super().forward(ray_samples)[FeatureFieldHeadNames.FEATURE]
        feat_clip = self.projector(feat_dino.view(-1, feat_dino.shape[-1]))
        return {"feat_dino": feat_dino, "feat_clip": feat_clip}
```

**Input**: Ray samples → **Output**: Feature vectors (single or dual)

### 9. Rendering Pipeline

**Implementation**: `/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/model_components/renderers.py`

#### RGB Rendering
```python
class RGBRenderer(nn.Module):
    def forward(self, rgb, weights, ...):
        # Composite samples along ray
        comp_rgb = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb, ray_indices=ray_indices, n_rays=num_rays
        )
```

#### Accumulation Rendering
```python
class AccumulationRenderer(nn.Module):
    def forward(cls, weights, ...):
        # Sum weights along ray
        accumulation = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
        )
```

#### Depth Rendering
```python
class DepthRenderer(nn.Module):
    def forward(self, weights, ray_samples, method="median"):
        if method == "median":
            # Find median depth
            depth = torch.quantile(depths, 0.5, dim=-1)
        elif method == "expected":
            # Expected depth
            depth = torch.sum(weights * depths, dim=-1)
```

**Input**: Field outputs + weights → **Output**: Rendered images (RGB, depth, accumulation, etc.)

## Complete Training Pipeline

### Training Iteration Flow (Step-by-Step)

```python
# TRAINING ITERATION FLOW
# Location: nerfstudio.engine.trainer.Trainer.train_iteration()

def train_iteration(step):
    # 1. DATA LOADING
    # Location: f3rm/feature_datamanager.py:147-161
    ray_bundle, batch = datamanager.next_train(step)
    # - Load RGB images from disk
    # - Extract pre-computed features from LazyFeatures
    # - Apply camera pose refinement via CameraOptimizer
    
    # 2. FORWARD PASS
    # Location: f3rm/model.py:280-330 (FeatureFieldModel.get_outputs())
    outputs = model.get_outputs(ray_bundle)
    
    # 3. LOSS COMPUTATION
    # Location: f3rm/model.py:341-368 (FeatureFieldModel.get_loss_dict())
    loss_dict = model.get_loss_dict(outputs, batch)
    
    # 4. BACKWARD PASS
    # Location: nerfstudio.engine.trainer.Trainer.backward()
    loss_dict["total_loss"].backward()
    
    # 5. OPTIMIZATION STEP
    # Location: nerfstudio.engine.trainer.Trainer.optimize()
    optimizer.step()
    optimizer.zero_grad()
```

### Detailed Forward Pass Breakdown

```python
# FORWARD PASS DETAILED FLOW
# Location: f3rm/model.py:280-330

def get_outputs(ray_bundle: RayBundle):
    # STEP 1: PROPOSAL SAMPLING
    # Location: nerfstudio/model_components/ray_samplers.py:572-616
    ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
        ray_bundle, density_fns=self.density_fns
    )
    # - Initial piecewise sampling (UniformLinDispPiecewiseSampler)
    # - Multi-stage proposal network evaluation
    # - PDF-based importance sampling
    
    # STEP 2: MAIN FIELD EVALUATION (NerfactoField)
    # Location: nerfstudio/fields/nerfacto_field.py:203-302
    field_outputs = self.field.forward(ray_samples, compute_normals=True)
    # - Scene contraction (spatial_distortions.py:40-91)
    # - Hash grid encoding (encodings.py:244-378)
    # - Density network (mlp_base)
    # - Color network (mlp_head with spherical harmonics + appearance)
    
    # STEP 3: FEATURE FIELD EVALUATION (FeatureField)
    # Location: f3rm/feature_field.py:75-85
    ff_outputs = self.feature_field(ray_samples)
    # - Separate hash grid encoding
    # - Optional positional encoding
    # - Feature MLP (tcnn.NetworkWithInputEncoding)
    
    # STEP 4: WEIGHT COMPUTATION
    # Location: nerfstudio/cameras/rays.py (RaySamples.get_weights())
    weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
    # - Volume rendering weight computation from density
    
    # STEP 5: RENDERING
    # Location: nerfstudio/model_components/renderers.py
    rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
    features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)
    depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
    accumulation = self.renderer_accumulation(weights=weights)
    
    return {"rgb": rgb, "feature": features, "depth": depth, "accumulation": accumulation}
```

### Loss Computation Breakdown

```python
# LOSS COMPUTATION DETAILED FLOW
# Location: f3rm/model.py:341-368

def get_loss_dict(outputs, batch, metrics_dict=None):
    # RGB LOSS
    # Location: nerfstudio/model_components/losses.py (MSELoss)
    loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
    
    # FEATURE LOSS (F3RM-specific)
    # Location: f3rm/model.py:354-356
    target_feats = batch["feature"].to(self.device)
    loss_dict["feature_loss"] = self.config.feat_loss_weight * F.mse_loss(
        outputs["feature"], target_feats
    )
    
    # PROPOSAL LOSSES (training only)
    if self.training:
        # Interlevel loss (proposal network consistency)
        # Location: nerfstudio/model_components/losses.py
        loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
            outputs["weights_list"], outputs["ray_samples_list"]
        )
        
        # Distortion loss (encourages compact sampling)
        # Location: nerfstudio/model_components/losses.py
        loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
        
        # Normal losses (if predict_normals=True)
        if self.config.predict_normals:
            # Location: nerfstudio/model_components/losses.py
            loss_dict["orientation_loss"] = self.config.orientation_loss_mult * orientation_loss(...)
            loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * pred_normal_loss(...)
    
    return loss_dict
```

## Key Differences: Documentation vs Implementation

### ✅ **Accurately Documented**

1. **Camera pose refinement** - Fully implemented with SO3xR3 parameterization
2. **Proposal sampling** - Hierarchical sampling with annealing implemented
3. **Scene contraction** - MipNeRF-360 contraction with L-infinity norm
4. **Hash encoding** - Multi-resolution hash grids with proper hash functions
5. **Piecewise sampling** - Uniform + linear disparity spacing

### ⚠️ **Partially Documented**

1. **Appearance embeddings** - Documented but not emphasized in pipeline overview
2. **Spherical harmonics encoding** - Used for directions but not mentioned in docs
3. **Proposal weight annealing** - Implemented but not clearly explained
4. **Hash grid as primary encoding** - Documentation emphasizes positional encoding but hash grids are the main encoding

### ❌ **Missing from Documentation**

1. **F3RM feature field extension** - Not mentioned in original docs
2. **Dual feature field architecture** - DINO+CLIP projection not documented
3. **Comprehensive seeding support** - F3RM-specific reproducibility features
4. **Vanilla NeRF comparison** - Original NeRF uses pure positional encoding vs Nerfacto's hash grids

## Input/Output Specifications

### Ray Bundle Input
- **Origins**: `[num_rays, 3]` - Ray origins in world coordinates
- **Directions**: `[num_rays, 3]` - Normalized ray directions  
- **Nears**: `[num_rays, 1]` - Near plane distances
- **Fars**: `[num_rays, 1]` - Far plane distances

### Ray Samples Output
- **Positions**: `[num_rays, num_samples, 3]` - 3D sample positions
- **Directions**: `[num_rays, num_samples, 3]` - Ray directions at samples
- **Deltas**: `[num_rays, num_samples, 1]` - Distance between samples

### Field Outputs
- **Density**: `[num_rays, num_samples, 1]` - Volume density
- **RGB**: `[num_rays, num_samples, 3]` - Color values
- **Features**: `[num_rays, num_samples, feature_dim]` - Feature vectors (F3RM)
- **Normals**: `[num_rays, num_samples, 3]` - Surface normals (optional)

### Final Rendered Outputs
- **RGB**: `[num_rays, 3]` - Final RGB image
- **Depth**: `[num_rays, 1]` - Depth map
- **Accumulation**: `[num_rays, 1]` - Opacity/accumulation
- **Features**: `[num_rays, feature_dim]` - Rendered features (F3RM)

## Performance Optimizations

1. **TCNN Integration** - Uses tiny-cuda-nn for fast hash encoding and MLPs
2. **Proposal Annealing** - Gradually increases proposal network influence
3. **Gradient Scaling** - Scales gradients by distance squared for stability
4. **Async CUDA Operations** - Non-blocking feature loading in F3RM
5. **Lazy Feature Loading** - Memory-efficient feature storage

## Data Flow Summary

### Training Data Flow
```
Images + Poses → CameraOptimizer → RayBundle → ProposalSampler → 
NerfactoField + FeatureField → VolumeRendering → LossComputation → 
BackwardPass → OptimizationStep
```

### Inference Data Flow
```
CameraParams → RayBundle → ProposalSampler → NerfactoField + FeatureField → 
VolumeRendering → PostProcessing → FinalOutputs
```

### Key File Locations for Each Step

| Step | Training Location | Inference Location |
|------|------------------|-------------------|
| **Data Loading** | `f3rm/feature_datamanager.py:147-161` | `f3rm/feature_datamanager.py:163-175` |
| **Camera Refinement** | `nerfstudio/cameras/camera_optimizers.py:104-138` | Same as training |
| **Proposal Sampling** | `nerfstudio/model_components/ray_samplers.py:572-616` | Same as training |
| **Main Field** | `nerfstudio/fields/nerfacto_field.py:203-302` | Same as training |
| **Feature Field** | `f3rm/feature_field.py:75-85` | Same as training |
| **Rendering** | `nerfstudio/model_components/renderers.py` | Same as training |
| **Loss Computation** | `f3rm/model.py:341-368` | N/A (inference) |
| **Post-Processing** | N/A (training) | `f3rm/model.py:310-337` |

This grounded analysis reveals that the Nerfacto implementation closely follows the documented pipeline while adding several optimizations and extensions (particularly in F3RM) that enhance both performance and functionality.

## Inference Pipeline Flow

### Inference Step-by-Step Flow

```python
# INFERENCE PIPELINE FLOW
# Location: nerfstudio.engine.trainer.Trainer.eval_iteration()

def eval_iteration(step):
    # 1. DATA LOADING (same as training)
    # Location: f3rm/feature_datamanager.py:163-175
    ray_bundle, batch = datamanager.next_eval(step)
    
    # 2. FORWARD PASS (same as training, but no gradients)
    # Location: f3rm/model.py:280-330
    with torch.no_grad():
        outputs = model.get_outputs(ray_bundle)
    
    # 3. METRICS COMPUTATION
    # Location: f3rm/model.py:330-340
    metrics_dict = model.get_metrics_dict(outputs, batch)
    
    # 4. IMAGE GENERATION (for visualization)
    # Location: f3rm/model.py:369-408
    metrics, images = model.get_image_metrics_and_images(outputs, batch)
```

### Novel View Synthesis Flow

```python
# NOVEL VIEW SYNTHESIS FLOW
# Location: nerfstudio.engine.trainer.Trainer.get_outputs_for_camera_ray_bundle()

def render_novel_view(camera_ray_bundle: RayBundle):
    # 1. RAY GENERATION
    # Location: nerfstudio.cameras.rays.py
    # - Generate rays from camera parameters
    # - Apply camera pose refinement if enabled
    
    # 2. FORWARD PASS (same as training)
    # Location: f3rm/model.py:280-330
    outputs = model.get_outputs(camera_ray_bundle)
    
    # 3. POST-PROCESSING (F3RM-specific)
    # Location: f3rm/model.py:310-337
    if feature_type in ["CLIP", "DINOCLIP"]:
        # PCA visualization
        outputs["feature_pca"], viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
            outputs["feature"], viewer_utils.pca_proj
        )
        
        # Language-guided similarity computation
        if viewer_utils.has_positives:
            clip_features = outputs["feature"]
            clip_features /= clip_features.norm(dim=-1, keepdim=True)
            
            if viewer_utils.has_negatives:
                # Paired softmax similarity
                text_embs = torch.cat([viewer_utils.pos_embed, viewer_utils.neg_embed], dim=0)
                raw_sims = clip_features @ text_embs.T
                # ... paired softmax computation
                outputs["similarity"] = sims
            else:
                # Simple cosine similarity
                sims = clip_features @ viewer_utils.pos_embed.T
                outputs["similarity"] = sims
    
    return outputs
```

### Real-time Viewer Flow

```python
# REAL-TIME VIEWER FLOW
# Location: nerfstudio.viewer.server.viewer_elements

def viewer_update():
    # 1. CAMERA UPDATE
    # Location: nerfstudio.viewer.server.viewer_elements
    # - User moves camera in viewer
    # - New camera parameters generated
    
    # 2. RAY GENERATION
    # Location: nerfstudio.cameras.rays.py
    camera_ray_bundle = camera.generate_rays()
    
    # 3. CHUNKED RENDERING
    # Location: nerfstudio.engine.trainer.Trainer.get_outputs_for_camera_ray_bundle()
    # - Split rays into chunks for memory efficiency
    # - Process each chunk separately
    for chunk in ray_chunks:
        chunk_outputs = model.get_outputs_for_camera_ray_bundle(chunk)
        # Merge chunk outputs
    
    # 4. GUI UPDATES (F3RM-specific)
    # Location: f3rm/model.py:150-180
    # - Language query processing via ViewerUtils
    # - PCA projection updates
    # - Similarity computation for language guidance
```

## Comparison with Vanilla NeRF

**Vanilla NeRF Implementation** (`/opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/models/vanilla_nerf.py`):

```python
# Vanilla NeRF uses pure positional encoding
position_encoding = NeRFEncoding(
    in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
)
direction_encoding = NeRFEncoding(
    in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
)

# Hierarchical sampling: uniform → PDF sampling
self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)
```

**Key Differences**:
1. **Encoding**: Vanilla NeRF uses sinusoidal positional encoding (10+4 frequencies), Nerfacto uses hash grids
2. **Sampling**: Vanilla NeRF uses 2-stage hierarchical sampling, Nerfacto uses multi-stage proposal networks
3. **Scene Handling**: Vanilla NeRF assumes bounded scenes, Nerfacto uses scene contraction for unbounded scenes
4. **Appearance**: Vanilla NeRF assumes constant lighting, Nerfacto uses per-image appearance embeddings
5. **Performance**: Nerfacto is significantly faster due to hash grid encoding and proposal sampling 