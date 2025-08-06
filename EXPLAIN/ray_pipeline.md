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

#### 1.1.1 Camera Pose Refinement via CameraOptimizer
**Location**: `CameraOptimizer.forward()` → `RayGenerator.forward()` → `Cameras.generate_rays()`

F3RM includes learnable camera pose refinement to handle noisy initial camera poses. This is implemented through the `CameraOptimizer` class which modifies camera poses during training.

**Configuration**:
```python
# From f3rm/f3rm_config.py
camera_optimizer=CameraOptimizerConfig(
    mode="SO3xR3",  # Rotation + Translation optimization
    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
)
```

**Camera Optimizer Initialization**:
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/cameras/camera_optimizers.py
class CameraOptimizer(nn.Module):
    def __init__(self, config, num_cameras, device, ...):
        # Initialize learnable parameters for each camera
        if self.config.mode in ("SO3xR3", "SE3"):
            # Key variable: 6 learnable parameters per camera (3 rotation + 3 translation)
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
```

**Camera Optimizer Forward Pass**:
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/cameras/camera_optimizers.py
def forward(self, indices: Int[Tensor, "camera_indices"]) -> Float[Tensor, "camera_indices 3 4"]:
    """Convert 6D parameters to 3x4 transformation matrices."""
    if self.config.mode == "SO3xR3":
        # Convert 6D parameters to 3x4 transformation matrix using Lie group exponential mapping
        return exp_map_SO3xR3(self.pose_adjustment[indices, :])
    elif self.config.mode == "SE3":
        return exp_map_SE3(self.pose_adjustment[indices, :])
```

**Integration into Ray Generation**:
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/model_components/ray_generators.py
class RayGenerator(nn.Module):
    def forward(self, ray_indices: Int[Tensor, "num_rays 3"]) -> RayBundle:
        c = ray_indices[:, 0]  # camera indices
        # Get camera optimizer transform for these cameras
        camera_opt_to_camera = self.pose_optimizer(c)
        
        # Generate rays with optimized camera poses
        ray_bundle = self.cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords,
            camera_opt_to_camera=camera_opt_to_camera,  # Apply pose refinement!
        )
```

**Application in Camera Ray Generation**:
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/cameras/cameras.py
def _generate_rays_from_coords(self, camera_indices, coords, camera_opt_to_camera=None, ...):
    # ... compute camera directions from intrinsics ...
    
    # CRITICAL: Apply camera optimizer adjustments to camera poses
    if camera_opt_to_camera is not None:
        c2w = pose_utils.multiply(c2w, camera_opt_to_camera)
    
    # ... continue with ray generation using adjusted poses ...
```

**Optimization Setup**:
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/data/datamanagers/base_datamanager.py
def get_param_groups(self) -> Dict[str, List[Parameter]]:
    """Include camera optimizer parameters in training optimization."""
    param_groups = {}
    camera_opt_params = list(self.train_camera_optimizer.parameters())
    if self.config.camera_optimizer.mode != "off":
        param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
    return param_groups
```

**Mathematical Foundation**:
The camera optimizer learns pose adjustments in the Lie algebra of SE(3) or SO(3)×R³:

```
For SO3xR3 mode:
pose_adjustment ∈ R^(num_cameras × 6)  # [ω_x, ω_y, ω_z, t_x, t_y, t_z]
T_adjustment = exp_map_SO3xR3(pose_adjustment)  # Convert to 3×4 transformation matrix
T_final = T_original @ T_adjustment  # Apply adjustment to original pose
```

**Training Process**:
1. **Initialization**: All pose adjustments start at zero (identity transformation)
2. **Forward Pass**: Camera poses are modified by learned adjustments during ray generation
3. **Loss Computation**: Photometric loss implicitly optimizes pose adjustments
4. **Backward Pass**: Gradients flow back to `pose_adjustment` parameters
5. **Parameter Update**: Adam optimizer updates pose adjustments

**Impact on Pipeline**:
- **Original poses**: Stored in `train_dataset.cameras.camera_to_worlds` (unchanged)
- **Optimized poses**: `original_pose @ learned_adjustment` (used during training)
- **Consistency**: Both training and inference use the same optimized poses
- **Memory**: Additional ~6 parameters per camera (minimal overhead)

**Configuration Options**:
```python
# Different optimization modes
mode="off"           # No pose optimization
mode="SO3xR3"        # Separate rotation (SO3) + translation (R3) optimization
mode="SE3"           # Joint SE(3) optimization

# Learning rate and scheduling
optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
scheduler=ExponentialDecaySchedulerConfig(max_steps=10000)
```

#### **1.1.2 Accessing Optimized Camera Poses**

**Important**: Optimized camera poses are **not stored as separate variables** - they're computed on-the-fly during ray generation. Here's how to access them:

**Method 1: Direct Access via Camera Optimizer Parameters**
```python
# Get the camera optimizer from the data manager
camera_optimizer = pipeline.datamanager.train_camera_optimizer

# Get the pose adjustment parameters (6 parameters per camera)
pose_adjustments = camera_optimizer.pose_adjustment  # Shape: [num_cameras, 6]

# Get the camera indices for the specific camera you want
camera_idx = len(pipeline.datamanager.train_dataset.cameras) - 1  # For the last camera

# Get the pose adjustment for this specific camera
camera_pose_adjustment = pose_adjustments[camera_idx:camera_idx+1]  # Shape: [1, 6]

# Convert to transformation matrix using the same method as the camera optimizer
from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
camera_opt_to_camera = exp_map_SO3xR3(camera_pose_adjustment)  # Shape: [1, 3, 4]

# Get the original camera pose (4x4 matrix)
original_c2w_34 = pipeline.datamanager.train_dataset.cameras[-1].camera_to_worlds.cpu().numpy()
original_c2w = np.vstack([original_c2w_34, np.array([[0, 0, 0, 1]])])

# Apply the camera optimizer adjustment
# Note: The camera optimizer applies: original_pose @ adjustment
optimized_c2w = original_c2w @ camera_opt_to_camera[0]  # Remove batch dimension

print("Original camera pose:")
print(original_c2w)
print("\nCamera optimizer adjustment:")
print(camera_opt_to_camera[0])
print("\nOptimized camera pose:")
print(optimized_c2w)
```

**Method 2: Access Through Ray Generator**
```python
# Get the ray generator
ray_generator = pipeline.datamanager.train_ray_generator

# Get the camera optimizer transform for a specific camera
camera_idx = torch.tensor([len(pipeline.datamanager.train_dataset.cameras) - 1])
camera_opt_to_camera = ray_generator.pose_optimizer(camera_idx)

print("Camera optimizer transform:")
print(camera_opt_to_camera.cpu().numpy())
```

**Key Points**:
- **No pre-existing variable**: Optimized poses are computed on-the-fly during ray generation
- **Formula**: `optimized_pose = original_pose @ camera_optimizer_adjustment`
- **Access pattern**: 
  - Original poses: `train_dataset.cameras.camera_to_worlds`
  - Optimizer adjustments: `train_camera_optimizer.pose_adjustment`
  - Optimized poses: Compute using the formula above

#### **1.1.3 6D Exponential Coordinates: Mathematical Foundation**

**Why 6D Exponential Coordinates?**

The camera optimizer uses **6D exponential coordinates** to represent pose adjustments in the **Lie algebra** of SE(3) or SO(3)×R³, which provides several key advantages:

**A. Smooth Optimization**
```python
# 6D representation: [ω_x, ω_y, ω_z, t_x, t_y, t_z]
pose_adjustment = torch.zeros((num_cameras, 6))  # Smooth, continuous space

# vs. direct 3x4 matrix representation:
# pose_adjustment = torch.zeros((num_cameras, 3, 4))  # Constrained, harder to optimize
```

**B. Lie Group Properties**
The exponential mapping preserves the **Lie group structure**:
```python
# exp_map_SO3xR3: R^6 → SE(3) preserves group operations
T_adjustment = exp_map_SO3xR3(pose_adjustment)
# This ensures: T1 @ T2 = exp(pose1) @ exp(pose2) = exp(pose1 + pose2)
```

**C. Gradient Flow**
6D coordinates provide **better gradient flow** during optimization:
- **Unconstrained**: All 6 parameters can vary freely
- **Smooth**: Small changes in parameters → small changes in pose
- **Stable**: No singularities or constraints to handle

**Implementation Details**:
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.10/site-packages/nerfstudio/cameras/lie_groups.py
def exp_map_SO3xR3(xi: Float[Tensor, "*batch 6"]) -> Float[Tensor, "*batch 3 4"]:
    """Convert 6D exponential coordinates to 3x4 transformation matrix."""
    # xi = [ω_x, ω_y, ω_z, t_x, t_y, t_z]
    omega = xi[..., :3]  # Rotation (angular velocity)
    v = xi[..., 3:]      # Translation (linear velocity)
    
    # Convert to rotation matrix using Rodrigues' formula
    R = exp_map_SO3(omega)
    
    # Convert to translation vector
    t = v  # For SO3xR3, translation is direct
    
    # Combine into 3x4 transformation matrix
    return torch.cat([R, t.unsqueeze(-1)], dim=-1)
```

**Alternative Representations**:
```python
# SE3 mode: 6D twist coordinates
# xi = [ω_x, ω_y, ω_z, v_x, v_y, v_z] where v is not direct translation
# More complex but preserves full SE(3) group structure
```

#### **1.1.4 Supervisory Signal for Pose Refinement**

**Implicit Supervision via Photometric Loss**

The camera pose refinement is **implicitly supervised** through the **photometric reconstruction loss**:

**Loss Function**:
```python
# From FeatureFieldModel.get_loss_dict() (f3rm/model.py)
def get_loss_dict(self, outputs, batch, metrics_dict=None):
    loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
    
    # RGB reconstruction loss (this implicitly optimizes camera poses)
    rgb_loss = F.mse_loss(outputs["rgb"], batch["image"])
    loss_dict["rgb_loss"] = rgb_loss
    
    # Feature loss (also affects pose optimization)
    feature_loss = F.mse_loss(outputs["feature"], batch["feature"])
    loss_dict["feature_loss"] = feature_loss
    
    return loss_dict
```

**Gradient Flow**:
```python
# The gradient flow works as follows:
# 1. Photometric loss computed between rendered and ground truth images
# 2. Gradients flow back through volume rendering
# 3. Gradients flow back through ray generation
# 4. Gradients flow back through camera pose optimization
# 5. Gradients update pose_adjustment parameters

# Simplified chain:
# pose_adjustment → camera_pose → ray_origins/directions → rendered_rgb → photometric_loss
#                                                                    ↑
#                                                              gradient flow
```

**Why This Works**:

**A. Geometric Consistency**
- **Correct poses** → **accurate ray directions** → **correct 3D sampling** → **good reconstruction**
- **Incorrect poses** → **wrong ray directions** → **poor 3D sampling** → **high reconstruction loss**

**B. Multi-View Consistency**
```python
# The system sees multiple views of the same scene
# If poses are wrong, the same 3D point will render differently from different views
# This inconsistency creates high photometric loss, driving pose correction
```

**Training Process**:
```python
# During training iteration:
for step in range(max_iterations):
    # 1. Generate rays with current pose adjustments
    ray_bundle = ray_generator(ray_indices)  # Uses optimized poses
    
    # 2. Render images with current poses
    outputs = model(ray_bundle)
    rendered_rgb = outputs["rgb"]
    
    # 3. Compute photometric loss
    loss = F.mse_loss(rendered_rgb, ground_truth_rgb)
    
    # 4. Backward pass - gradients flow to pose_adjustment
    loss.backward()
    
    # 5. Update pose adjustments
    optimizer.step()  # Updates pose_adjustment parameters
```

**No Explicit Pose Supervision**

**Key insight**: There's **no explicit pose supervision** (no ground truth poses). The system learns to refine poses by:

1. **Minimizing photometric reconstruction error**
2. **Enforcing multi-view consistency**
3. **Leveraging the 3D scene structure**

**Why This Approach Works**:

**A. Self-Supervised Learning**
- The scene itself provides supervision through photometric consistency
- No need for expensive pose annotation or ground truth

**B. Joint Optimization**
- Poses and scene geometry are optimized together
- This allows the system to find the best pose-scene fit

**C. Robust to Initial Noise**
- Can handle significant initial pose errors
- Gradually refines poses to improve reconstruction quality

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