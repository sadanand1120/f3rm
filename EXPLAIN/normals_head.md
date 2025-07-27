# Normals Head: Surface Normal Prediction

## Overview

The normals head predicts surface normal vectors at 3D points to capture scene geometry. F3RM supports **predicted normals** (learned via dedicated neural network head) and **ground-truth normals** (computed from density gradients) with specialized loss functions for geometric supervision via `NerfactoField` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/fields/nerfacto_field.py`).

## Neural Network Head vs Computed Output

### **Predicted Normals Head** (Neural Network - Learnable Parameters)
- **Location**: `NerfactoField.mlp_pred_normals` + `PredNormalsFieldHead`
- **Function**: Learned MLP that predicts surface normals
- **Parameters**: ~0.1M learnable parameters
- **Output**: `f_pred_normals: (x,y,z) → (n_x,n_y,n_z)` where ||n|| = 1

### **Ground Truth Normals** (Computed Output - No Learnable Parameters)
- **Location**: Computed from density gradients via automatic differentiation
- **Function**: Mathematical computation from density field gradients
- **Parameters**: None (computed)
- **Output**: `f_gt_normals: ∇σ(x,y,z) → (n_x,n_y,n_z)` where ||n|| = 1

## Mathematical Formulation

### Function Signature
```
f_pred_normals: (x, y, z) → (n_x, n_y, n_z) where ||n|| = 1
```
Surface normals are unit vectors pointing outward from surfaces.

### Volume Rendering Role
Normal vectors are integrated along rays using shared density weights:
```
Ĉ_normals = Σᵢ wᵢ nᵢ
```
Where `wᵢ` are volume rendering weights computed from density (shared with RGB and feature heads).

### Two Types of Normals

#### Ground-Truth Normals (Gradient-Based - Computed)
```python
# Computed from density gradients via automatic differentiation
∇σ(x,y,z) = (∂σ/∂x, ∂σ/∂y, ∂σ/∂z)
n_gt = -∇σ / ||∇σ||  # Negative gradient points outward
```

#### Predicted Normals (MLP-Based - Neural Network Head)
```python
# Learned via dedicated MLP head
n_pred = PredNormalsFieldHead(geo_features + pos_encoded)  # R^(15+12) → R^3
n_pred = n_pred / ||n_pred||                               # L2 normalize
```

## Pipeline Integration: Pre/Model_Forward/Post

### **Pre-Processing Stage** (Shared with all heads)
**Location**: `FeatureFieldModel.get_outputs()` → `NerfactoField.forward()`

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

### **Model Forward Pass Stage** (Shared with RGB + Density)
**Location**: `NerfactoField.forward()` → `FeatureFieldModel.get_outputs()`

```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)

# Normal outputs from NerfactoField:
normal_samples = field_outputs[FieldHeadNames.NORMALS]        # [N_rays, N_samples, 3]
pred_normal_samples = field_outputs[FieldHeadNames.PRED_NORMALS]  # [N_rays, N_samples, 3]
```

#### Neural Architecture

Normals are predicted using geometry features from the shared density network via `NerfactoField`:

##### Shared Geometry Features (with RGB + Density)
```python
# From NerfactoField geometry network
pos_encoded = HashGrid(xyz) + NeRFEncoding(xyz)  # R^44
geo_output = GeometryMLP(pos_encoded)            # R^16

density = geo_output[..., 0:1]      # R^1
geo_features = geo_output[..., 1:]  # R^15 (shared with RGB head)
```

##### Predicted Normals Head (Normals-specific)
```python
# Dedicated MLP for normal prediction via PredNormalsFieldHead
class PredNormalsFieldHead(FieldHead):
    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Tanh()) -> None:
        super().__init__(in_dim=in_dim, out_dim=3, field_head_name=FieldHeadNames.PRED_NORMALS, activation=activation)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Needed to normalize the output into valid normals."""
        out_tensor = super().forward(in_tensor)  # Linear layer + Tanh activation
        out_tensor = torch.nn.functional.normalize(out_tensor, dim=-1)  # L2 normalize
        return out_tensor
```

##### Ground-Truth Normals (Via Gradients)
```python
# Automatic differentiation of density via Field.get_normals()
def get_normals(self) -> Float[Tensor, "*batch 3"]:
    """Computes and returns a tensor of normals."""
    assert self._sample_locations is not None, "Sample locations must be set before calling get_normals."
    assert self._density_before_activation is not None, "Density must be set before calling get_normals."
    
    # Compute gradients w.r.t. positions
    normals = torch.autograd.grad(
        self._density_before_activation,
        self._sample_locations,
        grad_outputs=torch.ones_like(self._density_before_activation),
        retain_graph=True,
    )[0]  # [N_rays, N_samples, 3]
    
    # Normalize to unit vectors (negative for outward direction)
    normals = -torch.nn.functional.normalize(normals, dim=-1)
    return normals
```

### **Post-Processing Stage** (Independent rendering, shared weights)
**Location**: `NormalsRenderer.forward()` → `FeatureFieldModel.get_outputs()`

```python
# Shared: Volume rendering weights (computed from density)
weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

# Independent: Normal-specific rendering
normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
```

#### Normals Renderer Implementation
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/renderers.py
class NormalsRenderer(nn.Module):
    def forward(self, normals: Float[Tensor, "*bs num_samples 3"], 
                weights: Float[Tensor, "*bs num_samples 1"]) -> Float[Tensor, "*bs 3"]:
        """Weighted sum of normal vectors along ray."""
        normal_output = torch.sum(weights * normals, dim=-2)  # [N_rays, 3]
        return normal_output
```

## Training Data

### No Direct Normal Supervision
Like density, normals have **no direct ground truth** from images. Training uses:

1. **Orientation loss** - normals should align with ray directions at surfaces
2. **Predicted normal loss** - predicted normals match gradient-based normals
3. **Eikonal loss** (optional) - gradient magnitude regularization

### Orientation Loss Intuition
At surfaces, normals should be roughly perpendicular to the viewing direction:
```
surface normal ⊥ ray direction  ⟹  n · d ≈ 0
```

## Cross-Head Dependencies

### **Dependencies on Density Head**
- **Critical**: Normal rendering uses density weights via `ray_samples.get_weights(density)`
- **Shared Geometry**: Both use same geometry features from `GeometryMLP`
- **Gradient Computation**: Ground-truth normals computed from density gradients
- **Location**: `NerfactoField.get_normals()` and `ray_samples.get_weights()`

### **Dependencies on RGB Head**
- **Shared Geometry**: Both use same geometry features from `GeometryMLP`
- **Independent Computation**: Normals don't directly use RGB outputs
- **Location**: `NerfactoField.get_outputs()` (shared geometry network)

### **Independence from Feature Head**
- **Separate Computation**: Normals don't use feature outputs
- **Separate Architecture**: Features use `FeatureField`, normals use `NerfactoField`
- **Location**: `NerfactoField.forward()` vs `FeatureField.forward()`

## Training vs Inference Differences

### **Training-Only Operations**
```python
# Location: FeatureFieldModel.get_outputs() (training=True)
if self.training:
    # Store intermediate results for loss computation
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

### **Inference-Only Operations**
```python
# Location: FeatureFieldModel.get_outputs_for_camera_ray_bundle()
# Full image rendering in chunks
num_rays_per_chunk = 1 << 14  # From FeatureFieldModelConfig
# Process rays in chunks to manage memory
```

### **Shared Operations**
- Camera ray generation via `VanillaDataManager`
- Proposal sampling via `ProposalNetworkSampler`
- Spatial distortion via `SceneContraction`
- Field evaluation via `NerfactoField.forward()`
- Volume rendering via `NormalsRenderer`

## Loss Functions

### Orientation Loss
**Location**: `/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/losses.py`

```python
def orientation_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    viewdirs: Float[Tensor, "*bs 3"],
):
    """Orientation loss proposed in Ref-NeRF.
    Loss that encourages that all visible normals are facing towards the camera.
    """
    w = weights
    n = normals
    v = viewdirs * -1  # Flip view directions
    n_dot_v = (n * v[..., None, :]).sum(dim=-1)
    return (w[..., 0] * torch.fmin(torch.zeros_like(n_dot_v), n_dot_v) ** 2).sum(dim=-1)
```

### Predicted Normal Loss
**Location**: `/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/losses.py`

```python
def pred_normal_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    pred_normals: Float[Tensor, "*bs num_samples 3"],
):
    """Loss between normals calculated from density and normals from prediction network."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)
```

#### Properties
- **Orientation Loss Weight**: 0.0001 (from `NerfactoModelConfig.orientation_loss_mult`)
- **Predicted Normal Loss Weight**: 0.001 (from `NerfactoModelConfig.pred_normal_loss_mult`)
- **Purpose**: Ensure geometric consistency and surface quality
- **Training signal**: Self-supervised from density gradients and ray directions

## Neural Network Architecture Details

### Hash Grid Encoding (Shared with RGB + Density)
```python
# From NerfactoModelConfig
rgb_hash_config = {
    "num_levels": 16,                 # Multi-resolution levels
    "min_res": 16,                    # Base resolution
    "max_res": 2048,                  # Maximum resolution
    "log2_hashmap_size": 19,          # 2^19 = 512K entries per level
    "features_per_level": 2,          # Features per grid cell
    "implementation": "tcnn"          # TinyCUDA implementation
}

# Multi-resolution encoding via HashEncoding
encoded_pos = HashEncoding(contracted_pos)  # R^3 → R^(16*2) = R^32
```

### Geometry MLP (Shared with RGB + Density)
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/fields/nerfacto_field.py
self.mlp_base_mlp = MLP(
    in_dim=self.mlp_base_grid.get_out_dim(),  # 32 (hash grid)
    num_layers=2,                              # Hidden layers
    layer_width=64,                            # Hidden dimension
    out_dim=16,                                # 1 (density) + 15 (geometry features)
    activation=nn.ReLU(),
    out_activation=None,
    implementation=self.config.implementation,
)
```

### Predicted Normals Head (Normals-specific)
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/field_heads.py
class PredNormalsFieldHead(FieldHead):
    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Tanh()) -> None:
        super().__init__(in_dim=in_dim, out_dim=3, field_head_name=FieldHeadNames.PRED_NORMALS, activation=activation)

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Needed to normalize the output into valid normals."""
        out_tensor = super().forward(in_tensor)  # Linear layer + Tanh activation
        out_tensor = torch.nn.functional.normalize(out_tensor, dim=-1)  # L2 normalize
        return out_tensor
```

## Performance Characteristics

### Memory Usage
- **Hash Grid**: 16 × 2 × 2^19 × 4 bytes = 64MB (shared with RGB + Density)
- **MLP**: ~2M parameters for geometry network (shared)
- **PredNormals Head**: ~0.1M parameters (normals-specific)
- **Total**: ~66.1M parameters (mostly shared)

### Computational Complexity
- **Hash Grid Lookup**: O(1) per sample via TinyCUDA-NN
- **MLP Evaluation**: O(1) per sample
- **Gradient Computation**: O(1) per sample (automatic differentiation)
- **Volume Rendering**: O(N_samples) weighted sum per ray

### Quality Metrics
- **Geometric Accuracy**: Surface normal consistency
- **Orientation Quality**: Proper surface orientation
- **Training Stability**: Loss convergence and gradient flow

## Key Role in Unified Pipeline

### **Geometric Supervision**
The normals head provides geometric supervision to improve the overall scene quality:

1. **Surface Quality**: Ensures proper surface normals and orientations
2. **Geometric Consistency**: Aligns predicted normals with gradient-based normals
3. **Training Signal**: Provides additional supervision beyond RGB reconstruction
4. **Visual Quality**: Improves surface appearance and lighting consistency

### **Shared Architecture Benefits**
- **Computational Efficiency**: Uses same geometry network as RGB and density
- **Consistency**: All heads use same 3D scene representation
- **Memory Efficiency**: Shared hash grid and MLP parameters
- **Training Stability**: Coordinated optimization of all representations

### **Enabling Normal Computation**
```python
# From model config
predict_normals: bool = True  # Enable predicted normals head
```

The normals head enhances F3RM's geometric understanding by providing surface normal supervision, improving both visual quality and geometric consistency through shared architecture with the RGB and density heads. 