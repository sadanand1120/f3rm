# Ground Truth Normals Output: Density Gradient Computation

## Overview

The ground truth normals output computes surface normal vectors from density field gradients using automatic differentiation. Unlike neural network heads, ground truth normals are **computed outputs** with no learnable parameters - they're derived mathematically from density gradients via `Field.get_normals()` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/fields/base_field.py`).

## Computed Output vs Neural Network Head

### **Ground Truth Normals Output** (Computed Output - No Learnable Parameters)
- **Location**: `Field.get_normals()` - Mathematical computation via automatic differentiation
- **Function**: Computes surface normals from density field gradients
- **Parameters**: None (computed)
- **Output**: `f_gt_normals: ∇σ(x,y,z) → (n_x,n_y,n_z)` where ||n|| = 1

### **Physical Interpretation**
Ground truth normals represent the direction of steepest density increase:
- **Surface Normals**: Point outward from surfaces where density is high
- **Gradient Direction**: Normal to density level sets
- **Geometric Consistency**: Aligned with actual surface geometry

## Mathematical Formulation

### Function Signature
```
f_gt_normals: ∇σ(x,y,z) → (n_x, n_y, n_z) where ||n|| = 1
```
Where ∇σ is the gradient of the density field at position (x,y,z).

### Volume Rendering Foundation
Ground truth normal computation relies on the same density field used by all heads:
```
density = field_outputs[FieldHeadNames.DENSITY]  # From NerfactoField.get_density()
```

### Computation Method
```python
# From Field.get_normals() - Automatic differentiation
normals = torch.autograd.grad(
    density_before_activation,
    sample_locations,
    grad_outputs=torch.ones_like(density_before_activation),
    retain_graph=True,
)[0]  # [N_rays, N_samples, 3]

# Normalize to unit vectors (negative for outward direction)
normals = -torch.nn.functional.normalize(normals, dim=-1)
```

### Mathematical Derivation
The ground truth normals are computed as the negative gradient of the density field:
```
n_gt = -∇σ / ||∇σ||
```
Where:
- **∇σ**: Gradient of density field w.r.t. position (x,y,z)
- **Negative sign**: Ensures normals point outward from surfaces
- **Normalization**: Ensures unit vector magnitude

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

### **Model Forward Pass Stage** (Shared with RGB + Density + Predicted Normals)
**Location**: `NerfactoField.forward()` → `FeatureFieldModel.get_outputs()`

```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)

# Ground truth normals computed during field evaluation
gt_normal_samples = field_outputs[FieldHeadNames.NORMALS]  # [N_rays, N_samples, 3]
```

#### Neural Architecture

Ground truth normals are computed from the shared density network via `NerfactoField`:

##### Shared Geometry Features (with RGB + Density + Predicted Normals)
```python
# From NerfactoField geometry network
pos_encoded = HashGrid(xyz) + NeRFEncoding(xyz)  # R^44
geo_output = GeometryMLP(pos_encoded)            # R^16

density = geo_output[..., 0:1]      # R^1
geo_features = geo_output[..., 1:]  # R^15 (shared with RGB head)
```

##### Ground Truth Normals Computation (Via Gradients)
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/fields/base_field.py
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
gt_normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
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
Ground truth normals have **no direct ground truth** from images. They're computed from:

1. **Density gradients**: From neural network density predictions via automatic differentiation
2. **Mathematical computation**: Using gradient computation principles
3. **Physical interpretation**: Represents actual surface geometry

### Normal Properties
```python
# Physical constraints
||n_gt|| = 1  # Unit vectors
n_gt ∈ R³     # 3D normal vectors
```

## Cross-Head Dependencies

### **Critical Dependencies on Density Head**
- **Essential**: Ground truth normal computation uses density gradients via automatic differentiation
- **Shared Density Field**: Same density predictions used by RGB, features, depth, accumulation
- **Location**: `NerfactoField.get_density()` → `Field.get_normals()`

### **Dependencies on Proposal Sampling**
- **Sample Positions**: Ground truth normals computed at sampled positions
- **Position Gradients**: Automatic differentiation w.r.t. sample locations
- **Location**: `ProposalNetworkSampler.generate_ray_samples()`

### **Dependencies on Predicted Normals Head**
- **Supervision Target**: Ground truth normals provide supervision for predicted normals
- **Loss Computation**: Used in `pred_normal_loss()` to train predicted normals
- **Location**: `NerfactoModel.get_loss_dict()` - normal loss computation

### **Independence from RGB and Feature Heads**
- **No Direct Usage**: RGB and feature heads don't directly use ground truth normals
- **Indirect Influence**: Better geometry from normals improves overall scene quality
- **Location**: Independent computation via automatic differentiation

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
        field_outputs[FieldHeadNames.NORMALS].detach(),  # Ground truth normals
        field_outputs[FieldHeadNames.PRED_NORMALS],      # Predicted normals
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

## Loss Function

### No Direct Ground Truth Normal Loss
Ground truth normals have **no direct loss function** since they're computed outputs:

1. **No Ground Truth**: No normal supervision from images
2. **Supervision Target**: Used to supervise predicted normals
3. **Indirect Training**: Improved through density loss and normal supervision

### Supervision of Predicted Normals
Ground truth normals provide supervision for the predicted normals head:

```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/losses.py
def pred_normal_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],      # Ground truth normals
    pred_normals: Float[Tensor, "*bs num_samples 3"], # Predicted normals
):
    """Loss between normals calculated from density and normals from prediction network."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)
```

### Orientation Loss
Ground truth normals are also used in orientation loss:

```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/losses.py
def orientation_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],  # Ground truth normals
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

## Mathematical Details

### Gradient Computation
```python
# From Field.get_normals() - Automatic differentiation
normals = torch.autograd.grad(
    density_before_activation,  # Scalar density values
    sample_locations,          # 3D positions (x,y,z)
    grad_outputs=torch.ones_like(density_before_activation),
    retain_graph=True,
)[0]  # [N_rays, N_samples, 3]
```

### Normal Normalization
```python
# Ensure unit vectors pointing outward
normals = -torch.nn.functional.normalize(normals, dim=-1)
# Equivalent to: n_gt = -∇σ / ||∇σ||
```

### Volume Rendering Integration
```python
# Weighted sum along ray
gt_normals_rendered = torch.sum(weights * gt_normals, dim=-2)  # [N_rays, 3]
```

## Performance Characteristics

### Memory Usage
- **No Parameters**: Zero learnable parameters (computed output)
- **Gradient Storage**: Requires storing gradients for automatic differentiation
- **Shared Memory**: Uses same density field as other heads

### Computational Complexity
- **Gradient Computation**: O(1) per sample via automatic differentiation
- **Normalization**: O(1) per sample for L2 normalization
- **Memory Overhead**: Requires gradient computation graph

### Quality Metrics
- **Geometric Accuracy**: Normal consistency with scene geometry
- **Surface Quality**: Accurate surface normal estimation
- **Training Stability**: Stable gradient computation

## Key Role in Unified Pipeline

### **Geometric Supervision**
The ground truth normals output provides essential geometric supervision:

1. **Predicted Normal Training**: Supervises the predicted normals neural network head
2. **Geometric Consistency**: Ensures predicted normals match actual surface geometry
3. **Surface Quality**: Improves overall scene surface quality
4. **Training Signal**: Provides additional supervision beyond RGB reconstruction

### **Shared Architecture Benefits**
- **Computational Efficiency**: Uses same density field as other heads
- **Consistency**: All outputs use same 3D scene representation
- **Memory Efficiency**: No additional parameters (computed from existing density)
- **Training Stability**: Coordinated with density head optimization

### **Enabling Normal Supervision**
```python
# From model config
predict_normals: bool = True  # Enable predicted normals head and ground truth computation
```

### **Usage in F3RM**
```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
if self.config.predict_normals:
    gt_normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
    pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
    outputs["normals"] = self.normals_shader(gt_normals)      # Ground truth normals
    outputs["pred_normals"] = self.normals_shader(pred_normals)  # Predicted normals
```

### **Loss Computation**
```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 279-295)
if self.training and self.config.predict_normals:
    # Orientation loss using ground truth normals
    outputs["rendered_orientation_loss"] = orientation_loss(
        weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
    )
    # Predicted normal loss using ground truth normals as supervision
    outputs["rendered_pred_normal_loss"] = pred_normal_loss(
        weights.detach(),
        field_outputs[FieldHeadNames.NORMALS].detach(),  # Ground truth normals
        field_outputs[FieldHeadNames.PRED_NORMALS],      # Predicted normals
    )
```

The ground truth normals output provides crucial geometric supervision for the predicted normals head, computed efficiently from the same density field used by all other heads in the unified F3RM pipeline. 