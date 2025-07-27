# Depth Output: Ray Distance Computation

## Overview

The depth output computes the distance along rays to scene surfaces using volume rendering weights. Unlike neural network heads, depth is a **computed output** with no learnable parameters - it's derived mathematically from density weights and ray sample positions via `DepthRenderer` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/renderers.py`).

## Computed Output vs Neural Network Head

### **Depth Output** (Computed Output - No Learnable Parameters)
- **Location**: `DepthRenderer.forward()` - Mathematical computation
- **Function**: Computes ray distances from volume rendering weights
- **Parameters**: None (computed)
- **Output**: `f_depth: (weights, ray_samples) → depth` where depth ∈ [near_plane, far_plane]

### **Two Depth Methods**
F3RM computes two types of depth outputs:

1. **Median Depth**: `DepthRenderer(method="median")` - Distance where cumulative weight reaches 0.5
2. **Expected Depth**: `DepthRenderer(method="expected")` - Weighted average of sample distances

## Mathematical Formulation

### Function Signature
```
f_depth: (weights, ray_samples) → depth ∈ [near_plane, far_plane]
```
Where `weights` are volume rendering weights and `ray_samples` contain sample positions along rays.

### Volume Rendering Foundation
Depth computation relies on the same volume rendering weights used by all heads:
```
weights = ray_samples.get_weights(density)  # Shared with RGB, features, normals
```

### Two Computation Methods

#### Median Depth (Default)
```python
# From DepthRenderer.forward() (method="median")
steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [..., num_samples]
split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5  # [..., 1]
median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)  # [..., 1]
median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)  # [..., 1]
```

#### Expected Depth
```python
# From DepthRenderer.forward() (method="expected")
eps = 1e-10
steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
depth = torch.clip(depth, steps.min(), steps.max())
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

### **Model Forward Pass Stage** (Shared with all heads)
**Location**: `NerfactoField.forward()` → `FeatureFieldModel.get_outputs()`

```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)

# Shared: Volume rendering weights (computed from density)
weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
```

### **Post-Processing Stage** (Independent computation)
**Location**: `DepthRenderer.forward()` → `FeatureFieldModel.get_outputs()`

```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
with torch.no_grad():
    depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
```

#### Depth Renderer Implementation
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/renderers.py
class DepthRenderer(nn.Module):
    def __init__(self, method: Literal["median", "expected"] = "median") -> None:
        super().__init__()
        self.method = method

    def forward(self, weights: Float[Tensor, "*batch num_samples 1"], 
                ray_samples: RaySamples) -> Float[Tensor, "*batch 1"]:
        """Compute depth along ray using specified method."""
        if self.method == "median":
            # Find distance where cumulative weight reaches 0.5
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
            cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)
            split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5
            median_index = torch.searchsorted(cumulative_weights, split, side="left")
            median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)
            median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)
            return median_depth
        elif self.method == "expected":
            # Weighted average of sample distances
            eps = 1e-10
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
            depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
            depth = torch.clip(depth, steps.min(), steps.max())
            return depth
```

## Training Data

### No Direct Depth Supervision
Depth has **no direct ground truth** from images. It's computed from:

1. **Density weights**: From neural network density predictions
2. **Ray sample positions**: From proposal sampling
3. **Mathematical computation**: Using volume rendering principles

### Depth Range
```python
# From NerfactoModelConfig
near_plane: float = 0.05    # Minimum depth
far_plane: float = 1000.0   # Maximum depth
```

## Cross-Head Dependencies

### **Critical Dependencies on Density Head**
- **Essential**: Depth computation uses density weights via `ray_samples.get_weights(density)`
- **Shared Weights**: Same volume rendering weights used by RGB, features, normals
- **Location**: `NerfactoField.get_density()` → `ray_samples.get_weights()`

### **Dependencies on Proposal Sampling**
- **Ray Sample Positions**: Depth computation requires `ray_samples.frustums.starts/ends`
- **Sample Distribution**: Proposal networks determine where depth samples are taken
- **Location**: `ProposalNetworkSampler.generate_ray_samples()`

### **Independence from Other Heads**
- **No Neural Parameters**: Depth doesn't use learnable parameters from RGB/feature/normal heads
- **Mathematical Computation**: Pure mathematical derivation from weights and positions
- **Location**: `DepthRenderer.forward()` (independent computation)

## Training vs Inference Differences

### **Training-Only Operations**
```python
# Location: FeatureFieldModel.get_outputs() (training=True)
if self.training:
    # Store intermediate results for loss computation
    outputs["weights_list"] = weights_list
    outputs["ray_samples_list"] = ray_samples_list

# Proposal depth outputs (training only)
for i in range(self.config.num_proposal_iterations):
    outputs[f"prop_depth_{i}"] = self.renderer_depth(
        weights=weights_list[i], ray_samples=ray_samples_list[i]
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
- Volume rendering weight computation via `ray_samples.get_weights()`

## Loss Function

### No Direct Depth Loss
Depth has **no direct loss function** since it's a computed output:

1. **No Ground Truth**: No depth supervision from images
2. **Indirect Supervision**: Depth quality depends on density head accuracy
3. **Implicit Training**: Improved through density loss and other objectives

### Indirect Quality Improvement
Depth quality improves through:
- **Density Loss**: Better density predictions → better depth estimates
- **RGB Loss**: Better scene understanding → better geometry → better depth
- **Feature Loss**: Better semantic understanding → better scene structure

## Mathematical Details

### Ray Sample Positions
```python
# From RaySamples.frustums
starts = ray_samples.frustums.starts  # [N_rays, N_samples, 1] - start distances
ends = ray_samples.frustums.ends      # [N_rays, N_samples, 1] - end distances
steps = (starts + ends) / 2           # [N_rays, N_samples, 1] - midpoints
```

### Weight Computation
```python
# From ray_samples.get_weights()
weights = ray_samples.get_weights(density)  # [N_rays, N_samples, 1]
# weights[i,j] = α[i,j] * Π[k=0 to j-1] (1 - α[i,k])
# where α[i,j] = 1 - exp(-σ[i,j] * δ[i,j])
```

### Median Depth Algorithm
```python
# Find index where cumulative weight reaches 0.5
cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [N_rays, N_samples]
median_index = torch.searchsorted(cumulative_weights, 0.5, side="left")  # [N_rays, 1]
median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)  # [N_rays, 1]
```

### Expected Depth Algorithm
```python
# Weighted average of sample distances
depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
# Equivalent to: Σᵢ wᵢ * dᵢ / Σᵢ wᵢ
```

## Performance Characteristics

### Memory Usage
- **No Parameters**: Zero learnable parameters (computed output)
- **Temporary Storage**: Only stores computed depth values
- **Shared Memory**: Uses same weights as other heads

### Computational Complexity
- **Median Depth**: O(N_samples) for cumulative sum + binary search
- **Expected Depth**: O(N_samples) for weighted sum
- **Memory Efficient**: No additional neural network evaluation

### Quality Metrics
- **Geometric Accuracy**: Depth consistency with scene geometry
- **Surface Detection**: Accurate surface distance estimation
- **Range Validity**: Depth values within [near_plane, far_plane]

## Key Role in Unified Pipeline

### **Geometric Understanding**
The depth output provides essential geometric information:

1. **Surface Distance**: Distance to nearest surfaces along rays
2. **Scene Structure**: Understanding of 3D scene layout
3. **Visualization**: Depth maps for scene analysis
4. **Quality Assessment**: Depth consistency indicates scene quality

### **Shared Architecture Benefits**
- **Computational Efficiency**: Uses same weights as other heads
- **Consistency**: All outputs use same 3D scene representation
- **Memory Efficiency**: No additional parameters or computations
- **Training Stability**: Coordinated with density head optimization

### **Two Depth Variants**
```python
# From NerfactoModel.populate_modules()
self.renderer_depth = DepthRenderer(method="median")        # Default depth
self.renderer_expected_depth = DepthRenderer(method="expected")  # Expected depth
```

### **Usage in F3RM**
```python
# From FeatureFieldModel.get_outputs() (f3rm/model.py lines 198-260)
outputs = {
    "rgb": rgb,
    "accumulation": accumulation,
    "depth": depth,                    # Median depth
    "expected_depth": expected_depth,  # Expected depth
}
```

The depth output provides crucial geometric information for scene understanding and visualization, computed efficiently from the same volume rendering weights used by all other heads in the unified F3RM pipeline. 