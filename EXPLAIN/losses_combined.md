# Combined Loss Functions: Training Objectives

## Overview

F3RM training involves multiple loss terms that work together to learn RGB reconstruction, feature distillation, geometric consistency, and regularization. Understanding the balance between these losses is crucial for successful training via the `FeatureFieldModel.get_loss_dict()` method (`f3rm/model.py`).

## Total Loss Formulation

### Combined Loss Equation
```
L_total = L_rgb + λ_feat L_feat + λ_orient L_orient + λ_pred_norm L_pred_norm + λ_dist L_dist + λ_inter L_inter
```

Where λ terms are loss weights that balance different objectives.

## Individual Loss Components

### 1. RGB Reconstruction Loss (Primary)

#### MSE Loss via `NerfactoModel.get_loss_dict()`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/nerfacto.py
def get_loss_dict(self, outputs, batch, metrics_dict=None):
    loss_dict = {}
    image = batch["image"].to(self.device)
    pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
        pred_image=outputs["rgb"],
        pred_accumulation=outputs["accumulation"],
        gt_image=image,
    )
    loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)  # MSE loss
```

#### Properties
- **Weight**: 1.0 (baseline, highest priority, no explicit scaling)
- **Purpose**: Ensure photorealistic novel view synthesis
- **Training signal**: Direct supervision from input images via `batch["image"]`
- **Range**: [0, 1] since RGB values are in [0,1]³
- **Background handling**: Uses `RGBRenderer.blend_background_for_loss_computation()` for proper alpha blending

### 2. Feature Distillation Loss (Core Innovation)  

#### MSE Feature Loss via `FeatureFieldModel.get_loss_dict()`
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

### 3. Normal Consistency Losses (Geometry)

#### Orientation Loss via `orientation_loss()`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/losses.py
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

#### Predicted Normal Loss via `pred_normal_loss()`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/losses.py
def pred_normal_loss(
    weights: Float[Tensor, "*bs num_samples 1"],
    normals: Float[Tensor, "*bs num_samples 3"],
    pred_normals: Float[Tensor, "*bs num_samples 3"],
):
    """Loss between normals calculated from density and normals from prediction network."""
    return (weights[..., 0] * (1.0 - torch.sum(normals * pred_normals, dim=-1))).sum(dim=-1)
```

#### Integration in Training via `FeatureFieldModel.get_outputs()`
```python
# From f3rm/model.py
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

#### Properties
- **Orientation weight**: 1e-4 (from `NerfactoModelConfig.orientation_loss_mult`)
- **Predicted normal weight**: 1e-3 (from `NerfactoModelConfig.pred_normal_loss_mult`)
- **Purpose**: Improve geometric consistency and normal quality
- **Cost**: Expensive due to gradient computation for GT normals
- **Training only**: Only computed during training, not inference

### 4. Regularization Losses

#### Distortion Loss via `distortion_loss()`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/losses.py
def distortion_loss(weights_list, ray_samples_list):
    """From mipnerf360"""
    c = ray_samples_to_sdist(ray_samples_list[-1])
    w = weights_list[-1][..., 0]
    loss = torch.mean(lossfun_distortion(c, w))
    return loss

def lossfun_distortion(t, w):
    """Core distortion loss function."""
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3
    return loss_inter + loss_intra
```

#### Interlevel Loss via `interlevel_loss()`
```python  
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/losses.py
def interlevel_loss(weights_list, ray_samples_list) -> torch.Tensor:
    """Calculates the proposal loss in the MipNeRF-360 paper."""
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    assert len(ray_samples_list) > 0

    loss_interlevel = 0.0
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp))

    return loss_interlevel
```

#### Properties
- **Distortion weight**: 2e-3 (from `NerfactoModelConfig.distortion_loss_mult`)
- **Interlevel weight**: 1.0 (from `NerfactoModelConfig.interlevel_loss_mult`)
- **Purpose**: Improve sampling efficiency and reduce artifacts
- **Training only**: Only computed during training via `metrics_dict["distortion"]`

## Loss Integration in Training

### Complete Loss Computation via `FeatureFieldModel.get_loss_dict()`
```python
# From f3rm/model.py
def get_loss_dict(self, outputs, batch, metrics_dict=None):
    # Base losses from parent (RGB, distortion, interlevel)
    loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
    
    # Feature distillation loss
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

### Loss Weights Configuration
```python
# From f3rm/model.py
@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    feat_loss_weight: float = 1e-3           # Feature distillation

# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/nerfacto.py
@dataclass
class NerfactoModelConfig(ModelConfig):
    # RGB loss weight = 1.0 (implicit, no scaling)
    distortion_loss_mult: float = 0.002      # Distortion regularization
    interlevel_loss_mult: float = 1.0        # Interlevel consistency
    orientation_loss_mult: float = 0.0001    # Normal orientation  
    pred_normal_loss_mult: float = 0.001     # Predicted normals
```

### Metrics Computation via `FeatureFieldModel.get_metrics_dict()`
```python
# From f3rm/model.py
def get_metrics_dict(self, outputs, batch):
    metrics_dict = super().get_metrics_dict(outputs, batch)
    ft = self.kwargs["metadata"]["feature_type"]
    if ft == "DINOCLIP":
        metrics_dict["feature_dino_error"] = F.mse_loss(
            outputs["feature_dino"], batch["feature_dino"].to(self.device)
        )
        metrics_dict["feature_clip_error"] = F.mse_loss(
            outputs["feature_clip"], batch["feature_clip"].to(self.device)
        )
    else:
        target_feats = batch["feature"].to(self.device)
        metrics_dict["feature_error"] = F.mse_loss(outputs["feature"], target_feats)
    return metrics_dict
```

## Loss Balancing Strategies

### Typical Loss Magnitudes (Well-Tuned Model)
```python
# After ~10K training steps
loss_magnitudes = {
    "rgb_loss": 0.005,              # MSE in [0,1]³ space
    "feature_loss": 0.5,            # MSE in feature space (768D)
    "distortion_loss": 0.001,       # Regularization  
    "interlevel_loss": 0.0005,      # Consistency
    "orientation_loss": 0.002,      # Normal geometry
    "pred_normal_loss": 0.1,        # Normal consistency
}

# Weighted contributions to total loss
weighted_contributions = {
    "rgb": 0.005 * 1.0 = 0.005,           # Dominant
    "feature": 0.5 * 1e-3 = 0.0005,       # Moderate  
    "distortion": 0.001 * 2e-3 = 2e-6,    # Small
    "interlevel": 0.0005 * 1.0 = 0.0005,  # Moderate
    "orientation": 0.002 * 1e-4 = 2e-7,   # Tiny
    "pred_normal": 0.1 * 1e-3 = 1e-4,     # Small
}

# Total loss ≈ 0.006 (dominated by RGB reconstruction)
```

### Hyperparameter Tuning Guidelines

#### Feature Loss Weight (`feat_loss_weight`)
```python
# Too high (1e-1): Features dominate, poor RGB quality
"Loss at 1e-1": {
    "rgb_loss": 0.05,      # High, poor reconstruction
    "feature_loss": 0.1,   # Low, good distillation  
    "total": 0.015         # Feature term dominates
}

# Too low (1e-5): Features undertrained, poor semantic quality  
"Loss at 1e-5": {
    "rgb_loss": 0.003,     # Low, good reconstruction
    "feature_loss": 50.0,  # High, poor distillation
    "total": 0.0035        # Feature term negligible
}

# Well-balanced (1e-3): Both RGB and features train well
"Loss at 1e-3": {
    "rgb_loss": 0.005,     # Moderate, good reconstruction
    "feature_loss": 0.5,   # Moderate, good distillation  
    "total": 0.0055        # Balanced contributions
}
```

#### Normal Loss Weights
```python
# High normal weights can hurt RGB quality
"High normal weights": {
    "orientation_loss_mult": 1e-2,   # Too high
    "pred_normal_loss_mult": 1e-1,   # Too high
    "result": "Overly smooth geometry, blurry RGB"
}

# Low normal weights reduce geometric quality
"Low normal weights": {
    "orientation_loss_mult": 1e-6,   # Too low  
    "pred_normal_loss_mult": 1e-5,   # Too low
    "result": "Good RGB, but inconsistent normals"
}
```

## Loss Scheduling (Advanced)

### Warmup Strategies
```python
def get_scheduled_loss_weight(step, base_weight, warmup_steps=5000):
    """Gradually increase loss weight during training."""
    if step < warmup_steps:
        # Linear warmup from 0 to base_weight
        return base_weight * (step / warmup_steps)
    else:
        return base_weight

# Usage
feat_weight = get_scheduled_loss_weight(step, 1e-3, warmup_steps=5000)
```

### Adaptive Weighting
```python
def adaptive_feature_weight(rgb_loss, feature_loss, base_weight=1e-3):
    """Adjust feature weight based on relative losses."""
    ratio = rgb_loss / (feature_loss + 1e-8)
    
    if ratio > 0.01:  # RGB loss too high
        return base_weight * 0.5  # Reduce feature weight
    elif ratio < 0.001:  # RGB loss too low
        return base_weight * 2.0  # Increase feature weight
    else:
        return base_weight
```

## Debugging Loss Issues

### Common Problems and Solutions

#### Poor RGB Quality
```python
# Symptoms: Blurry images, high RGB loss
# Causes: Feature loss too high, normal losses too high
# Solutions:
- Reduce feat_loss_weight: 1e-3 → 1e-4
- Reduce normal loss weights
- Check if feature extraction is correct via FeatureDataManager
```

#### Poor Feature Quality  
```python
# Symptoms: Bad language queries, high feature loss
# Causes: Feature loss too low, feature extraction issues
# Solutions:
- Increase feat_loss_weight: 1e-3 → 1e-2  
- Verify feature extraction pipeline via extract_features_standalone.py
- Check feature field architecture (too small?)
```

#### Training Instability
```python
# Symptoms: Loss spikes, NaN values
# Causes: Learning rates too high, gradient explosion
# Solutions:
- Use gradient clipping
- Reduce learning rates (currently 1e-2 for fields, 1e-2 for feature_field)
- Enable mixed precision training (already enabled in F3RM config)
- Check for numerical issues in losses
```

The loss function design is critical for F3RM's success, balancing photorealistic reconstruction with semantic understanding through carefully tuned multi-objective optimization via the coordinated execution of RGB reconstruction, feature distillation, geometric consistency, and regularization losses. 