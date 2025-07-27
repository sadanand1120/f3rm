# Hyperparameters: Effects on Quality and Performance

## Overview

F3RM has numerous hyperparameters controlling model capacity, training dynamics, and quality trade-offs. This guide explains the most impactful parameters and provides tuning guidelines for different scenarios via the `FeatureFieldModelConfig` (`f3rm/model.py`), `NerfactoModelConfig` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/nerfacto.py`), and `F3RMTrainerConfig` (`f3rm/trainer.py`).

## Critical Architecture Parameters

### 1. Hash Grid Configuration

#### RGB Field Hash Grid via `NerfactoModelConfig`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/nerfacto.py
@dataclass
class NerfactoModelConfig(ModelConfig):
    num_levels: int = 16              # Multi-resolution levels
    base_res: int = 16               # Coarsest resolution
    max_res: int = 2048              # Finest resolution  
    log2_hashmap_size: int = 19      # 2^19 = 512K entries per level
    features_per_level: int = 2      # Features per grid cell
```

#### Feature Field Hash Grid via `FeatureFieldModelConfig`
```python
# From f3rm/model.py
@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    # Feature field uses different parameters
    feat_num_levels: int = 12           # Fewer levels than RGB
    feat_start_res: int = 16           # Same coarse resolution
    feat_max_res: int = 128            # Much lower max resolution
    feat_features_per_level: int = 8    # More features per cell
    feat_log2_hashmap_size: int = 19   # Same memory budget
```

#### Effects on Quality and Performance

**RGB Field Resolution (`max_res`)**
```python
# Low resolution (1024)
- Pro: Fast training/inference, low memory
- Con: Blurry textures, missing fine details
- Use case: Quick prototyping, limited GPU memory

# Medium resolution (2048) - Default
- Pro: Good detail/speed balance
- Con: Some fine textures may be blurred  
- Use case: Most production scenarios

# High resolution (4096)
- Pro: Exceptional detail, sharp textures
- Con: Slow training, high memory usage (>16GB)
- Use case: High-quality final results, powerful GPUs
```

**Hash Grid Levels (`num_levels`)**
```python
# Fewer levels (8-12)
- Pro: Faster training, less memory
- Con: Limited multi-scale representation
- Effect: May miss fine details or coarse structure

# More levels (16-20)  
- Pro: Rich multi-scale features
- Con: Diminishing returns, more memory
- Effect: Better quality but slower convergence
```

**Hash Table Size (`log2_hashmap_size`)**
```python
# Small tables (17: 128K entries)
- Pro: Low memory usage
- Con: Hash collisions reduce quality
- Effect: Blurry reconstruction, training instability

# Large tables (21: 2M entries)
- Pro: Fewer collisions, better quality
- Con: High memory usage
- Effect: Sharper results but may not fit on GPU
```

### 2. MLP Architecture

#### RGB Field MLPs via `NerfactoModelConfig`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/nerfacto.py
@dataclass
class NerfactoModelConfig(ModelConfig):
    hidden_dim: int = 64                # Geometry MLP width
    hidden_dim_color: int = 64          # Color MLP width
    # Geometry MLP depth is controlled by NerfactoField constructor
    # Color MLP depth is controlled by NerfactoField constructor
```

#### Feature Field MLP via `FeatureFieldModelConfig`
```python
# From f3rm/model.py
@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    feat_hidden_dim: int = 64           # Feature MLP width
    feat_num_layers: int = 2            # Feature MLP depth
```

#### Effects on Capacity and Speed

**MLP Width (`hidden_dim`)**
```python
# Narrow MLPs (32)
- Pro: Fast inference, low memory
- Con: Limited representational capacity
- Effect: May underfit complex scenes

# Wide MLPs (128)
- Pro: High capacity, better quality
- Con: Slower training/inference
- Effect: Can overfit on small datasets
```

**MLP Depth (`num_layers`)**
```python  
# Shallow networks (1-2 layers)
- Pro: Fast, stable training
- Con: Limited non-linearity
- Effect: Good for simple scenes

# Deep networks (3-4 layers)
- Pro: Complex representations
- Con: Harder to train, overfitting risk
- Effect: Better for complex lighting/materials
```

## Training Hyperparameters

### 3. Loss Weights

#### Feature Loss Weight via `FeatureFieldModelConfig`
```python
# From f3rm/model.py
@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    feat_loss_weight: float = 1e-3      # Core F3RM parameter
```

**Impact on RGB vs Feature Quality**
```python
# Very low (1e-5): RGB-focused training
results = {
    "rgb_quality": "Excellent",
    "feature_quality": "Poor", 
    "language_queries": "Inaccurate",
    "training_time": "Fast"
}

# Balanced (1e-3): Default setting  
results = {
    "rgb_quality": "Good", 
    "feature_quality": "Good",
    "language_queries": "Accurate",
    "training_time": "Standard"
}

# High (1e-1): Feature-focused training
results = {
    "rgb_quality": "Poor",
    "feature_quality": "Excellent", 
    "language_queries": "Very accurate",
    "training_time": "Slow"
}
```

#### Normal Loss Weights via `NerfactoModelConfig`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/nerfacto.py
@dataclass
class NerfactoModelConfig(ModelConfig):
    orientation_loss_mult: float = 0.0001     # Normal-ray orthogonality
    pred_normal_loss_mult: float = 0.001      # Predicted vs GT normals
```

### 4. Sampling Parameters

#### Ray Sampling via `FeatureDataManagerConfig`
```python
# From f3rm/f3rm_config.py
datamanager=FeatureDataManagerConfig(
    train_num_rays_per_batch=8192,    # Rays per training step
    eval_num_rays_per_batch=4096,     # Rays per eval step
)
```

#### Hierarchical Sampling via `NerfactoModelConfig`
```python
# From /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/nerfacto.py
@dataclass
class NerfactoModelConfig(ModelConfig):
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)  # Proposal stages
    num_nerf_samples_per_ray: int = 48                         # Final network samples
    num_proposal_iterations: int = 2                           # Number of proposal stages
```

**Effects on Quality vs Speed**
```python
# Fewer samples (32 final)
- Pro: 50% faster training/inference
- Con: Noisy rendering, missing details
- Use: Real-time applications

# More samples (96 final)  
- Pro: Higher quality, less noise
- Con: 2x slower rendering
- Use: High-quality offline rendering
```

### 5. Training Schedule

#### Learning Rates via `F3RMTrainerConfig`
```python
# From f3rm/f3rm_config.py
optimizers={
    "proposal_networks": {
        "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000)
    },
    "fields": {  # RGB field
        "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000)
    },
    "feature_field": {
        "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15), 
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000)
    }
}
```

#### Training Steps via `F3RMTrainerConfig`
```python
# From f3rm/f3rm_config.py
config=F3RMTrainerConfig(
    max_num_iterations=30000,         # Total training steps
    steps_per_eval_batch=500,         # Evaluation frequency
    steps_per_save=5000,              # Checkpoint frequency
)
```

## Performance Optimization Parameters

### 6. Memory Management

#### Chunk Sizes via `FeatureFieldModelConfig`
```python
# From f3rm/f3rm_config.py
model=FeatureFieldModelConfig(
    eval_num_rays_per_chunk=1 << 14,  # 16384 rays per chunk (inference)
)
viewer=ViewerConfig(
    num_rays_per_chunk=1 << 15        # 32768 rays per chunk (viewer)
)
```

**Memory vs Speed Trade-offs**
```python
# Small chunks (8K rays)
- Pro: Fits on 8GB GPUs
- Con: More overhead, slightly slower
- Use: Limited GPU memory

# Large chunks (32K rays)
- Pro: Maximum throughput  
- Con: Requires 16GB+ GPU memory
- Use: High-end GPUs, batch processing
```

#### Mixed Precision via `F3RMTrainerConfig`
```python
# From f3rm/f3rm_config.py
config=F3RMTrainerConfig(
    mixed_precision=True,            # Enable FP16 training
)
```

### 7. Feature Extraction Parameters

#### Feature Field Resolution Trade-off
```python
# The key insight: Features need less spatial resolution than RGB
RGB_field: {
    "max_res": 2048,           # Fine spatial detail for textures
    "num_levels": 16,          # Many resolution levels
    "features_per_level": 2    # Few features per location
}

Feature_field: {
    "feat_max_res": 128,       # Coarse spatial resolution
    "feat_num_levels": 12,     # Fewer levels  
    "feat_features_per_level": 8  # Rich features per location
}
```

#### Feature Type via `FeatureDataManagerConfig`
```python
# From f3rm/f3rm_config.py
datamanager=FeatureDataManagerConfig(
    feature_type="CLIP",              # Foundation model type
    enable_cache=True,                # Cache extracted features
)
```

## Scenario-Specific Tuning

### 8. Quick Prototyping (Fast Results)
```python
# Optimized for speed, acceptable quality
config_overrides = {
    "max_res": 1024,                    # Lower resolution
    "feat_max_res": 64,                 # Very coarse features
    "num_nerf_samples_per_ray": 32,     # Fewer samples
    "max_num_iterations": 15000,        # Shorter training
    "feat_loss_weight": 1e-4,           # Focus on RGB
    "mixed_precision": True             # Memory efficiency
}
# Result: ~2x faster training, 80% of quality
```

### 9. High Quality Production
```python
# Optimized for maximum quality
config_overrides = {
    "max_res": 4096,                    # Highest resolution
    "feat_max_res": 256,                # Detailed features
    "num_nerf_samples_per_ray": 64,     # More samples
    "max_num_iterations": 50000,        # Longer training
    "feat_loss_weight": 2e-3,           # Balanced emphasis
    "hidden_dim": 128,                  # Wider networks
    "log2_hashmap_size": 20             # Larger hash tables
}
# Result: ~3x slower training, 120% of quality
```

### 10. Memory-Constrained Training
```python
# For 8GB GPUs or large scenes
config_overrides = {
    "train_num_rays_per_batch": 4096,   # Smaller batches
    "eval_num_rays_per_chunk": 8192,    # Smaller chunks  
    "max_res": 1024,                    # Lower resolution
    "log2_hashmap_size": 18,            # Smaller hash tables
    "mixed_precision": True,            # Essential for memory
    "gradient_accumulation_steps": 2     # Accumulate gradients
}
# Result: Fits on 8GB GPU with minimal quality loss
```

### 11. Language-Heavy Applications
```python
# Emphasize feature quality for semantic tasks
config_overrides = {
    "feat_loss_weight": 5e-3,           # Higher feature emphasis
    "feat_max_res": 256,                # Detailed feature field
    "feat_hidden_dim": 128,             # Wider feature MLP
    "feature_type": "CLIP",             # Best for language
    "max_num_iterations": 40000         # More training for features
}
# Result: Excellent language queries, good RGB quality
```

## Hyperparameter Interaction Effects

### 12. Common Problematic Combinations

#### Memory Issues
```python
# This combination will likely OOM on most GPUs
problematic_config = {
    "max_res": 4096,
    "log2_hashmap_size": 21,
    "train_num_rays_per_batch": 16384,
    "mixed_precision": False
}
# Memory usage: >24GB
```

#### Training Instability  
```python
# High learning rates + high loss weights = unstable training
unstable_config = {
    "lr": 5e-2,                        # Too high
    "feat_loss_weight": 1e-1,          # Too high
    "orientation_loss_mult": 1e-2      # Too high
}
# Result: Loss spikes, NaN values, poor convergence
```

#### Poor Quality
```python
# Under-parameterized for complex scenes
poor_quality_config = {
    "max_res": 512,                    # Too low
    "hidden_dim": 32,                  # Too narrow
    "num_nerf_samples_per_ray": 16,    # Too few samples
    "feat_loss_weight": 1e-5           # Features ignored
}
# Result: Blurry RGB, no semantic understanding
```

## Automatic Hyperparameter Selection

### 13. Rules of Thumb

#### Based on Scene Complexity
```python
def auto_config(scene_complexity):
    if scene_complexity == "simple":  # Single object, uniform lighting
        return {
            "max_res": 1024,
            "hidden_dim": 32,
            "feat_loss_weight": 1e-4
        }
    elif scene_complexity == "complex":  # Multiple objects, complex lighting
        return {
            "max_res": 2048, 
            "hidden_dim": 64,
            "feat_loss_weight": 1e-3
        }
    else:  # "very_complex" - outdoor scenes, fine details
        return {
            "max_res": 4096,
            "hidden_dim": 128, 
            "feat_loss_weight": 2e-3
        }
```

#### Based on GPU Memory
```python
def memory_config(gpu_memory_gb):
    if gpu_memory_gb < 12:
        return {
            "train_num_rays_per_batch": 4096,
            "max_res": 1024,
            "log2_hashmap_size": 18,
            "mixed_precision": True
        }
    elif gpu_memory_gb < 24:
        return {
            "train_num_rays_per_batch": 8192,  # Default
            "max_res": 2048,                   # Default
            "log2_hashmap_size": 19,           # Default
            "mixed_precision": True
        }
    else:
        return {
            "train_num_rays_per_batch": 16384,
            "max_res": 4096,
            "log2_hashmap_size": 20,
            "mixed_precision": False  # Can afford FP32
        }
```

## Additional Critical Parameters

### 14. Seeding Configuration via `F3RMTrainerConfig`
```python
# From f3rm/trainer.py
@dataclass
class F3RMTrainerConfig(TrainerConfig):
    enable_comprehensive_seeding: bool = True
    """Whether to enable comprehensive seeding for reproducibility."""
    seed_deterministic_algorithms: bool = True
    """Whether to use deterministic algorithms (may impact performance)."""
    seed_warn_only: bool = False
    """If True, only warn about non-deterministic operations instead of erroring."""
    seed_cublas_workspace: bool = True
    """Whether to configure cuBLAS workspace for deterministic behavior."""
    print_seed_info: bool = True
    """Whether to print detailed seeding information at startup."""
```

### 15. Camera Optimization via `CameraOptimizerConfig`
```python
# From f3rm/f3rm_config.py
camera_optimizer=CameraOptimizerConfig(
    mode="SO3xR3",                    # Camera pose optimization
    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
)
```

### 16. Feature Field Positional Encoding via `FeatureFieldModelConfig`
```python
# From f3rm/model.py
@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    feat_use_pe: bool = True          # Enable positional encoding
    feat_pe_n_freq: int = 6           # Number of frequency bands
```

The key to successful F3RM training is balancing quality, speed, and memory constraints through careful hyperparameter selection. Start with the defaults and adjust based on your specific requirements and hardware constraints via the coordinated configuration of `FeatureFieldModelConfig`, `NerfactoModelConfig`, and `F3RMTrainerConfig`. 