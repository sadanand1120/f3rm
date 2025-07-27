# F3RM Technical Documentation

## Overview

F3RM (Feature Field for 3D Reconstruction and Manipulation) is a modified Nerfacto model that incorporates a feature head to distill 2D foundation models (like CLIP and DINO) into 3D neural fields. The system consists of **neural network heads** (with learnable parameters) and **computed outputs** (mathematical computations) via `FeatureFieldModel` (`f3rm/model.py`) which inherits from `NerfactoModel` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/nerfacto.py`).

## Neural Network Heads vs Computed Outputs

### **Neural Network Heads** (Learnable Parameters)
- **RGB Head**: `NerfactoField.mlp_head` - Standard NeRF color outputs via learnable MLP
- **Density Head**: `NerfactoField.mlp_base_mlp` - Volume density for ray marching via learnable MLP  
- **Feature Head**: `FeatureField.field` - High-dimensional feature vectors via learnable MLP
- **Predicted Normals Head**: `NerfactoField.mlp_pred_normals` + `PredNormalsFieldHead` - Surface normals via learnable MLP

### **Computed Outputs** (Mathematical Computations)
- **Ground Truth Normals**: Computed from density gradients via automatic differentiation
- **Depth**: Computed from weights and ray samples via `DepthRenderer`
- **Accumulation**: Computed from weights via `AccumulationRenderer` 
- **Expected Depth**: Computed from weights and ray samples via `DepthRenderer`

## Documentation Structure

### Field Heads (Neural Network Components)
- **[RGB Head](rgb_head.md)** - `NerfactoField.mlp_head` learns color outputs `f_rgb: (x,y,z,θ,φ) → (r,g,b)`
- **[Density Head](density_head.md)** - `NerfactoField.mlp_base_mlp` learns volume density `f_σ: (x,y,z) → σ`
- **[Feature Head](feature_head.md)** - `FeatureField.field` learns feature vectors `f_feat: (x,y,z) → f^d`
- **[Predicted Normals Head](normals_head.md)** - `NerfactoField.mlp_pred_normals` learns surface normals `f_n: (x,y,z) → (n_x,n_y,n_z)`

### Computed Outputs (Mathematical Components)
- **[Depth Output](depth_output.md)** - `DepthRenderer` computes ray distances `f_depth: (weights,ray_samples) → depth`
- **[Accumulation Output](accumulation_output.md)** - `AccumulationRenderer` computes ray opacity `f_accumulation: (weights) → accumulation`
- **[Ground Truth Normals Output](ground_truth_normals_output.md)** - `Field.get_normals()` computes surface normals `f_gt_normals: ∇σ(x,y,z) → (n_x,n_y,n_z)`

### Technical Deep Dives
- **[Ray Pipeline](ray_pipeline.md)** - Complete pre/model_fwd/post stage analysis via `FeatureFieldModel.get_outputs()`
- **[Model Architecture](model_architecture.md)** - Neural network architecture details via `FeatureFieldModelConfig` and `NerfactoModelConfig`
- **[Combined Losses](losses_combined.md)** - Training objectives via `FeatureFieldModel.get_loss_dict()` in `f3rm/model.py`
- **[Hyperparameters](hyperparameters.md)** - Configuration and tuning via `F3RMTrainerConfig` in `f3rm/trainer.py`

## Multi-Head Architecture

F3RM learns multiple representations via `FeatureFieldModel`:

- **RGB Field**: `NerfactoField.get_outputs()` produces photorealistic colors
- **Density Field**: `NerfactoField.get_density()` provides geometric foundation  
- **Feature Field**: `FeatureField.forward()` enables semantic understanding
- **Predicted Normals Field**: `NerfactoField.get_outputs()` with `predict_normals=True` improves geometry

## Core Innovation

The key innovation is **feature distillation** via `FeatureFieldModel.get_outputs()` (`f3rm/model.py` lines 198-260), where 2D foundation model features are distilled into a 3D neural field that shares volume rendering weights with the RGB field for consistency.

## Mathematical Foundation

The system uses volume rendering with discretized integration via `ray_samples.get_weights()`:
```
Ĉ = Σᵢ wᵢ Qᵢ
wᵢ = αᵢ Πⱼ₌₁ᶦ⁻¹ (1-αⱼ)  
αᵢ = 1 - exp(-σᵢδᵢ)
```

## Training vs Inference Operations

### Training Only Operations
- **Ground truth feature extraction**: `FeatureDataManager.next_train()` (`f3rm/feature_datamanager.py` lines 147-161) extracts features from images
- **Feature distillation**: `extract_features_for_dataset()` processes foundation model outputs
- **Multi-head training**: `FeatureFieldModel.get_outputs()` (`f3rm/model.py` lines 198-260) evaluates all heads
- **Loss computation**: `FeatureFieldModel.get_loss_dict()` (`f3rm/model.py` lines 279-295) combines multiple objectives
- **Proposal sampling**: `ProposalNetworkSampler` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/model_components/ray_samplers.py`) enables hierarchical sampling

### Inference Only Operations
- **Chunked rendering**: `eval_num_rays_per_chunk=1<<14` (`f3rm/f3rm_config.py`) for memory efficiency
- **Feature visualization**: `apply_pca_colormap_return_proj()` for PCA projection
- **Language queries**: `ViewerUtils.handle_language_queries()` (`f3rm/model.py` lines 65-85) for CLIP similarity
- **Camera optimization**: `CameraOptimizerConfig` (`f3rm/f3rm_config.py`) for pose refinement

### Shared Operations
- **Data management**: `VanillaDataManager` handles image loading and ray generation
- **Proposal networks**: `ProposalNetworkSampler` provides hierarchical sampling
- **Spatial distortion**: `SceneContraction` (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/field_components/spatial_distortions.py`) handles unbounded scenes
- **Field evaluation**: `NerfactoField.forward()` and `FeatureField.forward()` compute neural outputs
- **Volume rendering**: `RGBRenderer`, `FeatureRenderer`, etc. integrate along rays

## Getting Started

1. **Pipeline Overview**: Start with [ray_pipeline.md](ray_pipeline.md) to understand the complete flow via `FeatureFieldModel.get_outputs()`
2. **Architecture Details**: Review [model_architecture.md](model_architecture.md) for neural network structure via `FeatureFieldModelConfig` and `NerfactoModelConfig`
3. **Hyperparameter Tuning**: Consult [hyperparameters.md](hyperparameters.md) for configuration via `F3RMTrainerConfig` in `f3rm/trainer.py`

## Common Use Cases

- **Novel View Synthesis**: RGB field via `NerfactoField.get_outputs()` produces photorealistic images
- **Language-Guided Understanding**: Feature field via `FeatureField.forward()` enables text-to-3D queries
- **Geometric Analysis**: Predicted normals via `NerfactoField.mlp_pred_normals` improve surface quality
- **Semantic Segmentation**: Feature distillation via `FeatureFieldModel.get_outputs()` enables 3D understanding

## Technical Specifications

### Framework Components
- **PyTorch**: Core deep learning framework
- **TinyCUDA-NN**: Fast CUDA kernels via `tcnn.NetworkWithInputEncoding` in `f3rm/feature_field.py`
- **Nerfstudio**: Base NeRF infrastructure (`/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/`)
- **Open-CLIP**: Foundation model integration via `open_clip.create_model_and_transforms()`

### Architecture Details
- **Hash Grid Encoding**: Multi-resolution spatial encoding for RGB (`NerfactoModelConfig`) and Features (`FeatureFieldModelConfig`)
- **Volume Rendering**: Discretized integration via `ray_samples.get_weights()`
- **Feature Types**: CLIP (768D), DINO (384D), DINOCLIP (dual), ROBOPOINT (custom) via `FeatureDataManagerConfig`

### Configuration Files
- **Main Config**: `f3rm/f3rm_config.py` - Overall training configuration
- **Model Config**: `FeatureFieldModelConfig` in `f3rm/model.py` - Feature field parameters
- **Trainer Config**: `F3RMTrainerConfig` in `f3rm/trainer.py` - Training and seeding parameters
- **Data Config**: `FeatureDataManagerConfig` in `f3rm/feature_datamanager.py` - Data loading parameters

## Performance Characteristics

### Model Parameters
- **RGB Field**: ~66M parameters (shared hash grid + MLPs)
- **Feature Field**: ~192.5M parameters (separate hash grid + MLP)
- **Proposal Networks**: ~2M parameters (hierarchical sampling)

### Memory Usage
- **GPU Memory**: ~12GB with default settings
- **Batch Size**: 8192 rays per batch via `train_num_rays_per_batch`
- **Chunk Size**: 16384 rays per chunk via `eval_num_rays_per_chunk=1<<14`

### Training Performance
- **Speed**: ~30K iterations in 2-3 hours on RTX 3090
- **Learning Rate**: 1e-3 for fields, 1e-3 for feature field via `AdamOptimizerConfig`
- **Optimizer**: Adam with exponential decay via `ExponentialDecaySchedulerConfig`
- **Mixed Precision**: Enabled for faster training

### Feature Dimensions
- **CLIP ViT-L-14-336**: 768-dimensional features
- **DINO ViT-S**: 384-dimensional features  
- **DINOCLIP**: Dual features (768D + 384D)
- **ROBOPOINT**: Custom dimensions for robotics

### Quality Metrics
- **RGB Quality**: PSNR 25-30 dB, SSIM >0.9, LPIPS <0.1
- **Feature Quality**: Cosine similarity with ground truth features
- **Normal Quality**: Geometric consistency and surface orientation

## Implementation Details

### Key Classes and Files
- **FeatureFieldModel**: `f3rm/model.py` - Main model class (lines 125-337)
- **FeatureField**: `f3rm/feature_field.py` - Feature neural field (lines 1-100)
- **FeatureDataManager**: `f3rm/feature_datamanager.py` - Data loading (lines 147-161)
- **FeatureRenderer**: `f3rm/renderer.py` - Volume rendering for features
- **NerfactoField**: `/opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/fields/nerfacto_field.py` - RGB/density field

### Core Methods
- **get_outputs()**: `f3rm/model.py` lines 198-260 - Main forward pass
- **get_loss_dict()**: `f3rm/model.py` lines 279-295 - Loss computation
- **get_outputs_for_camera_ray_bundle()**: `f3rm/model.py` lines 296-337 - Inference
- **forward()**: `f3rm/feature_field.py` - Feature field evaluation
- **next_train()**: `f3rm/feature_datamanager.py` lines 147-161 - Training data loading

This documentation provides a comprehensive technical understanding of F3RM's architecture, implementation, and usage with concrete file paths and method locations throughout. 