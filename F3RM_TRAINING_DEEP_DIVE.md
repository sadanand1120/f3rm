# F3RM Training Arguments Deep Dive

This document provides a comprehensive understanding of how F3RM training arguments affect training performance, memory usage, and ETA. We'll explain each parameter from the perspective of the actual codebase implementation and the NeRF training process.

## Table of Contents

1. [Understanding F3RM Training Pipeline](#understanding-f3rm-training-pipeline)
2. [Training Arguments Deep Dive](#training-arguments-deep-dive)
3. [Multi-GPU Training Dynamics](#multi-gpu-training-dynamics)
4. [Performance Analysis](#performance-analysis)
5. [Memory Usage Analysis](#memory-usage-analysis)
6. [Best Practices](#best-practices)

## Understanding F3RM Training Pipeline

F3RM extends the standard NeRF training with feature distillation. The key components are:

### 1. Core Training Loop (Per Step)
```
For each training step:
1. Sample rays from images → RayBundle (batch_size = train_num_rays_per_batch)
2. Sample points along rays → RaySamples (num_samples_per_ray × batch_size points)
3. Forward pass through:
   - Main NeRF field (density + RGB)
   - Feature field (CLIP/DINO features)
4. Volume rendering → RGB + features
5. Loss computation (RGB loss + feature loss)
6. Backward pass + optimizer step
```

### 2. Data Management Architecture
F3RM uses `FeatureDataManager` which extends `VanillaDataManager`:

- **Feature Extraction**: Pre-extracts CLIP/DINO features for all images
- **LazyFeatures**: Memory-mapped feature storage for efficient access
- **Async GPU Transfer**: Features transferred asynchronously to reduce bottlenecks

## Training Arguments Deep Dive

### 1. `--max-num-iterations` 

**What it does:**
- Total number of training steps to perform
- Each step processes `train_num_rays_per_batch` rays

**Impact on:**
- **Train_Rays/sec**: No direct impact on throughput per step
- **ETA**: Directly proportional - doubling iterations doubles training time
- **VRAM**: No impact on memory usage per step
- **Final Quality**: More iterations generally = better convergence

**Code Reference:**
```python
# f3rm/f3rm_config.py:18
max_num_iterations=30000,  # Default in config
```

### 2. `--pipeline.datamanager.train-num-rays-per-batch` (✅ EXISTS)

**What it does:**
- Number of rays processed per training step
- Default: 8192 rays per batch
- Each ray gets sampled at multiple points (typically 48-96 points)
- **TRAINING ONLY**: This parameter only affects training, not evaluation

**Deep dive into ray processing:**
```python
# From f3rm/model.py:199-210
def get_outputs(self, ray_bundle: RayBundle):
    ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
        ray_bundle, density_fns=self.density_fns
    )
    # ray_samples shape: [batch_size, num_samples_per_ray, ...]
    # For 8192 rays × 48 samples = 393,216 points processed
```

**Impact on:**
- **Train_Rays/sec**: 
  - Higher batch size → potentially higher throughput (better GPU utilization)
  - But diminishing returns due to memory bandwidth limits
- **ETA**: 
  - Larger batches → fewer steps needed for same ray coverage → faster training
  - But each step takes longer
- **VRAM**: 
  - **Critical impact**: Linear scaling with batch size
  - Each ray requires memory for: origins, directions, RGB targets, feature targets
  - Total memory: `rays_per_batch × samples_per_ray × feature_dims`

**Memory calculation example:**
```
8192 rays × 48 samples × (3 RGB + 512 CLIP features) = ~200MB per batch
16384 rays would need ~400MB per batch
```

### 2.1. `--pipeline.datamanager.eval-num-rays-per-batch` (✅ EXISTS)

**What it does:**
- Number of rays processed per evaluation step
- Default: 8192 rays per batch
- **EVALUATION ONLY**: This parameter only affects evaluation, not training
- Similar to training version but used during validation/testing phases

**Impact on:**
- **Train_Rays/sec**: No direct impact on training throughput
- **ETA**: Minimal impact (evaluation is infrequent)
- **VRAM**: Can cause memory spikes during evaluation if set too high

### 2.2. `--pipeline.model.eval-num-rays-per-chunk` (✅ EXISTS)

**What it does:**
- Number of rays processed per forward pass during evaluation/inference
- Default: 4096 rays per chunk (32,768 in F3RM config for performance)
- **INFERENCE/EVALUATION ONLY**: Used for rendering full images during eval
- **Key difference**: This is about chunking for memory management, not training batches

**Why chunking is needed:**
```python
# From nerfstudio_changes/base_model.py:170-189
def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle):
    num_rays_per_chunk = self.config.eval_num_rays_per_chunk
    num_rays = len(camera_ray_bundle)  # e.g., 1024*1024 = 1M rays for full image
    
    for i in range(0, num_rays, num_rays_per_chunk):
        ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
        outputs = self.forward(ray_bundle=ray_bundle)
        # Process chunk and accumulate results
```

**Impact on:**
- **Train_Rays/sec**: No impact on training
- **ETA**: No impact on training time
- **VRAM**: 
  - **Critical for eval**: Controls peak memory during image rendering
  - Lower values → less VRAM but slower rendering
  - Higher values → more VRAM but faster rendering

### 2.3. `train-num-rays-per-chunk` (❌ DOES NOT EXIST)

**This parameter does not exist in nerfstudio/F3RM.** 

**Why it doesn't exist:**
- During training, rays are processed in batches determined by `train_num_rays_per_batch`
- No chunking is needed during training because batch sizes are controlled
- Chunking is only necessary during evaluation when rendering full images (millions of rays)
- Training only processes small batches (8192 rays), so no memory management chunking needed

### 3. `--pipeline.datamanager.train-num-images-to-sample-from`

**What it does:**
- Number of images loaded into CPU/GPU memory simultaneously for ray sampling
- Default: ∞ (load all images)
- If set to N < total_images, only N images are kept in memory at once

**Implementation details:**
```python
# From nerfstudio VanillaDataManager
self.train_image_dataloader = CacheDataloader(
    self.train_dataset,
    num_images_to_sample_from=self.config.train_num_images_to_sample_from,
    num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
    # ...
)
```

**Impact on:**
- **Train_Rays/sec**: 
  - Lower values → more frequent disk I/O → potential slowdowns
  - Higher values → better performance but more memory usage
- **ETA**: 
  - If too low → CPU becomes bottleneck due to frequent image loading
  - If sufficient → minimal impact
- **VRAM**: 
  - **Significant impact**: Each image + features must fit in memory
  - For CLIP: `num_images × H × W × 512 features × 4 bytes`
  - For high-res images with features: can easily exceed 10GB

**Real-world example:**
```
32 images × 1024×1024 × 512 CLIP features × 4 bytes = ~67GB
This explains why you might see more VRAM usage with more images!
```

### 4. `--pipeline.datamanager.train-num-times-to-repeat-images`

**What it does:**
- How many batches to sample from current loaded images before loading new ones
- Default: ∞ (never reload images)
- If set to N, reload images every N batches

**Impact on:**
- **Train_Rays/sec**: 
  - Lower values → more frequent reloading → slower training
  - Higher values → more stable throughput
- **ETA**: 
  - Frequent reloading adds overhead
- **VRAM**: 
  - Works with `train_num_images_to_sample_from` to control memory
  - No direct impact, but affects when memory spikes occur

### 5. `--pipeline.datamanager.eval-num-images-to-sample-from` & `eval-num-times-to-repeat-images`

**What they do:**
- Same as training versions but for evaluation/validation
- Only affect eval phases (every `steps_per_eval_batch` steps)

**Impact on:**
- **Train_Rays/sec**: No impact on training throughput
- **ETA**: Minimal impact (eval is infrequent)
- **VRAM**: Can cause memory spikes during eval if set too high

### 6. `--machine.num-devices` (Multi-GPU)

**What it does:**
- Number of GPUs to use for distributed training
- Each GPU gets a unique seed: `base_seed + rank`

**Critical understanding - Why more GPUs might increase ETA:**

#### Data Parallelism Implementation:
```python
# From f3rm/trainer.py:61-75
if world_size > 1:
    setup_distributed_seeding(
        rank=local_rank,
        world_size=world_size,
        base_seed=self.config.machine.seed
    )
```

#### Effective batch size scaling:
- **Per-GPU batch size**: `train_num_rays_per_batch / num_devices`
- **Global batch size**: `train_num_rays_per_batch` (same across all GPU counts)

**Example with 8192 rays:**
- 1 GPU: 8192 rays per GPU
- 2 GPUs: 4096 rays per GPU each
- 4 GPUs: 2048 rays per GPU each

#### Why Train_Rays/sec might increase but ETA increases:

1. **Smaller per-GPU batches**:
   - GPU utilization decreases with smaller batches
   - Memory bandwidth underutilized
   - More overhead relative to computation

2. **Synchronization overhead**:
   - AllReduce operations for gradient synchronization
   - Load balancing across GPUs
   - Network communication latency

3. **Feature loading bottleneck**:
```python
# From f3rm/feature_datamanager.py:126-130
def _gather_feats(self, feats, camera_idx, y_idx, x_idx):
    if isinstance(feats, LazyFeatures):
        out = [feats[int(ci), int(yi), int(xi)] for ci, yi, xi in zip(camera_idx, y_idx, x_idx)]
        cpu_batch = torch.stack(out, dim=0)
    return _async_to_cuda(cpu_batch, self.device)
```
   - Each GPU needs to load features independently
   - Smaller batches → less efficient memory access patterns
   - More frequent GPU-CPU transfers

4. **Fixed overhead amplification**:
   - Model forward/backward pass overhead
   - Feature field computation overhead
   - These become proportionally larger with smaller batches

## Multi-GPU Training Dynamics

### Memory Distribution

**With N GPUs:**
- **Model replication**: Each GPU holds full model copy (~3.9M parameters for F3RM)
- **Feature storage**: Features distributed or replicated based on implementation
- **Working memory**: `train_num_rays_per_batch / N` per GPU

### Optimal Multi-GPU Strategy

**For F3RM specifically:**

1. **2 GPUs**: Often optimal balance
   - 4096 rays per GPU still efficient
   - Reasonable synchronization overhead
   - Good memory utilization

2. **4+ GPUs**: Diminishing returns
   - <2048 rays per GPU → poor GPU utilization
   - Increased communication overhead
   - Feature loading becomes bottleneck

## Performance Analysis

### Factors affecting Train_Rays/sec:

1. **Primary factors** (in order of impact):
   - `train_num_rays_per_batch`: Higher = better up to memory limit
   - GPU memory bandwidth utilization
   - Feature extraction and loading efficiency

2. **Secondary factors**:
   - `train_num_images_to_sample_from`: Affects memory locality
   - Multi-GPU synchronization overhead
   - Model size (feature field complexity)

### Factors affecting ETA:

1. **Direct factors**:
   - `max_num_iterations`: Linear relationship
   - Steps per epoch (affected by image sampling params)

2. **Indirect factors**:
   - Per-step time (affected by batch size and GPU count)
   - Evaluation frequency and cost

## Memory Usage Analysis

### VRAM Breakdown for F3RM:

```
Total VRAM = Model + Features + Working Memory + Overhead

Model: ~64MB (NeRF field + Feature field)
Features: num_images × H × W × feature_dim × 4 bytes
Working Memory: batch_size × samples_per_ray × (RGB + features)
Overhead: ~1-2GB (CUDA context, intermediates)
```

### Memory optimization strategies:

1. **Reduce feature memory**:
   - Lower `train_num_images_to_sample_from`
   - Use LazyFeatures (memory-mapped)
   - Enable feature caching to disk

2. **Reduce working memory**:
   - Lower `train_num_rays_per_batch`
   - Reduce `eval_num_rays_per_chunk`

3. **Multi-GPU memory distribution**:
   - More GPUs → less memory per GPU
   - But synchronization overhead increases

## Best Practices

### For Maximum Training Speed:

1. **Single GPU (>16GB VRAM)**:
```bash
--pipeline.datamanager.train-num-rays-per-batch 16384
--pipeline.datamanager.train-num-images-to-sample-from -1  # All images
--pipeline.datamanager.train-num-times-to-repeat-images -1  # Never reload
```

2. **Multi-GPU (2 GPUs recommended)**:
```bash
--machine.num-devices 2
--pipeline.datamanager.train-num-rays-per-batch 8192  # 4096 per GPU
--pipeline.datamanager.train-num-images-to-sample-from 64
--pipeline.datamanager.train-num-times-to-repeat-images 1024
```

### For Memory-Constrained Systems:

1. **Low VRAM (<12GB)**:
```bash
--pipeline.datamanager.train-num-rays-per-batch 4096
--pipeline.datamanager.train-num-images-to-sample-from 16
--pipeline.datamanager.train-num-times-to-repeat-images 256
--pipeline.model.eval-num-rays-per-chunk 8192
```

2. **Extremely Low VRAM (<8GB)**:
```bash
--pipeline.datamanager.train-num-rays-per-batch 2048
--pipeline.datamanager.train-num-images-to-sample-from 8
--pipeline.datamanager.train-num-times-to-repeat-images 128
--pipeline.model.eval-num-rays-per-chunk 4096
```

### For Balanced Speed/Memory:

```bash
--max-num-iterations 120000
--pipeline.datamanager.train-num-rays-per-batch 8192
--pipeline.datamanager.train-num-images-to-sample-from 32
--pipeline.datamanager.train-num-times-to-repeat-images 512
--pipeline.datamanager.eval-num-images-to-sample-from 32
--pipeline.datamanager.eval-num-times-to-repeat-images 512
--machine.num-devices 2
```

### Understanding Your Training Metrics:

- **Train_Rays/sec increasing**: Good GPU utilization, efficient batching
- **Train_Rays/sec decreasing**: Memory bottleneck, poor GPU utilization
- **ETA increasing with more GPUs**: Overhead > parallelization benefit
- **High memory usage**: Increase image sampling constraints

### Debugging Performance Issues:

1. **Low Train_Rays/sec**:
   - Increase `train_num_rays_per_batch` if memory allows
   - Check if feature loading is bottleneck
   - Verify sufficient `train_num_images_to_sample_from`

2. **High memory usage**:
   - Reduce `train_num_images_to_sample_from`
   - Lower `train_num_rays_per_batch`
   - Check feature caching efficiency

3. **ETA increasing with more GPUs**:
   - Reduce GPU count to 1-2
   - Increase per-GPU batch size
   - Check network bandwidth for multi-node setups

## Key Takeaways & Corrections (Updated based on Discussion)

### 1. ❌ CORRECTED: Iterations ≠ Epochs

**Your understanding was CORRECT!**

- `max_num_iterations` is **NOT** epochs
- **Epoch** = One full pass through all pixels/rays in the training dataset
- **Iteration/Step** = One forward+backward pass with a batch of rays

**Formula for calculating epochs:**
```
epochs = (max_num_iterations × train_num_rays_per_batch) / total_pixel_count_training_images

Example:
- Training images: 100 images × 1024×1024 = ~105M pixels
- max_num_iterations: 120,000
- train_num_rays_per_batch: 8192

epochs = (120,000 × 8192) / 105M ≈ 9.4 epochs
```

### 2. ✅ VRAM vs GPU Utilization Parameters

**Parameters that can cause OOM (Hard Limits):**
- `train_num_rays_per_batch` - Linear VRAM scaling, **WILL CRASH** if too high
- `train_num_images_to_sample_from` - Feature memory, **WILL CRASH** if too high  
- `eval_num_rays_per_chunk` - Evaluation memory, **WILL CRASH** during rendering

**Parameters that only affect GPU utilization (Soft Limits):**
- `train_num_times_to_repeat_images` - Only affects I/O frequency, no VRAM impact
- `eval_num_times_to_repeat_images` - Only affects eval I/O, no VRAM impact

**Concrete Example:**
```bash
# This WILL crash with OOM:
--pipeline.datamanager.train-num-rays-per-batch 32768  # Too much VRAM

# This WON'T crash, just slower training:
--pipeline.datamanager.train-num-times-to-repeat-images 10  # Frequent reloading
```

### 3. ✅ Memory Bandwidth Limits Explained

**"Diminishing returns due to memory bandwidth limits"** means:

**Toy Example:**
```
GPU Memory Bandwidth: 1000 GB/s theoretical
Current utilization with 8192 rays: 800 GB/s (80%)
Doubling to 16384 rays: 950 GB/s (95%) - only 19% improvement!
Tripling to 24576 rays: 990 GB/s (99%) - only 4% more improvement!

Result: Train_Rays/sec increases sub-linearly with batch size
```

**Why this happens:**
- GPU cores finish computation, wait for memory
- Memory controller becomes bottleneck
- Larger batches don't help beyond memory saturation point

### 4. ✅ ETA vs Batch Size Confusion RESOLVED

**Your confusion was valid!** Here's what's happening:

**ETA is affected by batch size due to GPU efficiency, not step count:**

```python
# Steps = max_num_iterations (FIXED, e.g., 120,000 steps)
# But TIME per step varies dramatically with batch size due to GPU architecture:

Small batch (2048 rays):  0.15 seconds/step → ETA = 120,000 × 0.15s = 5 hours
Large batch (8192 rays):  0.08 seconds/step → ETA = 120,000 × 0.08s = 2.7 hours
```

**Why smaller batches are SLOWER per step:**

1. **GPU Underutilization**: 
   - RTX 4090 has 16,384 CUDA cores
   - 2048 rays → only 12.5% of cores busy, 87.5% idle
   - 8192 rays → 50% of cores busy, much better utilization

2. **Memory Bandwidth Inefficiency**:
   - Small batches → scattered memory access → poor bandwidth utilization
   - Large batches → coalesced memory access → better bandwidth utilization

3. **Fixed Overhead Amortization**:
   - Every GPU kernel has launch overhead (~0.01ms)
   - Small batch: overhead is 6.7% of total time (0.01ms / 0.15ms)
   - Large batch: overhead is 1.25% of total time (0.01ms / 0.08ms)

4. **F3RM Feature Field Overhead**:
   - Feature field computation has high fixed cost
   - Small batches → poor amortization of feature extraction overhead
   - Large batches → better amortization

**Key insight**: It's not about compute capability, it's about **efficiency of using that capability**!

### 5. ✅ eval_num_rays_per_chunk Deep Dive

**Lines 37-39 in f3rm_config.py:**
```python
# To support more GPUs, we reduce the num rays per chunk. The default was 1 << 15 which uses ~16GB of GPU
# memory when training and using viewer. 1 << 14 uses ~12GB of GPU memory in comparison.
model=FeatureFieldModelConfig(eval_num_rays_per_chunk=1 << 15),  # = 32,768
```

**What this means:**
- **During evaluation/rendering**: Full image = 1M+ rays (1024×1024)
- **Memory chunking**: Process 32,768 rays at a time to avoid OOM
- **F3RM uses 32,768 vs nerfstudio default 4,096** = 8× larger chunks = faster rendering
- **Trade-off**: Faster rendering but uses more VRAM (16GB vs 12GB)

### 6. ✅ steps_per_eval_batch Explained

**What it does:**
- **Frequency** of evaluation/validation runs during training  
- Set to `500` in f3rm_config.py line 16: `steps_per_eval_batch=500`
- Every 500 training steps → run evaluation on validation set

**Code reference:**
```python
# f3rm/f3rm_config.py:16
steps_per_eval_batch=500,  # Run eval every 500 training steps
```

**Impact on training:**
- **Lower values** (e.g., 100): More frequent evaluation, slower overall training
- **Higher values** (e.g., 2000): Less frequent evaluation, faster training but less monitoring

### 7. ✅ Consistent Terminology (FIXED)

**Going forward, this document uses:**

- **Epoch** = One complete pass through ALL training data pixels
- **Iteration/Step** = One forward+backward pass with one batch of rays  
- **max_num_iterations** = Total number of training steps (NOT epochs)

**Corrected Performance Analysis:**

**ETA Direct Factors:**
- `max_num_iterations`: Total training steps (linear relationship)
- Time per step (affected by batch size, GPU efficiency, multi-GPU overhead)

**Steps per epoch calculation:**
```python
total_pixels = sum(image_width × image_height for all training images)
rays_per_step = train_num_rays_per_batch
steps_per_epoch = total_pixels / rays_per_step

# Example: 100 images × 1024² pixels = 104M pixels
# With 8192 rays/step: 104M / 8192 = ~12,700 steps per epoch
```

### 8. ✅ Multi-GPU ETA Paradox Explained

**Why Train_Rays/sec increases but ETA also increases:**

1. **Ray throughput increases** (more total rays processed across all GPUs)
2. **But per-step time increases** due to:
   - Synchronization overhead (AllReduce operations)
   - Smaller per-GPU batches → poor GPU utilization  
   - Feature loading becomes bottleneck with smaller batches

**Net result**: Higher ray/sec but longer total training time!

**Recommendation**: 2 GPUs is often optimal for F3RM, 4+ GPUs show diminishing returns.

### 9. ⚠️ CRITICAL: Image Sampling Parameters and Epoch Calculation

**Your observation is ABSOLUTELY CORRECT!** 

The `train_num_images_to_sample_from` and `train_num_times_to_repeat_images` parameters **directly affect epoch coverage** and must be carefully coordinated with `max_num_iterations`.

**How it works in practice:**

1. **Image Sampling Cycle**: Every `train_num_times_to_repeat_images` iterations, F3RM reloads a fresh set of `train_num_images_to_sample_from` images.

2. **Dataset Coverage**: To ensure proper epoch coverage, you need:
   ```python
   # For full dataset coverage:
   total_image_reloads_needed = total_training_images / train_num_images_to_sample_from
   iterations_per_reload = train_num_times_to_repeat_images
   min_iterations_for_full_coverage = total_image_reloads_needed * iterations_per_reload
   
   # Example with your settings:
   # 100 training images, sample 32 at a time, repeat 512 times each
   total_image_reloads_needed = 100 / 32 = 3.125 ≈ 4 reloads needed
   min_iterations_for_full_coverage = 4 * 512 = 2,048 iterations
   ```

3. **Epoch Calculation (REFINED)**: 
   ```python
   # Traditional calculation (INCOMPLETE):
   epochs = (max_num_iterations × train_num_rays_per_batch) / total_pixels

   # COMPLETE calculation considering image sampling:
   unique_images_seen_per_cycle = train_num_images_to_sample_from
   iterations_per_cycle = train_num_times_to_repeat_images
   total_cycles = max_num_iterations / iterations_per_cycle
   total_unique_images_seen = total_cycles × unique_images_seen_per_cycle
   
   # Effective epoch coverage:
   dataset_coverage = min(1.0, total_unique_images_seen / total_training_images)
   epochs = (max_num_iterations × train_num_rays_per_batch × dataset_coverage) / total_pixels
   ```

**Real-world example with your settings:**
```bash
# Your command:
--max-num-iterations 120000
--train-num-images-to-sample-from 32  
--train-num-times-to-repeat-images 512

# Analysis:
total_cycles = 120,000 / 512 = 234.375 cycles
unique_images_seen = 234.375 * 32 = 7,500 image views
# If you have 100 training images: 7,500 / 100 = 75 times each image is seen
# If you have 200 training images: 7,500 / 200 = 37.5 times each image is seen
```

**⚠️ Key Insights:**

1. **Under-sampling Risk**: If `train_num_images_to_sample_from` is too small relative to your dataset, you might not see all images equally.

2. **Over-sampling Risk**: If `train_num_times_to_repeat_images` is too large, you'll see the same images repeatedly, potentially causing overfitting.

3. **Optimal Strategy**: 
   ```bash
   # For balanced coverage:
   train_num_images_to_sample_from = total_training_images  # See all images
   train_num_times_to_repeat_images = large_number  # Don't reload frequently
   
   # For memory-constrained systems:
   train_num_images_to_sample_from = memory_limited_value
   train_num_times_to_repeat_images = max_num_iterations / (total_training_images / train_num_images_to_sample_from)
   ```

**Updated Best Practice:**
```bash
# Calculate based on your dataset size:
total_images=100  # Your training images
target_epochs=10

# Method 1: Load all images (if memory allows)
--train-num-images-to-sample-from -1  # All images
--train-num-times-to-repeat-images -1  # Never reload

# Method 2: Memory-constrained balanced sampling
--train-num-images-to-sample-from 32
--train-num-times-to-repeat-images $((max_iterations * 32 / (target_epochs * total_images)))
```

This understanding should help you optimize F3RM training for your specific hardware and dataset constraints. 