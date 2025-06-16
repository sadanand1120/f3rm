# F3RM Training Arguments Deep Dive

This document provides a comprehensive understanding of how F3RM training arguments affect training performance, memory usage, and ETA. Each parameter is explained with precise technical details and practical implications.

## Table of Contents

1. [Understanding F3RM Training Pipeline](#understanding-f3rm-training-pipeline)
2. [Training Arguments Deep Dive](#training-arguments-deep-dive)
3. [Multi-GPU Training Dynamics](#multi-gpu-training-dynamics)
4. [Performance Analysis](#performance-analysis)
5. [Memory Usage Analysis](#memory-usage-analysis)
6. [Best Practices](#best-practices)
7. [FAQ](#faq)

## Understanding F3RM Training Pipeline

F3RM extends standard NeRF training with feature distillation. Understanding the core loop is essential for optimizing performance.

### Core Training Loop (Per Step)
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

### Key Terminology

- **Iteration/Step**: One forward+backward pass with one batch of rays
- **Epoch**: One complete pass through ALL training data pixels
- **max_num_iterations**: Total number of training steps (NOT epochs)

**Critical Understanding**: `max_num_iterations` is the total number of training steps, not epochs. Each step processes a batch of rays, not a full pass through the dataset.

## Training Arguments Deep Dive

### 1. `--max-num-iterations`

**What it does:**
- Total number of training steps to perform
- Each step processes `train_num_rays_per_batch` rays
- **NOT equivalent to epochs** - this is a common misunderstanding

**Epoch Calculation:**
```python
# Basic formula (see image sampling section for complete version):
total_pixels = sum(image_width × image_height for all training images)
rays_per_step = train_num_rays_per_batch
steps_per_epoch = total_pixels / rays_per_step
epochs = max_num_iterations / steps_per_epoch

# Example: 100 images × 1024² = ~105M pixels, 8192 rays/step
# steps_per_epoch = 105M / 8192 ≈ 12,817 steps
# epochs = 120,000 / 12,817 ≈ 9.4 epochs
```

**Impact on:**
- **Train_Rays/sec**: No direct impact on throughput per step
- **ETA**: Directly proportional - doubling iterations doubles training time
- **VRAM**: No impact on memory usage per step

### 2. `--pipeline.datamanager.train-num-rays-per-batch` (✅ EXISTS)

**What it does:**
- Number of rays processed per training step
- Default: 8192 rays per batch
- **TRAINING ONLY**: This parameter only affects training, not evaluation

**Deep Technical Details:**
```python
# From f3rm/model.py - each ray gets sampled at multiple points
# 8192 rays × 48 samples per ray = 393,216 3D points processed per step

# Memory requirements per batch:
# rays_per_batch × samples_per_ray × (3 RGB + 512 CLIP features) × 4 bytes
# 8192 × 48 × 515 × 4 = ~810MB working memory per batch
```

**Impact on Train_Rays/sec:**
Higher batch sizes generally improve throughput due to **GPU architecture realities**:

```python
# GPU has thousands of cores (e.g., RTX 4090 has 16,384 CUDA cores)
# Small batch utilization:
2048 rays: Only 12.5% of cores busy → 87.5% cores idle → poor efficiency
# Large batch utilization:
8192 rays: 50% of cores busy → much better efficiency
16384 rays: 100% of cores busy → optimal utilization (if memory allows)
```

**Why larger batches are faster per step:**
1. **Better GPU utilization**: More cores actively computing
2. **Memory bandwidth efficiency**: Coalesced memory access patterns
3. **Fixed overhead amortization**: Kernel launch overhead is constant per step
4. **Feature field overhead**: F3RM's feature extraction has high fixed cost

**Impact on ETA:**
Although `max_num_iterations` is fixed, **time per step varies dramatically** with batch size:

```python
# Same number of steps, but different time per step:
Small batch (2048 rays): 0.15 seconds/step → ETA = 120,000 × 0.15s = 5.0 hours
Large batch (8192 rays):  0.08 seconds/step → ETA = 120,000 × 0.08s = 2.7 hours
```

**Memory Bandwidth Limits:**
```python
# Diminishing returns example:
GPU Memory Bandwidth: 1000 GB/s theoretical
8192 rays:  800 GB/s utilization (80%) 
16384 rays: 950 GB/s utilization (95%) - only 19% improvement
24576 rays: 990 GB/s utilization (99%) - only 4% more improvement
```

**Impact on VRAM:**
- **Critical impact**: Linear scaling with batch size
- **Hard limit**: Will cause OOM crash if too high
- Each ray requires memory for: origins, directions, RGB targets, feature targets

### 3. `--pipeline.datamanager.eval-num-rays-per-batch` (✅ EXISTS)

**What it does:**
- Number of rays processed per evaluation step
- Default: 8192 rays per batch
- **EVALUATION ONLY**: Used during validation phases every `steps_per_eval_batch` steps

**Impact on:**
- **Train_Rays/sec**: No direct impact on training throughput
- **ETA**: Minimal impact (evaluation runs every 500 steps by default)
- **VRAM**: Can cause memory spikes during evaluation if set too high

### 4. `--pipeline.model.eval-num-rays-per-chunk` (✅ EXISTS)

**What it does:**
- Number of rays processed per forward pass during image rendering
- Default: 4096 rays per chunk (F3RM uses 32,768 for performance)
- **INFERENCE/EVALUATION ONLY**: Used for rendering full images during evaluation

**Why chunking is needed:**
```python
# Full image rendering: 1024×1024 = 1,048,576 rays
# Without chunking: Would need ~4GB VRAM just for one image
# With chunking: Process 32,768 rays at a time = manageable memory usage

# From f3rm_config.py:37-39
# "To support more GPUs, we reduce the num rays per chunk. 
# The default was 1 << 15 (32,768) which uses ~16GB of GPU memory
# when training and using viewer. 1 << 14 (16,384) uses ~12GB."
```

**Impact on:**
- **Train_Rays/sec**: No impact on training
- **ETA**: No impact on training time
- **VRAM**: Controls peak memory during image rendering (hard limit)

### 5. `train-num-rays-per-chunk` (❌ DOES NOT EXIST)

**This parameter does not exist** because training doesn't need chunking:
- Training processes controlled batches (8192 rays)
- Chunking is only needed for evaluation when rendering full images (1M+ rays)
- Training batch sizes are already memory-manageable

### 6. `--pipeline.datamanager.train-num-images-to-sample-from`

**What it does:**
- Number of images loaded into memory simultaneously for ray sampling
- Default: ∞ (load all images)
- **Critical for dataset coverage**: Directly affects which images are seen during training

**Memory Impact:**
```python
# VRAM calculation for features:
# num_images × H × W × feature_dim × 4 bytes
# 32 images × 1024² × 512 CLIP features × 4 bytes = ~67GB

# This explains why more images can dramatically increase VRAM usage!
```

**Impact on Dataset Coverage:**
This parameter fundamentally changes how much of your dataset is seen:

```python
# If train_num_images_to_sample_from = 32 and you have 100 training images:
# Only 32 images are loaded at a time
# You'll cycle through different sets of 32 images during training
# BUT: you might not see all 100 images equally often
```

**Impact on:**
- **Train_Rays/sec**: Lower values → more frequent disk I/O → potential slowdowns
- **ETA**: If too low → CPU becomes bottleneck due to frequent image loading
- **VRAM**: **Significant impact** - each image + features must fit in memory (hard limit)

### 7. `--pipeline.datamanager.train-num-times-to-repeat-images`

**What it does:**
- How many training steps to use current loaded images before loading new ones
- Default: ∞ (never reload images)
- **Works with above parameter** to control memory usage and dataset coverage

**Critical Relationship with Dataset Coverage:**
```python
# Complete epoch calculation considering image sampling:
iterations_per_image_cycle = train_num_times_to_repeat_images
images_per_cycle = train_num_images_to_sample_from
total_cycles = max_num_iterations / iterations_per_image_cycle
total_unique_image_views = total_cycles × images_per_cycle

# Dataset coverage:
dataset_coverage = min(1.0, total_unique_image_views / total_training_images)

# Example with your settings:
# 120,000 iterations, 32 images per cycle, 512 iterations per cycle
total_cycles = 120,000 / 512 = 234 cycles
total_unique_image_views = 234 × 32 = 7,488
# If 100 training images: 7,488 / 100 = 74.9 times each image seen
# If 200 training images: 7,488 / 200 = 37.4 times each image seen
```

**Impact on:**
- **Train_Rays/sec**: Lower values → more frequent reloading → slower training
- **ETA**: Frequent reloading adds overhead
- **VRAM**: No direct impact, but affects when memory spikes occur

### 8. `--pipeline.datamanager.eval-num-images-to-sample-from` & `eval-num-times-to-repeat-images`

**What they do:**
- Same as training versions but for evaluation/validation
- Only affect eval phases (every `steps_per_eval_batch` steps)
- Default: `steps_per_eval_batch = 500` (set in f3rm_config.py:16)

**Impact on:**
- **Train_Rays/sec**: No impact on training throughput
- **ETA**: Minimal impact (eval is infrequent)
- **VRAM**: Can cause memory spikes during eval if set too high

### 9. `--machine.num-devices` (Multi-GPU)

**What it does:**
- Number of GPUs to use for distributed training
- Uses data parallelism: each GPU processes a subset of the batch

**Critical Understanding - The Multi-GPU Paradox:**

**Why Train_Rays/sec might increase but ETA also increases:**

1. **Batch Size Division:**
   ```python
   # Global batch size remains constant
   per_gpu_batch_size = train_num_rays_per_batch / num_devices
   
   # Example with 8192 rays:
   1 GPU: 8192 rays per GPU (good utilization)
   2 GPUs: 4096 rays per GPU each (still reasonable)
   4 GPUs: 2048 rays per GPU each (poor utilization)
   ```

2. **GPU Utilization Degradation:**
   ```python
   # RTX 4090 has 16,384 CUDA cores
   1 GPU with 8192 rays: 50% core utilization
   4 GPUs with 2048 rays each: 12.5% core utilization per GPU
   ```

3. **Synchronization Overhead:**
   - AllReduce operations for gradient synchronization
   - Communication latency between GPUs
   - Load balancing overhead

4. **Feature Loading Bottleneck:**
   ```python
   # From f3rm/feature_datamanager.py - each GPU loads features independently
   # Smaller batches → less efficient memory access patterns
   # More frequent GPU-CPU transfers
   ```

**Result:** More total rays processed per second, but longer time per training step.

## Multi-GPU Training Dynamics

### Memory Distribution
- **Model replication**: Each GPU holds full model copy (~64MB for F3RM)
- **Feature storage**: Features may be replicated across GPUs
- **Working memory**: `train_num_rays_per_batch / num_devices` per GPU

### Optimal Multi-GPU Strategy for F3RM

**2 GPUs**: Often optimal balance
- 4096 rays per GPU still provides good utilization
- Reasonable synchronization overhead
- Good memory distribution

**4+ GPUs**: Diminishing returns
- <2048 rays per GPU → poor GPU utilization
- Increased communication overhead
- Feature loading becomes bottleneck

## Performance Analysis

### Train_Rays/sec Factors (in order of impact):
1. **train_num_rays_per_batch**: Higher = better (up to memory limit)
2. **GPU utilization efficiency**: Depends on batch size per GPU
3. **Memory bandwidth utilization**: Affects throughput ceiling
4. **Feature extraction efficiency**: F3RM-specific bottleneck
5. **Multi-GPU synchronization overhead**: Increases with more GPUs

### ETA Factors:
1. **max_num_iterations**: Direct linear relationship
2. **Time per step**: Affected by batch size, GPU count, efficiency
3. **Evaluation frequency**: Minor impact (every 500 steps)

## Memory Usage Analysis

### VRAM Breakdown for F3RM:
```
Total VRAM = Model + Features + Working Memory + Overhead

Model: ~64MB (NeRF field + Feature field)
Features: train_num_images_to_sample_from × H × W × 512 × 4 bytes
Working Memory: train_num_rays_per_batch × samples_per_ray × (RGB + features)
Overhead: ~1-2GB (CUDA context, gradients, optimizer states)
```

### Memory Optimization Strategies:

**Hard Limits (will cause OOM crash):**
- `train_num_rays_per_batch`: Linear VRAM scaling
- `train_num_images_to_sample_from`: Feature memory scaling
- `eval_num_rays_per_chunk`: Evaluation memory scaling

**Soft Limits (only affect efficiency):**
- `train_num_times_to_repeat_images`: Only affects I/O frequency
- `eval_num_times_to_repeat_images`: Only affects eval I/O

## Best Practices

### For Maximum Training Speed (>16GB VRAM):
```bash
--max-num-iterations 120000
--pipeline.datamanager.train-num-rays-per-batch 16384
--pipeline.datamanager.train-num-images-to-sample-from -1  # All images
--pipeline.datamanager.train-num-times-to-repeat-images -1  # Never reload
--machine.num-devices 1
```

### For Multi-GPU (2 GPUs recommended):
```bash
--max-num-iterations 120000
--pipeline.datamanager.train-num-rays-per-batch 8192  # 4096 per GPU
--pipeline.datamanager.train-num-images-to-sample-from 64
--pipeline.datamanager.train-num-times-to-repeat-images 1024
--machine.num-devices 2
```

### For Memory-Constrained Systems (<12GB VRAM):
```bash
--max-num-iterations 120000
--pipeline.datamanager.train-num-rays-per-batch 4096
--pipeline.datamanager.train-num-images-to-sample-from 16
--pipeline.datamanager.train-num-times-to-repeat-images 256
--pipeline.model.eval-num-rays-per-chunk 8192
--machine.num-devices 1
```

### For Balanced Dataset Coverage:
```python
# Calculate optimal image sampling for your dataset:
total_images = 100  # Your training images
target_coverage = 1.0  # See each image at least once

# Method 1: Load all images (if memory allows)
train_num_images_to_sample_from = -1  # All images
train_num_times_to_repeat_images = -1  # Never reload

# Method 2: Memory-constrained balanced sampling
train_num_images_to_sample_from = 32  # Memory-limited
cycles_needed = math.ceil(total_images / 32)  # 4 cycles for 100 images
train_num_times_to_repeat_images = max_num_iterations // cycles_needed
```

## FAQ

### Q: Why does my ETA increase when I add more GPUs?

**A:** This is the "Multi-GPU Paradox." More GPUs increase total ray throughput but also increase time per step due to:
1. Smaller per-GPU batches → poor GPU utilization
2. Synchronization overhead between GPUs
3. Feature loading becomes bottleneck with smaller batches

**Solution:** Use 2 GPUs maximum for F3RM. More GPUs show diminishing returns.

### Q: How do I calculate actual epochs from my training settings?

**A:** Complete formula considering image sampling:
```python
# Basic calculation:
total_pixels = sum(H × W for all training images)
rays_per_epoch = total_pixels
steps_per_epoch = rays_per_epoch / train_num_rays_per_batch

# Adjusted for image sampling:
cycles = max_num_iterations / train_num_times_to_repeat_images
images_seen = cycles × train_num_images_to_sample_from
coverage = min(1.0, images_seen / total_training_images)
effective_epochs = (max_num_iterations / steps_per_epoch) × coverage
```

### Q: What's the difference between batch and chunk parameters?

**A:** 
- **Batch**: How many rays to process in one training/eval step (affects learning)
- **Chunk**: How many rays to process in one forward pass during inference (memory management only)
- Training uses batches, full image rendering uses chunks

### Q: Why do smaller batch sizes sometimes use MORE VRAM per GPU?

**A:** This seems counterintuitive but can happen due to:
1. **Fixed memory overhead**: Model, optimizer states, CUDA context remain constant
2. **Poor memory allocation**: Smaller batches may cause memory fragmentation
3. **Multi-GPU inefficiency**: Synchronization buffers scale with number of GPUs, not batch size

### Q: How do I optimize for my specific dataset size?

**A:** Calculate your requirements:
```bash
# For 100 training images, 1024² resolution:
total_pixels = 100 × 1024² = ~105M pixels

# For 10 epochs with 8192 rays/batch:
total_rays_needed = 10 × 105M = 1.05B rays
steps_needed = 1.05B / 8192 = ~128,000 steps

# Set max_num_iterations ≥ 128,000 for full coverage
```

### Q: What causes "Train_Rays/sec" to fluctuate during training?

**A:** Common causes:
1. **Feature loading**: Spikes when loading new images
2. **Evaluation overhead**: Drops every 500 steps during eval
3. **Memory management**: Garbage collection or memory reallocation
4. **Thermal throttling**: GPU reducing clock speeds when hot
5. **System background tasks**: Other processes using GPU/CPU

### Q: How do I debug OOM errors?

**A:** Reduce parameters in this order:
1. `train_num_rays_per_batch` (biggest impact)
2. `train_num_images_to_sample_from` (second biggest)
3. `eval_num_rays_per_chunk` (if OOM during evaluation)
4. Use fewer GPUs (reduces per-GPU memory requirements)

### Q: What's the relationship between learning rate and batch size?

**A:** Larger batches generally need higher learning rates:
```python
# Rule of thumb: scale learning rate with batch size
base_lr = 1e-4  # for 8192 rays
scaled_lr = base_lr × (your_batch_size / 8192)

# But be careful: F3RM uses Adam optimizer which is less sensitive to this
```

This comprehensive guide should help you understand and optimize F3RM training for your specific hardware and dataset constraints. 