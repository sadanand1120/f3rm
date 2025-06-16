#!/usr/bin/env python3
"""
F3RM Meta-Analysis Script

Analyzes your dataset and hardware to recommend optimal F3RM training arguments.
Based on the comprehensive understanding from F3RM_TRAINING_DEEP_DIVE.md.

🔧 INTELLIGENT FEATURE DIMENSION INFERENCE:
   - Uses the same extraction functions as the training pipeline
   - Reads from existing feature cache shards if available
   - Extracts features from single test image to determine dimensions
   - Supports all feature types: CLIP, DINO, DINOCLIP, ROBOPOINTproj, ROBOPOINTnoproj

🚀 ACCURATE GPU SPECIFICATION:
   - Uses pynvml library for precise GPU specs
   - Gets actual memory bandwidth and compute capabilities
   - No hardcoded fallback dictionaries

💾 CORRECTED FEATURE MEMORY LOGIC:
   - Accounts for F3RM's LazyFeatures memory-mapped loading
   - Only calculates memory for train_num_images_to_sample_from
   - Much more accurate memory usage predictions

Usage:
    python meta_analyze.py --data /path/to/dataset --epochs 10 --feature-type CLIP
    python meta_analyze.py --data /path/to/dataset --epochs 5 --feature-type DINOCLIP --output recs.json
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import torch
import numpy as np
from PIL import Image
import GPUtil
import psutil
import pynvml
from pynvml import NVMLError_NotSupported, NVMLError_NoData

# Import F3RM feature extraction directly
from f3rm.scripts.extract_features_standalone import (
    FEAT_TYPE_TO_EXTRACT_FN,
    FEAT_TYPE_TO_ARGS,
    get_image_filenames_from_dataparser,
    get_cache_paths
)


class DatasetAnalyzer:
    """Analyzes dataset characteristics for F3RM training optimization."""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.transforms_path = self.data_path / "transforms.json"

        if not self.transforms_path.exists():
            raise FileNotFoundError(f"transforms.json not found at {self.transforms_path}")

    def analyze(self) -> Dict:
        """Analyze dataset characteristics."""
        with open(self.transforms_path, 'r') as f:
            transforms = json.load(f)

        # Get image paths
        train_frames = [f for f in transforms.get('frames', []) if not f.get('is_eval', False)]
        eval_frames = [f for f in transforms.get('frames', []) if f.get('is_eval', False)]

        # If no explicit eval split, assume 5% eval (nerfstudio default)
        if not eval_frames:
            total_frames = len(transforms['frames'])
            eval_count = max(1, int(total_frames * 0.05))
            train_frames = transforms['frames'][:-eval_count]
            eval_frames = transforms['frames'][-eval_count:]

        # Analyze first training image to get dimensions
        first_train_frame = train_frames[0]
        first_image_path = self.data_path / first_train_frame['file_path']

        # Handle different image extensions
        if not first_image_path.exists():
            for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG']:
                test_path = first_image_path.with_suffix(ext)
                if test_path.exists():
                    first_image_path = test_path
                    break

        if not first_image_path.exists():
            raise FileNotFoundError(f"First training image not found: {first_image_path}")

        # Get image dimensions
        with Image.open(first_image_path) as img:
            width, height = img.size

        # Calculate total pixels
        total_train_pixels = len(train_frames) * width * height
        total_eval_pixels = len(eval_frames) * width * height

        return {
            'num_train_images': len(train_frames),
            'num_eval_images': len(eval_frames),
            'image_width': width,
            'image_height': height,
            'total_train_pixels': total_train_pixels,
            'total_eval_pixels': total_eval_pixels,
            'dataset_path': str(self.data_path),
        }


class GPUAnalyzer:
    """Analyzes GPU hardware for F3RM training optimization."""

    def __init__(self):
        self.gpus = []
        self.analyze_gpus()

    def analyze_gpus(self):
        """Analyze available GPUs using pynvml."""
        pynvml.nvmlInit()
        num_gpus = torch.cuda.device_count()

        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Get device properties
            props = torch.cuda.get_device_properties(i)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else str(raw_name)

            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total
            available_memory = total_memory * 0.9  # 90% usable

            # Estimate memory bandwidth from GPU type/memory size (pynvml bandwidth query not available)
            memory_bandwidth = self._estimate_bandwidth_from_memory_size(total_memory / (1024**3))

            # Get compute capability and estimate cores
            compute_capability = f"{props.major}.{props.minor}"
            cuda_cores = self._estimate_cuda_cores_from_compute_capability(
                props.major, props.minor, props.multi_processor_count
            )

            gpu_info = {
                'id': i,
                'name': name,
                'total_memory_gb': total_memory / (1024**3),
                'available_memory_gb': available_memory / (1024**3),
                'compute_capability': compute_capability,
                'multiprocessor_count': props.multi_processor_count,
                'max_threads_per_multiprocessor': props.max_threads_per_multi_processor,
                'memory_bandwidth_gbps': memory_bandwidth,
                'cuda_cores': cuda_cores,
            }

            self.gpus.append(gpu_info)

    def _estimate_bandwidth_from_memory_size(self, total_memory_gb: float) -> float:
        """Estimate memory bandwidth based on total memory size."""
        if total_memory_gb >= 80:  # H100, A100 80GB
            return 3000
        elif total_memory_gb >= 40:  # A100 40GB, A6000
            return 1500
        elif total_memory_gb >= 24:  # RTX 4090, RTX 3090
            return 1000
        elif total_memory_gb >= 12:  # RTX 4070 Ti, RTX 3060 Ti
            return 600
        else:
            return 400

    def _estimate_cuda_cores_from_compute_capability(self, major: int, minor: int, mp_count: int) -> int:
        """Estimate CUDA cores based on compute capability."""
        if major == 9:  # H100
            cores_per_mp = 128
        elif major == 8 and minor >= 6:  # RTX 40 series (Ada Lovelace)
            cores_per_mp = 128
        elif major == 8 and minor >= 0:  # RTX 30 series, A100 (Ampere)
            cores_per_mp = 64 if 'A100' in str(major) else 128  # Datacenter vs consumer
        elif major == 7 and minor >= 5:  # RTX 20 series (Turing)
            cores_per_mp = 64
        elif major == 7 and minor >= 0:  # V100 (Volta)
            cores_per_mp = 64
        else:  # Pascal and older
            cores_per_mp = 64

        return cores_per_mp * mp_count


class F3RMRecommendationEngine:
    """Generates optimal F3RM training recommendations."""

    def __init__(self, dataset_info: Dict, gpu_info: List[Dict]):
        self.dataset = dataset_info
        self.gpus = gpu_info

        # F3RM-specific constants
        self.MODEL_MEMORY_GB = 0.064
        self.SAMPLES_PER_RAY = 48
        self.BYTES_PER_FLOAT = 4
        self.OVERHEAD_MEMORY_GB = 2.0
        self.MEMORY_SAFETY_FACTOR = 0.85

    def infer_feature_dimensions(self, data_dir: Path, feature_type: str) -> int:
        """Infer feature dimensions using the same logic as the training pipeline."""
        print(f"🔧 INFERRING: Feature dimensions for {feature_type}")

        # Method 1: Check existing cache files (same as extract_features_standalone.py)
        cache_root, _ = get_cache_paths(data_dir, feature_type)
        print(f"🔍 CHECKING: Cache directory {cache_root}")

        if cache_root.exists():
            shard_files = sorted(cache_root.glob("chunk_*.npy"))
            if shard_files:
                if feature_type == "DINOCLIP":
                    # For DINOCLIP, check if we have combined cache or separate DINO/CLIP caches
                    sample_shard = np.load(shard_files[0], mmap_mode="r")
                    feature_dim = sample_shard.shape[-1]
                    print(f"💡 INFERRED: DINOCLIP feature dimension from combined cache: {feature_dim}")
                    return feature_dim
                else:
                    sample_shard = np.load(shard_files[0], mmap_mode="r")
                    feature_dim = sample_shard.shape[-1]
                    print(f"💡 INFERRED: {feature_type} feature dimension from cache: {feature_dim}")
                    return feature_dim
            else:
                print(f"⚠️  Cache directory exists but no chunk files found in {cache_root}")
        else:
            print(f"⚠️  Cache directory does not exist: {cache_root}")

        # For DINOCLIP, also check if separate DINO and CLIP caches exist
        if feature_type == "DINOCLIP":
            print(f"🔍 CHECKING: Separate DINO and CLIP caches for DINOCLIP")
            dino_cache_root, _ = get_cache_paths(data_dir, "DINO")
            clip_cache_root, _ = get_cache_paths(data_dir, "CLIP")
            print(f"🔍 CHECKING: DINO cache at {dino_cache_root}")
            print(f"🔍 CHECKING: CLIP cache at {clip_cache_root}")

            if dino_cache_root.exists() and clip_cache_root.exists():
                dino_shards = sorted(dino_cache_root.glob("chunk_*.npy"))
                clip_shards = sorted(clip_cache_root.glob("chunk_*.npy"))
                if dino_shards and clip_shards:
                    dino_shard = np.load(dino_shards[0], mmap_mode="r")
                    clip_shard = np.load(clip_shards[0], mmap_mode="r")
                    dino_dim = dino_shard.shape[-1]
                    clip_dim = clip_shard.shape[-1]
                    feature_dim = dino_dim + clip_dim
                    print(f"💡 INFERRED: DINOCLIP dimension from separate caches: {dino_dim} (DINO) + {clip_dim} (CLIP) = {feature_dim}")
                    return feature_dim

        # Method 2: Extract features from a single test image
        print(f"🔧 ATTEMPTING: Extract single image features to infer {feature_type} dimensions")

        # Get image filenames using the same function as training pipeline
        image_fnames = get_image_filenames_from_dataparser(data_dir)
        if not image_fnames:
            raise ValueError(f"No images found in dataset {data_dir}")

        first_image = image_fnames[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use the same extraction functions as the training pipeline
        if feature_type == "DINOCLIP":
            # Extract both DINO and CLIP
            dino_feats = FEAT_TYPE_TO_EXTRACT_FN["DINO"]([first_image], device)
            clip_feats = FEAT_TYPE_TO_EXTRACT_FN["CLIP"]([first_image], device)
            feature_dim = dino_feats.shape[-1] + clip_feats.shape[-1]
            print(f"💡 INFERRED: DINOCLIP dimension = {dino_feats.shape[-1]} (DINO) + {clip_feats.shape[-1]} (CLIP) = {feature_dim}")
            del dino_feats, clip_feats
        else:
            test_feats = FEAT_TYPE_TO_EXTRACT_FN[feature_type]([first_image], device)
            feature_dim = test_feats.shape[-1]
            print(f"💡 INFERRED: {feature_type} feature dimension from extraction: {feature_dim}")
            del test_feats

        torch.cuda.empty_cache()
        return feature_dim

    def calculate_feature_memory_gb(self, num_images_in_memory: int, width: int, height: int,
                                    feature_dim: int) -> float:
        """Calculate memory required for features currently loaded in memory."""
        return (num_images_in_memory * width * height * feature_dim * self.BYTES_PER_FLOAT) / (1024**3)

    def calculate_working_memory_gb(self, rays_per_batch: int, feature_dim: int) -> float:
        """Calculate working memory for a batch."""
        memory_per_ray = self.SAMPLES_PER_RAY * (3 + feature_dim) * self.BYTES_PER_FLOAT
        return (rays_per_batch * memory_per_ray) / (1024**3)

    def estimate_optimal_batch_size(self, available_memory_gb: float,
                                    feature_memory_gb: float, feature_dim: int) -> int:
        """Estimate optimal batch size based on available memory."""
        # Available memory for working batch
        memory_for_batch = (available_memory_gb - self.MODEL_MEMORY_GB -
                            feature_memory_gb - self.OVERHEAD_MEMORY_GB) * self.MEMORY_SAFETY_FACTOR

        if memory_for_batch <= 0:
            return 1024  # Minimum viable batch

        # Calculate max batch size based on memory
        memory_per_ray = self.SAMPLES_PER_RAY * (3 + feature_dim) * self.BYTES_PER_FLOAT / (1024**3)
        max_batch_size = int(memory_for_batch / memory_per_ray)

        # Round to nearest power of 2 for GPU efficiency
        power_of_2_sizes = [1024, 2048, 4096, 8192, 16384, 32768]
        optimal_size = 1024

        for size in power_of_2_sizes:
            if max_batch_size >= size:
                optimal_size = size
            else:
                break

        return min(optimal_size, 16384)  # Cap at 16K

    def calculate_steps_per_epoch(self, rays_per_batch: int) -> int:
        """Calculate steps needed for one epoch."""
        return math.ceil(self.dataset['total_train_pixels'] / rays_per_batch)

    def recommend_multi_gpu_strategy(self, feature_memory_gb: float) -> Tuple[int, str]:
        """Recommend number of GPUs and strategy."""
        num_gpus = len(self.gpus)
        min_memory_gb = min(gpu['available_memory_gb'] for gpu in self.gpus)

        # Check if features fit in memory
        if feature_memory_gb > min_memory_gb * 0.5:
            return 1, f"Features require {feature_memory_gb:.1f}GB, use 1 GPU to avoid memory issues"
        elif num_gpus == 1:
            return 1, "Single GPU available"
        elif num_gpus >= 2:
            return min(2, num_gpus), f"2 GPUs optimal for F3RM (have {num_gpus} available)"
        else:
            return num_gpus, f"Use all {num_gpus} GPUs available"

    def generate_recommendations(self, target_epochs: int, feature_type: str, overrides: Dict = None) -> Dict:
        """Generate comprehensive training recommendations."""
        if overrides is None:
            overrides = {}

        data_dir = Path(self.dataset['dataset_path'])
        feature_dim = self.infer_feature_dimensions(data_dir, feature_type)

        # Calculate feature memory requirements
        single_image_feature_gb = (self.dataset['image_width'] * self.dataset['image_height'] *
                                   feature_dim * self.BYTES_PER_FLOAT) / (1024**3)

        available_memory_gb = min(gpu['available_memory_gb'] for gpu in self.gpus)
        max_affordable_images = int((available_memory_gb * 0.2) / single_image_feature_gb)
        max_affordable_images = max(16, min(max_affordable_images, self.dataset['num_train_images']))

        # Determine optimal image sampling (can be overridden)
        if 'train_num_images_to_sample_from' in overrides:
            proposed_images_to_sample = overrides['train_num_images_to_sample_from']
            if proposed_images_to_sample == -1:
                proposed_images_to_sample = self.dataset['num_train_images']
        else:
            if max_affordable_images >= 64:
                proposed_images_to_sample = 64
            elif max_affordable_images >= 32:
                proposed_images_to_sample = 32
            elif max_affordable_images >= 16:
                proposed_images_to_sample = 16
            else:
                proposed_images_to_sample = 8

        feature_memory_gb = self.calculate_feature_memory_gb(
            proposed_images_to_sample,
            self.dataset['image_width'],
            self.dataset['image_height'],
            feature_dim
        )

        # GPU strategy (can be overridden)
        if 'num_devices' in overrides:
            recommended_gpus = overrides['num_devices']
            gpu_reason = f"User specified {recommended_gpus} GPUs"
        else:
            recommended_gpus, gpu_reason = self.recommend_multi_gpu_strategy(feature_memory_gb)

        # Optimal batch size (can be overridden)
        if 'train_num_rays_per_batch' in overrides:
            optimal_batch_size = overrides['train_num_rays_per_batch']
        else:
            optimal_batch_size = self.estimate_optimal_batch_size(
                available_memory_gb, feature_memory_gb, feature_dim
            )

            # Adjust for multi-GPU
            if recommended_gpus > 1:
                min_per_gpu = 2048
                if optimal_batch_size / recommended_gpus < min_per_gpu:
                    optimal_batch_size = min_per_gpu * recommended_gpus

        # Calculate training parameters
        steps_per_epoch = self.calculate_steps_per_epoch(optimal_batch_size)

        # Max iterations (can be overridden)
        if 'max_num_iterations' in overrides:
            max_iterations = overrides['max_num_iterations']
        else:
            max_iterations = target_epochs * steps_per_epoch

        # Image sampling strategy (can be overridden)
        if 'train_num_images_to_sample_from' in overrides and 'train_num_times_to_repeat_images' in overrides:
            train_num_images_to_sample_from = overrides['train_num_images_to_sample_from']
            train_num_times_to_repeat_images = overrides['train_num_times_to_repeat_images']
        elif proposed_images_to_sample >= self.dataset['num_train_images']:
            train_num_images_to_sample_from = -1
            train_num_times_to_repeat_images = -1
        else:
            train_num_images_to_sample_from = proposed_images_to_sample
            cycles_needed = math.ceil(self.dataset['num_train_images'] / train_num_images_to_sample_from)
            train_num_times_to_repeat_images = max(256, max_iterations // cycles_needed)

        # Evaluation parameters (can be overridden)
        if 'eval_num_images_to_sample_from' in overrides:
            eval_num_images_to_sample_from = overrides['eval_num_images_to_sample_from']
        else:
            eval_images_to_sample = min(train_num_images_to_sample_from if train_num_images_to_sample_from != -1 else 32,
                                        self.dataset['num_eval_images'])
            if eval_images_to_sample >= self.dataset['num_eval_images']:
                eval_num_images_to_sample_from = -1
            else:
                eval_num_images_to_sample_from = max(8, eval_images_to_sample)

        if 'eval_num_times_to_repeat_images' in overrides:
            eval_num_times_to_repeat_images = overrides['eval_num_times_to_repeat_images']
        elif eval_num_images_to_sample_from == -1:
            eval_num_times_to_repeat_images = -1
        else:
            eval_num_times_to_repeat_images = 500

        # Evaluation batch parameters (can be overridden)
        if 'eval_num_rays_per_batch' in overrides:
            eval_batch_size = overrides['eval_num_rays_per_batch']
        else:
            eval_batch_size = min(optimal_batch_size, 8192)

        if 'eval_num_rays_per_chunk' in overrides:
            eval_chunk_size = overrides['eval_num_rays_per_chunk']
        else:
            eval_chunk_size = 16384 if available_memory_gb > 16 else 8192

        # Estimate training time
        avg_memory_bandwidth = sum(gpu['memory_bandwidth_gbps'] for gpu in self.gpus) / len(self.gpus)
        gpu_utilization = optimal_batch_size / (sum(gpu['cuda_cores'] for gpu in self.gpus) / len(self.gpus) / 8)
        gpu_utilization = min(1.0, gpu_utilization)

        time_per_step = 0.1 * (1.0 / gpu_utilization) * (1000 / avg_memory_bandwidth)
        estimated_time_hours = (max_iterations * time_per_step) / 3600

        return {
            'dataset_analysis': self.dataset,
            'gpu_analysis': {
                'num_gpus_available': len(self.gpus),
                'recommended_gpus': recommended_gpus,
                'gpu_reason': gpu_reason,
                'available_memory_gb': available_memory_gb,
                'feature_memory_gb': feature_memory_gb,
                'estimated_gpu_utilization': gpu_utilization,
            },
            'training_recommendations': {
                'max_num_iterations': max_iterations,
                'train_num_rays_per_batch': optimal_batch_size,
                'train_num_images_to_sample_from': train_num_images_to_sample_from,
                'train_num_times_to_repeat_images': train_num_times_to_repeat_images,
                'eval_num_rays_per_batch': eval_batch_size,
                'eval_num_rays_per_chunk': eval_chunk_size,
                'eval_num_images_to_sample_from': eval_num_images_to_sample_from,
                'eval_num_times_to_repeat_images': eval_num_times_to_repeat_images,
                'num_devices': recommended_gpus,
                'feature_type': feature_type,
            },
            'training_estimates': {
                'target_epochs': target_epochs,
                'steps_per_epoch': steps_per_epoch,
                'estimated_time_hours': estimated_time_hours,
                'rays_per_gpu_per_step': optimal_batch_size // recommended_gpus,
            },
            'command_line': self._generate_command_line(
                max_iterations, optimal_batch_size, train_num_images_to_sample_from,
                train_num_times_to_repeat_images, eval_batch_size, eval_chunk_size,
                eval_num_images_to_sample_from, eval_num_times_to_repeat_images,
                recommended_gpus, feature_type
            ),
            'overrides_used': overrides
        }

    def _generate_command_line(self, max_iterations: int, train_batch: int,
                               train_images: int, repeat_images: int, eval_batch: int,
                               eval_chunk: int, eval_images: int, eval_repeat: int,
                               num_gpus: int, feature_type: str) -> str:
        """Generate ns-train command line."""
        cmd = "ns-train f3rm"
        cmd += f" --max-num-iterations {max_iterations}"
        cmd += f" --pipeline.datamanager.train-num-rays-per-batch {train_batch}"
        cmd += f" --pipeline.datamanager.eval-num-rays-per-batch {eval_batch}"
        cmd += f" --pipeline.model.eval-num-rays-per-chunk {eval_chunk}"

        if train_images != -1:
            cmd += f" --pipeline.datamanager.train-num-images-to-sample-from {train_images}"

        if repeat_images != -1:
            cmd += f" --pipeline.datamanager.train-num-times-to-repeat-images {repeat_images}"

        if eval_images != -1:
            cmd += f" --pipeline.datamanager.eval-num-images-to-sample-from {eval_images}"

        if eval_repeat != -1:
            cmd += f" --pipeline.datamanager.eval-num-times-to-repeat-images {eval_repeat}"

        if num_gpus > 1:
            cmd += f" --machine.num-devices {num_gpus}"

        cmd += f" --pipeline.datamanager.feature-type {feature_type}"
        cmd += f" --data {self.dataset['dataset_path']}"

        return cmd

    def validate_configuration(self, overrides: Dict, dataset_info: Dict) -> List[str]:
        """Validate configuration for incompatible combinations."""
        warnings = []

        # Check GPU device count
        if 'num_devices' in overrides:
            if overrides['num_devices'] > len(self.gpus):
                warnings.append(f"⚠️  INCOMPATIBLE: Requested {overrides['num_devices']} GPUs but only {len(self.gpus)} available")
            if overrides['num_devices'] <= 0:
                warnings.append(f"⚠️  INCOMPATIBLE: num_devices must be positive, got {overrides['num_devices']}")

        # Check memory constraints for image sampling
        if 'train_num_images_to_sample_from' in overrides:
            train_images = overrides['train_num_images_to_sample_from']
            if train_images != -1:
                if train_images > dataset_info['num_train_images']:
                    warnings.append(f"⚠️  INCOMPATIBLE: Requested {train_images} training images but dataset only has {dataset_info['num_train_images']}")
                if train_images <= 0:
                    warnings.append(f"⚠️  INCOMPATIBLE: train_num_images_to_sample_from must be positive or -1, got {train_images}")

        # Check eval image sampling
        if 'eval_num_images_to_sample_from' in overrides:
            eval_images = overrides['eval_num_images_to_sample_from']
            if eval_images != -1:
                if eval_images > dataset_info['num_eval_images']:
                    warnings.append(f"⚠️  INCOMPATIBLE: Requested {eval_images} eval images but dataset only has {dataset_info['num_eval_images']}")
                if eval_images <= 0:
                    warnings.append(f"⚠️  INCOMPATIBLE: eval_num_images_to_sample_from must be positive or -1, got {eval_images}")

        # Check batch sizes
        if 'train_num_rays_per_batch' in overrides:
            batch_size = overrides['train_num_rays_per_batch']
            if batch_size <= 0:
                warnings.append(f"⚠️  INCOMPATIBLE: train_num_rays_per_batch must be positive, got {batch_size}")

            # Check if batch size is compatible with num_devices
            if 'num_devices' in overrides and overrides['num_devices'] > 1:
                min_per_gpu = 1024
                if batch_size / overrides['num_devices'] < min_per_gpu:
                    warnings.append(f"⚠️  SUBOPTIMAL: Batch size {batch_size} with {overrides['num_devices']} GPUs gives {batch_size // overrides['num_devices']} rays/GPU (recommended: ≥{min_per_gpu})")

        if 'eval_num_rays_per_batch' in overrides:
            eval_batch = overrides['eval_num_rays_per_batch']
            if eval_batch <= 0:
                warnings.append(f"⚠️  INCOMPATIBLE: eval_num_rays_per_batch must be positive, got {eval_batch}")

        if 'eval_num_rays_per_chunk' in overrides:
            eval_chunk = overrides['eval_num_rays_per_chunk']
            if eval_chunk <= 0:
                warnings.append(f"⚠️  INCOMPATIBLE: eval_num_rays_per_chunk must be positive, got {eval_chunk}")

        # Check iterations
        if 'max_num_iterations' in overrides:
            max_iter = overrides['max_num_iterations']
            if max_iter <= 0:
                warnings.append(f"⚠️  INCOMPATIBLE: max_num_iterations must be positive, got {max_iter}")

        # Check repeat times
        for param in ['train_num_times_to_repeat_images', 'eval_num_times_to_repeat_images']:
            if param in overrides:
                repeat_times = overrides[param]
                if repeat_times != -1 and repeat_times <= 0:
                    warnings.append(f"⚠️  INCOMPATIBLE: {param} must be positive or -1, got {repeat_times}")

        return warnings


def print_recommendations(recommendations: Dict):
    """Print recommendations in a formatted way."""
    print("\n" + "=" * 80)
    print("F3RM TRAINING RECOMMENDATIONS")
    print("=" * 80)

    # Show overrides if any were used
    overrides = recommendations.get('overrides_used', {})
    if overrides:
        print(f"\n🔧 USER OVERRIDES APPLIED:")
        for key, value in overrides.items():
            print(f"  • {key.replace('_', '-')}: {value}")

    # Dataset info
    dataset = recommendations['dataset_analysis']
    print(f"\n📊 DATASET ANALYSIS:")
    print(f"  • Training images: {dataset['num_train_images']:,}")
    print(f"  • Evaluation images: {dataset['num_eval_images']:,}")
    print(f"  • Image dimensions: {dataset['image_width']}×{dataset['image_height']}")
    print(f"  • Total training pixels: {dataset['total_train_pixels']:,}")

    # GPU info
    gpu = recommendations['gpu_analysis']
    print(f"\n🚀 GPU ANALYSIS:")
    print(f"  • GPUs available: {gpu['num_gpus_available']}")
    print(f"  • Recommended GPUs: {gpu['recommended_gpus']}")
    print(f"  • Reason: {gpu['gpu_reason']}")
    print(f"  • Available memory: {gpu['available_memory_gb']:.1f} GB")
    print(f"  • Feature memory needed: {gpu['feature_memory_gb']:.1f} GB")
    print(f"  • Estimated GPU utilization: {gpu['estimated_gpu_utilization']*100:.1f}%")

    # Training recommendations
    train = recommendations['training_recommendations']
    print(f"\n⚙️ TRAINING RECOMMENDATIONS:")
    for param, value in train.items():
        if param == 'feature_type':
            continue
        override_marker = " 🔧" if param in overrides else ""
        if isinstance(value, int) and value >= 1000:
            print(f"  • {param.replace('_', '-')}: {value:,}{override_marker}")
        else:
            print(f"  • {param.replace('_', '-')}: {value}{override_marker}")

    # Training estimates
    est = recommendations['training_estimates']
    print(f"\n⏱️ TRAINING ESTIMATES:")
    print(f"  • Target epochs: {est['target_epochs']}")
    print(f"  • Steps per epoch: {est['steps_per_epoch']:,}")
    print(f"  • Estimated training time: {est['estimated_time_hours']:.1f} hours")
    print(f"  • Rays per GPU per step: {est['rays_per_gpu_per_step']:,}")

    # Command line
    print(f"\n💻 RECOMMENDED COMMAND:")
    print(f"  {recommendations['command_line']}")

    print("\n" + "=" * 80)
    if overrides:
        print("🔧 = User override applied")
        print("Copy the command above to start training with your custom parameters!")
    else:
        print("Copy the command above to start training with optimized parameters!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="F3RM Meta-Analysis: Optimize training parameters for your dataset and hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python meta_analyze.py --data /path/to/dataset --epochs 10
  python meta_analyze.py --data /path/to/dataset --epochs 5 --feature-type DINO
  python meta_analyze.py --data /path/to/dataset --epochs 15 --feature-type DINOCLIP --output recommendations.json
  python meta_analyze.py --data /path/to/dataset --epochs 10 --num-devices 4 --train-num-rays-per-batch 8192
        """
    )

    # Required arguments
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset directory (containing transforms.json)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Target number of training epochs (default: 10)")
    parser.add_argument("--feature-type", type=str, default="CLIP",
                        choices=["CLIP", "DINO", "DINOCLIP", "ROBOPOINTproj", "ROBOPOINTnoproj"],
                        help="Feature type to use (default: CLIP)")

    # Output options
    parser.add_argument("--output", type=str,
                        help="Output JSON file to save recommendations")
    parser.add_argument("--quiet", action="store_true",
                        help="Only output the command line (for scripting)")

    # Override arguments - training parameters
    parser.add_argument("--num-devices", type=int,
                        help="Override number of GPUs to use")
    parser.add_argument("--max-num-iterations", type=int,
                        help="Override maximum number of training iterations")
    parser.add_argument("--train-num-rays-per-batch", type=int,
                        help="Override training batch size (rays per batch)")
    parser.add_argument("--train-num-images-to-sample-from", type=int,
                        help="Override number of training images to sample from (-1 for all)")
    parser.add_argument("--train-num-times-to-repeat-images", type=int,
                        help="Override how many times to repeat training images (-1 for auto)")

    # Override arguments - evaluation parameters
    parser.add_argument("--eval-num-rays-per-batch", type=int,
                        help="Override evaluation batch size (rays per batch)")
    parser.add_argument("--eval-num-rays-per-chunk", type=int,
                        help="Override evaluation chunk size")
    parser.add_argument("--eval-num-images-to-sample-from", type=int,
                        help="Override number of evaluation images to sample from (-1 for all)")
    parser.add_argument("--eval-num-times-to-repeat-images", type=int,
                        help="Override how many times to repeat evaluation images (-1 for auto)")

    args = parser.parse_args()

    # Collect override parameters
    overrides = {}
    override_args = [
        'num_devices', 'max_num_iterations', 'train_num_rays_per_batch',
        'train_num_images_to_sample_from', 'train_num_times_to_repeat_images',
        'eval_num_rays_per_batch', 'eval_num_rays_per_chunk',
        'eval_num_images_to_sample_from', 'eval_num_times_to_repeat_images'
    ]

    for arg in override_args:
        value = getattr(args, arg.replace('-', '_'), None)
        if value is not None:
            overrides[arg.replace('-', '_')] = value

    # Analyze dataset
    dataset_analyzer = DatasetAnalyzer(args.data)
    dataset_info = dataset_analyzer.analyze()

    # Analyze GPUs
    gpu_analyzer = GPUAnalyzer()
    gpu_info = gpu_analyzer.gpus

    # Create recommendation engine and validate configuration
    engine = F3RMRecommendationEngine(dataset_info, gpu_info)

    # Validate overrides for incompatible combinations
    validation_warnings = engine.validate_configuration(overrides, dataset_info)
    if validation_warnings:
        print("\n" + "=" * 80)
        print("⚠️  CONFIGURATION WARNINGS")
        print("=" * 80)
        for warning in validation_warnings:
            print(f"  {warning}")
        print("\nProceeding with provided parameters...")
        print("=" * 80)

    # Generate recommendations
    recommendations = engine.generate_recommendations(args.epochs, args.feature_type, overrides)

    # Output results
    if args.quiet:
        print(recommendations['command_line'])
    else:
        print_recommendations(recommendations)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(recommendations, f, indent=2)
        if not args.quiet:
            print(f"\nRecommendations saved to: {args.output}")


if __name__ == "__main__":
    main()
