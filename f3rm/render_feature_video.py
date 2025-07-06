#!/usr/bin/env python3
"""
Custom F3RM Feature Video Renderer

This script renders feature and feature_pca videos from camera paths,
ensuring consistent PCA visualization across all frames.

Usage:
    python render_feature_video.py --config path/to/config.yml --camera-path path/to/camera_path.json --output path/to/output.mp4 --render-type feature_pca
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

from f3rm.pca_colormap import apply_pca_colormap_return_proj

console = Console()


def load_camera_path(camera_path_file: Path) -> List[Dict]:
    """Load camera path from JSON file."""
    with open(camera_path_file, 'r') as f:
        camera_path = json.load(f)

    # Handle different JSON structures
    if "camera_path" in camera_path:
        poses = camera_path["camera_path"]
    elif "frames" in camera_path:
        poses = camera_path["frames"]
    elif isinstance(camera_path, list):
        poses = camera_path
    else:
        raise ValueError(f"Unknown camera path format. Keys: {list(camera_path.keys())}")

    # Debug: print first pose structure
    if poses:
        console.print(f"[dim]First pose keys: {list(poses[0].keys())}")
        if "camera_to_world" in poses[0]:
            console.print(f"[dim]camera_to_world shape: {np.array(poses[0]['camera_to_world']).shape}")
        elif "transform_matrix" in poses[0]:
            console.print(f"[dim]transform_matrix shape: {np.array(poses[0]['transform_matrix']).shape}")

    return poses


def create_camera_from_pose(
    pose_dict: Dict,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    device: torch.device,
) -> Cameras:
    """Create a camera from pose dictionary."""
    # Extract camera-to-world matrix - handle different formats
    if "camera_to_world" in pose_dict:
        c2w_data = pose_dict["camera_to_world"]
    elif "transform_matrix" in pose_dict:
        c2w_data = pose_dict["transform_matrix"]
    else:
        raise ValueError("No camera pose found in pose_dict. Expected 'camera_to_world' or 'transform_matrix'")

    # Convert to tensor and ensure proper shape
    c2w = torch.tensor(c2w_data, dtype=torch.float32, device=device)

    # Handle different input shapes
    if c2w.dim() == 1:
        # Flattened 4x4 matrix
        c2w = c2w.reshape(4, 4)
    elif c2w.dim() == 2:
        # Already a 2D matrix
        pass
    else:
        raise ValueError(f"Unexpected camera pose shape: {c2w.shape}")

    # Ensure we have the right dimensions and extract 3x4 part
    if c2w.shape == (4, 4):
        c2w_3x4 = c2w[:3, :4]
    elif c2w.shape == (3, 4):
        c2w_3x4 = c2w
    else:
        raise ValueError(f"Unexpected camera pose shape: {c2w.shape}")

    # Create camera
    camera = Cameras(
        camera_to_worlds=c2w_3x4[None, :, :],  # Add batch dimension
        fx=torch.tensor([fx], dtype=torch.float32, device=device),
        fy=torch.tensor([fy], dtype=torch.float32, device=device),
        cx=torch.tensor([cx], dtype=torch.float32, device=device),
        cy=torch.tensor([cy], dtype=torch.float32, device=device),
        width=torch.tensor([width], dtype=torch.int32, device=device),
        height=torch.tensor([height], dtype=torch.int32, device=device),
        camera_type=CameraType.PERSPECTIVE,
    )
    return camera


def render_frame(
    pipeline,
    camera: Cameras,
    render_type: Literal["rgb", "feature", "feature_pca"],
    pca_proj: Optional[torch.Tensor] = None,
    pca_min: Optional[torch.Tensor] = None,
    pca_max: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Render a single frame with chunking for memory efficiency."""
    with torch.no_grad():
        # Get camera ray bundle
        ray_bundle = camera.generate_rays(camera_indices=0)

        # Get model outputs with chunking for large images
        height, width = camera.height[0].item(), camera.width[0].item()

        # Use chunked rendering for large images
        if height * width > 1000000:  # > 1M pixels
            # Render in chunks to avoid OOM
            chunk_size = 1 << 12  # 4096 rays per chunk
            outputs = {}

            # Split ray bundle into chunks
            num_rays = ray_bundle.shape[0]
            for i in range(0, num_rays, chunk_size):
                end_idx = min(i + chunk_size, num_rays)
                chunk_bundle = ray_bundle[i:end_idx]

                # Get chunk outputs
                chunk_outputs = pipeline.model.get_outputs_for_camera_ray_bundle(chunk_bundle)

                # Accumulate outputs
                for key, value in chunk_outputs.items():
                    if key not in outputs:
                        outputs[key] = []
                    outputs[key].append(value)

            # Concatenate chunks
            for key in outputs:
                outputs[key] = torch.cat(outputs[key], dim=0)

            # Reshape to image dimensions
            for key in outputs:
                if outputs[key].dim() == 2:
                    outputs[key] = outputs[key].reshape(height, width, -1)
                elif outputs[key].dim() == 1:
                    outputs[key] = outputs[key].reshape(height, width)
        else:
            # Small image, render normally
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

        if render_type == "rgb":
            rendered = outputs["rgb"]
        elif render_type == "feature":
            # For raw features, just take first 3 channels and normalize
            features = outputs["feature"]
            if features.shape[-1] >= 3:
                rendered = features[..., :3]
                # Normalize to [0, 1]
                rendered = (rendered - rendered.min()) / (rendered.max() - rendered.min() + 1e-8)
            else:
                # If less than 3 channels, duplicate to make RGB
                rendered = features.repeat(1, 1, 3)[:, :, :3]
                rendered = (rendered - rendered.min()) / (rendered.max() - rendered.min() + 1e-8)
        elif render_type == "feature_pca":
            features = outputs["feature"]
            # Apply consistent PCA
            rendered, pca_proj, pca_min, pca_max = apply_pca_colormap_return_proj(
                features, pca_proj, pca_min, pca_max
            )
        else:
            raise ValueError(f"Unknown render type: {render_type}")

        # Convert to numpy and ensure values are in [0, 1]
        rendered_np = rendered.cpu().numpy()
        rendered_np = np.clip(rendered_np, 0, 1)

        return rendered_np, pca_proj, pca_min, pca_max


def render_video(
    config_path: Path,
    camera_path_file: Path,
    output_path: Path,
    render_type: Literal["rgb", "feature", "feature_pca"] = "feature_pca",
    fps: int = 24,
    quality: int = 8,  # CRF value for H.264 (lower = better quality)
    resolution_scale: float = 1.0,
) -> None:
    """Render a video from camera path."""

    console.print(f"[bold green]Loading model from {config_path}...")

    # Load model
    config, pipeline, checkpoint_path, step = eval_setup(config_path)
    device = pipeline.device

    console.print(f"[bold green]Loaded checkpoint from step {step}")

    # Load camera path
    camera_poses = load_camera_path(camera_path_file)
    console.print(f"[bold green]Loaded {len(camera_poses)} camera poses")

    # Get camera intrinsics from the model's training data
    train_cameras = pipeline.datamanager.train_dataset.cameras
    fx = float(train_cameras.fx[0])
    fy = float(train_cameras.fy[0])
    cx = float(train_cameras.cx[0])
    cy = float(train_cameras.cy[0])
    width = int(train_cameras.width[0] * resolution_scale)
    height = int(train_cameras.height[0] * resolution_scale)

    # Adjust intrinsics for resolution scaling
    fx *= resolution_scale
    fy *= resolution_scale
    cx *= resolution_scale
    cy *= resolution_scale

    console.print(f"[bold green]Rendering at {width}x{height} resolution")
    console.print(f"[bold green]Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Prepare video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Initialize PCA projection for consistency
    pca_proj = None
    pca_min = None
    pca_max = None

    console.print(f"[bold green]Rendering {render_type} video...")

    # Create progress bar
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Rendering frames", total=len(camera_poses))

        # Render frames
        for i, pose_dict in enumerate(camera_poses):
            # Create camera
            camera = create_camera_from_pose(
                pose_dict, fx, fy, cx, cy, width, height, device
            )

            # Render frame
            rendered_np, pca_proj, pca_min, pca_max = render_frame(
                pipeline, camera, render_type, pca_proj, pca_min, pca_max
            )

            # Convert to uint8 and BGR for OpenCV
            frame = (rendered_np * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write frame
            out.write(frame_bgr)

            # Update progress
            progress.update(task, advance=1)

            # Clear GPU cache periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()

    # Release video writer
    out.release()

    # If output is .mp4, re-encode with better compression
    if output_path.suffix.lower() == '.mp4':
        temp_path = output_path.with_suffix('.temp.mp4')
        os.rename(output_path, temp_path)

        # Re-encode with H.264 and better compression
        cmd = [
            'ffmpeg', '-y', '-i', str(temp_path),
            '-c:v', 'libx264', '-crf', str(quality),
            '-preset', 'medium', '-pix_fmt', 'yuv420p',
            str(output_path)
        ]

        console.print(f"[bold green]Re-encoding video with H.264...")
        try:
            import subprocess
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_path)
            console.print(f"[bold green]Video saved to {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg fails, just use the original
            os.rename(temp_path, output_path)
            console.print(f"[yellow]FFmpeg not available, saved uncompressed video to {output_path}")
    else:
        console.print(f"[bold green]Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render F3RM feature videos")
    parser.add_argument("--config", type=Path, required=True, help="Path to model config.yml")
    parser.add_argument("--camera-path", type=Path, required=True, help="Path to camera path JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output video path")
    parser.add_argument(
        "--render-type",
        choices=["rgb", "feature", "feature_pca"],
        default="feature_pca",
        help="Type of rendering to perform"
    )
    parser.add_argument("--fps", type=int, default=15, help="Video FPS")
    parser.add_argument("--quality", type=int, default=8, help="Video quality (CRF, lower=better)")
    parser.add_argument("--resolution-scale", type=float, default=0.5, help="Resolution scaling factor (default: 0.5 for speed)")

    args = parser.parse_args()

    # Validate inputs
    if not args.config.exists():
        console.print(f"[bold red]Config file not found: {args.config}")
        return

    if not args.camera_path.exists():
        console.print(f"[bold red]Camera path file not found: {args.camera_path}")
        return

    console.print(f"[bold blue]F3RM Feature Video Renderer")
    console.print(f"[bold blue]Config: {args.config}")
    console.print(f"[bold blue]Camera path: {args.camera_path}")
    console.print(f"[bold blue]Output: {args.output}")
    console.print(f"[bold blue]Render type: {args.render_type}")

    try:
        render_video(
            args.config,
            args.camera_path,
            args.output,
            args.render_type,
            args.fps,
            args.quality,
            args.resolution_scale,
        )
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        raise


if __name__ == "__main__":
    main()
