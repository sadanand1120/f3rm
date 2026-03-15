#!/usr/bin/env python3
"""Export F3RM pointcloud data (raw tensors + visualization colors)."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d
import torch
from nerfstudio.utils.eval_utils import eval_setup
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from f3rm.pca_colormap import apply_pca_colormap_return_proj

console = Console()


@dataclass
class ExportOptions:
    config_path: Path
    output_dir: Path
    num_points: int = 1_000_000
    bbox_min: float = -1.0
    bbox_max: float = 1.0
    feature_dtype: str = "float16"
    seed: int = 42


@dataclass
class SampledTensors:
    points: torch.Tensor
    rgbs: torch.Tensor
    features: torch.Tensor
    foreground_logits: torch.Tensor
    foreground_probs: torch.Tensor
    pred_normals_raw: Optional[torch.Tensor]
    pred_normals_rgb: Optional[torch.Tensor]
    normals_raw: Optional[torch.Tensor]
    normals_rgb: Optional[torch.Tensor]


class FeaturePointcloudExporter:
    """Class-based pointcloud exporter for current F3RM model outputs."""

    def __init__(self, options: ExportOptions):
        self.options = options
        self.pipeline = None
        self.model = None
        self.device = None

    def run(self) -> None:
        self._set_seed(self.options.seed)
        self._setup_pipeline()
        self._configure_sampling_for_export()
        start = time.perf_counter()
        sampled = self._sample_points()
        console.print(
            f"[cyan]Sampled {sampled.points.shape[0]:,} points in {time.perf_counter() - start:.2f}s[/]"
        )
        self._save_outputs(sampled)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_pipeline(self) -> None:
        console.print(f"[bold blue]Loading model from[/] {self.options.config_path}")
        _, pipeline, _, step = eval_setup(config_path=self.options.config_path, test_mode="test")
        pipeline.eval()
        self.pipeline = pipeline
        self.model = pipeline.model
        self.device = torch.device(pipeline.device)
        console.print(f"[green]Loaded checkpoint step {step}[/]")

    def _configure_sampling_for_export(self) -> None:
        """Increase spatial coverage by forcing cache resampling every export iteration."""
        dataloader = getattr(self.pipeline.datamanager, "train_image_dataloader", None)
        if dataloader is None or not hasattr(dataloader, "num_times_to_repeat_images"):
            return

        old_repeat = dataloader.num_times_to_repeat_images
        dataloader.num_times_to_repeat_images = 0
        dataloader.first_time = True
        dataloader.cached_collated_batch = None
        console.print(
            "[cyan]Export sampling:[/] set train image cache resampling "
            f"from every {old_repeat} iters to every iter for better coverage."
        )

    @torch.no_grad()
    def _sample_points(self) -> SampledTensors:
        tensor_lists: Dict[str, List[torch.Tensor]] = {
            "points": [],
            "rgbs": [],
            "features": [],
            "foreground_logits": [],
            "foreground_probs": [],
            "pred_normals_rgb": [],
            "normals_rgb": [],
        }

        collected = 0
        total_candidate_rays = 0
        train_dataset = getattr(self.pipeline.datamanager, "train_dataset", None)
        cameras_seen = (
            torch.zeros(len(train_dataset), dtype=torch.bool)
            if train_dataset is not None and hasattr(train_dataset, "__len__")
            else None
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Sampling rays", total=self.options.num_points)
            while collected < self.options.num_points:
                ray_bundle, batch = self.pipeline.datamanager.next_train(0)
                outputs = self.model(ray_bundle)

                points = ray_bundle.origins + ray_bundle.directions * outputs["depth"]
                total_candidate_rays += int(points.shape[0])

                if cameras_seen is not None and isinstance(batch, dict) and "indices" in batch:
                    cam_ids = batch["indices"][:, 0].detach().to("cpu", dtype=torch.long)
                    cameras_seen[cam_ids.unique()] = True

                mask = self._build_valid_mask(points, outputs)
                if not torch.any(mask):
                    continue

                points_m = points[mask]
                rgbs_m = outputs["rgb"][mask]
                features_m = outputs["feature"][mask]
                fg_logits_m = outputs["foreground_logits"][mask]
                fg_probs_m = torch.softmax(fg_logits_m.float(), dim=-1)[..., 1:2]

                tensor_lists["points"].append(points_m.cpu())
                tensor_lists["rgbs"].append(rgbs_m.cpu())
                tensor_lists["features"].append(features_m.cpu())
                tensor_lists["foreground_logits"].append(fg_logits_m.cpu())
                tensor_lists["foreground_probs"].append(fg_probs_m.cpu())

                if "pred_normals" in outputs:
                    tensor_lists["pred_normals_rgb"].append(outputs["pred_normals"][mask].cpu())
                if "normals" in outputs:
                    tensor_lists["normals_rgb"].append(outputs["normals"][mask].cpu())

                collected += int(points_m.shape[0])
                progress.update(task, completed=min(collected, self.options.num_points))

        if cameras_seen is not None:
            console.print(f"[cyan]Train cameras covered:[/] {int(cameras_seen.sum())}/{cameras_seen.numel()}")
        if total_candidate_rays > 0:
            kept_pct = 100.0 * collected / total_candidate_rays
            console.print(
                f"[cyan]Accepted rays after filters:[/] {collected:,}/{total_candidate_rays:,} ({kept_pct:.1f}%)"
            )

        concatenated = {
            key: (torch.cat(value, dim=0) if value else None)
            for key, value in tensor_lists.items()
        }

        num_total = int(concatenated["points"].shape[0])
        if num_total > self.options.num_points:
            idx = torch.randperm(num_total)[: self.options.num_points]
            for key, value in concatenated.items():
                if value is not None:
                    concatenated[key] = value[idx]

        pred_normals_rgb = concatenated["pred_normals_rgb"]
        normals_rgb = concatenated["normals_rgb"]

        pred_normals_raw = None
        if pred_normals_rgb is not None:
            pred_normals_raw = torch.clamp(pred_normals_rgb.float(), 0.0, 1.0) * 2.0 - 1.0

        normals_raw = None
        if normals_rgb is not None:
            normals_raw = torch.clamp(normals_rgb.float(), 0.0, 1.0) * 2.0 - 1.0

        return SampledTensors(
            points=concatenated["points"].float(),
            rgbs=concatenated["rgbs"].float(),
            features=concatenated["features"],
            foreground_logits=concatenated["foreground_logits"].float(),
            foreground_probs=concatenated["foreground_probs"].float(),
            pred_normals_raw=pred_normals_raw,
            pred_normals_rgb=pred_normals_rgb.float() if pred_normals_rgb is not None else None,
            normals_raw=normals_raw,
            normals_rgb=normals_rgb.float() if normals_rgb is not None else None,
        )

    def _build_valid_mask(self, points: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = torch.isfinite(points).all(dim=-1)
        mask = mask & torch.isfinite(outputs["rgb"]).all(dim=-1)
        mask = mask & torch.isfinite(outputs["feature"]).all(dim=-1)
        mask = mask & torch.isfinite(outputs["foreground_logits"]).all(dim=-1)
        in_bbox = ((points >= self.options.bbox_min) & (points <= self.options.bbox_max)).all(dim=-1)
        mask = mask & in_bbox
        return mask

    def _save_outputs(self, sampled: SampledTensors) -> None:
        out_dir = self.options.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        console.print("[cyan][1/4][/cyan] Computing PCA colors...")
        feature_pca, pca_proj, pca_min, pca_max = apply_pca_colormap_return_proj(sampled.features.float())
        console.print(f"[green]Done[/] PCA colors ({time.perf_counter() - t0:.2f}s)")

        feature_dtype = np.float16 if self.options.feature_dtype == "float16" else np.float32
        feature_file = "features_float16.npy" if feature_dtype == np.float16 else "features_float32.npy"

        t1 = time.perf_counter()
        console.print("[cyan][2/4][/cyan] Saving raw arrays...")
        np.save(out_dir / "points.npy", sampled.points.numpy().astype(np.float32))
        np.save(out_dir / "rgbs.npy", sampled.rgbs.numpy().astype(np.float32))
        np.save(out_dir / feature_file, sampled.features.numpy().astype(feature_dtype))
        np.save(out_dir / "feature_pca.npy", feature_pca.numpy().astype(np.float32))
        np.save(out_dir / "foreground_logits_raw.npy", sampled.foreground_logits.numpy().astype(np.float32))
        np.save(out_dir / "foreground_prob_raw.npy", sampled.foreground_probs.numpy().astype(np.float32))

        if sampled.pred_normals_raw is not None:
            np.save(out_dir / "pred_normals_raw.npy", sampled.pred_normals_raw.numpy().astype(np.float32))
        if sampled.pred_normals_rgb is not None:
            np.save(out_dir / "pred_normals_rgb.npy", sampled.pred_normals_rgb.numpy().astype(np.float32))
        if sampled.normals_raw is not None:
            np.save(out_dir / "normals_raw.npy", sampled.normals_raw.numpy().astype(np.float32))
        if sampled.normals_rgb is not None:
            np.save(out_dir / "normals_rgb.npy", sampled.normals_rgb.numpy().astype(np.float32))

        pca_payload = {
            "projection_matrix": pca_proj.numpy(),
            "min_values": pca_min.numpy(),
            "max_values": pca_max.numpy(),
            "feature_dim": int(sampled.features.shape[-1]),
            "num_points": int(sampled.points.shape[0]),
        }
        with open(out_dir / "pca_params.pkl", "wb") as f:
            pickle.dump(pca_payload, f)
        console.print(f"[green]Done[/] raw arrays ({time.perf_counter() - t1:.2f}s)")

        # Save visualization pointclouds
        t2 = time.perf_counter()
        console.print("[cyan][3/4][/cyan] Saving PLY pointclouds...")
        self._save_ply(out_dir / "pointcloud_rgb.ply", sampled.points, sampled.rgbs)
        self._save_ply(out_dir / "pointcloud_feature_pca.ply", sampled.points, feature_pca.float())

        additional_files: Dict[str, str] = {}

        foreground_prob_rgb = self.model.prob_from_probs_shader(sampled.foreground_probs.to(self.device)).cpu().float()
        self._save_ply(out_dir / "pointcloud_foreground_prob_rgb.ply", sampled.points, foreground_prob_rgb)
        additional_files["foreground_prob_rgb"] = "pointcloud_foreground_prob_rgb.ply"

        if sampled.pred_normals_rgb is not None:
            self._save_ply(out_dir / "pointcloud_pred_normals_rgb.ply", sampled.points, sampled.pred_normals_rgb)
            additional_files["pred_normals_rgb"] = "pointcloud_pred_normals_rgb.ply"

        if sampled.normals_rgb is not None:
            self._save_ply(out_dir / "pointcloud_normals_rgb.ply", sampled.points, sampled.normals_rgb)
            additional_files["normals_rgb"] = "pointcloud_normals_rgb.ply"
        console.print(f"[green]Done[/] PLY pointclouds ({time.perf_counter() - t2:.2f}s)")

        t3 = time.perf_counter()
        console.print("[cyan][4/4][/cyan] Writing metadata...")
        metadata = {
            "num_points": int(sampled.points.shape[0]),
            "feature_dim": int(sampled.features.shape[-1]),
            "bbox_min": sampled.points.amin(dim=0).tolist(),
            "bbox_max": sampled.points.amax(dim=0).tolist(),
            "compressed_features": feature_dtype == np.float16,
            "files": {
                "rgb_pointcloud": "pointcloud_rgb.ply",
                "pca_pointcloud": "pointcloud_feature_pca.ply",
                "points": "points.npy",
                "rgbs": "rgbs.npy",
                "features": feature_file,
                "feature_pca": "feature_pca.npy",
                "pca_params": "pca_params.pkl",
                "foreground_logits_raw": "foreground_logits_raw.npy",
                "foreground_prob_raw": "foreground_prob_raw.npy",
                **additional_files,
            },
            "additional_outputs": list(additional_files.keys()),
            "raw_outputs": {
                "pred_normals_raw": sampled.pred_normals_raw is not None,
                "normals_raw": sampled.normals_raw is not None,
                "foreground_logits_raw": True,
                "foreground_prob_raw": True,
            },
        }

        if sampled.pred_normals_raw is not None:
            metadata["files"]["pred_normals_raw"] = "pred_normals_raw.npy"
            metadata["files"]["pred_normals_rgb_raw"] = "pred_normals_rgb.npy"
        if sampled.normals_raw is not None:
            metadata["files"]["normals_raw"] = "normals_raw.npy"
            metadata["files"]["normals_rgb_raw"] = "normals_rgb.npy"

        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        console.print(f"[green]Done[/] metadata ({time.perf_counter() - t3:.2f}s)")

        console.print("[bold green]Export complete[/]")
        console.print(f"[green]Output directory:[/] {out_dir}")

    @staticmethod
    def _save_ply(path: Path, points: torch.Tensor, colors: torch.Tensor) -> None:
        points_np = points.numpy().astype(np.float64)
        colors_np = np.clip(colors.numpy().astype(np.float64), 0.0, 1.0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors_np)
        o3d.io.write_point_cloud(str(path), pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export pointcloud data from a trained F3RM model")
    parser.add_argument("--config", type=Path, required=True, help="Path to model config.yml")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write outputs")
    parser.add_argument("--num-points", type=int, default=1_000_000, help="Number of points to export")
    parser.add_argument("--bbox-min", type=float, default=-1.0, help="Bounding box min value (cube filter)")
    parser.add_argument("--bbox-max", type=float, default=1.0, help="Bounding box max value (cube filter)")
    parser.add_argument(
        "--feature-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Saved feature dtype",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    options = ExportOptions(
        config_path=args.config,
        output_dir=args.output_dir,
        num_points=args.num_points,
        bbox_min=args.bbox_min,
        bbox_max=args.bbox_max,
        feature_dtype=args.feature_dtype,
        seed=args.seed,
    )
    FeaturePointcloudExporter(options).run()
