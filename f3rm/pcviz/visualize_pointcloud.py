#!/usr/bin/env python3
"""Visualize exported F3RM pointcloud artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d
import torch
from rich.console import Console

from f3rm.features.utils import compute_similarity_scores, parse_comma_separated_labels

console = Console()


@dataclass
class ViewOptions:
    export_dir: Path
    view: str = "rgb"
    display: str = "open3d"
    show_axes: bool = False
    viser_host: str = "0.0.0.0"
    viser_port: int = 8012
    viser_share_url: bool = False


class PointcloudViewer:
    """Simple metadata-driven pointcloud viewer."""

    VIEW_KEY_TO_FILE_KEY = {
        "rgb": "rgb_pointcloud",
        "feature_pca": "pca_pointcloud",
        "foreground_prob": "foreground_prob_rgb",
        "pred_normals": "pred_normals_rgb",
        "normals": "normals_rgb",
    }

    def __init__(self, options: ViewOptions):
        self.options = options
        self.metadata = self._load_metadata(options.export_dir)
        self._points_cache: np.ndarray | None = None
        self._features_cache: np.ndarray | None = None
        self._rgbs_cache: np.ndarray | None = None
        self._clip_model = None
        self._clip_tokenizer = None
        self._clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> None:
        if self.options.display == "open3d":
            self._run_open3d()
            return
        if self.options.display == "viser":
            self._run_viser()
            return
        raise ValueError(f"Unsupported display mode: {self.options.display}")

    def _load_metadata(self, export_dir: Path) -> Dict:
        metadata_path = export_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_pointcloud_path(self, view_name: str) -> Path:
        if view_name in ("similarity_semfeat", "similarity_rgb"):
            raise ValueError(f"`{view_name}` is not a file-backed pointcloud view.")
        file_key = self.VIEW_KEY_TO_FILE_KEY.get(view_name)
        if file_key is None:
            valid = ", ".join(sorted(self.VIEW_KEY_TO_FILE_KEY))
            raise ValueError(f"Unsupported --view '{view_name}'. Valid options: {valid}")

        mapped = self.metadata["files"].get(file_key)
        if mapped is None:
            available = ", ".join(sorted(self.available_views()))
            raise ValueError(
                f"View '{view_name}' not present in this export. Available views: {available}"
            )

        if not mapped.endswith(".ply"):
            raise ValueError(f"View '{view_name}' must map to a .ply file, got: {mapped}")

        ply_path = self.options.export_dir / mapped
        if not ply_path.exists():
            raise FileNotFoundError(f"Pointcloud file not found: {ply_path}")
        return ply_path

    def _load_selected_pointcloud(self) -> o3d.geometry.PointCloud:
        ply_path = self._resolve_pointcloud_path(self.options.view)
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if pcd.is_empty():
            raise ValueError(f"Loaded empty pointcloud: {ply_path}")
        return pcd

    @staticmethod
    def _axes_lineset(length: float = 1.0) -> o3d.geometry.LineSet:
        points = [[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]]
        lines = [[0, 1], [0, 2], [0, 3]]
        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def _run_open3d(self) -> None:
        if self.options.view in ("similarity_semfeat", "similarity_rgb"):
            raise ValueError(f"`{self.options.view}` view is supported only in `--display viser` mode.")
        pcd = self._load_selected_pointcloud()
        console.print(f"[green]Rendering view with Open3D:[/] {self.options.view}")
        geometries = [pcd]
        if self.options.show_axes:
            geometries.append(self._axes_lineset(length=1.0))
        o3d.visualization.draw_geometries(geometries)

    def _run_viser(self) -> None:
        try:
            from viser import ViserServer
        except ImportError as exc:
            raise ImportError("Viser mode requires `viser` to be installed in this environment.") from exc

        server = ViserServer(host=self.options.viser_host, port=self.options.viser_port)
        actual_port = getattr(getattr(server, "_server", None), "_port", self.options.viser_port)
        console.print(
            f"[green]Viser running:[/] http://{self.options.viser_host}:{actual_port} "
            "(use SSH port forwarding if remote)"
        )

        if self.options.show_axes:
            console.print("[yellow]`--show-axes` is ignored in viser mode (use viewer's world-axes toggle).[/]")

        available = self.available_views(include_similarity=True)
        selected = self.options.view if self.options.view in available else available[0]
        pointcloud_handle = {"h": None}

        with server.gui.add_folder("Pointcloud"):
            update_button = server.gui.add_button("Update")
            view_dropdown = server.gui.add_dropdown("View", options=available, initial_value=selected)
            semfeat_positive = server.gui.add_text("SemFeat Positives", initial_value="")
            semfeat_negative = server.gui.add_text("SemFeat Negatives", initial_value="")
            semfeat_temp = server.gui.add_text("SemFeat Softmax Temp", initial_value="0.1")
            semfeat_threshold = server.gui.add_text("SemFeat Threshold [0-1]", initial_value="0.0")
            rgb_target = server.gui.add_text("RGB Target [R,G,B]", initial_value="255,255,255")
            rgb_threshold = server.gui.add_text("RGB Threshold [0-1]", initial_value="0.5")

        def set_similarity_gui_visible() -> None:
            is_semfeat = view_dropdown.value == "similarity_semfeat"
            is_rgb = view_dropdown.value == "similarity_rgb"
            semfeat_positive.visible = is_semfeat
            semfeat_negative.visible = is_semfeat
            semfeat_temp.visible = is_semfeat
            semfeat_threshold.visible = is_semfeat
            rgb_target.visible = is_rgb
            rgb_threshold.visible = is_rgb

        def render_view(
            view_name: str,
            semfeat_positive_text: str,
            semfeat_negative_text: str,
            semfeat_softmax_temp: float,
            semfeat_thresh: float,
            rgb_target_text: str,
            rgb_thresh: float,
        ) -> None:
            if view_name == "similarity_semfeat":
                points, colors = self._render_similarity_semfeat_pointcloud(
                    semfeat_positive_text,
                    semfeat_negative_text,
                    semfeat_softmax_temp,
                    semfeat_thresh,
                )
            elif view_name == "similarity_rgb":
                points, colors = self._render_similarity_rgb_pointcloud(
                    target_rgb_text=rgb_target_text,
                    threshold_rgb=rgb_thresh,
                )
            else:
                pcd = o3d.io.read_point_cloud(str(self._resolve_pointcloud_path(view_name)))
                if pcd.is_empty():
                    raise ValueError(f"Loaded empty pointcloud for view '{view_name}'")
                points = np.asarray(pcd.points, dtype=np.float32)
                colors = np.asarray(pcd.colors, dtype=np.float32)
            if pointcloud_handle["h"] is not None:
                pointcloud_handle["h"].remove()
            pointcloud_handle["h"] = server.scene.add_point_cloud(
                "/pointcloud", points=points, colors=colors, point_size=0.002, point_shape="circle"
            )
            console.print(f"[cyan]Rendered view:[/] {view_name} ({points.shape[0]:,} points)")

        @view_dropdown.on_update
        def _(_event) -> None:
            set_similarity_gui_visible()

        @update_button.on_click
        def _(_event) -> None:
            try:
                render_view(
                    view_dropdown.value,
                    semfeat_positive.value,
                    semfeat_negative.value,
                    float(semfeat_temp.value),
                    float(semfeat_threshold.value),
                    rgb_target.value,
                    float(rgb_threshold.value),
                )
            except Exception as exc:
                console.print(f"[red]Failed to render {view_dropdown.value}:[/] {exc}")

        set_similarity_gui_visible()
        if selected in ("similarity_semfeat", "similarity_rgb"):
            console.print("[yellow]Similarity view selected. Fill fields and click Update.[/]")
        else:
            render_view(
                selected,
                semfeat_positive.value,
                semfeat_negative.value,
                float(semfeat_temp.value),
                float(semfeat_threshold.value),
                rgb_target.value,
                float(rgb_threshold.value),
            )

        if self.options.viser_share_url:
            try:
                share_url = server.request_share_url(verbose=True)
                if share_url is None:
                    console.print("[yellow]Viser could not create share URL (network/firewall may block it).[/]")
            except Exception as exc:
                console.print(f"[yellow]Viser share URL request failed:[/] {type(exc).__name__}: {exc}")

        try:
            import time

            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            console.print("[yellow]Stopping Viser.[/]")

    def _has_similarity_semfeat_data(self) -> bool:
        files = self.metadata.get("files", {})
        points_file = files.get("points")
        features_file = files.get("features")
        if points_file is None or features_file is None:
            return False
        return (self.options.export_dir / points_file).exists() and (self.options.export_dir / features_file).exists()

    def _has_similarity_rgb_data(self) -> bool:
        files = self.metadata.get("files", {})
        points_file = files.get("points")
        rgbs_file = files.get("rgbs")
        if points_file is None or rgbs_file is None:
            return False
        return (self.options.export_dir / points_file).exists() and (self.options.export_dir / rgbs_file).exists()

    def _get_points_array(self) -> np.ndarray:
        if self._points_cache is None:
            points_file = self.metadata["files"]["points"]
            self._points_cache = np.load(self.options.export_dir / points_file, mmap_mode="r")
        return self._points_cache

    def _get_features_array(self) -> np.ndarray:
        if self._features_cache is None:
            features_file = self.metadata["files"]["features"]
            self._features_cache = np.load(self.options.export_dir / features_file, mmap_mode="r")
        return self._features_cache

    def _get_rgbs_array(self) -> np.ndarray:
        if self._rgbs_cache is None:
            rgbs_file = self.metadata["files"]["rgbs"]
            self._rgbs_cache = np.load(self.options.export_dir / rgbs_file, mmap_mode="r")
        return self._rgbs_cache

    def _init_clip(self) -> None:
        if self._clip_model is not None and self._clip_tokenizer is not None:
            return
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("Similarity view requires `open_clip_torch` to be installed.") from exc

        from f3rm.features.clip_extract import CLIPArgs

        model, _, _ = open_clip.create_model_and_transforms(
            CLIPArgs.model_name,
            pretrained=CLIPArgs.model_pretrained,
            device=self._clip_device,
        )
        model.eval()
        self._clip_model = model
        self._clip_tokenizer = open_clip.get_tokenizer(CLIPArgs.model_name)

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        self._init_clip()
        tokens = self._clip_tokenizer(texts).to(self._clip_device)
        embeds = self._clip_model.encode_text(tokens).float()
        return embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _scalar_to_heatmap(values: np.ndarray) -> np.ndarray:
        """Map scalar values in [0,1] to a high-contrast heatmap RGB."""
        x = np.clip(values.reshape(-1), 0.0, 1.0).astype(np.float32)
        # Blue -> Cyan -> Green -> Yellow -> Red
        knots = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
        cols = np.array(
            [
                [0.05, 0.05, 0.55],
                [0.00, 0.65, 1.00],
                [0.00, 0.95, 0.35],
                [1.00, 0.95, 0.00],
                [1.00, 0.10, 0.00],
            ],
            dtype=np.float32,
        )
        rgb = np.empty((x.shape[0], 3), dtype=np.float32)
        rgb[:, 0] = np.interp(x, knots, cols[:, 0])
        rgb[:, 1] = np.interp(x, knots, cols[:, 1])
        rgb[:, 2] = np.interp(x, knots, cols[:, 2])
        return rgb

    def _save_similarity_distribution_plot(self, values: np.ndarray, out_name: str, title: str) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        x = values.reshape(-1).astype(np.float32)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(x, bins=100, color="#2E86DE", alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Similarity")
        ax.set_ylabel("Count")
        fig.tight_layout()
        out_path = Path(".") / out_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        console.print(f"[cyan]Saved distribution:[/] {out_path}")

    @torch.no_grad()
    def _render_similarity_semfeat_pointcloud(
        self,
        positive_text: str,
        negative_text: str,
        softmax_temp: float,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        positives = parse_comma_separated_labels(positive_text)
        if not positives:
            raise ValueError("Similarity view requires at least one positive label.")
        negatives = parse_comma_separated_labels(negative_text)
        softmax_temp = max(float(softmax_temp), 1e-6)
        threshold = np.clip(float(threshold), 0.0, 1.0)

        points = np.asarray(self._get_points_array(), dtype=np.float32)
        features_np = self._get_features_array()
        if features_np.shape[0] != points.shape[0]:
            raise ValueError(
                f"Feature/point count mismatch: features={features_np.shape[0]}, points={points.shape[0]}"
            )

        pos_embed = self._encode_texts(positives).mean(dim=0, keepdim=True)
        pos_embed = pos_embed / pos_embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        neg_embed = self._encode_texts(negatives) if negatives else None

        scalar_chunks: List[np.ndarray] = []
        chunk_size = 200_000
        for start in range(0, features_np.shape[0], chunk_size):
            end = min(start + chunk_size, features_np.shape[0])
            feat_chunk = torch.from_numpy(np.asarray(features_np[start:end], dtype=np.float32)).to(self._clip_device)
            sims_or_probs = compute_similarity_scores(
                clip_features=feat_chunk,
                pos_embed=pos_embed,
                neg_embed=neg_embed,
                softmax_temp=softmax_temp,
            )
            if neg_embed is None:
                probs = torch.clamp((sims_or_probs + 1.0) * 0.5, 0.0, 1.0)
            else:
                probs = sims_or_probs

            scalar = probs.detach().cpu().numpy().astype(np.float32).reshape(-1)
            scalar_chunks.append(scalar)

        scalar_all = np.concatenate(scalar_chunks, axis=0)
        colors = self._scalar_to_heatmap(scalar_all)
        colors[scalar_all < threshold] = 0.0
        return points, colors

    @staticmethod
    def _parse_rgb_target(rgb_text: str) -> np.ndarray:
        parts = [p.strip() for p in rgb_text.split(",")]
        if len(parts) != 3:
            raise ValueError("RGB target must be in the format R,G,B with 3 comma-separated values.")
        rgb = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
        return np.clip(rgb, 0.0, 255.0)

    @torch.no_grad()
    def _render_similarity_rgb_pointcloud(self, target_rgb_text: str, threshold_rgb: float) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(self._get_points_array(), dtype=np.float32)
        rgbs = np.asarray(self._get_rgbs_array(), dtype=np.float32)
        if rgbs.shape[0] != points.shape[0]:
            raise ValueError(f"RGB/point count mismatch: rgbs={rgbs.shape[0]}, points={points.shape[0]}")

        target_rgb = self._parse_rgb_target(target_rgb_text)
        threshold_rgb = np.clip(float(threshold_rgb), 0.0, 1.0)

        rgbs_255 = np.clip(rgbs * 255.0, 0.0, 255.0)
        diff = np.abs(rgbs_255 - target_rgb.reshape(1, 3))

        dist = np.linalg.norm(diff, axis=1)
        similarity = 1.0 - (dist / (np.sqrt(3.0) * 255.0))
        similarity = np.clip(similarity.astype(np.float32), 0.0, 1.0)
        self._save_similarity_distribution_plot(
            similarity,
            out_name="similarity_rgb_distribution.png",
            title=f"RGB Similarity Distribution | target={target_rgb_text} | threshold={threshold_rgb:.3f}",
        )

        colors = self._scalar_to_heatmap(similarity)
        colors[similarity < threshold_rgb] = 0.0
        return points, colors

    def available_views(self, include_similarity: bool = False) -> List[str]:
        available = []
        for view_name in self.VIEW_KEY_TO_FILE_KEY:
            try:
                self._resolve_pointcloud_path(view_name)
                available.append(view_name)
            except (ValueError, FileNotFoundError):
                pass
        if include_similarity and self._has_similarity_semfeat_data():
            available.append("similarity_semfeat")
        if include_similarity and self._has_similarity_rgb_data():
            available.append("similarity_rgb")
        return available


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pointcloud outputs exported by f3rm/pcviz/export_pointcloud.py")
    parser.add_argument("--export-dir", type=Path, required=True, help="Directory containing metadata.json and PLY files")
    parser.add_argument(
        "--view",
        type=str,
        default="rgb",
        choices=["rgb", "feature_pca", "foreground_prob", "pred_normals", "normals", "similarity_semfeat", "similarity_rgb"],
        help="Which exported view to render",
    )
    parser.add_argument(
        "--display",
        type=str,
        default="open3d",
        choices=["open3d", "viser"],
        help="Display backend",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Show unit-length XYZ axes at origin (X=red, Y=green, Z=blue)",
    )
    parser.add_argument("--viser-host", type=str, default="0.0.0.0", help="Host for viser display server.")
    parser.add_argument("--viser-port", type=int, default=7007, help="Port for viser display server.")
    parser.add_argument("--viser-share-url", action="store_true", help="Request a public share URL from viser.")
    args = parser.parse_args()

    options = ViewOptions(
        export_dir=args.export_dir,
        view=args.view,
        display=args.display,
        show_axes=args.show_axes,
        viser_host=args.viser_host,
        viser_port=args.viser_port,
        viser_share_url=args.viser_share_url,
    )
    viewer = PointcloudViewer(options)
    available = viewer.available_views(include_similarity=(options.display == "viser"))
    if options.view not in available:
        available_str = ", ".join(sorted(available)) if available else "none"
        raise ValueError(f"Requested --view '{options.view}' is unavailable. Available: {available_str}")

    console.print(f"[blue]Available views:[/] {', '.join(sorted(available))}")
    viewer.run()
