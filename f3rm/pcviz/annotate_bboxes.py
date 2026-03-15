#!/usr/bin/env python3
"""Annotate 3D bounding boxes on exported pointclouds using viser."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from rich.console import Console

console = Console()


@dataclass
class BoxAnnotation:
    id: str
    label: str
    center: List[float]
    size: List[float]
    rotation_deg: List[float]  # [rx, ry, rz]
    color_rgb: List[int]


def _parse_color_rgb(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"color must be R,G,B, got: {text}")
    rgb = [int(round(float(parts[0]))), int(round(float(parts[1]))), int(round(float(parts[2])))]
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError(f"color must be in [0,255], got: {text}")
    return rgb


def _format_color(color_rgb: List[int]) -> str:
    return f"{color_rgb[0]},{color_rgb[1]},{color_rgb[2]}"


def _rotation_matrix_xyz(rotation_deg: List[float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(rotation_deg, dtype=np.float32))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def _box_corners(center: np.ndarray, size: np.ndarray, rotation_deg: List[float]) -> np.ndarray:
    half = size * 0.5
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float32,
    )
    rot = _rotation_matrix_xyz(rotation_deg)
    return local @ rot.T + center[None, :]


def _wireframe_points(corners: np.ndarray, points_per_edge: int = 24) -> np.ndarray:
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    t = np.linspace(0.0, 1.0, points_per_edge, dtype=np.float32)
    out = []
    for a, b in edges:
        p0, p1 = corners[a], corners[b]
        out.append(p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None])
    return np.concatenate(out, axis=0)


def _points_inside_box(points: np.ndarray, box: BoxAnnotation) -> np.ndarray:
    center = np.asarray(box.center, dtype=np.float32)
    size = np.maximum(np.asarray(box.size, dtype=np.float32), 1e-6)
    rot = _rotation_matrix_xyz(box.rotation_deg)
    local = (points - center[None, :]) @ rot
    return (np.abs(local) <= (size[None, :] * 0.5)).all(axis=1)


def _local_axes_points(
    center: np.ndarray, rotation_deg: List[float], size: np.ndarray, points_per_axis: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    rot = _rotation_matrix_xyz(rotation_deg)
    axis_len = max(0.02, float(np.min(np.maximum(size, 1e-6)) * 0.35))
    t = np.linspace(0.0, axis_len, points_per_axis, dtype=np.float32)
    dirs = [rot[:, 0], rot[:, 1], rot[:, 2]]
    cols = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]
    pts, cts = [], []
    for d, c in zip(dirs, cols):
        p = center[None, :] + t[:, None] * d[None, :]
        pts.append(p)
        cts.append(np.repeat(c[None, :], p.shape[0], axis=0))
    return np.concatenate(pts, axis=0), np.concatenate(cts, axis=0)


class BBoxAnnotator:
    def __init__(self, export_dir: Path, host: str, port: int, share_url: bool, max_points: int):
        self.export_dir = export_dir
        self.host = host
        self.port = port
        self.share_url = share_url
        self.max_points = max_points

        self.metadata = self._load_metadata()
        self.points, self.colors = self._load_points_rgb()
        self.scene_min = self.points.min(axis=0)
        self.scene_max = self.points.max(axis=0)
        self.scene_extent = np.maximum(self.scene_max - self.scene_min, 1e-3)

        self.boxes: Dict[str, BoxAnnotation] = self._load_boxes()
        self._next_id = self._compute_next_id()
        self._handles: Dict[str, object] = {}
        self._active_axes_handle = None

    def _load_metadata(self) -> Dict:
        path = self.export_dir / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metadata: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_points_rgb(self) -> tuple[np.ndarray, np.ndarray]:
        files = self.metadata.get("files", {})
        points_file = files.get("points")
        rgbs_file = files.get("rgbs")
        if points_file is None or rgbs_file is None:
            raise ValueError("metadata.json missing files.points or files.rgbs")
        points = np.asarray(np.load(self.export_dir / points_file), dtype=np.float32)
        colors = np.asarray(np.load(self.export_dir / rgbs_file), dtype=np.float32)
        if points.shape[0] != colors.shape[0]:
            raise ValueError(f"points/rgbs size mismatch: {points.shape[0]} vs {colors.shape[0]}")
        if points.shape[0] > self.max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(points.shape[0], size=self.max_points, replace=False)
            points = points[idx]
            colors = colors[idx]
        return points, np.clip(colors, 0.0, 1.0)

    def _boxes_path(self) -> Path:
        return self.export_dir / "bboxes.json"

    def _load_boxes(self) -> Dict[str, BoxAnnotation]:
        path = self._boxes_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        boxes = {}
        for item in payload.get("boxes", []):
            if "rotation_deg" not in item:
                raise ValueError("Invalid bbox schema: each box must contain `rotation_deg`.")
            box = BoxAnnotation(**item)
            boxes[box.id] = box
        return boxes

    def _save_boxes(self) -> None:
        payload = {
            "version": 2,
            "coordinate_frame": "export_points",
            "boxes": [asdict(self.boxes[k]) for k in sorted(self.boxes.keys())],
        }
        out_path = self._boxes_path()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        console.print(f"[green]Saved annotations:[/] {out_path}")

    def _compute_next_id(self) -> int:
        best = 0
        for key in self.boxes.keys():
            if key.startswith("box_"):
                try:
                    best = max(best, int(key.split("_", 1)[1]))
                except ValueError:
                    pass
        return best + 1

    def _new_default_box(self) -> BoxAnnotation:
        center = self.points.mean(axis=0)
        size = np.maximum(self.scene_extent * 0.15, 0.05)
        box_id = f"box_{self._next_id:03d}"
        self._next_id += 1
        return BoxAnnotation(
            id=box_id,
            label=box_id,
            center=[float(center[0]), float(center[1]), float(center[2])],
            size=[float(size[0]), float(size[1]), float(size[2])],
            rotation_deg=[0.0, 0.0, 0.0],
            color_rgb=[255, 64, 64],
        )

    def run(self) -> None:
        try:
            from viser import ViserServer
        except ImportError as exc:
            raise ImportError("This script requires `viser` in your environment.") from exc

        server = ViserServer(host=self.host, port=self.port)
        actual_port = getattr(getattr(server, "_server", None), "_port", self.port)
        console.print(f"[green]Viser running:[/] http://{self.host}:{actual_port}")
        server.scene.add_point_cloud(
            "/rgb_pointcloud",
            points=self.points,
            colors=self.colors,
            point_size=0.002,
            point_shape="circle",
        )

        if self.share_url:
            try:
                server.request_share_url(verbose=True)
            except Exception as exc:
                console.print(f"[yellow]Share URL request failed:[/] {type(exc).__name__}: {exc}")

        center_pad = 0.25 * self.scene_extent
        center_min = self.scene_min - center_pad
        center_max = self.scene_max + center_pad
        center_step = np.maximum(self.scene_extent / 400.0, 1e-4)
        size_max = np.maximum(self.scene_extent * 1.5, 0.05)
        size_step = np.maximum(size_max / 400.0, 1e-4)

        options = sorted(self.boxes.keys()) if self.boxes else ["<none>"]
        selected = options[0]
        sync = {"busy": False}

        with server.gui.add_folder("BBoxes"):
            add_btn = server.gui.add_button("Add Box")
            dup_btn = server.gui.add_button("Duplicate Box")
            del_btn = server.gui.add_button("Delete Box")
            save_btn = server.gui.add_button("Save")
            select = server.gui.add_dropdown("Active Box", options=options, initial_value=selected)
            label_text = server.gui.add_text("Label", initial_value="")
            color_text = server.gui.add_text("Color R,G,B", initial_value="255,64,64")

            cx = server.gui.add_slider("Center X", min=float(center_min[0]), max=float(center_max[0]), step=float(center_step[0]), initial_value=0.0)
            cy = server.gui.add_slider("Center Y", min=float(center_min[1]), max=float(center_max[1]), step=float(center_step[1]), initial_value=0.0)
            cz = server.gui.add_slider("Center Z", min=float(center_min[2]), max=float(center_max[2]), step=float(center_step[2]), initial_value=0.0)

            sx = server.gui.add_slider("Size X", min=1e-4, max=float(size_max[0]), step=float(size_step[0]), initial_value=0.2)
            sy = server.gui.add_slider("Size Y", min=1e-4, max=float(size_max[1]), step=float(size_step[1]), initial_value=0.2)
            sz = server.gui.add_slider("Size Z", min=1e-4, max=float(size_max[2]), step=float(size_step[2]), initial_value=0.2)

            rx = server.gui.add_slider("Rot X (deg)", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
            ry = server.gui.add_slider("Rot Y (deg)", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
            rz = server.gui.add_slider("Rot Z (deg)", min=-180.0, max=180.0, step=1.0, initial_value=0.0)

            apply_meta_btn = server.gui.add_button("Apply Label/Color")
            stats_text = server.gui.add_markdown("Points in box: n/a")

        def refresh_dropdown(active: str | None = None) -> None:
            ids = sorted(self.boxes.keys())
            if not ids:
                select.options = ["<none>"]
                select.value = "<none>"
                return
            select.options = ids
            select.value = active if active in ids else ids[0]

        def render_boxes() -> None:
            for handle in self._handles.values():
                handle.remove()
            self._handles.clear()
            for box_id, box in self.boxes.items():
                corners = _box_corners(
                    center=np.asarray(box.center, dtype=np.float32),
                    size=np.asarray(np.maximum(np.asarray(box.size, dtype=np.float32), 1e-6), dtype=np.float32),
                    rotation_deg=box.rotation_deg,
                )
                edge_pts = _wireframe_points(corners, points_per_edge=24)
                color = np.asarray(box.color_rgb, dtype=np.float32)[None, :] / 255.0
                edge_colors = np.repeat(color, edge_pts.shape[0], axis=0)
                self._handles[box_id] = server.scene.add_point_cloud(
                    f"/bboxes/{box_id}",
                    points=edge_pts,
                    colors=edge_colors,
                    point_size=0.01,
                    point_shape="circle",
                )

        def render_active_box_axes() -> None:
            if self._active_axes_handle is not None:
                self._active_axes_handle.remove()
                self._active_axes_handle = None
            box_id = select.value
            if box_id not in self.boxes:
                return
            box = self.boxes[box_id]
            center = np.asarray(box.center, dtype=np.float32)
            size = np.asarray(box.size, dtype=np.float32)
            pts, cols = _local_axes_points(center=center, rotation_deg=box.rotation_deg, size=size, points_per_axis=32)
            self._active_axes_handle = server.scene.add_point_cloud(
                "/bboxes/active_axes",
                points=pts,
                colors=cols,
                point_size=0.012,
                point_shape="circle",
            )

        def update_stats() -> None:
            box_id = select.value
            if box_id not in self.boxes:
                stats_text.content = "Points in box: n/a"
                return
            mask = _points_inside_box(self.points, self.boxes[box_id])
            stats_text.content = f"Points in box: **{int(mask.sum()):,}** / {self.points.shape[0]:,}"

        def sync_controls_from_selected() -> None:
            sync["busy"] = True
            box_id = select.value
            if box_id in self.boxes:
                box = self.boxes[box_id]
                label_text.value = box.label
                color_text.value = _format_color(box.color_rgb)
                cx.value, cy.value, cz.value = float(box.center[0]), float(box.center[1]), float(box.center[2])
                sx.value, sy.value, sz.value = float(box.size[0]), float(box.size[1]), float(box.size[2])
                rx.value, ry.value, rz.value = (
                    float(box.rotation_deg[0]),
                    float(box.rotation_deg[1]),
                    float(box.rotation_deg[2]),
                )
            else:
                label_text.value = ""
                color_text.value = "255,64,64"
            sync["busy"] = False
            render_active_box_axes()
            update_stats()

        def update_selected_from_sliders() -> None:
            if sync["busy"]:
                return
            box_id = select.value
            if box_id not in self.boxes:
                return
            box = self.boxes[box_id]
            box.center = [float(cx.value), float(cy.value), float(cz.value)]
            box.size = [max(float(sx.value), 1e-4), max(float(sy.value), 1e-4), max(float(sz.value), 1e-4)]
            box.rotation_deg = [float(rx.value), float(ry.value), float(rz.value)]
            render_boxes()
            render_active_box_axes()
            update_stats()

        @select.on_update
        def _(_event) -> None:
            sync_controls_from_selected()

        @cx.on_update
        @cy.on_update
        @cz.on_update
        @sx.on_update
        @sy.on_update
        @sz.on_update
        @rx.on_update
        @ry.on_update
        @rz.on_update
        def _(_event) -> None:
            update_selected_from_sliders()

        @add_btn.on_click
        def _(_event) -> None:
            box = self._new_default_box()
            self.boxes[box.id] = box
            refresh_dropdown(active=box.id)
            render_boxes()
            sync_controls_from_selected()

        @dup_btn.on_click
        def _(_event) -> None:
            src_id = select.value
            if src_id not in self.boxes:
                return
            src = self.boxes[src_id]
            new_box = BoxAnnotation(
                id=f"box_{self._next_id:03d}",
                label=f"{src.label}_copy",
                center=[src.center[0] + 0.03, src.center[1], src.center[2]],
                size=list(src.size),
                rotation_deg=list(src.rotation_deg),
                color_rgb=list(src.color_rgb),
            )
            self._next_id += 1
            self.boxes[new_box.id] = new_box
            refresh_dropdown(active=new_box.id)
            render_boxes()
            sync_controls_from_selected()

        @del_btn.on_click
        def _(_event) -> None:
            box_id = select.value
            if box_id in self.boxes:
                self.boxes.pop(box_id)
                render_boxes()
                refresh_dropdown(active=None)
                sync_controls_from_selected()

        @apply_meta_btn.on_click
        def _(_event) -> None:
            box_id = select.value
            if box_id not in self.boxes:
                return
            try:
                box = self.boxes[box_id]
                box.label = label_text.value.strip() or box.id
                box.color_rgb = _parse_color_rgb(color_text.value)
                render_boxes()
                render_active_box_axes()
                update_stats()
            except Exception as exc:
                console.print(f"[red]Failed to apply label/color:[/] {exc}")

        @save_btn.on_click
        def _(_event) -> None:
            self._save_boxes()

        render_boxes()
        sync_controls_from_selected()
        console.print("[cyan]Use sliders to move/resize/rotate boxes. Ctrl+C to quit.[/]")

        try:
            import time

            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            console.print("[yellow]Stopping annotator.[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate 3D bounding boxes on exported pointclouds.")
    parser.add_argument("--export-dir", type=Path, required=True, help="Export dir containing metadata.json, points.npy, rgbs.npy")
    parser.add_argument("--viser-host", type=str, default="0.0.0.0", help="Viser host")
    parser.add_argument("--viser-port", type=int, default=7011, help="Viser port")
    parser.add_argument("--viser-share-url", action="store_true", help="Request a public share URL from viser")
    parser.add_argument("--max-points", type=int, default=1_500_000, help="Max points to render in viewer")
    args = parser.parse_args()

    annotator = BBoxAnnotator(
        export_dir=args.export_dir,
        host=args.viser_host,
        port=args.viser_port,
        share_url=args.viser_share_url,
        max_points=args.max_points,
    )
    annotator.run()
