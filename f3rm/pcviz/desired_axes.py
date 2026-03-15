#!/usr/bin/env python3
"""Apply an in-place exported-pointcloud frame transform to a desired axes frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
from rich.console import Console

console = Console()


def rotation_matrix_xyz(rotation_deg: List[float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(rotation_deg, dtype=np.float32))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def matrix_to_euler_xyz_deg(rot: np.ndarray) -> List[float]:
    sy = -float(rot[2, 0])
    sy = float(np.clip(sy, -1.0, 1.0))
    cy = float(np.sqrt(max(0.0, 1.0 - sy * sy)))
    if cy > 1e-6:
        rx = np.arctan2(rot[2, 1], rot[2, 2])
        ry = np.arctan2(sy, cy)
        rz = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        rx = 0.0
        ry = np.arctan2(sy, cy)
        rz = np.arctan2(-rot[0, 1], rot[1, 1])
    return [float(np.rad2deg(rx)), float(np.rad2deg(ry)), float(np.rad2deg(rz))]


def to_desired_points(points: np.ndarray, translation: np.ndarray, rot_desired_to_original: np.ndarray) -> np.ndarray:
    # Row-vector form of p_d = R^T (p_o - t): (p_o - t) @ R
    return (points - translation[None, :]) @ rot_desired_to_original


def to_desired_vectors(vectors: np.ndarray, rot_desired_to_original: np.ndarray) -> np.ndarray:
    # Row-vector form of v_d = R^T v_o
    return vectors @ rot_desired_to_original


def transform_normals_rgb(normals_rgb: np.ndarray, rot_desired_to_original: np.ndarray) -> np.ndarray:
    raw = np.clip(normals_rgb, 0.0, 1.0) * 2.0 - 1.0
    raw_d = to_desired_vectors(raw, rot_desired_to_original)
    rgb_d = np.clip((raw_d + 1.0) * 0.5, 0.0, 1.0)
    return rgb_d.astype(np.float32)


def load_metadata(export_dir: Path) -> Dict:
    metadata_path = export_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(export_dir: Path, metadata: Dict) -> None:
    metadata_path = export_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def transform_points_array(export_dir: Path, metadata: Dict, t: np.ndarray, r_do: np.ndarray) -> np.ndarray:
    points_rel = metadata["files"].get("points")
    if points_rel is None:
        raise ValueError("metadata.files.points not found")
    points_path = export_dir / points_rel
    points = np.asarray(np.load(points_path), dtype=np.float32)
    points_d = to_desired_points(points, t, r_do).astype(np.float32)
    np.save(points_path, points_d)
    metadata["bbox_min"] = points_d.min(axis=0).tolist()
    metadata["bbox_max"] = points_d.max(axis=0).tolist()
    return points_d


def transform_normals_arrays(export_dir: Path, metadata: Dict, r_do: np.ndarray) -> List[str]:
    transformed = []
    files = metadata.get("files", {})

    raw_keys = ["pred_normals_raw", "normals_raw"]
    for key in raw_keys:
        rel = files.get(key)
        if rel is None:
            continue
        path = export_dir / rel
        arr = np.asarray(np.load(path), dtype=np.float32)
        arr_d = to_desired_vectors(arr, r_do).astype(np.float32)
        np.save(path, arr_d)
        transformed.append(rel)

    rgb_keys = ["pred_normals_rgb_raw", "normals_rgb_raw"]
    for key in rgb_keys:
        rel = files.get(key)
        if rel is None:
            continue
        path = export_dir / rel
        arr = np.asarray(np.load(path), dtype=np.float32)
        arr_d = transform_normals_rgb(arr, r_do)
        np.save(path, arr_d)
        transformed.append(rel)

    return transformed


def transform_plys(export_dir: Path, metadata: Dict, t: np.ndarray, r_do: np.ndarray) -> List[str]:
    transformed = []
    files = metadata.get("files", {})
    normal_color_keys = {"pred_normals_rgb", "normals_rgb"}

    for key, rel in files.items():
        if not isinstance(rel, str) or not rel.endswith(".ply"):
            continue
        ply_path = export_dir / rel
        if not ply_path.exists():
            continue
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if pcd.is_empty():
            continue

        pts = np.asarray(pcd.points, dtype=np.float32)
        pts_d = to_desired_points(pts, t, r_do)
        pcd.points = o3d.utility.Vector3dVector(pts_d.astype(np.float64))

        if pcd.has_normals():
            normals = np.asarray(pcd.normals, dtype=np.float32)
            normals_d = to_desired_vectors(normals, r_do)
            pcd.normals = o3d.utility.Vector3dVector(normals_d.astype(np.float64))

        if key in normal_color_keys and pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float32)
            colors_d = transform_normals_rgb(colors, r_do)
            pcd.colors = o3d.utility.Vector3dVector(colors_d.astype(np.float64))

        ok = o3d.io.write_point_cloud(str(ply_path), pcd)
        if ok:
            transformed.append(rel)
        else:
            console.print(f"[yellow]Failed writing PLY:[/] {ply_path}")

    return transformed


def transform_bboxes(export_dir: Path, t: np.ndarray, r_do: np.ndarray) -> bool:
    bboxes_path = export_dir / "bboxes.json"
    if not bboxes_path.exists():
        return False

    with bboxes_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    boxes = payload.get("boxes", [])
    changed = False
    for box in boxes:
        center = np.asarray(box["center"], dtype=np.float32)
        center_d = to_desired_points(center[None, :], t, r_do)[0]
        box["center"] = center_d.astype(np.float32).tolist()

        rot_o = rotation_matrix_xyz(box["rotation_deg"])
        rot_d = r_do.T @ rot_o
        box["rotation_deg"] = matrix_to_euler_xyz_deg(rot_d)
        changed = True

    if "desired_axes_transform" in payload:
        payload["applied_desired_axes_transform"] = payload["desired_axes_transform"]
        payload.pop("desired_axes_transform", None)

    payload["coordinate_frame"] = "desired_axes"
    payload["boxes_frame"] = "desired_axes"
    payload["inplace_desired_axes_applied"] = True

    if changed:
        with bboxes_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return changed


def save_transform_record(export_dir: Path, t: np.ndarray, r_do: np.ndarray, output_path: Path | None, changed_files: List[str]) -> Path:
    out = output_path if output_path is not None else (export_dir / "desired_axes_transform.json")
    payload = {
        "version": 1,
        "description": "Transform applied in-place to move export into desired frame.",
        "translation_desired_origin_in_original": t.astype(np.float32).tolist(),
        "rotation_deg_desired_to_original_xyz": matrix_to_euler_xyz_deg(r_do),
        "rotation_matrix_desired_to_original": r_do.astype(np.float32).tolist(),
        "rotation_matrix_original_to_desired": r_do.T.astype(np.float32).tolist(),
        "changed_files": sorted(set(changed_files)),
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def make_axes_points(origin: np.ndarray, rot_do: np.ndarray, axis_length: float, points_per_axis: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    dirs = [rot_do[:, 0], rot_do[:, 1], rot_do[:, 2]]
    cols = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]
    t = np.linspace(0.0, axis_length, points_per_axis, dtype=np.float32)
    pts, cts = [], []
    for d, c in zip(dirs, cols):
        p = origin[None, :] + t[:, None] * d[None, :]
        pts.append(p)
        cts.append(np.repeat(c[None, :], p.shape[0], axis=0))
    return np.concatenate(pts, axis=0), np.concatenate(cts, axis=0)


def apply_inplace_transform(export_dir: Path, t: np.ndarray, r_do: np.ndarray, transform_record_path: Path | None) -> Path:
    metadata = load_metadata(export_dir)
    changed_files: List[str] = []

    console.print("[cyan]Transforming points.npy + metadata bbox...[/]")
    _ = transform_points_array(export_dir, metadata, t, r_do)
    changed_files.append(metadata["files"]["points"])
    changed_files.append("metadata.json")

    console.print("[cyan]Transforming normals arrays (if present)...[/]")
    changed_files.extend(transform_normals_arrays(export_dir, metadata, r_do))

    console.print("[cyan]Transforming PLY pointclouds (if present)...[/]")
    changed_files.extend(transform_plys(export_dir, metadata, t, r_do))

    save_metadata(export_dir, metadata)

    console.print("[cyan]Transforming bboxes.json (if present)...[/]")
    if transform_bboxes(export_dir, t, r_do):
        changed_files.append("bboxes.json")

    return save_transform_record(export_dir, t, r_do, transform_record_path, changed_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive desired-axes tool with in-place apply.")
    parser.add_argument("--export-dir", type=Path, required=True, help="Export directory containing metadata.json and artifacts.")
    parser.add_argument("--viser-host", type=str, default="0.0.0.0", help="Viser host")
    parser.add_argument("--viser-port", type=int, default=7012, help="Viser port")
    parser.add_argument("--viser-share-url", action="store_true", help="Request a public share URL from viser")
    parser.add_argument("--max-points", type=int, default=1_500_000, help="Max points to display for responsiveness")
    parser.add_argument("--init-translation", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("TX", "TY", "TZ"), help="Initial desired-origin translation")
    parser.add_argument("--init-rotation-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("RX", "RY", "RZ"), help="Initial desired-frame XYZ Euler rotation (deg)")
    parser.add_argument("--transform-record-path", type=Path, default=None, help="Optional output path for transform record JSON.")
    args = parser.parse_args()

    try:
        from viser import ViserServer
    except ImportError as exc:
        raise ImportError("This script requires `viser` in your environment.") from exc

    export_dir = args.export_dir
    metadata = load_metadata(export_dir)
    points_rel = metadata["files"].get("points")
    rgbs_rel = metadata["files"].get("rgbs")
    if points_rel is None or rgbs_rel is None:
        raise ValueError("metadata files must contain `points` and `rgbs`")

    points = np.asarray(np.load(export_dir / points_rel), dtype=np.float32)
    rgbs = np.clip(np.asarray(np.load(export_dir / rgbs_rel), dtype=np.float32), 0.0, 1.0)
    if points.shape[0] != rgbs.shape[0]:
        raise ValueError(f"points/rgbs size mismatch: {points.shape[0]} vs {rgbs.shape[0]}")

    if points.shape[0] > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(points.shape[0], size=args.max_points, replace=False)
        points = points[idx]
        rgbs = rgbs[idx]

    scene_min = points.min(axis=0)
    scene_max = points.max(axis=0)
    scene_extent = np.maximum(scene_max - scene_min, 1e-3)
    center_limit = float(np.max(np.abs(np.concatenate([scene_min, scene_max]))) + np.max(scene_extent))
    center_limit = max(center_limit, 1.0) * 2.0
    center_step = float(max(np.max(scene_extent) / 400.0, 1e-4))
    axis_len = float(max(np.max(scene_extent) * 0.2, 0.05))

    server = ViserServer(host=args.viser_host, port=args.viser_port)
    actual_port = getattr(getattr(server, "_server", None), "_port", args.viser_port)
    console.print(f"[green]Viser running:[/] http://{args.viser_host}:{actual_port}")
    server.scene.add_point_cloud("/rgb_pointcloud", points=points, colors=rgbs, point_size=0.002, point_shape="circle")

    if args.viser_share_url:
        try:
            server.request_share_url(verbose=True)
        except Exception as exc:
            console.print(f"[yellow]Share URL request failed:[/] {type(exc).__name__}: {exc}")

    with server.gui.add_folder("Desired Axes (Body-Fixed Deltas)"):
        dtx = server.gui.add_slider("dTx", min=-center_limit, max=center_limit, step=center_step, initial_value=0.0)
        dty = server.gui.add_slider("dTy", min=-center_limit, max=center_limit, step=center_step, initial_value=0.0)
        dtz = server.gui.add_slider("dTz", min=-center_limit, max=center_limit, step=center_step, initial_value=0.0)
        drx = server.gui.add_slider("dRx (deg)", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
        dry = server.gui.add_slider("dRy (deg)", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
        drz = server.gui.add_slider("dRz (deg)", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
        zero_delta_btn = server.gui.add_button("Zero Delta Sliders")
        reset_pose_btn = server.gui.add_button("Reset Pose")
        apply_btn = server.gui.add_button("Apply In-Place")
        status = server.gui.add_markdown("Status: preview only")

    state = {
        "axes_handle": None,
        "applied": False,
        "sync": False,
        "pose_t": np.array(
            [float(args.init_translation[0]), float(args.init_translation[1]), float(args.init_translation[2])],
            dtype=np.float32,
        ),
        "pose_r": rotation_matrix_xyz(
            [float(args.init_rotation_deg[0]), float(args.init_rotation_deg[1]), float(args.init_rotation_deg[2])]
        ),
        "last_delta_t": np.zeros(3, dtype=np.float32),
        "last_delta_r": np.zeros(3, dtype=np.float32),
    }

    def render_desired_axes() -> None:
        if state["axes_handle"] is not None:
            state["axes_handle"].remove()
        t = state["pose_t"]
        r = state["pose_r"]
        pts, cols = make_axes_points(origin=t, rot_do=r, axis_length=axis_len)
        euler = matrix_to_euler_xyz_deg(r)
        state["axes_handle"] = server.scene.add_point_cloud("/axes/desired", points=pts, colors=cols, point_size=0.012, point_shape="circle")
        status.content = (
            "Status: preview only\n"
            f"current t=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})\n"
            f"current r=({euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}) deg"
        )

    def reset_delta_sliders() -> None:
        state["sync"] = True
        dtx.value = 0.0
        dty.value = 0.0
        dtz.value = 0.0
        drx.value = 0.0
        dry.value = 0.0
        drz.value = 0.0
        state["last_delta_t"] = np.zeros(3, dtype=np.float32)
        state["last_delta_r"] = np.zeros(3, dtype=np.float32)
        state["sync"] = False

    def refresh_status_with_delta() -> None:
        t = state["pose_t"]
        euler = matrix_to_euler_xyz_deg(state["pose_r"])
        status.content = (
            "Status: preview only\n"
            f"current t=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})\n"
            f"current r=({euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}) deg\n"
            f"pending dT=({float(dtx.value):.4f}, {float(dty.value):.4f}, {float(dtz.value):.4f}) "
            f"dR=({float(drx.value):.1f}, {float(dry.value):.1f}, {float(drz.value):.1f})"
        )

    def apply_slider_increment_live() -> None:
        if state["sync"] or state["applied"]:
            return
        cur_dt = np.array([float(dtx.value), float(dty.value), float(dtz.value)], dtype=np.float32)
        cur_dr = np.array([float(drx.value), float(dry.value), float(drz.value)], dtype=np.float32)
        inc_dt = cur_dt - state["last_delta_t"]
        inc_dr = cur_dr - state["last_delta_r"]
        if np.allclose(inc_dt, 0.0) and np.allclose(inc_dr, 0.0):
            refresh_status_with_delta()
            return

        # Body-fixed live update: compose by incremental delta in current local frame.
        state["pose_t"] = state["pose_t"] + state["pose_r"] @ inc_dt
        state["pose_r"] = state["pose_r"] @ rotation_matrix_xyz(inc_dr.tolist())
        state["last_delta_t"] = cur_dt
        state["last_delta_r"] = cur_dr
        render_desired_axes()

    @dtx.on_update
    @dty.on_update
    @dtz.on_update
    @drx.on_update
    @dry.on_update
    @drz.on_update
    def _(_event) -> None:
        apply_slider_increment_live()

    @zero_delta_btn.on_click
    def _(_event) -> None:
        reset_delta_sliders()
        refresh_status_with_delta()

    @reset_pose_btn.on_click
    def _(_event) -> None:
        if state["applied"]:
            status.content = "Status: already applied in-place in this session. Restart tool to continue."
            return
        state["pose_t"] = np.array(
            [float(args.init_translation[0]), float(args.init_translation[1]), float(args.init_translation[2])],
            dtype=np.float32,
        )
        state["pose_r"] = rotation_matrix_xyz(
            [float(args.init_rotation_deg[0]), float(args.init_rotation_deg[1]), float(args.init_rotation_deg[2])]
        )
        reset_delta_sliders()
        render_desired_axes()

    @apply_btn.on_click
    def _(_event) -> None:
        if state["applied"]:
            status.content = "Status: already applied in this session. Restart tool to apply another transform."
            return
        try:
            record_path = apply_inplace_transform(export_dir, state["pose_t"], state["pose_r"], args.transform_record_path)
            status.content = f"Status: applied in-place\nrecord: {record_path}"
            console.print(f"[green]Done. Transform record:[/] {record_path}")
            state["applied"] = True
        except Exception as exc:
            status.content = f"Status: apply failed - {exc}"
            console.print(f"[red]Apply failed:[/] {exc}")

    render_desired_axes()
    console.print(
        "[cyan]Use dT/dR sliders for live body-fixed updates (w.r.t current frame). "
        "Click 'Apply In-Place' once when ready.[/]"
    )

    try:
        import time

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("[yellow]Stopping desired_axes tool.[/]")
