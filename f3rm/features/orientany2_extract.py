import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from sam2.features.utils import AsyncMultiWrapper

from f3rm.features.orientany2.orientanyv2_main import OrientAnyV2
from f3rm.features.orientany_extract import (
    _compute_object_orientation_features,
    _create_pixel_data,
    _filter_masks_by_size,
    _get_camera_transforms,
)
from f3rm.features.utils import (
    BatchFeatureLoader,
    build_transform_lookup,
    get_nerf_ccs_to_orig_nerf_world,
    get_orig_to_final_nerf_world_transform_scale,
    resolve_devices_and_workers,
    run_async_in_any_context,
)
from f3rm.manual.instance_axes_annotator import AxesAnnotator
from f3rm.shaders import VectorShader


class ORIENTANY2Args:
    min_instance_percent: float = 1.0
    batch_size_per_gpu: int = 4

    @classmethod
    def id_dict(cls):
        return {
            "min_instance_percent": float(cls.min_instance_percent),
        }


def parse_orientany2_feature_type(feature_type: str) -> List[str]:
    if not feature_type.startswith("ORIENTANY2_"):
        raise ValueError(f"Invalid ORIENTANY2 feature type: {feature_type}. Must start with 'ORIENTANY2_'")
    prompts_part = feature_type[len("ORIENTANY2_"):]
    if prompts_part == "":
        return []
    return [w.lower() for w in prompts_part.split("_") if w.strip()]


def _normalize_alpha(alpha_raw: Any) -> int:
    try:
        alpha_val = int(round(float(alpha_raw)))
    except (TypeError, ValueError):
        alpha_val = 1
    allowed = [0, 1, 2, 4]
    if alpha_val in allowed:
        return alpha_val
    return min(allowed, key=lambda v: abs(v - alpha_val))


def _build_orientany2_instance_features(
    phi: float,
    theta_elev: float,
    delta: float,
    alpha_raw: Any,
    nerf_ccs1_to_final_nerf_world: np.ndarray,
) -> np.ndarray:
    alpha = _normalize_alpha(alpha_raw)

    main_axes = _compute_object_orientation_features(phi, theta_elev, delta, nerf_ccs1_to_final_nerf_world)
    main_u_x = main_axes[:3].astype(np.float32)
    u_z = main_axes[3:6].astype(np.float32)

    u_x_slots = [np.zeros(3, dtype=np.float32) for _ in range(4)]
    if alpha > 0:
        offsets_deg = [k * (360.0 / alpha) for k in range(alpha)]
        for k, offset_deg in enumerate(offsets_deg[:4]):
            if k == 0:
                u_x_slots[k] = main_u_x
                continue
            u_x_k = _compute_object_orientation_features(
                (phi + offset_deg) % 360.0,
                theta_elev,
                delta,
                nerf_ccs1_to_final_nerf_world,
            )[:3]
            u_x_slots[k] = u_x_k.astype(np.float32)

    # Layout: [alpha, u_z(3), u_x0(3), u_x1(3), u_x2(3), u_x3(3)].
    feature = np.concatenate(
        [np.array([float(alpha)], dtype=np.float32), u_z, u_x_slots[0], u_x_slots[1], u_x_slots[2], u_x_slots[3]],
        axis=0,
    )
    return feature.astype(np.float16)


def _draw_orientany2_instance_axes(
    axes_frame: np.ndarray,
    instance_center: Tuple[int, int],
    instance_feat_arr: np.ndarray,
    R_final_nerf_world_to_nerf_ccs1: np.ndarray,
    axis_length: int = 70,
    axis_thickness: int = 4,
) -> np.ndarray:
    alpha = _normalize_alpha(instance_feat_arr[0])
    u_z_world = instance_feat_arr[1:4]
    if np.linalg.norm(u_z_world) < 1e-6:
        return axes_frame

    axes_to_draw = ("x", "z") if alpha == 4 else ("x", "y", "z")
    if alpha == 0:
        # Build any stable orthonormal frame around u_z; only Z will be drawn.
        z_hat = u_z_world / np.linalg.norm(u_z_world)
        aux = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(z_hat, aux))) > 0.95:
            aux = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_hat = np.cross(aux, z_hat)
        x_norm = np.linalg.norm(x_hat)
        if x_norm < 1e-6:
            aux = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            x_hat = np.cross(aux, z_hat)
            x_norm = np.linalg.norm(x_hat)
        if x_norm < 1e-6:
            return axes_frame
        x_hat = x_hat / x_norm
        y_hat = np.cross(z_hat, x_hat)
        y_norm = np.linalg.norm(y_hat)
        if y_norm < 1e-6:
            return axes_frame
        y_hat = y_hat / y_norm

        R_objw_to_final_nerf_world = np.column_stack([x_hat, y_hat, z_hat])
        R_objw_to_nerf_ccs1 = R_final_nerf_world_to_nerf_ccs1 @ R_objw_to_final_nerf_world
        return AxesAnnotator.visualize_rotation_matrix(
            axes_frame,
            instance_center,
            R_objw_to_nerf_ccs1,
            axis_length=axis_length,
            axis_thickness=axis_thickness,
            axes_to_draw=("z",),
        )

    for k in range(alpha):
        u_x_world = instance_feat_arr[4 + 3 * k: 7 + 3 * k]
        if np.linalg.norm(u_x_world) < 1e-6:
            continue
        u_y_world = np.cross(u_z_world, u_x_world)
        if np.linalg.norm(u_y_world) < 1e-6:
            continue
        u_y_world = u_y_world / np.linalg.norm(u_y_world)
        R_objw_to_final_nerf_world = np.column_stack([u_x_world, u_y_world, u_z_world])
        R_objw_to_nerf_ccs1 = R_final_nerf_world_to_nerf_ccs1 @ R_objw_to_final_nerf_world
        axes_frame = AxesAnnotator.visualize_rotation_matrix(
            axes_frame,
            instance_center,
            R_objw_to_nerf_ccs1,
            axis_length=axis_length,
            axis_thickness=axis_thickness,
            axes_to_draw=axes_to_draw,
        )

    return axes_frame


class ORIENTANY2Worker:
    def __init__(self, device: torch.device, data_dir: Path, sam3_feature_type: str):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.data_dir = Path(data_dir)
        self.sam3_feature_type = sam3_feature_type

        sam3_root = self.data_dir / "features" / self.sam3_feature_type.lower()
        if not (sam3_root / "meta.pt").exists():
            raise FileNotFoundError(f"Missing SAM3 meta: {sam3_root / 'meta.pt'}")

        sam3_meta = torch.load(sam3_root / "meta.pt")
        self.feat_image_fnames = [str(p) for p in sam3_meta["image_fnames"]]
        self.sam3_loader = BatchFeatureLoader(self.data_dir, self.sam3_feature_type, self.feat_image_fnames, self.device)

        ckpt_path = Path(__file__).parent / "orientany2" / "rotmod_realrotaug_best.pt"
        self.orient_any = OrientAnyV2(str(ckpt_path), device=self.device)

        transforms_path = self.data_dir / "transforms.json"
        if not transforms_path.exists():
            raise ValueError("transforms.json not found")
        T_orig_to_final_nerf_world, orig_to_final_nerf_world_scale = get_orig_to_final_nerf_world_transform_scale(str(transforms_path))
        dataset_transforms_data = json.load(open(transforms_path, "r"))
        self.transforms_lookup = build_transform_lookup(dataset_transforms_data["frames"])
        self.T_orig_to_final_nerf_world = T_orig_to_final_nerf_world
        self.orig_to_final_nerf_world_scale = orig_to_final_nerf_world_scale

    async def compute_orientany2_for_image_async(self, image_path: str, debug: bool = False) -> Dict[str, Any]:
        del debug
        try:
            idx = self.feat_image_fnames.index(str(image_path))
        except ValueError as exc:
            raise ValueError(f"Image path not found in SAM3 meta order: {image_path}") from exc

        masks = np.asarray(self.sam3_loader[idx])
        if masks.ndim == 4:
            masks = masks[:, 0, ...]
        elif masks.ndim == 2:
            masks = masks[None, ...]
        masks = masks.astype(bool)

        h, w = masks.shape[-2:]
        obj_masks = _filter_masks_by_size(masks, ORIENTANY2Args.min_instance_percent)
        if not obj_masks:
            return {"pixel_data": _create_pixel_data(h, w, []), "instance_features": {}}

        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        nerf_ccs1_to_final_nerf_world, _ = _get_camera_transforms(
            image_path,
            self.transforms_lookup,
            self.T_orig_to_final_nerf_world,
            self.orig_to_final_nerf_world_scale,
        )

        instance_features: Dict[int, List[float]] = {}
        for mask in obj_masks:
            instance_img_array = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
            instance_img_array[..., :3] = img_array * mask[..., None]
            instance_img_array[..., 3] = mask * 255
            instance_img = Image.fromarray(instance_img_array, "RGBA")

            rm_bkg_img = self.orient_any.preprocess_remove_bkg(instance_img, do_remove_background=False)
            outs = self.orient_any.get_model_outputs(rm_bkg_img)

            instance_feat = _build_orientany2_instance_features(
                phi=float(outs["phi"]),
                theta_elev=float(outs["theta_elev"]),
                delta=float(outs["delta"]),
                alpha_raw=outs.get("alpha", 1),
                nerf_ccs1_to_final_nerf_world=nerf_ccs1_to_final_nerf_world,
            )
            instance_features[len(instance_features) + 1] = instance_feat.tolist()
            del instance_img, rm_bkg_img, outs

        pixel_data = _create_pixel_data(h, w, obj_masks)
        return {"pixel_data": pixel_data, "instance_features": instance_features}


class ORIENTANY2Extractor:
    def __init__(
        self,
        device: torch.device,
        data_dir: Optional[Path] = None,
        text_prompts: Optional[List[str]] = None,
        sam3_feature_type: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.verbose = verbose
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.text_prompts = text_prompts
        self.sam3_feature_type = sam3_feature_type or self._infer_sam3_feature_type(text_prompts)

        if self.data_dir is None:
            raise ValueError("ORIENTANY2Extractor requires data_dir to locate precomputed SAM3 shards")

        sam3_root = self.data_dir / "features" / self.sam3_feature_type.lower()
        if not (sam3_root / "meta.pt").exists():
            raise FileNotFoundError(
                f"Missing SAM3 meta: {sam3_root / 'meta.pt'} (expected for ORIENTANY2). "
                f"Run SAM3 extraction for feature type '{self.sam3_feature_type}' first."
            )
        if not list(sam3_root.glob("image_*.npz")):
            raise FileNotFoundError(f"Missing SAM3 per-image features under {sam3_root}")

        devices_param, num_workers = resolve_devices_and_workers(device, ORIENTANY2Args.batch_size_per_gpu)
        if verbose:
            print(f"Initializing ORIENTANY2 workers (using {self.sam3_feature_type} masks)")
        self.client = AsyncMultiWrapper(
            ORIENTANY2Worker,
            num_objects=num_workers,
            devices=devices_param,
            data_dir=self.data_dir,
            sam3_feature_type=self.sam3_feature_type,
        )
        self.num_workers = num_workers

    @staticmethod
    def _infer_sam3_feature_type(text_prompts: Optional[List[str]]) -> str:
        if text_prompts is None or len(text_prompts) == 0:
            return "SAM3_"
        joined = "_".join([p.lower() for p in text_prompts])
        return f"SAM3_{joined}"

    async def extract_batch_async(self, image_paths: List[str], debug: bool = False) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting ORIENTANY2 features", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_orientany2_async(path, self.client, debug=debug) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="ORIENTANY2", leave=False)
            results.extend(batch_results)
            gc.collect()
        return results


async def process_single_image_orientany2_async(image_path: str, orientany2_client: AsyncMultiWrapper, debug: bool = False) -> Dict[str, Any]:
    return await orientany2_client.compute_orientany2_for_image_async(image_path, debug=debug)


def examine_saved(orientany2_feat_dir: str):
    """Create .mp4 video of saved ORIENTANY2 features with side-by-side visualization."""
    meta_path = os.path.join(orientany2_feat_dir, "meta.pt")
    assert os.path.exists(meta_path), f"ORIENTANY2 meta not found at {meta_path}"

    meta = torch.load(meta_path)
    image_fnames = meta["image_fnames"]
    n_images = len(image_fnames)

    data_dir = Path(orientany2_feat_dir).parent.parent
    transforms_path = data_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found at {transforms_path}")

    T_orig_to_final_nerf_world, scale = get_orig_to_final_nerf_world_transform_scale(str(transforms_path))
    dataset_transforms_data = json.load(open(transforms_path, "r"))
    transforms_lookup = build_transform_lookup(dataset_transforms_data["frames"])

    first_pixel_path = os.path.join(orientany2_feat_dir, "image_000000_pixel.npy")
    first_inst_path = os.path.join(orientany2_feat_dir, "image_000000_instances.json")
    if not (os.path.exists(first_pixel_path) and os.path.exists(first_inst_path)):
        raise FileNotFoundError(f"ORIENTANY2 feature files not found in {orientany2_feat_dir}")

    pixel_data = np.load(first_pixel_path)
    H, W = pixel_data.shape[:2]

    video_path = os.path.join(orientany2_feat_dir, "features_viz.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 2.0, (W * 3, H))
    vector_shader = VectorShader()

    for i in tqdm(range(n_images), desc="Creating ORIENTANY2 features video"):
        pixel_path = os.path.join(orientany2_feat_dir, f"image_{i:06d}_pixel.npy")
        inst_path = os.path.join(orientany2_feat_dir, f"image_{i:06d}_instances.json")
        if not (os.path.exists(pixel_path) and os.path.exists(inst_path)):
            continue

        pixel_data = np.load(pixel_path)
        with open(inst_path, "r") as f:
            instance_features = json.load(f)

        image_path = image_fnames[i]
        if not os.path.exists(image_path):
            continue
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        obj_mask = pixel_data[..., 1] > 0.5
        if not np.any(obj_mask):
            frame_bgr = cv2.cvtColor(np.zeros((H, W * 3, 3), dtype=np.uint8), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            continue

        full_features = np.zeros((H, W, 18), dtype=np.float16)
        full_features[..., 16:18] = pixel_data[..., :2]
        for instance_id, instance_feat in instance_features.items():
            mask = pixel_data[..., 2] == int(instance_id)
            if np.any(mask) and isinstance(instance_feat, list):
                full_features[mask, :16] = np.asarray(instance_feat, dtype=np.float16)

        u_x_main = full_features[..., 4:7]
        u_z = full_features[..., 1:4]

        axes_frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(Path(image_path).name, transforms_lookup)
        nerf_ccs1_to_final_nerf_world = T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
        nerf_ccs1_to_final_nerf_world[:3, 3] *= scale
        R_final_nerf_world_to_nerf_ccs1 = nerf_ccs1_to_final_nerf_world[:3, :3].T

        for instance_id, instance_feat in instance_features.items():
            instance_feat_arr = np.asarray(instance_feat, dtype=np.float32)
            instance_mask = pixel_data[..., 2] == int(instance_id)
            if not np.any(instance_mask):
                continue

            ys, xs = np.where(instance_mask)
            instance_center = (int(xs.mean()), int(ys.mean()))
            axes_frame = _draw_orientany2_instance_axes(
                axes_frame=axes_frame,
                instance_center=instance_center,
                instance_feat_arr=instance_feat_arr,
                R_final_nerf_world_to_nerf_ccs1=R_final_nerf_world_to_nerf_ccs1,
                axis_length=70,
                axis_thickness=4,
            )

        axes_frame = cv2.cvtColor(axes_frame, cv2.COLOR_BGR2RGB)
        obj_tensor = torch.from_numpy(obj_mask).float().unsqueeze(-1)
        ux_rgb = vector_shader(torch.from_numpy(u_x_main).float(), valid_mask=obj_tensor)
        uz_rgb = vector_shader(torch.from_numpy(u_z).float(), valid_mask=obj_tensor)
        ux_rgb = (ux_rgb * 255).clamp(0, 255).byte().numpy()
        uz_rgb = (uz_rgb * 255).clamp(0, 255).byte().numpy()

        side_by_side = np.hstack([axes_frame, ux_rgb, uz_rgb])
        out.write(cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

    out.release()
    assert os.path.exists(video_path), f"Video not created at {video_path}"


if __name__ == "__main__":
    # examine_saved("datasets/f3rm/opt/objaverse/car2/features/orientany2_")

    data_root = Path("datasets/f3rm/opt/objaverse/car2")
    image_dir = data_root / "images"
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    image_paths = [str(p) for p in image_paths[10:13]]  # Just 3 for demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = ORIENTANY2Extractor(device=device, data_dir=data_root, text_prompts=None, verbose=True)
    features_data = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths, debug=True))
    print(f"Extracted {len(features_data)} feature maps")

    vis_count = min(1, len(image_paths))
    fig, axes = plt.subplots(vis_count, 3, figsize=(18, 4 * vis_count))

    transforms_path = data_root / "transforms.json"
    T_orig_to_final_nerf_world, scale = get_orig_to_final_nerf_world_transform_scale(str(transforms_path))
    dataset_transforms_data = json.load(open(transforms_path, "r"))
    transforms_lookup = build_transform_lookup(dataset_transforms_data["frames"])
    vector_shader = VectorShader()

    for i in tqdm(range(vis_count), desc="Visualizing results"):
        data = features_data[i]
        pixel_data = data["pixel_data"]
        instance_features = data["instance_features"]

        rgb = Image.open(image_paths[i]).convert("RGB")
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB {i+1}")
        axes[i, 0].axis("off")

        obj_mask = pixel_data[..., 1]
        axes[i, 1].imshow(obj_mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Object Mask")
        axes[i, 1].axis("off")

        if instance_features:
            h, w, _ = pixel_data.shape
            full_features = np.zeros((h, w, 18), dtype=np.float16)
            full_features[..., 16:18] = pixel_data[..., :2]

            for instance_id, instance_feat in instance_features.items():
                mask = pixel_data[..., 2] == int(instance_id)
                if np.any(mask) and isinstance(instance_feat, list):
                    full_features[mask, :16] = np.asarray(instance_feat, dtype=np.float16)

            u_x_main = full_features[..., 4:7]
            u_x_tensor = torch.from_numpy(u_x_main).float()
            obj_tensor = torch.from_numpy(obj_mask).float().unsqueeze(-1)
            orient_rgb = vector_shader(u_x_tensor, valid_mask=obj_tensor)
            orient_rgb = (orient_rgb * 255).clamp(0, 255).byte().numpy()
            orient_bgr = cv2.cvtColor(orient_rgb, cv2.COLOR_RGB2BGR)

            nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(Path(image_paths[i]).name, transforms_lookup)
            nerf_ccs1_to_final_nerf_world = T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
            nerf_ccs1_to_final_nerf_world[:3, 3] *= scale
            R_final_nerf_world_to_nerf_ccs1 = nerf_ccs1_to_final_nerf_world[:3, :3].T

            for instance_id, instance_feat in instance_features.items():
                instance_mask = pixel_data[..., 2] == int(instance_id)
                if not np.any(instance_mask):
                    continue
                ys, xs = np.where(instance_mask)
                instance_center = (int(xs.mean()), int(ys.mean()))
                orient_bgr = _draw_orientany2_instance_axes(
                    axes_frame=orient_bgr,
                    instance_center=instance_center,
                    instance_feat_arr=np.asarray(instance_feat, dtype=np.float32),
                    R_final_nerf_world_to_nerf_ccs1=R_final_nerf_world_to_nerf_ccs1,
                    axis_length=70,
                    axis_thickness=4,
                )

            orient_rgb = cv2.cvtColor(orient_bgr, cv2.COLOR_BGR2RGB)
            axes[i, 2].imshow(orient_rgb)
            axes[i, 2].set_title("Main u_x RGB + Symmetry Axes")
        else:
            axes[i, 2].text(0.5, 0.5, "No instances", ha="center", va="center", transform=axes[i, 2].transAxes)
            axes[i, 2].set_title("No Instances Found")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()
