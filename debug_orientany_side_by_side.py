import torch
import numpy as np
np.set_printoptions(precision=2, suppress=True)
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
import torch.nn.functional as F
import json
import os
from typing import List

from nerfstudio.utils.eval_utils import eval_setup
from f3rm.features.clipsam_extract import CLIPSAMExtractor, run_async_in_any_context
from f3rm.features.orientany.orientany_main import OrientAny
from f3rm.manual.instance_axes_annotator import AxesAnnotator
from f3rm.features.utils import (
    get_nerf_ccs_to_normal_ccs_T,
    build_transform_lookup,
    get_nerf_ccs_to_orig_nerf_world,
    get_orig_to_final_nerf_world_transform_scale,
)
from f3rm.features.orientany.homography import Homography
from f3rm.shaders import VectorShader


class OrientAnyComparison:
    """Compare OrientAny ground truth vs NeRF model predictions."""

    def __init__(self, data_dir: Path, config_path: Path):
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)

        # Setup devices and models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.orient_any = OrientAny("f3rm/features/orientany/ckpts", "ronormsigma1_dino_weight.pt")

        # Load NeRF pipeline
        _, self.pipeline, _, _ = eval_setup(config_path=self.config_path, test_mode="test")
        self.pipeline.eval()
        self.pipeline.model.config.eval_num_rays_per_chunk = min(
            getattr(self.pipeline.model.config, "eval_num_rays_per_chunk", 8192), 4096
        )

        # Load transforms for coordinate conversion
        self._load_transforms()

    def _load_transforms(self):
        """Load camera transforms for coordinate conversion."""
        transforms_path = self.data_dir / "transforms.json"
        self.T_orig_to_final_nerf_world, self.scale = get_orig_to_final_nerf_world_transform_scale(str(transforms_path))
        dataset_transforms = json.load(open(transforms_path, "r"))
        self.transforms_lookup = build_transform_lookup(dataset_transforms["frames"])

    def _find_cam_for_image(self, image_path: str):
        """Find camera split and index for given image."""
        image_path = str(image_path)
        train_imgs = [str(p) for p in self.pipeline.datamanager.train_dataset.image_filenames]
        eval_imgs = [str(p) for p in self.pipeline.datamanager.eval_dataset.image_filenames]
        if image_path in train_imgs:
            return "train", train_imgs.index(image_path)
        if image_path in eval_imgs:
            return "eval", eval_imgs.index(image_path)
        raise ValueError("Image not found in train/eval datasets: " + image_path)

    def _compute_cam_to_world_normal_R(self, image_path: str):
        """Compute camera to world rotation matrix."""
        nerf_ccs1_to_orig_world = get_nerf_ccs_to_orig_nerf_world(Path(image_path).name, self.transforms_lookup)
        nerf_ccs1_to_final_world = self.T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_world
        nerf_ccs1_to_final_world[:3, 3] *= self.scale
        normal_ccs2_to_final_world = np.eye(4) @ np.linalg.inv(get_nerf_ccs_to_normal_ccs_T())
        normal_ccs1_to_final_world = nerf_ccs1_to_final_world @ np.linalg.inv(get_nerf_ccs_to_normal_ccs_T())
        normal_ccs1_to_normal_ccs2 = np.linalg.inv(normal_ccs2_to_final_world) @ normal_ccs1_to_final_world
        return normal_ccs1_to_normal_ccs2[:3, :3].astype(np.float32)

    @torch.no_grad()
    def _render_orientany_outputs(self, image_path: str):
        """Render OrientAny outputs from NeRF model."""
        split, local_cam_idx = self._find_cam_for_image(str(image_path))
        cams = (self.pipeline.datamanager.eval_ray_generator.cameras if split == 'eval'
                else self.pipeline.datamanager.train_ray_generator.cameras)
        c_tensor = torch.tensor([local_cam_idx], device=cams.device)
        # Camera optimizer only applies to train cameras; eval cameras don't have trained adjustments
        camera_opt_to_camera = None if split == 'eval' else self.pipeline.model.camera_optimizer(c_tensor)
        ray_bundle = cams.generate_rays(camera_indices=local_cam_idx, camera_opt_to_camera=camera_opt_to_camera)

        outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(
            ray_bundle, render_features=False, render_orientany=True
        )
        return outputs["orientany_rx"].cpu(), outputs["orientany_rz"].cpu()

    def get_orientany_gt_prediction(self, instance_img_array: np.ndarray, image_path: str):
        """Get OrientAny ground truth prediction (direct model inference)."""
        # Process with OrientAny
        instance_img = Image.fromarray(instance_img_array, 'RGBA')
        rm_bkg_img = self.orient_any.preprocess_remove_bkg(instance_img, do_remove_background=False)
        outs = self.orient_any.get_model_outputs(rm_bkg_img, viz_distn=False)

        # Extract logits and get argmax predictions
        gaus_ax_logits = torch.from_numpy(outs['gaus_ax_logits']).to(self.device)  # 360D
        gaus_pl_logits = torch.from_numpy(outs['gaus_pl_logits']).to(self.device)  # 180D
        gaus_ro_logits = torch.from_numpy(outs['gaus_ro_logits']).to(self.device)  # 360D

        # Get rotation matrix using argmax (same as extraction logic)
        ax_pred = torch.argmax(gaus_ax_logits).item()
        pl_pred = torch.argmax(gaus_pl_logits).item()
        ro_pred = torch.argmax(gaus_ro_logits).item()

        # Convert to OrientAny angles (same as extraction logic)
        phi = float(ax_pred)
        theta_elev = float(pl_pred) - 90.0
        delta = float(ro_pred) - self.orient_any.model_config['ro_offset']

        # Get R_objw_to_cam1 (camera 1 coordinate system)
        R_objw_to_normal_ccs1 = self.orient_any.get_R_objw2cam(phi, theta_elev, delta)
        R_objw_to_nerf_ccs1 = get_nerf_ccs_to_normal_ccs_T()[:3, :3].T @ R_objw_to_normal_ccs1

        # Return rotation matrix directly in NeRF CCS (camera coordinate system)
        confidence_gt = outs['confidence']
        return R_objw_to_nerf_ccs1, confidence_gt

    def get_nerf_model_prediction_per_instance(self, image_path: str, clipsam_masks: List[dict]):
        """Get NeRF model prediction rotation matrices per instance in NeRF CCS."""
        # Render OrientAny outputs from NeRF
        orientany_rx, orientany_rz = self._render_orientany_outputs(image_path)  # (H, W, 3)

        # Get camera transform for this image
        nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(Path(image_path).name, self.transforms_lookup)
        nerf_ccs1_to_final_nerf_world = self.T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
        nerf_ccs1_to_final_nerf_world[:3, 3] *= self.scale
        R_final_nerf_world_to_nerf_ccs1 = nerf_ccs1_to_final_nerf_world[:3, :3].T

        instance_rotations = []

        for mask in clipsam_masks:
            seg = mask["segmentation"]

            # Average vectors over this instance's pixels
            mask_flat = torch.from_numpy(seg.reshape(-1)).bool()
            if not mask_flat.any():
                # Empty instance, use identity rotation
                instance_rotations.append(np.eye(3))
                continue

            rx_mean = orientany_rx.view(-1, 3)[mask_flat].mean(dim=0).numpy()
            rz_mean = orientany_rz.view(-1, 3)[mask_flat].mean(dim=0).numpy()

            # Normalize vectors
            rx_mean = rx_mean / np.linalg.norm(rx_mean)
            rz_mean = rz_mean / np.linalg.norm(rz_mean)

            # Convert vectors to rotation matrix in final NeRF world frame
            ry_mean = np.cross(rz_mean, rx_mean)
            ry_mean = ry_mean / np.linalg.norm(ry_mean)
            R_objw_to_final_nerf_world = np.column_stack([rx_mean, ry_mean, rz_mean])

            # Transform to NeRF CCS (camera coordinate system)
            R_objw_to_nerf_ccs1 = R_final_nerf_world_to_nerf_ccs1 @ R_objw_to_final_nerf_world
            instance_rotations.append(R_objw_to_nerf_ccs1)

        return instance_rotations, orientany_rx, orientany_rz


if __name__ == "__main__":
    # Configuration
    # DATA_DIR = Path("datasets/f3rm/opt/objaverse/car2")
    # CONFIG_PATH = Path("centdx4_outputs/car2_ori_testing_perp_centenc/f3rm/2025-10-11_180207/config.yml")
    DATA_DIR = Path("datasets/f3rm/opt/facchair1")
    CONFIG_PATH = Path("/robodata/smodak/repos/f3rm/jan_outputs/facchair1_centenc/f3rm/2026-01-29_063758/config.yml")
    TEXT_PROMPTS = None
    VIDEO_PATH = "orientany_comparison1.mp4"

    # Get image paths
    image_dir = DATA_DIR / "images"
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    # image_paths = image_paths[:5]  # Process first 5 images

    # Extract CLIPSAM instances for all images at once (efficient batch processing)
    print("=== OrientAny Comparison Demo ===")
    print(f"Extracting CLIPSAM instances for {len(image_paths)} images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clipsam_extractor = CLIPSAMExtractor(device=device, data_dir=DATA_DIR, text_prompts=TEXT_PROMPTS, verbose=True)
    clipsam_instances = run_async_in_any_context(lambda: clipsam_extractor.extract_batch_async([str(p) for p in image_paths]))

    # Cleanup CLIPSAM extractor
    del clipsam_extractor
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize comparison class
    comparator = OrientAnyComparison(DATA_DIR, CONFIG_PATH)

    print(f"Processing {len(image_paths)} images...")

    # Example 1: Visualize OrientAny GT predictions with instance axes
    print("\n1. OrientAny GT predictions with instance axes:")
    image_path = image_paths[0]
    clipsam_masks = clipsam_instances[0]

    if clipsam_masks:
        # PIL Image loading
        origin_img = Image.open(image_path).convert('RGB')
        img_array = np.array(origin_img)

        # Convert to BGR for axes drawing (like in orientany_extract.py)
        axes_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Get camera transform for this image
        nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(Path(image_path).name, comparator.transforms_lookup)
        nerf_ccs1_to_final_nerf_world = comparator.T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
        nerf_ccs1_to_final_nerf_world[:3, 3] *= comparator.scale
        R_final_nerf_world_to_nerf_ccs1 = nerf_ccs1_to_final_nerf_world[:3, :3].T

        for j, mask in enumerate(clipsam_masks):
            seg = mask["segmentation"]

            ys, xs = np.where(seg)
            instance_center = (int(xs.mean()), int(ys.mean()))

            # Create instance image for OrientAny GT
            instance_img_array = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
            instance_img_array[..., :3] = img_array * seg[..., None]
            instance_img_array[..., 3] = seg.astype(np.uint8) * 255

            # Get OrientAny GT prediction for this instance
            R_objw_to_nerf_ccs1_gt, confidence = comparator.get_orientany_gt_prediction(instance_img_array, str(image_path))

            # Draw axes for this instance
            axes_image = AxesAnnotator.visualize_rotation_matrix(
                axes_image,  # Already BGR, no conversion needed
                instance_center,
                R_objw_to_nerf_ccs1_gt,
                axis_length=70,  # Match orientany_extract.py
                axis_thickness=4
            )

        # Convert back to RGB for display
        axes_image = cv2.cvtColor(axes_image, cv2.COLOR_BGR2RGB)

        # matplotlib visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(axes_image)
        plt.title(f"OrientAny GT Predictions ({len(clipsam_masks)} instances)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        print(f"No instances found for {Path(image_path).name}")

    # Example 2: Visualize only NeRF model prediction (per instance)
    print("\n2. NeRF Model prediction per instance:")
    image_path = image_paths[0]
    clipsam_masks = clipsam_instances[0]

    if clipsam_masks:
        # PIL Image loading
        origin_img = Image.open(image_path).convert('RGB')
        img_array = np.array(origin_img)

        # Get NeRF model predictions per instance
        instance_rotations, orientany_rx, orientany_rz = comparator.get_nerf_model_prediction_per_instance(image_path, clipsam_masks)

        # Convert to BGR for axes drawing
        axes_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Draw axes for each instance
        for j, mask in enumerate(clipsam_masks):
            seg = mask["segmentation"]
            ys, xs = np.where(seg)
            instance_center = (int(xs.mean()), int(ys.mean()))

            # Use the rotation matrix for this instance
            R_objw_to_nerf_ccs1_pred = instance_rotations[j]

            axes_image = AxesAnnotator.visualize_rotation_matrix(
                axes_image,  # Already BGR, no conversion needed
                instance_center,
                R_objw_to_nerf_ccs1_pred,
                axis_length=70,
                axis_thickness=4
            )

        # VectorShader visualization for rx and rz
        vector_shader = VectorShader()

        # Create a combined mask for all instances
        combined_mask = np.zeros_like(clipsam_masks[0]["segmentation"], dtype=bool)
        for mask in clipsam_masks:
            combined_mask |= mask["segmentation"]

        fg_tensor = torch.from_numpy(combined_mask).float().unsqueeze(-1)

        # Convert rx and rz to RGB using VectorShader
        rx_rgb = vector_shader(orientany_rx, valid_mask=fg_tensor)
        rz_rgb = vector_shader(orientany_rz, valid_mask=fg_tensor)

        # Convert to numpy uint8 for display
        rx_rgb_np = (rx_rgb * 255).clamp(0, 255).byte().numpy()
        rz_rgb_np = (rz_rgb * 255).clamp(0, 255).byte().numpy()

        # matplotlib visualization - side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Axes visualization
        axes[0].imshow(cv2.cvtColor(axes_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"NeRF Model ({len(clipsam_masks)} instances)")
        axes[0].axis('off')

        # RX vector visualization
        axes[1].imshow(rx_rgb_np)
        axes[1].set_title("RX Vector (VectorShader)")
        axes[1].axis('off')

        # RZ vector visualization
        axes[2].imshow(rz_rgb_np)
        axes[2].set_title("RZ Vector (VectorShader)")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        print(f"No instances found for {Path(image_path).name}")

    # Example 3: Side-by-side comparison for all images
    print("\n3. Side-by-side comparison:")

    # Step 1: Compute all predictions first
    print("Computing predictions for all images...")
    results = []

    for i, image_path in enumerate(tqdm(image_paths, desc="Computing predictions")):
        clipsam_masks = clipsam_instances[i]

        if not clipsam_masks:
            print(f"No CLIPSAM instances found in {Path(image_path).name}")
            continue

        # PIL Image loading
        origin_img = Image.open(image_path).convert('RGB')
        img_array = np.array(origin_img)

        # Get NeRF model predictions per instance
        instance_rotations, _, _ = comparator.get_nerf_model_prediction_per_instance(image_path, clipsam_masks)

        # Store results with instance information
        results.append({
            'image_path': image_path,
            'img_array': img_array,
            'clipsam_masks': clipsam_masks,
            'instance_rotations': instance_rotations,
        })

    # Step 2: Visualize all results and create video
    print(f"\nVisualizing side-by-side comparisons for {len(results)} images and creating video...")

    if results:
        # Initialize video writer
        first_result = results[0]
        first_img_array = first_result['img_array']
        h, w = first_img_array.shape[:2]

        video_w, video_h = w * 2, h
        fps = 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = VIDEO_PATH
        video_writer = None

        try:
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (video_w, video_h))

            if not video_writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {video_path}")

            for result in tqdm(results, desc="Visualizing and creating video"):
                image_path = result['image_path']
                img_array = result['img_array']
                clipsam_masks = result['clipsam_masks']
                instance_rotations = result['instance_rotations']

                # Get camera transform for this image
                nerf_ccs1_to_orig_nerf_world = get_nerf_ccs_to_orig_nerf_world(Path(image_path).name, comparator.transforms_lookup)
                nerf_ccs1_to_final_nerf_world = comparator.T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_nerf_world
                nerf_ccs1_to_final_nerf_world[:3, 3] *= comparator.scale
                R_final_nerf_world_to_nerf_ccs1 = nerf_ccs1_to_final_nerf_world[:3, :3].T

                # Draw OrientAny GT axes (per instance)
                axes_gt = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR once
                for j, mask in enumerate(clipsam_masks):
                    seg = mask["segmentation"]

                    ys, xs = np.where(seg)
                    instance_center = (int(xs.mean()), int(ys.mean()))

                    # Create instance image for OrientAny GT
                    instance_img_array = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
                    instance_img_array[..., :3] = img_array * seg[..., None]
                    instance_img_array[..., 3] = seg.astype(np.uint8) * 255

                    # Get OrientAny GT prediction for this instance
                    R_objw_to_nerf_ccs1_gt, confidence = comparator.get_orientany_gt_prediction(instance_img_array, str(image_path))

                    axes_gt = AxesAnnotator.visualize_rotation_matrix(
                        axes_gt,  # Already BGR, no conversion needed
                        instance_center,
                        R_objw_to_nerf_ccs1_gt,
                        axis_length=70,  # Match orientany_extract.py
                        axis_thickness=4
                    )

                # Draw NeRF prediction axes (per instance)
                axes_pred = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR once
                for j, mask in enumerate(clipsam_masks):
                    seg = mask["segmentation"]

                    ys, xs = np.where(seg)
                    instance_center = (int(xs.mean()), int(ys.mean()))

                    # Use the rotation matrix for this instance
                    R_objw_to_nerf_ccs1_pred = instance_rotations[j]

                    axes_pred = AxesAnnotator.visualize_rotation_matrix(
                        axes_pred,  # Already BGR, no conversion needed
                        instance_center,
                        R_objw_to_nerf_ccs1_pred,
                        axis_length=70,
                        axis_thickness=4
                    )

                # Create side-by-side frame for video
                side_by_side_frame = np.hstack([axes_gt, axes_pred])

                # Add text overlay for video
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2

                # GT label (instance-based)
                gt_text = f"OrientAny GT ({len(clipsam_masks)} instances)"
                cv2.putText(side_by_side_frame, gt_text, (10, 30), font, font_scale, (255, 255, 255), font_thickness)
                cv2.putText(side_by_side_frame, gt_text, (10, 30), font, font_scale, (0, 0, 0), font_thickness - 1)

                # Pred label
                pred_text = "NeRF Model"
                cv2.putText(side_by_side_frame, pred_text, (w + 10, 30), font, font_scale, (255, 255, 255), font_thickness)
                cv2.putText(side_by_side_frame, pred_text, (w + 10, 30), font, font_scale, (0, 0, 0), font_thickness - 1)

                # Image name at bottom
                image_name = Path(image_path).name
                name_text_size = cv2.getTextSize(image_name, font, font_scale, font_thickness)[0]
                name_x = (video_w - name_text_size[0]) // 2
                cv2.putText(side_by_side_frame, image_name, (name_x, video_h - 10), font, font_scale, (255, 255, 255), font_thickness)
                cv2.putText(side_by_side_frame, image_name, (name_x, video_h - 10), font, font_scale, (0, 0, 0), font_thickness - 1)

                # Write frame to video
                video_writer.write(side_by_side_frame)

                # Side-by-side matplotlib visualization
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                axes[0].imshow(cv2.cvtColor(axes_gt, cv2.COLOR_BGR2RGB))
                axes[0].set_title(f"OrientAny GT ({len(clipsam_masks)} instances)")
                axes[0].axis('off')

                axes[1].imshow(cv2.cvtColor(axes_pred, cv2.COLOR_BGR2RGB))
                axes[1].set_title(f"NeRF Model ({len(clipsam_masks)} instances)")
                axes[1].axis('off')

                # plt.suptitle(f"OrientAny Comparison: {Path(image_path).name}", fontsize=14)
                # plt.tight_layout()
                # plt.show()
                plt.close(fig)

        except Exception as e:
            print(f"Error during video creation: {e}")

        finally:
            # Ensure video writer is properly released
            if video_writer is not None:
                video_writer.release()
                print(f"Video writer released.")

            print(f"Video saved to: {video_path}")
    else:
        print("No results to visualize.")
