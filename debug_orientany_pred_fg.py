import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from nerfstudio.utils.eval_utils import eval_setup

from f3rm.features.orientany.orientany_main import OrientAny
from f3rm.manual.instance_axes_annotator import AxesAnnotator
from f3rm.features.utils import (
    get_nerf_ccs_to_normal_ccs_T,
    build_transform_lookup,
    get_nerf_ccs_to_orig_nerf_world,
    get_orig_to_final_nerf_world_transform_scale,
)
from f3rm.features.foreground_extract import FOREGROUNDExtractor, run_async_in_any_context


# Hardcoded inputs
DATA_DIR = Path("datasets/f3rm/opt/caterpillar")
IMAGE_PATH = DATA_DIR / "images/frame_00004.jpg"
CONFIG_PATH = Path("centopt1_outputs/caterp_cstext_ori_emasched/f3rm/2025-09-10_162852/config.yml")
TEXT_PROMPTS = ["toy"]


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
orient_any = OrientAny("f3rm/features/orientany/ckpts", "ronormsigma1_dino_weight.pt")
fg_extractor = FOREGROUNDExtractor(device=device, data_dir=DATA_DIR, text_prompts=TEXT_PROMPTS, verbose=True)


def _find_cam_for_image(pipeline, image_path: str):
    image_path = str(image_path)
    train_imgs = [str(p) for p in pipeline.datamanager.train_dataset.image_filenames]
    eval_imgs = [str(p) for p in pipeline.datamanager.eval_dataset.image_filenames]
    if image_path in train_imgs:
        return "train", train_imgs.index(image_path)
    if image_path in eval_imgs:
        return "eval", eval_imgs.index(image_path)
    raise ValueError("Image not found in train/eval datasets: " + image_path)


@torch.no_grad()
def _render_outputs_for_image(pipeline, split: str, local_cam_idx: int):
    cams = pipeline.datamanager.eval_ray_generator.cameras if split == 'eval' else pipeline.datamanager.train_ray_generator.cameras
    c_tensor = torch.tensor([local_cam_idx], device=cams.device)
    # Camera optimizer only applies to train cameras; eval cameras don't have trained adjustments
    camera_opt_to_camera = None if split == 'eval' else pipeline.model.camera_optimizer(c_tensor)
    ray_bundle = cams.generate_rays(camera_indices=local_cam_idx, camera_opt_to_camera=camera_opt_to_camera)
    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(
        ray_bundle,
        render_features=False,
        render_orientany=True,
    )
    cam_opt_np = camera_opt_to_camera[0].cpu().numpy() if camera_opt_to_camera is not None else None
    return outputs, cams, cam_opt_np


def _compute_cam_to_world_normal_R(image_path: str):
    transforms_path = DATA_DIR / "transforms.json"
    T_orig_to_final_nerf_world, scale = get_orig_to_final_nerf_world_transform_scale(str(transforms_path))
    dataset_transforms = __import__("json").load(open(transforms_path, "r"))
    lookup = build_transform_lookup(dataset_transforms["frames"])
    nerf_ccs1_to_orig_world = get_nerf_ccs_to_orig_nerf_world(Path(image_path).name, lookup)
    nerf_ccs1_to_final_world = T_orig_to_final_nerf_world @ nerf_ccs1_to_orig_world
    nerf_ccs1_to_final_world[:3, 3] *= scale
    normal_ccs2_to_final_world = np.eye(4) @ np.linalg.inv(get_nerf_ccs_to_normal_ccs_T())
    normal_ccs1_to_final_world = nerf_ccs1_to_final_world @ np.linalg.inv(get_nerf_ccs_to_normal_ccs_T())
    normal_ccs1_to_normal_ccs2 = np.linalg.inv(normal_ccs2_to_final_world) @ normal_ccs1_to_final_world
    R_normal_cam1_to_normal_world = normal_ccs1_to_normal_ccs2[:3, :3].astype(np.float32)
    return R_normal_cam1_to_normal_world


def _softmax_logits(logits: torch.Tensor):
    return torch.softmax(logits, dim=-1)


if __name__ == "__main__":
    # Load pipeline/model
    _, pipeline, _, _ = eval_setup(config_path=CONFIG_PATH, test_mode="test")
    pipeline.eval()
    # Reduce VRAM by lowering rays per chunk
    pipeline.model.config.eval_num_rays_per_chunk = min(getattr(pipeline.model.config, "eval_num_rays_per_chunk", 8192), 4096)
    torch.cuda.empty_cache()

    split, local_cam_idx = _find_cam_for_image(pipeline, str(IMAGE_PATH))
    outputs, cams, _ = _render_outputs_for_image(pipeline, split, local_cam_idx)

    if "orientany_logits" not in outputs:
        raise RuntimeError("Model did not return orientany_logits. Ensure orientany_enable=True and render_orientany=True.")

    logits = outputs["orientany_logits"].cpu()  # (H, W, 902) on CPU
    H, W, _ = logits.shape

    # Split logits and compute probs
    ax_logits = logits[..., 0:360]
    pl_logits = logits[..., 360:540]
    ro_logits = logits[..., 540:900]
    # fg_logits = logits[..., 900:902]

    # Foreground mask via FOREGROUNDExtractor (consistent with debug_orientany_foreground.py)
    fg_map = run_async_in_any_context(lambda: fg_extractor.extract_batch_async([str(IMAGE_PATH)]))[0]
    fg_mask = (fg_map[..., 1] > 0.5)
    # Cleanup foreground extractor to free VRAM/CPU RAM
    del fg_map
    del fg_extractor
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()

    ax_probs_world = _softmax_logits(ax_logits.float())
    pl_probs_world = _softmax_logits(pl_logits.float())
    ro_probs_world = _softmax_logits(ro_logits.float())

    # Average world-frame distributions over foreground pixels
    # NOTE: Model predictions are already in WORLD frame (stored by orientany_extract.py)
    mask_flat = torch.from_numpy(fg_mask.reshape(-1)).bool()
    ax_mean_world = ax_probs_world.view(-1, 360)[mask_flat].mean(dim=0)
    pl_mean_world = pl_probs_world.view(-1, 180)[mask_flat].mean(dim=0)
    ro_mean_world = ro_probs_world.view(-1, 360)[mask_flat].mean(dim=0)

    # Transform world distributions -> camera frame for visualization
    R_cam_to_world = _compute_cam_to_world_normal_R(str(IMAGE_PATH))
    R_world_to_cam = np.linalg.inv(R_cam_to_world).astype(np.float32)
    R_world_to_cam_t = torch.from_numpy(R_world_to_cam).to(device)

    ax_cam_vis, pl_cam_vis, ro_cam_vis = orient_any.push_distributions_to_new_view(
        ax_mean_world.to(device), pl_mean_world.to(device), ro_mean_world.to(device),
        R_world_to_cam_t, batch_phi=32, batch_theta=32, device=device, show_progress=False
    )

    # Final angles via argmax (now in camera frame)
    phi = torch.argmax(ax_cam_vis).item()                    # 0..359 degrees
    theta_elev = torch.argmax(pl_cam_vis).item() - 90        # 0..179 → -90..+89 degrees
    delta = torch.argmax(ro_cam_vis).item() - orient_any.model_config['ro_offset']  # 0..359 → -180..+179 degrees

    # Draw axes on image at foreground centroid
    origin_img = Image.open(IMAGE_PATH).convert('RGB')
    img_array = np.array(origin_img)
    ys, xs = np.where(fg_mask)
    if len(ys) == 0:
        ys, xs = np.array([img_array.shape[0] // 2]), np.array([img_array.shape[1] // 2])
    mask_center = (int(xs.mean()), int(ys.mean()))

    R_objw_to_normal_ccs = orient_any.get_R_objw2cam(phi, theta_elev, delta)
    R_objw_to_nerf_ccs = get_nerf_ccs_to_normal_ccs_T()[:3, :3].T @ R_objw_to_normal_ccs
    axes_image = AxesAnnotator.visualize_rotation_matrix(
        cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR),
        mask_center,
        R_objw_to_nerf_ccs,
        axis_length=90,
        axis_thickness=4
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(axes_image, cv2.COLOR_BGR2RGB))
    plt.title(f"OrientAny (pred world→cam): φ={phi}°, θ={theta_elev}°, δ={delta}°")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
