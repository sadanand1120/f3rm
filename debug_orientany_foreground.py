import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
import torch.nn.functional as F

from f3rm.features.foreground_extract import FOREGROUNDExtractor, run_async_in_any_context
from f3rm.features.orientany.orientany_main import OrientAny
from f3rm.manual.instance_axes_annotator import AxesAnnotator
from f3rm.features.utils import get_nerf_ccs_to_normal_ccs_T, get_conf_temp_scaled_logits


# Hardcoded inputs
DATA_DIR = Path("datasets/f3rm/opt/caterpillar")
TEXT_PROMPTS = ["toy"]
image_dir = DATA_DIR / "images"
image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
image_paths = image_paths[:10]

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
foreground_extractor = FOREGROUNDExtractor(device=device, data_dir=DATA_DIR, text_prompts=TEXT_PROMPTS, verbose=True)
orient_any = OrientAny("f3rm/features/orientany/ckpts", "ronormsigma1_dino_weight.pt")

# Extract all foreground maps at once
foreground_maps = run_async_in_any_context(lambda: foreground_extractor.extract_batch_async([str(p) for p in image_paths]))
foreground_masks = [fg_map[..., 1] > 0.5 for fg_map in foreground_maps]

for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
    origin_img = Image.open(image_path).convert('RGB')
    foreground_mask = foreground_masks[i]

    # Final visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)

    axes[0, 0].imshow(origin_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(foreground_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title("Foreground Mask")
    axes[0, 1].axis('off')

    if np.any(foreground_mask):
        # Create instance image from foreground
        img_array = np.array(origin_img.copy())
        instance_img_array = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
        instance_img_array[..., :3] = img_array * foreground_mask[..., None]
        instance_img_array[..., 3] = foreground_mask.astype(np.uint8) * 255
        instance_img = Image.fromarray(instance_img_array, 'RGBA')

        # Process with OrientAny
        rm_bkg_img = orient_any.preprocess_remove_bkg(instance_img, do_remove_background=False)
        outs = orient_any.get_model_outputs(rm_bkg_img, viz_distn=False)
        R_objw_to_normal_ccs = orient_any.get_R_objw2cam(outs['phi'], outs['theta_elev'], outs['delta'])
        R_objw_to_nerf_ccs = get_nerf_ccs_to_normal_ccs_T()[:3, :3].T @ R_objw_to_normal_ccs

        # Calculate center of foreground mask
        foreground_coords = np.where(foreground_mask)
        mask_center_y = int(np.mean(foreground_coords[0]))
        mask_center_x = int(np.mean(foreground_coords[1]))
        mask_center = (mask_center_x, mask_center_y)

        axes_image = AxesAnnotator.visualize_rotation_matrix(
            cv2.cvtColor(np.array(origin_img), cv2.COLOR_RGB2BGR),
            mask_center,
            R_objw_to_nerf_ccs,
            axis_length=90,
            axis_thickness=4
        )

        axes[0, 2].imshow(cv2.cvtColor(axes_image, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f"OrientAny: φ={outs['phi']:.1f}°, θ={outs['theta_elev']:.1f}°, δ={outs['delta']:.1f}°, conf={outs['confidence']:.2f}")

        # Extract logits and create distributions
        gaus_ax_logits = torch.from_numpy(outs['gaus_ax_logits']).float()
        gaus_pl_logits = torch.from_numpy(outs['gaus_pl_logits']).float()
        gaus_ro_logits = torch.from_numpy(outs['gaus_ro_logits']).float()

        gaus_ax_distn = F.softmax(gaus_ax_logits, dim=-1).numpy()
        gaus_pl_distn = F.softmax(gaus_pl_logits, dim=-1).numpy()
        gaus_ro_distn = F.softmax(gaus_ro_logits, dim=-1).numpy()

        # Plot distributions
        azimuth_x_labels = np.linspace(0, 360, len(gaus_ax_distn))
        axes[1, 0].plot(azimuth_x_labels, gaus_ax_distn, 'b-', label='Original')
        axes[1, 0].set_title(f'Azimuth (φ) Distribution - conf={outs["confidence"]:.2f}')
        axes[1, 0].set_xlabel('Angle (degrees)')
        axes[1, 0].set_ylabel('Probability')

        polar_x_labels = np.linspace(-90, 90, len(gaus_pl_distn))
        axes[1, 1].plot(polar_x_labels, gaus_pl_distn, 'b-', label='Original')
        axes[1, 1].set_title(f'Polar (θ) Distribution - conf={outs["confidence"]:.2f}')
        axes[1, 1].set_xlabel('Angle (degrees)')
        axes[1, 1].set_ylabel('Probability')

        roll_range = orient_any.model_config['ro_range']
        roll_x_labels = np.linspace(-roll_range // 2, roll_range // 2, len(gaus_ro_distn))
        axes[1, 2].plot(roll_x_labels, gaus_ro_distn, 'b-', label='Original')
        axes[1, 2].set_title(f'Roll (δ) Distribution - conf={outs["confidence"]:.2f}')
        axes[1, 2].set_xlabel('Angle (degrees)')
        axes[1, 2].set_ylabel('Probability')

        # Add confidence-scaled distributions in red
        conf_scaled_ax_logits = get_conf_temp_scaled_logits(gaus_ax_logits, outs['confidence'], drop_exp_factor=4)
        conf_scaled_pl_logits = get_conf_temp_scaled_logits(gaus_pl_logits, outs['confidence'], drop_exp_factor=4)
        conf_scaled_ro_logits = get_conf_temp_scaled_logits(gaus_ro_logits, outs['confidence'], drop_exp_factor=4)

        conf_scaled_ax_distn = F.softmax(conf_scaled_ax_logits, dim=-1).numpy()
        conf_scaled_pl_distn = F.softmax(conf_scaled_pl_logits, dim=-1).numpy()
        conf_scaled_ro_distn = F.softmax(conf_scaled_ro_logits, dim=-1).numpy()

        axes[1, 0].plot(azimuth_x_labels, conf_scaled_ax_distn, 'r-', label='Conf-scaled')
        axes[1, 0].legend()
        axes[1, 1].plot(polar_x_labels, conf_scaled_pl_distn, 'r-', label='Conf-scaled')
        axes[1, 1].legend()
        axes[1, 2].plot(roll_x_labels, conf_scaled_ro_distn, 'r-', label='Conf-scaled')
        axes[1, 2].legend()
    else:
        axes[0, 2].imshow(np.zeros_like(np.array(origin_img)))
        axes[0, 2].set_title("No OrientAny (no foreground)")

        # Empty second row when no foreground
        for j in range(3):
            axes[1, j].imshow(np.zeros_like(np.array(origin_img)))
            axes[1, j].set_title("No Distribution (no foreground)")
            axes[1, j].axis('off')

    axes[0, 2].axis('off')
    plt.tight_layout()
    plt.show()
