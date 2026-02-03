import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm

from f3rm.features.sam2_extract import SAM2Args, make_sam2_extractor, extract_sam2_features
from sam2.features.utils import SAM2utils
from f3rm.features.orientany.orientany_main import OrientAny
from f3rm.features.orientany.homography import Homography

IMAGE_PATH = "f3rm/features/orientany/tt2.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
origin_img = Image.open(IMAGE_PATH).convert('RGB')
print(f"Image size: {origin_img.size}")

print("Extracting SAM2 auto-masks...")
auto_masks_per_image = extract_sam2_features([IMAGE_PATH], device=device, verbose=True)
auto_masks = auto_masks_per_image[0]

inst_mask, _ = SAM2utils.auto_masks_to_instance_mask(
    auto_masks,
    min_iou=float(SAM2Args.pred_iou_thresh) if SAM2Args.pred_iou_thresh is not None else 0.0,
    min_area=float(SAM2Args.min_mask_region_area) if SAM2Args.min_mask_region_area is not None else 0.0,
    assign_by="area",
    start_from="low",
)

if inst_mask is None:
    print("No valid instances found")
    exit()

instance_ids = np.unique(inst_mask)
instance_ids = instance_ids[instance_ids > 0]
print(f"Found {len(instance_ids)} instances: {instance_ids}")

total_pixels = origin_img.width * origin_img.height
filtered_instance_ids = []
for inst_id in instance_ids:
    instance_pixels = np.sum(inst_mask == inst_id)
    instance_percent = (instance_pixels / total_pixels) * 100
    if instance_percent > 10:
        filtered_instance_ids.append(inst_id)
        print(f"Instance {inst_id}: {instance_pixels} pixels ({instance_percent:.1f}% of image)")
    else:
        print(f"Instance {inst_id}: {instance_pixels} pixels ({instance_percent:.1f}% of image) - SKIPPED")

instance_ids = filtered_instance_ids
print(f"Processing {len(instance_ids)} instances after size filtering")

print("Visualizing SAM2 masks...")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(origin_img)
axes[0].set_title("Original Image")
axes[0].axis('off')

if inst_mask is not None:
    viz_mask, cmap, norm = SAM2utils.make_viz_mask_and_cmap(inst_mask)
    axes[1].imshow(viz_mask, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title(f"Instance Mask ({len(instance_ids)} instances)")
else:
    axes[1].imshow(np.zeros_like(np.array(origin_img)))
    axes[1].set_title("No instances found")
axes[1].axis('off')

plt.tight_layout()
plt.show()

orient_any = OrientAny("f3rm/features/orientany/ckpts", "ronormsigma1_dino_weight.pt")

for inst_id in tqdm(instance_ids, desc="Processing instances"):
    instance_mask = (inst_mask == inst_id).astype(np.uint8)

    img_array = np.array(origin_img.copy())
    instance_img_array = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
    instance_img_array[..., :3] = img_array * instance_mask[..., None]
    instance_img_array[..., 3] = instance_mask * 255
    instance_img = Image.fromarray(instance_img_array, 'RGBA')
    plt.imshow(instance_img.convert('RGB'))
    plt.show()

    rm_bkg_img = orient_any.preprocess_remove_bkg(instance_img, do_remove_background=False)
    plt.imshow(rm_bkg_img.convert('RGB'))
    plt.show()
    outs = orient_any.get_model_outputs(rm_bkg_img, viz_distn=True)
    R_objw2cam = orient_any.get_R_objw2cam(outs['phi'], outs['theta_elev'], outs['delta'])

    print(f"Instance {inst_id}: φ={outs['phi']:.1f}°, θ={outs['theta_elev']:.1f}°, δ={outs['delta']:.1f}°, conf={outs['confidence']:.2f}")
    result_img, _ = orient_any.draw_axes_on_image(rm_bkg_img, R_objw2cam, radius=16, axes_len=2)
    plt.imshow(result_img)
    plt.title(f"Instance {inst_id}: φ={outs['phi']:.1f}°, θ={outs['theta_elev']:.1f}°, δ={outs['delta']:.1f}°")
    plt.axis('off')
    plt.show()
