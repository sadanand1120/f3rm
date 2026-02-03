#!/usr/bin/env python3

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


def create_masks_from_rgba(images_dir, mask_dir):
    """Extract alpha channel from RGBA images and create masks"""
    mask_dir.mkdir(exist_ok=True)

    # Process each RGBA image in the dataset
    for img_path in sorted(images_dir.glob("*.png")):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:  # RGBA
            alpha = img[:, :, 3]
            # Convert: alpha=0 → 0 (black/ignored), alpha>0 → 255 (white/valid)
            mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
            # Use same filename as image but in masks directory
            mask_path = mask_dir / img_path.name
            cv2.imwrite(str(mask_path), mask)


def update_transforms_json(transforms_path):
    """Add mask_path entries to transforms.json"""
    with open(transforms_path, 'r') as f:
        data = json.load(f)

    for frame in data["frames"]:
        img_path = Path(frame["file_path"])
        # Use same filename as image but in masks directory
        mask_path = f"./masks/{img_path.name}"
        frame["mask_path"] = mask_path

    with open(transforms_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset directory (contains transforms.json)")
    args = parser.parse_args()

    dataset_dir = Path(args.data)
    images_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    transforms_path = dataset_dir / "transforms.json"

    create_masks_from_rgba(images_dir, mask_dir)
    update_transforms_json(transforms_path)
    print(f"Created masks in {mask_dir} and updated {transforms_path}")
