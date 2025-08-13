import gc
import asyncio
import glob
from typing import List

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from sam2.features.client.dino_client import DINOFeaturesUnified
from sam2.features.utils import AsyncMultiWrapper, apply_pca_colormap
from f3rm.features.utils import resolve_devices_and_workers, run_async_in_any_context


class DINOArgs:
    model_type: str = "dinov2_vitl14"
    load_size: int = 2048  # -1 to use smallest side size, -x to use scaled of smallest side
    stride: int = None
    facet: str = "token"
    layer: int = -1
    bin: bool = False
    batch_size_per_gpu: int = 4

    @classmethod
    def id_dict(cls):
        return {
            "model_type": cls.model_type,
            "load_size": cls.load_size,
            "stride": cls.stride,
            "facet": cls.facet,
            "layer": cls.layer,
            "bin": cls.bin,
        }


class DINOExtractor:
    def __init__(self, device: torch.device, verbose: bool = False) -> None:
        devices_param, num_workers = resolve_devices_and_workers(device, DINOArgs.batch_size_per_gpu)
        if verbose:
            print("Initializing DINO client")
        self.client = AsyncMultiWrapper(DINOFeaturesUnified, num_objects=num_workers, devices=devices_param)
        self.num_workers = num_workers
        if verbose:
            print("Warming up DINO workers...")
        tiny = Image.new("RGB", (8, 8), color=0)
        for _ in range(self.num_workers):
            _ = self.client.extract_features(
                image=tiny,
                model_type=DINOArgs.model_type,
                stride=DINOArgs.stride,
                ret_pca=False,
                ret_patches=True,
                load_size=32,
                facet=DINOArgs.facet,
                layer=DINOArgs.layer,
                bin=DINOArgs.bin,
                interpolation_mode="bilinear",
                tensor_format="HWC",
                padding_mode="constant",
            )

    async def extract_batch_async(self, image_paths: List[str]) -> torch.Tensor:
        batches = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting DINO features", leave=False):
            chunk = image_paths[i:i + self.num_workers]
            tasks = [
                self.client.extract_features_async(
                    image=path,
                    model_type=DINOArgs.model_type,
                    stride=DINOArgs.stride,
                    ret_pca=False,
                    ret_patches=True,
                    load_size=DINOArgs.load_size,
                    facet=DINOArgs.facet,
                    layer=DINOArgs.layer,
                    bin=DINOArgs.bin,
                    interpolation_mode="bilinear",
                    tensor_format="HWC",
                    padding_mode="constant",
                ) for path in chunk
            ]
            results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="DINO tasks", leave=False)
            batch_tensor = torch.stack([r.cpu() for r in results], dim=0)
            batches.append(batch_tensor)
            gc.collect()
        return torch.cat(batches, dim=0) if batches else torch.empty(0)


def make_dino_extractor(device: torch.device, verbose: bool = False) -> DINOExtractor:
    return DINOExtractor(device=device, verbose=verbose)


def extract_dino_features(image_paths: List[str], device: torch.device, verbose=False) -> torch.Tensor:
    extractor = make_dino_extractor(device, verbose=verbose)
    return run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))


if __name__ == "__main__":
    image_dir = "datasets/f3rm/panda/scene_001/images"
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png"))
    image_paths = image_paths[:4]
    print(f"Found {len(image_paths)} images in {image_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = extract_dino_features(image_paths, device=device, verbose=True)
    print(f"DINO features shape: {feats.shape}")
    if feats.numel() > 0:
        pca_img = apply_pca_colormap(feats[0], niter=5, q_min=0.01, q_max=0.99)
        plt.figure(figsize=(6, 6))
        plt.imshow(pca_img.cpu().numpy())
        plt.title("DINO PCA Visualization (first image)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
