import gc
import asyncio
import glob
from typing import List, Optional

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from sam2.features.client.clip_client import CLIPFeaturesUnified
from sam2.features.utils import AsyncMultiWrapper, apply_pca_colormap
from f3rm.features.utils import resolve_devices_and_workers, run_async_in_any_context


class CLIPArgs:
    model_name: str = "ViT-L-14-336-quickgelu"   # open_clip.list_pretrained() lists all available models
    model_pretrained: str = "openai"
    load_size: int = 2048
    skip_center_crop: bool = True
    batch_size_per_gpu: int = 4
    agg_scales: List[float] = [0.25, 0.5, 1.0, 1.5]
    agg_weights: Optional[List[float]] = [1.5, 3, 6, 3]

    @classmethod
    def id_dict(cls):
        return {
            "model_name": cls.model_name,
            "model_pretrained": cls.model_pretrained,
            "load_size": cls.load_size,
            "skip_center_crop": cls.skip_center_crop,
            "agg_scales": cls.agg_scales,
            "agg_weights": cls.agg_weights,
        }


class CLIPExtractor:
    def __init__(self, device: torch.device, verbose: bool = False) -> None:
        devices_param, num_workers = resolve_devices_and_workers(device, CLIPArgs.batch_size_per_gpu)
        if verbose:
            print("Initializing CLIP client")
        self.client = AsyncMultiWrapper(CLIPFeaturesUnified, num_objects=num_workers, devices=devices_param)
        self.num_workers = num_workers
        if verbose:
            print("Warming up CLIP workers...")
        tiny = Image.new("RGB", (8, 8), color=0)
        for _ in range(self.num_workers):
            _ = self.client.extract_features_agg(
                image=tiny,
                model_name=CLIPArgs.model_name,
                model_pretrained=CLIPArgs.model_pretrained,
                agg_scales=[1.0],
                agg_weights=None,
                ret_pca=False,
                ret_patches=True,
                load_size=32,
                center_crop=False,
                interpolation_mode="bilinear",
                tensor_format="HWC",
                padding_mode="constant",
                return_meta=False,
                ret_internal_feats=False,
            )

    async def extract_batch_async(self, image_paths: List[str]) -> torch.Tensor:
        batches = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting CLIP features", leave=False):
            chunk = image_paths[i:i + self.num_workers]
            tasks = [
                self.client.extract_features_agg_async(
                    image=path,
                    model_name=CLIPArgs.model_name,
                    model_pretrained=CLIPArgs.model_pretrained,
                    agg_scales=CLIPArgs.agg_scales,
                    agg_weights=CLIPArgs.agg_weights,
                    ret_pca=False,
                    ret_patches=True,
                    load_size=CLIPArgs.load_size,
                    center_crop=not CLIPArgs.skip_center_crop,
                    interpolation_mode="bilinear",
                    tensor_format="HWC",
                    padding_mode="constant",
                    return_meta=False,
                    ret_internal_feats=False,
                )
                for path in chunk
            ]
            results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="CLIP tasks", leave=False)
            batch_tensor = torch.stack([r.cpu() for r in results], dim=0)
            batches.append(batch_tensor)
            gc.collect()
        return torch.cat(batches, dim=0) if batches else torch.empty(0)


def make_clip_extractor(device: torch.device, verbose: bool = False) -> CLIPExtractor:
    return CLIPExtractor(device=device, verbose=verbose)


def extract_clip_features(image_paths: List[str], device: torch.device, verbose=False) -> torch.Tensor:
    extractor = make_clip_extractor(device, verbose=verbose)
    return run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))


if __name__ == "__main__":
    image_dir = "datasets/f3rm/panda/scene_001/images"
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png"))
    image_paths = image_paths[:4]
    print(f"Found {len(image_paths)} images in {image_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = extract_clip_features(image_paths, device=device, verbose=True)
    print(f"CLIP features shape: {feats.shape}")
    if feats.numel() > 0:
        pca_img = apply_pca_colormap(feats[0], niter=5, q_min=0.01, q_max=0.99)
        plt.figure(figsize=(6, 6))
        plt.imshow(pca_img.cpu().numpy())
        plt.title("CLIP PCA Visualization (first image)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
