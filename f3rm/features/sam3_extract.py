import gc
import asyncio
import glob
import os
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from sam2.features.utils import AsyncMultiWrapper
from sam3.sam3_main import SAM3Main

from f3rm.features.utils import BatchFeatureLoader, resolve_devices_and_workers, run_async_in_any_context


class SAM3Args:
    bpe_path: str = "/robodata/smodak/repos/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    confidence_threshold: float = 0.5
    resolution: int = 1008  # multiple of 168
    precision: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    compile: bool = False
    batch_size_per_gpu: int = 2

    @classmethod
    def id_dict(cls):
        return {
            "bpe_path": cls.bpe_path,
            "confidence_threshold": float(cls.confidence_threshold),
            "resolution": int(cls.resolution),
            "precision": str(cls.precision),
            "compile": bool(cls.compile),
        }


def parse_sam3_feature_type(feature_type: str) -> List[str]:
    if not feature_type.startswith("SAM3_"):
        raise ValueError(f"Invalid SAM3 feature type: {feature_type}. Must start with 'SAM3_'")
    prompts_part = feature_type[len("SAM3_"):]
    if prompts_part == "":
        return []
    return [w.lower() for w in prompts_part.split("_") if w.strip()]


class SAM3Worker:
    def __init__(self, device: torch.device, data_dir: Path, text_prompts: Optional[List[str]] = None):
        self.device = torch.device(device)
        self.data_dir = Path(data_dir)
        self.text_prompts = text_prompts

        text_root = self.data_dir / "features" / "text"
        if self.text_prompts is None:
            meta_path = text_root / "meta.pt"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing TEXT meta: {meta_path}")
            meta = torch.load(meta_path)
            self.feat_image_fnames = [str(p) for p in meta["image_fnames"]]
            if not list(text_root.glob("image_*.json")):
                raise FileNotFoundError(f"Missing TEXT per-image files under {text_root}")
            self.text_loader = BatchFeatureLoader(self.data_dir, "TEXT", self.feat_image_fnames, self.device)
        else:
            self.text_loader = None
            self.feat_image_fnames = None

        # Build SAM3 model once per worker
        self.sam3 = SAM3Main(
            bpe_path=SAM3Args.bpe_path,
            confidence_threshold=SAM3Args.confidence_threshold,
            resolution=SAM3Args.resolution,
            precision=SAM3Args.precision,
            device=self.device,
            compile=SAM3Args.compile,
        )

    @torch.inference_mode()
    def _compute_masks_for_image(self, image_path: str) -> np.ndarray:
        if self.text_loader is not None:
            try:
                idx = self.feat_image_fnames.index(str(image_path))
            except ValueError:
                raise ValueError(f"Image path not found in TEXT meta order: {image_path}")
            prompts = self.text_loader[idx]
        else:
            prompts = self.text_prompts

        if not prompts:
            img = Image.open(image_path)
            return np.zeros((0, 1, img.height, img.width), dtype=np.bool_)

        mask_tensors = []
        base_shape = None
        for prompt in prompts:
            masks, _ = self.sam3.process_image(image_path=image_path, prompt=prompt)
            if masks is None:
                continue
            if isinstance(masks, list):
                if len(masks) == 0:
                    continue
                mask_tensor = torch.stack(masks, dim=0)
            else:
                mask_tensor = masks
            if mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(1)
            mask_tensor = mask_tensor.to("cpu").to(torch.bool)
            if base_shape is None:
                base_shape = mask_tensor.shape[-2:]
            mask_tensors.append(mask_tensor)

        if not mask_tensors:
            if base_shape is None:
                img = Image.open(image_path)
                base_shape = (img.height, img.width)
            h, w = base_shape
            return np.zeros((0, 1, h, w), dtype=np.bool_)

        stacked = torch.cat(mask_tensors, dim=0)
        return stacked.numpy()

    async def compute_masks_for_image_async(self, image_path: str) -> np.ndarray:
        return await asyncio.to_thread(self._compute_masks_for_image, image_path)


class SAM3Extractor:
    def __init__(self, device: torch.device, data_dir: Optional[Path] = None, text_prompts: Optional[List[str]] = None, verbose: bool = False) -> None:
        self.device = device
        self.verbose = verbose
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.text_prompts = text_prompts

        if self.data_dir is None:
            raise ValueError("SAM3Extractor requires data_dir to locate TEXT shards")

        if text_prompts is None:
            text_root = self.data_dir / "features" / "text"
            if not (text_root / "meta.pt").exists():
                raise FileNotFoundError(f"Missing TEXT meta: {text_root / 'meta.pt'}")
            if not list(text_root.glob("image_*.json")):
                raise FileNotFoundError(f"Missing TEXT per-image features under {text_root} (required when using per-image text prompts)")

        devices_param, num_workers = resolve_devices_and_workers(device, SAM3Args.batch_size_per_gpu)
        if verbose:
            print("Initializing SAM3 workers")
        self.client = AsyncMultiWrapper(SAM3Worker, num_objects=num_workers, devices=devices_param, data_dir=self.data_dir, text_prompts=self.text_prompts)
        self.num_workers = num_workers

    async def extract_batch_async(self, image_paths: List[str]) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting SAM3 masks", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_sam3_async(path, self.client) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="SAM3", leave=False)
            results.extend(batch_results)
            gc.collect()
        return results


async def process_single_image_sam3_async(image_path: str, sam3_client: AsyncMultiWrapper) -> np.ndarray:
    return await sam3_client.compute_masks_for_image_async(image_path)


def examine_saved(sam3_feat_dir: str):
    meta_path = os.path.join(sam3_feat_dir, "meta.pt")
    assert os.path.exists(meta_path), f"SAM3 meta not found at {meta_path}"

    meta = torch.load(meta_path)
    image_fnames = meta["image_fnames"]
    n_images = len(image_fnames)

    sample = np.load(os.path.join(sam3_feat_dir, "image_000000.npz"))
    masks = sample["masks"]
    if masks.ndim == 3:
        masks = masks[:, None, ...]
    _, _, H, W = masks.shape

    video_path = os.path.join(sam3_feat_dir, "features_viz.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 2.0, (W, H))

    for i in tqdm(range(n_images), desc="Creating SAM3 features video"):
        feat_path = os.path.join(sam3_feat_dir, f"image_{i:06d}.npz")
        data = np.load(feat_path)
        mask_stack = data["masks"]
        if mask_stack.ndim == 3:
            mask_stack = mask_stack[:, None, ...]
        masks = [seg[0] for seg in mask_stack]
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        overlaid = SAM3Main.overlay_masks_on_image(blank, masks)
        out.write(cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))

    out.release()
    assert os.path.exists(video_path), f"Video not created at {video_path}"


if __name__ == "__main__":
    data_root = Path("datasets/f3rm/opt/caterpillar")
    image_dir = data_root / "images"
    image_paths = sorted(glob.glob(str(image_dir / "*.jpg")) + glob.glob(str(image_dir / "*.png")))
    image_paths = image_paths[:6]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Found {len(image_paths)} images in {image_dir}")
    extractor = SAM3Extractor(device=device, data_dir=data_root, text_prompts=["toy"], verbose=True)
    mask_batches = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))

    print("Visualizing first two results (Demo 1)...")
    for idx in range(min(2, len(mask_batches))):
        img = Image.open(image_paths[idx]).convert("RGB")
        masks = [seg[0] if seg.ndim == 3 else seg for seg in mask_batches[idx]]
        SAM3Main.show_masks(img, masks)

    print("\n" + "=" * 50)
    print("DEMO 2: Full pipeline (save -> load -> visualize)")
    print("=" * 50)

    test_dir = Path("test_sam3_pipeline")
    test_dir.mkdir(exist_ok=True)
    for i, masks in enumerate(mask_batches):
        np.savez_compressed(test_dir / f"image_{i:06d}.npz", masks=masks.astype(np.bool_))

    reloaded = []
    for i in range(len(mask_batches)):
        data = np.load(test_dir / f"image_{i:06d}.npz")
        re_masks = data["masks"]
        reloaded.append(re_masks)

    print("Visualizing reloaded data...")
    for idx in range(min(2, len(reloaded))):
        img = Image.open(image_paths[idx]).convert("RGB")
        masks = [seg[0] if seg.ndim == 3 else seg for seg in reloaded[idx]]
        SAM3Main.show_masks(img, masks)

    plt.close('all')
    gc.collect()
    shutil.rmtree(test_dir, ignore_errors=True)
    print(f"Cleaned up test directory: {test_dir}")
