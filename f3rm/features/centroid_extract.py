import asyncio
import gc
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from sam2.features.utils import AsyncMultiWrapper

from f3rm.features.utils import (
    BatchFeatureLoader,
    compose_prefixed_feature_type,
    ensure_sam3_feature_cache,
    infer_sam3_feature_type,
    load_sam3_image_fnames,
    normalize_sam3_masks,
    parse_prefixed_feature_type,
    resolve_devices_and_workers,
    run_async_in_any_context,
)


FACTOR_SAM3D_P3D_TO_NERF_GL = np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)
CENTROID_FILL_VALUE = np.asarray([-99.0, -99.0, -99.0, -99.0], dtype=np.float16)


class CENTROIDArgs:
    batch_size_per_gpu: int = 32

    @classmethod
    def id_dict(cls):
        return {}


def parse_centroid_feature_type(feature_type: str) -> List[str]:
    return parse_prefixed_feature_type(feature_type, "CENTROID_")


def infer_sam3d_feature_type(text_prompts: Optional[List[str]]) -> str:
    return compose_prefixed_feature_type("SAM3D_", text_prompts)


class CENTROIDWorker:
    def __init__(
        self,
        device: torch.device,
        data_dir: Path,
        sam3_feature_type: str = "SAM3_",
        sam3d_feature_type: str = "SAM3D_",
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.data_dir = Path(data_dir)
        self.sam3_feature_type = sam3_feature_type
        self.sam3d_feature_type = sam3d_feature_type

        self.feat_image_fnames = load_sam3_image_fnames(self.data_dir, self.sam3_feature_type)
        self._image_to_index = {fname: i for i, fname in enumerate(self.feat_image_fnames)}
        self.sam3_loader = BatchFeatureLoader(self.data_dir, self.sam3_feature_type, self.feat_image_fnames, self.device)
        self.sam3d_root = self.data_dir / "features" / self.sam3d_feature_type.lower()

    @staticmethod
    def _load_centroid_gl(instance_record: dict) -> np.ndarray:
        centroid_path = Path(instance_record["centroid_path"])
        centroid_meta = json.loads(centroid_path.read_text())
        centroid_p3d = np.asarray(centroid_meta["centroid_sam3d_p3d"], dtype=np.float32).reshape(3)
        return centroid_p3d * FACTOR_SAM3D_P3D_TO_NERF_GL

    def _compute_centroid_for_image(self, image_path: str) -> np.ndarray:
        try:
            idx = self._image_to_index[str(image_path)]
        except KeyError as exc:
            raise ValueError(f"Image path not found in SAM3 meta order: {image_path}") from exc

        raw_masks = self.sam3_loader[idx]
        masks = normalize_sam3_masks(raw_masks, image_path=image_path)
        h, w = masks.shape[-2:]
        centroid_map = np.broadcast_to(CENTROID_FILL_VALUE, (h, w, 4)).copy()

        manifest_path = self.sam3d_root / Path(image_path).stem / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing SAM3D manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())

        for instance_record in manifest["instances"]:
            sam3_mask_index = int(instance_record["mask_index"])
            if sam3_mask_index >= masks.shape[0]:
                continue
            centroid_path = instance_record.get("centroid_path")
            if not centroid_path or not Path(centroid_path).exists():
                continue
            centroid_gl = self._load_centroid_gl(instance_record).astype(np.float16)
            packed_value = np.concatenate(
                [np.asarray([float(sam3_mask_index)], dtype=np.float16), centroid_gl],
                axis=0,
            )
            centroid_map[masks[sam3_mask_index]] = packed_value

        return centroid_map

    async def compute_centroid_for_image_async(self, image_path: str) -> np.ndarray:
        return await asyncio.to_thread(self._compute_centroid_for_image, image_path)


class CENTROIDExtractor:
    def __init__(
        self,
        device: torch.device,
        data_dir: Optional[Path] = None,
        text_prompts: Optional[List[str]] = None,
        sam3_feature_type: Optional[str] = None,
        sam3d_feature_type: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.verbose = verbose
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.sam3_feature_type = sam3_feature_type or infer_sam3_feature_type(text_prompts)
        self.sam3d_feature_type = sam3d_feature_type or infer_sam3d_feature_type(text_prompts)

        if self.data_dir is None:
            raise ValueError("CENTROIDExtractor requires data_dir to locate precomputed SAM3/SAM3D shards")

        ensure_sam3_feature_cache(self.data_dir, self.sam3_feature_type, consumer="CENTROID", require_npz=True)
        if not (self.data_dir / "features" / self.sam3d_feature_type.lower()).exists():
            raise FileNotFoundError(
                f"Missing SAM3D feature root: {self.data_dir / 'features' / self.sam3d_feature_type.lower()}"
            )

        devices_param, num_workers = resolve_devices_and_workers(device, CENTROIDArgs.batch_size_per_gpu)
        if verbose:
            print(
                f"Initializing CENTROID workers (using {self.sam3_feature_type} masks, "
                f"{self.sam3d_feature_type} centroids)"
            )
        self.client = AsyncMultiWrapper(
            CENTROIDWorker,
            num_objects=num_workers,
            devices=devices_param,
            data_dir=self.data_dir,
            sam3_feature_type=self.sam3_feature_type,
            sam3d_feature_type=self.sam3d_feature_type,
        )
        self.num_workers = num_workers

    async def extract_batch_async(self, image_paths: List[str]) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting CENTROID maps", leave=False):
            batch_paths = image_paths[i:i + self.num_workers]
            tasks = [process_single_image_centroid_async(path, self.client) for path in batch_paths]
            batch_results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="CENTROID", leave=False)
            results.extend(batch_results)
            gc.collect()
        return results


async def process_single_image_centroid_async(image_path: str, centroid_client: AsyncMultiWrapper) -> np.ndarray:
    return await centroid_client.compute_centroid_for_image_async(image_path)


if __name__ == "__main__":
    data_root = Path("datasets/f3rm/opt/betaipad/small")
    image_dir = data_root / "images"
    image_paths = [str(p) for p in sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))[:8]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = CENTROIDExtractor(device=device, data_dir=data_root, text_prompts=None, verbose=True)
    maps = run_async_in_any_context(lambda: extractor.extract_batch_async(image_paths))
    print(f"Extracted {len(maps)} centroid maps, sample shape: {maps[0].shape if maps else None}")
