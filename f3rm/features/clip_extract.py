import asyncio
import gc
import glob
import os
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Sequence, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm

from f3rm.features.utils import AsyncMultiWrapper, apply_pca_colormap, resolve_devices_and_workers, run_async_in_any_context


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _emit_timing(name: str, duration: float, enabled: bool) -> None:
    if enabled:
        print(f"[F3RM_TIMING] {name}={duration:.6f}")


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


def _resolve_resized_hw(width: int, height: int, size: int) -> tuple[int, int]:
    if width <= height:
        new_width = size
        new_height = max(1, int(round(height * size / width)))
    else:
        new_height = size
        new_width = max(1, int(round(width * size / height)))
    return new_height, new_width


def _load_rgb_tensor(image: Union[str, Path, Image.Image]) -> torch.Tensor:
    if isinstance(image, (str, Path)):
        return read_image(str(image), mode=ImageReadMode.RGB).to(torch.float32).div_(255.0)
    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)
    raise TypeError("image must be a path or PIL image")


def _pad_to_multiple_bchw(batch: torch.Tensor, patch_size: int, mode: str = "constant") -> torch.Tensor:
    pad_h = (-batch.shape[-2]) % patch_size
    pad_w = (-batch.shape[-1]) % patch_size
    if pad_h == 0 and pad_w == 0:
        return batch
    if mode == "constant":
        return F.pad(batch, (0, pad_w, 0, pad_h), mode=mode, value=0.0)
    return F.pad(batch, (0, pad_w, 0, pad_h), mode=mode)


def _preprocess_rgb(
    image: Union[str, Path, Image.Image],
    load_size: Optional[int],
    center_crop: bool,
    patch_size: int,
    padding_mode: str,
) -> torch.Tensor:
    tensor = _load_rgb_tensor(image)
    if load_size is not None:
        height, width = tensor.shape[-2:]
        resolved_size = abs(load_size) * min(width, height) if load_size < 0 else load_size
        resized_height, resized_width = _resolve_resized_hw(width, height, int(resolved_size))
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        if center_crop:
            crop_height = min(resized_height, int(resolved_size))
            crop_width = min(resized_width, int(resolved_size))
            top = max((resized_height - crop_height) // 2, 0)
            left = max((resized_width - crop_width) // 2, 0)
            tensor = tensor[:, top:top + crop_height, left:left + crop_width]

    mean = tensor.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(CLIP_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return _pad_to_multiple_bchw(tensor.unsqueeze(0), patch_size, mode=padding_mode).squeeze(0)


def _interpolate_positional_embedding(
    positional_embedding: torch.Tensor,
    x: torch.Tensor,
    patch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if positional_embedding.ndim != 2:
        raise ValueError("Expected 2-D positional_embedding")

    num_patches = x.shape[1] - 1
    num_original_patches = positional_embedding.shape[0] - 1
    if num_patches == num_original_patches and height == width:
        return positional_embedding.to(x.dtype)

    dim = x.shape[-1]
    class_pos_embed = positional_embedding[:1]
    patch_pos_embed = positional_embedding[1:]
    grid_h = height // patch_size
    grid_w = width // patch_size
    if grid_h * grid_w != num_patches:
        raise ValueError("Number of patches does not match positional embedding interpolation target")

    grid_h_f = grid_h + 0.1
    grid_w_f = grid_w + 0.1
    patch_per_axis = int(np.sqrt(num_original_patches))
    patch_pos_embed_interp = F.interpolate(
        patch_pos_embed.reshape(1, patch_per_axis, patch_per_axis, dim).permute(0, 3, 1, 2),
        scale_factor=(grid_h_f / patch_per_axis, grid_w_f / patch_per_axis),
        mode="bicubic",
        align_corners=False,
        recompute_scale_factor=False,
    )
    if int(grid_h_f) != patch_pos_embed_interp.shape[-2] or int(grid_w_f) != patch_pos_embed_interp.shape[-1]:
        raise ValueError("Positional embedding interpolation failed")

    patch_pos_embed_interp = patch_pos_embed_interp.permute(0, 2, 3, 1).reshape(-1, dim)
    return torch.cat([class_pos_embed, patch_pos_embed_interp], dim=0).to(x.dtype)


class _CLIPWorker(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-L-14-336-quickgelu",
        pretrained: str = "openai",
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Please install open-clip-torch") from exc

        requested_device = torch.device(device)
        self.device = torch.device(requested_device if requested_device.type != "cuda" or torch.cuda.is_available() else "cpu")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision="fp32",
        )
        self.model = model.eval().to(self.device)

        visual = self.model.visual
        required_attrs = [
            "conv1",
            "class_embedding",
            "positional_embedding",
            "ln_pre",
            "transformer",
            "ln_post",
            "proj",
            "patch_size",
        ]
        missing = [name for name in required_attrs if not hasattr(visual, name)]
        if missing:
            raise RuntimeError(
                f"OpenCLIP visual tower missing expected attrs {missing}. "
                "This extractor expects a ViT-style OpenCLIP visual encoder."
            )

    @property
    def patch_size(self) -> int:
        patch_size = self.model.visual.patch_size
        if isinstance(patch_size, int):
            return int(patch_size)
        return int(patch_size[0])

    def _get_patch_encodings(self, image_batch: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual
        _, _, height, width = image_batch.shape
        x = visual.conv1(image_batch)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(x.dtype)
        class_token = class_token + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_token, x], dim=1)
        x = x + _interpolate_positional_embedding(
            visual.positional_embedding,
            x,
            patch_size=self.patch_size,
            height=height,
            width=width,
        )
        x = visual.ln_pre(x)
        *layers, last_resblock = visual.transformer.resblocks
        if layers:
            x = torch.nn.Sequential(*layers)(x)
        v_in_proj_weight = last_resblock.attn.in_proj_weight[-last_resblock.attn.embed_dim:]
        v_in_proj_bias = last_resblock.attn.in_proj_bias[-last_resblock.attn.embed_dim:]
        v_in = F.linear(last_resblock.ln_1(x), v_in_proj_weight, v_in_proj_bias)
        x = F.linear(v_in, last_resblock.attn.out_proj.weight, last_resblock.attn.out_proj.bias)
        x = x[:, 1:, :]
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        return x

    @torch.inference_mode()
    def encode_dense(self, batch: torch.Tensor, return_device_tensor: bool = False) -> torch.Tensor:
        batch = batch.to(torch.float32)
        batch = batch.to(device=self.device, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                patch_tokens = self._get_patch_encodings(batch)
        else:
            patch_tokens = self._get_patch_encodings(batch)
        gh = batch.shape[-2] // self.patch_size
        gw = batch.shape[-1] // self.patch_size
        patch_tokens = patch_tokens.reshape(batch.shape[0], gh, gw, -1).float()
        if return_device_tensor:
            return patch_tokens
        return patch_tokens.cpu()

    def _extract_single_scale(
        self,
        image: Union[str, Path, Image.Image],
        load_size: Optional[int],
        center_crop: bool,
        padding_mode: str,
    ) -> torch.Tensor:
        batch = _preprocess_rgb(
            image=image,
            load_size=load_size,
            center_crop=center_crop,
            patch_size=self.patch_size,
            padding_mode=padding_mode,
        ).unsqueeze(0)
        return self.encode_dense(batch, return_device_tensor=True)[0]

    @torch.inference_mode()
    def extract_agg(
        self,
        image: Union[str, Path, Image.Image],
        agg_scales: Optional[Sequence[float]] = None,
        agg_weights: Optional[Sequence[float]] = None,
        load_size: Optional[int] = 1024,
        center_crop: bool = False,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "constant",
    ) -> torch.Tensor:
        scales = list(agg_scales or [1.0])
        if 1.0 not in scales:
            raise ValueError("agg_scales must include 1.0 as the reference scale")
        if agg_weights is not None and len(agg_weights) != len(scales):
            raise ValueError("agg_weights must match agg_scales length")

        weights = [float(weight) for weight in agg_weights] if agg_weights is not None else [1.0] * len(scales)
        ref_idx = scales.index(1.0)
        ref_patch = self._extract_single_scale(
            image=image,
            load_size=load_size,
            center_crop=center_crop,
            padding_mode=padding_mode,
        )
        h_ref, w_ref, d_ref = ref_patch.shape
        agg_tensor = torch.zeros((d_ref, h_ref, w_ref), device=ref_patch.device, dtype=ref_patch.dtype)

        ref_weight = float(weights[ref_idx])
        if ref_weight > 0.0:
            agg_tensor.add_(ref_patch.permute(2, 0, 1), alpha=ref_weight)
        weight_accum = max(ref_weight, 0.0)

        for idx, scale in enumerate(scales):
            if idx == ref_idx:
                continue
            if load_size is None:
                raise ValueError("Multi-scale aggregation requires a concrete load_size")
            scaled_load_size = int(round(load_size * float(scale)))
            if scaled_load_size <= 0:
                continue
            patch_features = self._extract_single_scale(
                image=image,
                load_size=scaled_load_size,
                center_crop=center_crop,
                padding_mode=padding_mode,
            )
            patch_features = patch_features.permute(2, 0, 1).unsqueeze(0)
            if interpolation_mode in ("bilinear", "bicubic", "trilinear"):
                patch_features = F.interpolate(
                    patch_features,
                    size=(h_ref, w_ref),
                    mode=interpolation_mode,
                    align_corners=False,
                )
            else:
                patch_features = F.interpolate(patch_features, size=(h_ref, w_ref), mode=interpolation_mode)
            weight = float(weights[idx])
            if weight > 0.0:
                agg_tensor.add_(patch_features.squeeze(0), alpha=weight)
                weight_accum += weight

        if weight_accum <= 0.0:
            raise ValueError("Sum of weights must be > 0")
        return (agg_tensor / weight_accum).permute(1, 2, 0).contiguous().cpu()

    async def extract_agg_async(self, **kwargs) -> torch.Tensor:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.extract_agg(**kwargs))


class CLIPExtractor:
    def __init__(self, device: torch.device, verbose: bool = False) -> None:
        devices_param, num_workers = resolve_devices_and_workers(device, CLIPArgs.batch_size_per_gpu)
        if verbose:
            print("Initializing CLIP workers")
            print(f"[F3RM_INFO] extract.num_workers={num_workers}")
        init_start = perf_counter()
        self.client = AsyncMultiWrapper(
            _CLIPWorker,
            num_objects=num_workers,
            devices=devices_param,
            model_name=CLIPArgs.model_name,
            pretrained=CLIPArgs.model_pretrained,
        )
        self.num_workers = num_workers
        _emit_timing("extract.worker_init_s", perf_counter() - init_start, verbose)
        if verbose:
            print("Warming up CLIP workers...")
        warmup_start = perf_counter()
        tiny = Image.new("RGB", (8, 8), color=0)
        for _ in range(self.num_workers):
            _ = self.client.extract_agg(
                image=tiny,
                agg_scales=[1.0],
                agg_weights=None,
                load_size=32,
                center_crop=False,
                interpolation_mode="bilinear",
                padding_mode="constant",
            )
        _emit_timing("extract.worker_warmup_s", perf_counter() - warmup_start, verbose)

    async def _extract_batch_async(self, image_paths: List[str]) -> torch.Tensor:
        batches = []
        for i in tqdm(range(0, len(image_paths), self.num_workers), desc="Extracting CLIP features", leave=False):
            chunk = image_paths[i:i + self.num_workers]
            tasks = [
                self.client.extract_agg_async(
                    image=path,
                    agg_scales=CLIPArgs.agg_scales,
                    agg_weights=CLIPArgs.agg_weights,
                    load_size=CLIPArgs.load_size,
                    center_crop=not CLIPArgs.skip_center_crop,
                    interpolation_mode="bilinear",
                    padding_mode="constant",
                )
                for path in chunk
            ]
            results = await AsyncMultiWrapper.async_run_tasks(tasks, desc="CLIP tasks", leave=False)
            batches.append(torch.stack([result.cpu() for result in results], dim=0))
            gc.collect()
        return torch.cat(batches, dim=0) if batches else torch.empty(0)

    def extract_batch(self, image_paths: List[str]) -> torch.Tensor:
        return run_async_in_any_context(lambda: self._extract_batch_async(image_paths))


def examine_saved(clip_feat_dir: str):
    """Create .mp4 video of saved CLIP features with PCA visualization."""
    meta_path = os.path.join(clip_feat_dir, "meta.pt")
    assert os.path.exists(meta_path), f"CLIP meta not found at {meta_path}"

    meta = torch.load(meta_path)
    image_fnames = meta["image_fnames"]
    n_images = len(image_fnames)

    first_feat = np.load(os.path.join(clip_feat_dir, "image_000000.npy"))
    H, W = first_feat.shape[:2]

    video_path = os.path.join(clip_feat_dir, "features_viz.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 2.0, (W, H))

    for i in tqdm(range(n_images), desc="Creating CLIP features video"):
        feat_path = os.path.join(clip_feat_dir, f"image_{i:06d}.npy")
        feat = torch.from_numpy(np.load(feat_path)).float()
        pca_img = apply_pca_colormap(feat, niter=5, q_min=0.01, q_max=0.99)
        frame = (pca_img.cpu().numpy() * 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    assert os.path.exists(video_path), f"Video not created at {video_path}"


if __name__ == "__main__":
    image_dir = "datasets/f3rm/panda/scene_001/images"
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png"))[:4]
    print(f"Found {len(image_paths)} images in {image_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = CLIPExtractor(device=device, verbose=True).extract_batch(image_paths)
    print(f"CLIP features shape: {tuple(feats.shape)}")

    pca_img = apply_pca_colormap(feats[0], niter=5, q_min=0.01, q_max=0.99)
    plt.figure(figsize=(6, 6))
    plt.imshow(pca_img.cpu().numpy())
    plt.title("CLIP PCA Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
