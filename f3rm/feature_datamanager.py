import concurrent.futures
import gc
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.utils.rich_utils import CONSOLE

from f3rm.features.extract_features_standalone import extract_features_for_dataset
from f3rm.ray_generator import FeatureRayGenerator


@dataclass
class FeatureDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: FeatureDataManager)
    feature_type: Literal["CLIP"] = "CLIP"
    enable_cache: bool = True
    """Whether to cache extracted features."""
    pin_cpu_feature_cache: bool = True
    cpu_feature_cache_images: int = 128
    gpu_feature_cache_images: int = 16


class UInt8InputDataset(InputDataset):
    """Train dataset variant that keeps cached images in uint8 until after pixel sampling."""

    def __getitem__(self, image_idx: int) -> Dict:
        return self.get_data(image_idx, image_type="uint8")


class PrefetchCacheDataloader(CacheDataloader):
    """Train-image cache dataloader that prepares the next sampled window ahead of the refresh boundary."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefetch_executor = None
        self._prefetch_future = None
        if not self.cache_all_images and self.num_times_to_repeat_images != -1:
            self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _submit_prefetch(self) -> None:
        if self._prefetch_executor is None or self._prefetch_future is not None:
            return
        self._prefetch_future = self._prefetch_executor.submit(self._get_collated_batch)

    def _next_collated_batch(self):
        if self._prefetch_future is None:
            collated_batch = self._get_collated_batch()
        else:
            collated_batch = self._prefetch_future.result()
            self._prefetch_future = None
        self._submit_prefetch()
        return collated_batch

    def __iter__(self):
        self._submit_prefetch()
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or (
                self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images
            ):
                self.num_repeated = 0
                collated_batch = self._next_collated_batch()
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch

    def __del__(self):
        if self._prefetch_executor is not None:
            try:
                self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass


class FeatureDataManager(VanillaDataManager):
    config: FeatureDataManagerConfig

    def create_train_dataset(self) -> InputDataset:
        return UInt8InputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def setup_train(self):
        """Use a prefetched train-image cache to hide image-window refresh stalls."""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = PrefetchCacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = FeatureRayGenerator(self.train_dataset.cameras.to(self.device))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_ray_generator = FeatureRayGenerator(self.train_dataset.cameras.to(self.device))
        self.eval_ray_generator = FeatureRayGenerator(self.eval_dataset.cameras.to(self.device))

        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        loader_kwargs = {
            "image_fnames": image_fnames,
            "data_dir": self.config.dataparser.data,
            "batch_size": 128,
            "enable_cache": self.config.enable_cache,
            "pin_cpu_tensors": self.config.pin_cpu_feature_cache,
            "force": False,
        }
        self.feature_loader = extract_features_for_dataset(
            feature_type=self.config.feature_type,
            device=self.device,
            max_cpu_images=self.config.cpu_feature_cache_images,
            max_gpu_images=self.config.gpu_feature_cache_images,
            **loader_kwargs,
        )
        CONSOLE.print(f"Created batch loader for {self.config.feature_type} features")

        self.train_dataset.metadata["feature_type"] = self.config.feature_type
        self.train_dataset.metadata["feature_dim"] = self.feature_loader.C

        feat_h, feat_w = self.feature_loader.H, self.feature_loader.W
        im_h = set(self.train_dataset.cameras.image_height.squeeze().tolist())
        im_w = set(self.train_dataset.cameras.image_width.squeeze().tolist())
        assert len(im_h) == 1, "All images must have the same height"
        assert len(im_w) == 1, "All images must have the same width"
        im_h, im_w = im_h.pop(), im_w.pop()
        self.feat_scale_h = feat_h / im_h
        self.feat_scale_w = feat_w / im_w
        CONSOLE.print(
            f"Feat h: {feat_h}, Feat w: {feat_w}, Feat c: {self.feature_loader.C}, Im h: {im_h}, Im w: {im_w}"
        )
        CONSOLE.print(f"Feat scale h: {self.feat_scale_h}, Feat scale w: {self.feat_scale_w}")

        self._train_window_cache: Dict[str, torch.Tensor | int] = {}
        self._eval_window_cache: Dict[str, torch.Tensor | int] = {}

        torch.cuda.empty_cache()
        gc.collect()

    def _populate_batch_features(self, image_batch: Dict, batch: Dict, is_eval: bool) -> None:
        ray_indices = batch["indices"]
        camera_idx = ray_indices[:, 0]
        y_feat = (ray_indices[:, 1] * self.feat_scale_h).long()
        x_feat = (ray_indices[:, 2] * self.feat_scale_w).long()

        cache = self._eval_window_cache if is_eval else self._train_window_cache
        image = image_batch["image"]
        token = image.data_ptr() if torch.is_tensor(image) else id(image)
        if cache.get("token") == token:
            feature_window, lookup = cache["feature"], cache["lookup"]  # type: ignore[assignment]
        else:
            image_ids = image_batch["image_idx"]
            loader_ids = image_ids + len(self.train_dataset) if is_eval else image_ids
            feature_dict = self.feature_loader.load_batch_images(loader_ids)
            feature_window = torch.stack([feature_dict[int(idx)] for idx in loader_ids.tolist()], dim=0)
            lookup = torch.full(
                (len(self.eval_dataset) if is_eval else len(self.train_dataset),),
                -1,
                dtype=torch.long,
                device=image_ids.device,
            )
            lookup[image_ids] = torch.arange(len(image_ids), dtype=torch.long, device=image_ids.device)
            cache.clear()
            cache["token"] = token
            cache["feature"] = feature_window
            cache["lookup"] = lookup

        window_idx = lookup[camera_idx]
        if feature_window.ndim == 4:
            batch["feature"] = feature_window[window_idx, y_feat, x_feat, :]
        else:
            batch["feature"] = feature_window[window_idx, y_feat, x_feat]

    def _prepare_sampled_image(self, batch: Dict) -> None:
        image = batch["image"]
        if image.dtype == torch.uint8:
            if image.device != self.device:
                image = image.to(self.device, non_blocking=True)
            batch["image"] = image.float().mul_(1.0 / 255.0)
            return
        if image.device != self.device:
            batch["image"] = image.to(self.device, non_blocking=True)

    def _next_batch(self, image_iter, pixel_sampler, ray_generator, count_attr: str, is_eval: bool):
        setattr(self, count_attr, getattr(self, count_attr) + 1)
        image_batch = next(image_iter)
        assert pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = pixel_sampler.sample(image_batch)
        ray_bundle = ray_generator(batch["indices"])
        self._prepare_sampled_image(batch)
        self._populate_batch_features(image_batch, batch, is_eval=is_eval)
        return ray_bundle, batch

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        del step
        return self._next_batch(
            self.iter_train_image_dataloader, self.train_pixel_sampler, self.train_ray_generator, "train_count", False
        )

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        del step
        return self._next_batch(
            self.iter_eval_image_dataloader, self.eval_pixel_sampler, self.eval_ray_generator, "eval_count", True
        )
