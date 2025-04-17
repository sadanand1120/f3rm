import torch
from PIL import Image

from robopoint.model.builder import load_pretrained_model
from robopoint.utils import disable_torch_init
from robopoint.mm_utils import process_images, get_model_name_from_path
disable_torch_init()

import gc
from typing import List
from einops import rearrange
from tqdm import tqdm


class ROBOPOINTArgs:
    model_name: str = "wentao-yuan/robopoint-v1-vicuna-v1.5-13b"
    batch_size: int = 64
    patch_size: int = 14

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the ROBOPOINT model parameters."""
        return {
            "model_name": cls.model_name,
            "batch_size": cls.batch_size,
            "patch_size": cls.patch_size,
        }


@torch.inference_mode()
def _process(image_paths: List[str], device: torch.device, do_proj: bool = True) -> torch.Tensor:
    model_path = ROBOPOINTArgs.model_name
    batch_size = ROBOPOINTArgs.batch_size
    patch_size = ROBOPOINTArgs.patch_size
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )

    vision_tower = model.get_model().get_vision_tower()
    mm_projector = model.get_model().mm_projector

    # Preprocess the images
    images = [Image.open(path).convert('RGB') for path in image_paths]
    preprocessed_images = process_images(images, image_processor, model.config)
    preprocessed_images = preprocessed_images.half()  # (b, 3, h, w)
    print(f"Preprocessed {len(images)} images into {preprocessed_images.shape}")

    # Get visual embeddings for the images
    embeddings = []
    for i in tqdm(
        range(0, len(preprocessed_images), batch_size),
        desc="Extracting visual features",
    ):
        batch = preprocessed_images[i: i + batch_size].to(device)
        img_feat = vision_tower(batch)
        if do_proj:
            img_feat = mm_projector(img_feat)
        embeddings.append(img_feat.cpu())
        del batch, img_feat
        torch.cuda.empty_cache()
    embeddings = torch.cat(embeddings, dim=0)

    # Reshape embeddings from flattened patches to patch height and width
    h_in, w_in = preprocessed_images.shape[-2:]
    h_out = h_in // patch_size
    w_out = w_in // patch_size
    embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
    print(f"Extracted visual embeddings of shape {embeddings.shape}")

    # Delete and clear memory to be safe
    del model
    del preprocessed_images
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings


@torch.inference_mode()
def extract_robopoint_proj_features(image_paths: List[str], device: torch.device) -> torch.Tensor:
    return _process(image_paths, device, do_proj=True)


@torch.inference_mode()
def extract_robopoint_noproj_features(image_paths: List[str], device: torch.device) -> torch.Tensor:
    return _process(image_paths, device, do_proj=False)
