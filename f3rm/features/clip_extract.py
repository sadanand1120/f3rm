import gc
from typing import List

import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F

import open_clip


class CLIPArgs:
    model_name: str = "ViT-L-14-336-quickgelu"
    model_pretrained: str = "openai"
    load_size: int = 1024
    skip_center_crop: bool = True
    batch_size: int = 16

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the CLIP model parameters."""
        return {
            "model_name": cls.model_name,
            "model_pretrained": cls.model_pretrained,
            "load_size": cls.load_size,
            "skip_center_crop": cls.skip_center_crop,
        }

@torch.inference_mode()
def get_patch_encodings(model, image_batch):
    from f3rm.features.utils import interpolate_positional_embedding

    _, _, w, h = image_batch.shape
    x = model.visual.conv1(image_batch)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + interpolate_positional_embedding(model.visual.positional_embedding, x, patch_size=model.visual.patch_size[0], w=w, h=h)
    x = model.visual.ln_pre(x)
    *layers, last_resblock = model.visual.transformer.resblocks
    penultimate = nn.Sequential(*layers)
    x = penultimate(x)
    v_in_proj_weight = last_resblock.attn.in_proj_weight[-last_resblock.attn.embed_dim:]
    v_in_proj_bias = last_resblock.attn.in_proj_bias[-last_resblock.attn.embed_dim:]
    v_in = F.linear(last_resblock.ln_1(x), v_in_proj_weight, v_in_proj_bias)
    x = F.linear(v_in, last_resblock.attn.out_proj.weight, last_resblock.attn.out_proj.bias)
    x = x[:, 1:, :]   # Extract the patch tokens, not the class token
    x = model.visual.ln_post(x)
    if model.visual.proj is not None:
        x = x @ model.visual.proj
    return x


@torch.no_grad()
def extract_clip_features(image_paths: List[str], device: torch.device, verbose=False) -> torch.Tensor:
    from f3rm.features.utils import get_preprocess

    model, _, _ = open_clip.create_model_and_transforms(CLIPArgs.model_name, pretrained=CLIPArgs.model_pretrained, device=device)
    preprocess = get_preprocess(resize=CLIPArgs.load_size, do_center_crop=not CLIPArgs.skip_center_crop)
    if verbose:
        print(f"Loaded CLIP model {CLIPArgs.model_name}")

    embeddings = []
    h_in = w_in = None
    for i in tqdm(range(0, len(image_paths), CLIPArgs.batch_size),
                desc="Preprocessing & extracting CLIP features", leave=verbose):
        batch_paths = image_paths[i : i + CLIPArgs.batch_size]
        # open & preprocess each image in the batch
        batch_tensors = [preprocess(Image.open(p)) for p in batch_paths]
        if h_in is None:
            _, h_in, w_in = batch_tensors[0].shape
        batch = torch.stack(batch_tensors).to(device)
        # get embeddings, move to CPU, and store
        embeddings.append(get_patch_encodings(model, batch).cpu())
        del batch
        torch.cuda.empty_cache()

    embeddings = torch.cat(embeddings, dim=0)
    if verbose:
        print(f"Processed {len(image_paths)} images into {embeddings.shape}")

    # Reshape embeddings from flattened patches to patch height and width
    if CLIPArgs.model_name.startswith("ViT"):
        h_out = h_in // model.visual.patch_size[0]
        w_out = w_in // model.visual.patch_size[0]
    elif CLIPArgs.model_name.startswith("RN"):
        h_out = max(h_in / w_in, 1.0) * model.visual.attnpool.spacial_dim
        w_out = max(w_in / h_in, 1.0) * model.visual.attnpool.spacial_dim
        h_out, w_out = int(h_out), int(w_out)
    else:
        raise ValueError(f"Unknown CLIP model name: {CLIPArgs.model_name}")
    embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
    if verbose:
        print(f"Extracted CLIP embeddings of shape {embeddings.shape}")

    # Delete and clear memory to be safe
    del model
    del preprocess
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings
