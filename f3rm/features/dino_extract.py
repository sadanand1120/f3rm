import gc
from typing import List

import torch
from einops import rearrange
from tqdm import tqdm


class DINOArgs:
    model_type: str = "dinov2_vitl14"
    load_size: int = -1  # -1 to use smallest side size, -x to use scaled of smallest side
    stride: int = None
    facet: str = "token"
    layer: int = -1
    bin: bool = False
    batch_size: int = 2

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the DINO model parameters."""
        return {
            "model_type": cls.model_type,
            "load_size": cls.load_size,
            "stride": cls.stride,
            "facet": cls.facet,
            "layer": cls.layer,
            "bin": cls.bin,
        }


@torch.no_grad()
def extract_dino_features(image_paths: List[str], device: torch.device, verbose=False) -> torch.Tensor:
    from f3rm.features.dino.dino_vit_extractor import ViTExtractor

    extractor = ViTExtractor(model_type=DINOArgs.model_type, stride=DINOArgs.stride)
    if verbose:
        print(f"Loaded DINO model {DINOArgs.model_type}")

    embeddings = []
    for i in tqdm(range(0, len(image_paths), DINOArgs.batch_size),
                  desc="Preprocessing & extracting DINO features", leave=verbose):
        batch_paths = image_paths[i: i + DINOArgs.batch_size]
        # open & preprocess each image in the batch
        batch_tensors = [
            extractor.preprocess(p, DINOArgs.load_size, allow_crop=True)[0]
            for p in batch_paths
        ]
        batch = torch.cat(batch_tensors, dim=0).to(device)
        # extract descriptors, move to CPU, and store
        emb = extractor.extract_descriptors(
            batch, DINOArgs.layer, DINOArgs.facet, DINOArgs.bin
        ).cpu()
        embeddings.append(emb)
        del batch, emb
        torch.cuda.empty_cache()

    embeddings = torch.cat(embeddings, dim=0)
    # reshape to (batch, height, width, channels)
    height, width = extractor.num_patches
    embeddings = rearrange(embeddings, "b 1 (h w) c -> b h w c", h=height, w=width)
    if verbose:
        print(f"Extracted DINO embeddings of shape {embeddings.shape}")

    # cleanup
    del extractor
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings
