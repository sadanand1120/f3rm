from PIL import Image
from sklearn.decomposition import PCA
from copy import deepcopy
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop, Compose, Resize, InterpolationMode, ToTensor, Normalize
from pathlib import Path


def interpolate_positional_embedding(
    positional_embedding: torch.Tensor, x: torch.Tensor, patch_size: int, w: int, h: int
):
    """
    Interpolate the positional encoding for CLIP to the number of patches in the image given width and height.
    Modified from DINO ViT `interpolate_pos_encoding` method.
    https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L174
    """
    assert positional_embedding.ndim == 2, "pos_encoding must be 2D"

    # Number of patches in input
    num_patches = x.shape[1] - 1
    # Original number of patches for square images
    num_og_patches = positional_embedding.shape[0] - 1

    if num_patches == num_og_patches and w == h:
        # No interpolation needed
        return positional_embedding.to(x.dtype)

    dim = x.shape[-1]
    class_pos_embed = positional_embedding[:1]  # (1, dim)
    patch_pos_embed = positional_embedding[1:]  # (num_og_patches, dim)

    # Compute number of tokens
    w0 = w // patch_size
    h0 = h // patch_size
    assert w0 * h0 == num_patches, "Number of patches does not match"

    # Add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1

    # Interpolate
    patch_per_ax = int(np.sqrt(num_og_patches))
    patch_pos_embed_interp = torch.nn.functional.interpolate(
        patch_pos_embed.reshape(1, patch_per_ax, patch_per_ax, dim).permute(0, 3, 1, 2),
        # (1, dim, patch_per_ax, patch_per_ax)
        scale_factor=(w0 / patch_per_ax, h0 / patch_per_ax),
        mode="bicubic",
        align_corners=False,
        recompute_scale_factor=False,
    )  # (1, dim, w0, h0)
    assert (
        int(w0) == patch_pos_embed_interp.shape[-2] and int(h0) == patch_pos_embed_interp.shape[-1]
    ), "Interpolation error."

    patch_pos_embed_interp = patch_pos_embed_interp.permute(0, 2, 3, 1).reshape(-1, dim)  # (w0 * h0, dim)
    # Concat class token embedding and interpolated patch embeddings
    pos_embed_interp = torch.cat([class_pos_embed, patch_pos_embed_interp], dim=0)  # (w0 * h0 + 1, dim)
    return pos_embed_interp.to(x.dtype)


@torch.inference_mode()
def preprocess(image_path, load_size, patch_size, mean, std, allow_crop=False):
    """
    Preprocesses an image before extraction.
    :param image_path: path to image to be extracted, or a PIL image.
    :param load_size: optional. Size to resize image before the rest of preprocessing. -1 to use smallest side size.
    :param allow_crop: optional. If True, crop the image to be divisible by the patch size.
    :return: a tuple containing:
                (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                (2) the pil image in relevant dimensions
    """
    if isinstance(image_path, str) or isinstance(image_path, Path):
        pil_image = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, Image.Image):
        pil_image = image_path.convert('RGB')
    pil_image = transforms.ToTensor()(pil_image)
    if allow_crop:
        height, width = pil_image.shape[1:]   # C x H x W
        cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
        pil_image = pil_image[:, :cropped_height, :cropped_width]
    else:
        cropped_width, cropped_height = pil_image.shape[2], pil_image.shape[1]
    if load_size is not None:
        if load_size < 0:
            load_size = abs(load_size) * min(pil_image.shape[1:])
        pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.BICUBIC)(pil_image)
    prep = transforms.Compose([
        transforms.Normalize(mean=mean, std=std)
    ])
    prep_img = prep(pil_image)
    prep_img = prep_img[None, ...]
    return prep_img, pil_image, cropped_height, cropped_width


def get_preprocess(resize=224, do_center_crop=True, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    """
    clip: mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    dino: dino_vit_extractor's mean and std
    """
    def _convert_to_rgb(image):
        return image.convert('RGB')

    if do_center_crop:
        preprocess = Compose([
            Resize(resize, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(resize),
            _convert_to_rgb,
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
    else:
        preprocess = Compose([
            Resize(resize, interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
    return preprocess


def compute_new_dims(orig_size, short_side=224, round_multiple=14):
    """
    Scales the input (width, height) so that the smaller side becomes 'short_side',
    then rounds both dimensions to a multiple of 'round_multiple'.
    """
    w, h = orig_size

    # Figure out which side is smaller and compute scale factor
    if w <= h:
        scale = short_side / w
    else:
        scale = short_side / h

    # Scale both dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Round each dimension to the nearest multiple of 'round_multiple'
    new_w = (new_w + round_multiple - 1) // round_multiple * round_multiple
    new_h = (new_h + round_multiple - 1) // round_multiple * round_multiple

    return (new_w, new_h)


def resize_to_match_aspect_ratio(pil_img, target_ratio_tuple=(1280, 720)):
    _img = deepcopy(pil_img)
    target_w, target_h = target_ratio_tuple
    target_aspect = target_w / target_h
    w, h = _img.size
    current_aspect = w / h
    if abs(current_aspect - target_aspect) < 1e-6:
        return _img  # Already matches aspect ratio
    if current_aspect > target_aspect:
        # Image is too wide → increase height
        new_h = round(w / target_aspect)
        new_size = (w, new_h)
    else:
        # Image is too tall → increase width
        new_w = round(h * target_aspect)
        new_size = (new_w, h)
    return _img.resize(new_size, Image.LANCZOS)


@torch.inference_mode()
def run_pca(tokens, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(tokens)
    projected_tokens = pca.transform(tokens)
    return projected_tokens


@torch.inference_mode()
def run_pca_2(tokens, n_components=3, proj_V=None, low_rank_min=None, low_rank_max=None, niter=5, q_min=0.01, q_max=0.99):
    """
    Adapted from the distilled-feature-fields code; uses torch.pca_lowrank 
    and quantile-based clamping for PCA-based projection to 3 channels.
    """
    tokens_t = torch.as_tensor(tokens, dtype=torch.float32)

    # 1) Compute PCA basis if not provided
    if proj_V is None:
        mean = tokens_t.mean(dim=0)
        shifted = tokens_t - mean
        # Perform low-rank approximation (PCA) in PyTorch
        U, S, V = torch.pca_lowrank(shifted, q=n_components, niter=niter)
        proj_V = V[:, :n_components]  # top n_components

    # 2) Project into 3D
    projected_tokens = tokens_t @ proj_V

    # 3) Compute quantile-based min/max if not provided
    if low_rank_min is None:
        low_rank_min = torch.quantile(projected_tokens, q_min, dim=0)
    if low_rank_max is None:
        low_rank_max = torch.quantile(projected_tokens, q_max, dim=0)

    # 4) Scale to [0,1] and clamp
    projected_tokens = (projected_tokens - low_rank_min) / (low_rank_max - low_rank_min)
    projected_tokens = projected_tokens.clamp(0, 1)

    # Return the 3D result plus the projection matrix & min/max for reuse
    return projected_tokens.numpy(), proj_V, low_rank_min, low_rank_max


@torch.inference_mode()
def viz_pca3(projected_tokens, grid_size, orig_img_width, orig_img_height, resample=Image.LANCZOS) -> Image:
    t = torch.tensor(projected_tokens)
    t_min = t.min(dim=0, keepdim=True).values
    t_max = t.max(dim=0, keepdim=True).values
    normalized_t = (t - t_min) / (t_max - t_min)
    array = (normalized_t * 255).byte().numpy()
    array = array.reshape(*grid_size, 3)
    return Image.fromarray(array).resize((orig_img_width, orig_img_height), resample=resample)


@torch.inference_mode()
def viz_pca3_2(projected_tokens, grid_size, orig_img_width, orig_img_height, resample=Image.LANCZOS) -> Image.Image:
    """
    Adapted from the distilled-feature-fields code; Take a (N x 3) array in [0,1], reshape to the specified grid_size, and map to RGB image.
    """
    # Convert [0,1] -> [0,255], reshape to (H, W, 3)
    arr = (projected_tokens * 255).astype("uint8").reshape(*grid_size, 3)
    img = Image.fromarray(arr)
    return img.resize((orig_img_width, orig_img_height), resample=resample)
