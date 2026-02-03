#!/usr/bin/env python3
"""
High-resolution zero-shot segmentation with DINO-txt

 • prompt-ensemble text embeddings
 • ≈ 791 sliding-window crops per 1 MP
 • k-means on *pre-block-11* tokens  → sharp edges
 • clusters re-encoded with *final* tokens, then zero-shot classified
"""

from __future__ import annotations
import contextlib
import math
import random
import sys
import urllib.request
import os
from pathlib import Path
from typing import Sequence, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import torchvision.transforms.functional as TF
from tqdm import tqdm
from sklearn.decomposition import PCA

# ──────────────────────────── DINO-v2 import ────────────────────────────── #
# REPO_PATH = "./dinov2"                # change if you pip-installed dinov2
# sys.path.append(REPO_PATH)
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.hub.dinotxt import (
    dinov2_vitl14_reg4_dinotxt_tet1280d20h24l,
    get_tokenizer,
)

# ─────────────────────────────── constants ───────────────────────────────── #
IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
CLASS_NAMES = ["dog", "chair", "bowl", "tupperware", "wooden floor"]
PROMPT_TEMPLATES = (
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a photo of the hard to see {c}.",
    lambda c: f"a low resolution photo of the {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a hard to see {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a pixelated photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a jpeg corrupted photo of a {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a rendering of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a doodle of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a pixelated photo of a {c}.",
    lambda c: f"itap of the {c}.",
    lambda c: f"a jpeg corrupted photo of the {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"the plushie {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"itap of a {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"itap of my {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a tattoo of the {c}.",
)

AREA_FRACS = (0.01, 0.10, 1.00)
STRIDE_RATIO = 0.40                     # stride = 0.40 · side
JITTER_RATIO = 0.10                     # 10 % corner jitter
PRE_BLOCK = 11                       # ViT-L block for “pre” tokens
NUM_CLUSTERS = 32
MAX_KMEANS_SAMPLES = 20_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(".")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
random.seed(0)
torch.manual_seed(0)

# ───────────────────────── helper utilities ─────────────────────────────── #


def download_image(url: str) -> Image.Image:
    with urllib.request.urlopen(url) as resp:
        return Image.open(resp).convert("RGB")


def prepare_model():
    model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().to(DEVICE).eval()
    tokenizer = get_tokenizer()
    return model, tokenizer


@torch.no_grad()
def build_text_embeddings(model, tokenizer, names: Sequence[str]) -> torch.Tensor:
    prompts, owner = [], []
    for c, n in enumerate(names):
        for tpl in PROMPT_TEMPLATES:
            prompts.append(tpl(n))
            owner.append(c)

    tok = tokenizer.tokenize(prompts).to(DEVICE)
    emb = model.encode_text(tok)[:, 1024:]                # [N,256]
    C, D = len(names), emb.size(1)
    agg, cnt = torch.zeros(C, D, device=DEVICE), torch.zeros(C, device=DEVICE)
    for e, c in zip(emb, owner):
        agg[c] += e
        cnt[c] += 1
    return F.normalize(agg / cnt.unsqueeze(1), p=2, dim=1)  # [C,D]

# ───────────────────── crop generator (projective quadrilaterals) ────────────────────── #


def gen_quad_crops(h: int, w: int, area_fracs=AREA_FRACS, jitter=JITTER_RATIO) -> list:
    """
    Generate random quadrilateral crops for projective sampling.
    Returns a list of 4x2 numpy arrays (corners in (x, y) order).
    """
    crops = []
    for af in area_fracs:
        side = max(16, int(round(math.sqrt(af) * min(h, w))))
        stride = max(8, int(round(side * STRIDE_RATIO)))
        for y0 in range(0, h - side + 1, stride):
            for x0 in range(0, w - side + 1, stride):
                # Generate a square, then jitter corners
                base = np.array([
                    [x0, y0],
                    [x0 + side, y0],
                    [x0 + side, y0 + side],
                    [x0, y0 + side],
                ], dtype=np.float32)
                noise = np.random.uniform(-jitter, jitter, size=(4, 2)) * side
                quad = base + noise
                quad = np.clip(quad, [0, 0], [w - 1, h - 1])
                crops.append(quad)
    print(f"Generated quad crops: {len(crops)}")
    return crops


def get_projective_grid(quad, out_h, out_w, device, img_h, img_w):
    """
    Given a quadrilateral (4x2), return a grid for grid_sample that maps a (out_h, out_w) square to the quad.
    """
    # Target: regular grid in output square
    tgt = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)
    # Compute projective transform (homography)
    import cv2
    H, _ = cv2.findHomography(tgt, quad)
    # Generate meshgrid for output
    grid_y, grid_x = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing='ij')
    ones = np.ones_like(grid_x)
    coords = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3).T  # [3, N]
    mapped = H @ coords
    mapped = mapped[:2] / mapped[2:]
    mapped = mapped.T.reshape(out_h, out_w, 2)
    # Normalize to [-1, 1] for grid_sample
    mapped[..., 0] = mapped[..., 0] / (img_w - 1) * 2 - 1
    mapped[..., 1] = mapped[..., 1] / (img_h - 1) * 2 - 1
    return torch.from_numpy(mapped).to(device).float()

# ───────────────────── token extractor (pre / post) ──────────────────────── #


@torch.no_grad()
def encode_patches(model, img_t: torch.Tensor,
                   pre_layer: int | None = None) -> torch.Tensor:
    """
    Return patch tokens [P,D] with *all* special tokens removed.
      • pre_layer=None  → last-layer patch tokens (already stripped by helper)
      • pre_layer>=0    → tokens before transformer block ‹pre_layer›
    """
    ctx = torch.autocast("cuda") if DEVICE.type == "cuda" else contextlib.nullcontext()
    with ctx:
        if pre_layer is None:
            _, patch = model.get_visual_class_and_patch_tokens(img_t)
            return patch.squeeze(0)                               # [P,D]

        inter = model.visual_model.backbone.get_intermediate_layers(img_t, n=pre_layer + 1)[pre_layer]
        inter = inter.squeeze(0)                                  # [*,D]
        start = 1 + model.visual_model.backbone.model.num_register_tokens                    # skip CLS + regs
        return inter[start:]                                      # [P,D]

# ──────────────────── sliding-window feature stack (projective) ───────────────────────── #


@torch.no_grad()
def aggregate_features(model, preprocess, pil_img: Image.Image, use_pre: bool, return_hits=False) -> torch.Tensor:
    H, W = pil_img.height, pil_img.width
    fmap, hits = None, torch.zeros((H, W), device=DEVICE)
    # Convert PIL image to normalized torch tensor
    img_tensor = TF.to_tensor(pil_img).unsqueeze(0).to(DEVICE)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    quad_crops = gen_quad_crops(H, W)
    out_size = 224  # DINOv2 default input size
    for quad in tqdm(quad_crops, desc=f"Aggregating {'pre' if use_pre else 'post'}-block features"):
        grid = get_projective_grid(quad, out_size, out_size, DEVICE, H, W)
        grid = grid.unsqueeze(0)
        crop_tensor = F.grid_sample(img_tensor, grid, mode='bilinear', align_corners=False)
        toks = encode_patches(model, crop_tensor, PRE_BLOCK if use_pre else None)
        if fmap is None:
            fmap = torch.zeros((toks.size(1), H, W), device=DEVICE)
        # Map crop back to image using the quad's bounding box
        # For simplicity, use the bounding box of the quad
        x0, y0 = quad[:, 0].min().astype(int), quad[:, 1].min().astype(int)
        x1, y1 = quad[:, 0].max().astype(int), quad[:, 1].max().astype(int)
        # Determine patch grid shape
        P = toks.size(0)
        w_p = int(round(math.sqrt(P)))
        if P % w_p != 0:
            for w_p in range(w_p, 0, -1):
                if P % w_p == 0:
                    break
        h_p = P // w_p
        grid_feat = toks.movedim(1, 0).reshape(fmap.size(0), h_p, w_p)
        grid_feat = F.interpolate(grid_feat.unsqueeze(0), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0]
        fmap[:, y0:y1, x0:x1] += grid_feat
        hits[y0:y1, x0:x1] += 1
    if return_hits:
        return fmap / hits.clamp(min=1), hits
    return fmap / hits.clamp(min=1)  # [D,H,W]

# ──────────────────────── k-means on pre-tokens ──────────────────────────── #


def gpu_kmeans(x: torch.Tensor, k: int, n_iter: int = 20) -> torch.Tensor:
    """
    Run k-means on GPU using faiss if available, else torch. x: [N, D] (float32, cuda)
    Returns cluster assignments [N] (int64)
    """
    try:
        import faiss
        x_np = x.detach().cpu().numpy().astype('float32') if not x.is_cuda else x.detach().contiguous().cpu().numpy().astype('float32')
        res = faiss.StandardGpuResources()
        kmeans = faiss.Kmeans(x.size(1), k, niter=n_iter, gpu=True)
        kmeans.train(x_np, res)
        _, I = kmeans.index.search(x_np, 1)
        return torch.from_numpy(I.squeeze(1)).to(x.device)
    except ImportError:
        # Fallback: simple PyTorch k-means
        N, D = x.shape
        c = x[torch.randperm(N)[:k]].clone()  # [k, D]
        pbar = tqdm(range(n_iter), desc="PyTorch k-means (fallback)")
        for _ in pbar:
            dist = torch.cdist(x, c)
            labels = dist.argmin(dim=1)
            for i in range(k):
                if (labels == i).any():
                    c[i] = x[labels == i].mean(dim=0)
        dist = torch.cdist(x, c)
        labels = dist.argmin(dim=1)
        return labels


def kmeans_on_pixels(fmap_pre: torch.Tensor) -> np.ndarray:
    D, H, W = fmap_pre.shape
    flat = fmap_pre.permute(1, 2, 0).reshape(-1, D).contiguous().to(DEVICE)
    labels = gpu_kmeans(flat, NUM_CLUSTERS, n_iter=20)
    return labels.cpu().numpy().reshape(H, W).astype(np.int16)

# ───────────── centroids from post-tokens + zero-shot class ──────────────── #


def centroids_from_labels(fmap_post: torch.Tensor,
                          labels: np.ndarray) -> torch.Tensor:
    D, H, W = fmap_post.shape
    flat_f = fmap_post.reshape(D, -1)        # [D,HW]
    flat_l = torch.from_numpy(labels.reshape(-1)).to(fmap_post.device)
    cents = torch.zeros(NUM_CLUSTERS, D, device=fmap_post.device)
    for k in range(NUM_CLUSTERS):
        m = flat_l == k
        if m.any():
            cents[k] = flat_f[:, m].mean(dim=1)
    return F.normalize(cents, p=2, dim=1)


def centroid_zero_shot(cents: torch.Tensor,
                       text_emb: torch.Tensor) -> np.ndarray:
    return (cents @ text_emb.T).argmax(1).cpu().numpy()

# ───────────────────────── drawing helpers ───────────────────────────────── #


def save_reference(pil_img: Image.Image, path: Path) -> np.ndarray:
    rgb = np.asarray(pil_img).astype(np.float32) / 255.0
    plt.imsave(path, rgb)
    return rgb


def save_overlay(rgb: np.ndarray, idx: np.ndarray, labels: Sequence[str],
                 path: Path, alpha: float = 0.7):
    H, W = idx.shape
    over = np.zeros((H, W, 4), dtype=np.float32)
    cmap = plt.get_cmap("tab10", len(labels))
    for i in range(len(labels)):
        over[idx == i] = (*cmap(i)[:3], alpha)
    over[..., :3] *= rgb

    plt.figure(figsize=(6, 4))
    plt.imshow(over)
    plt.axis("off")
    for i, txt in enumerate(labels):
        plt.text(10, 40 + 30 * i, txt, color=cmap(i)[:3], fontsize=10,
                 bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300)
    plt.close()


def save_kmeans_clusters(idx: np.ndarray, path: Path, n_clusters: int):
    H, W = idx.shape
    cmap = plt.get_cmap("tab20", n_clusters)
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(n_clusters):
        rgb[idx == i] = cmap(i)[:3]
    plt.imsave(path, rgb)

# ───────────────────── PCA colorization helper ───────────────────── #


def pca_color(feat: torch.Tensor, out_hw=None) -> np.ndarray:
    """
    feat: [D, H, W] torch tensor
    out_hw: (H, W) to resize output
    Returns: [H, W, 3] np.ndarray
    """
    D, H, W = feat.shape
    x = feat.permute(1, 2, 0).reshape(-1, D).cpu().numpy()
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x)
    x_pca = (x_pca - x_pca.min()) / (x_pca.max() - x_pca.min() + 1e-8)
    img = x_pca.reshape(H, W, 3)
    if out_hw is not None and (H, W) != out_hw:
        import cv2
        img = cv2.resize(img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return img

# ───────────────────── composite visualization ───────────────────── #


def save_composite_viz(pil_img, fmap_pre, hits, kmeans_labels, out_path):
    import cv2
    H, W = pil_img.height, pil_img.width
    # Input image
    img_np = np.asarray(pil_img).astype(np.float32) / 255.0
    # PCA single (downsampled)
    pca_single = pca_color(F.interpolate(fmap_pre.unsqueeze(0), size=(54, 64), mode='bilinear', align_corners=False)[0], out_hw=(54, 64))
    # PCA high-res
    pca_high = pca_color(fmap_pre, out_hw=(H, W))
    # Counts
    counts = hits.cpu().numpy()
    counts_img = (counts - counts.min()) / (counts.max() - counts.min() + 1e-8)
    counts_img = np.stack([counts_img] * 3, axis=-1)
    # K-means
    kmeans_rgb = np.zeros((H, W, 3), dtype=np.float32)
    cmap = plt.get_cmap("tab20", np.max(kmeans_labels) + 1)
    for i in range(np.max(kmeans_labels) + 1):
        kmeans_rgb[kmeans_labels == i] = cmap(i)[:3]
    # Compose
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(img_np)
    axs[0].set_title(f"Input [{H}, {W}]")
    axs[0].axis('off')
    axs[1].imshow(pca_single)
    axs[1].set_title(f"PCA single [54, 64]")
    axs[1].axis('off')
    axs[2].imshow(pca_high)
    axs[2].set_title(f"PCA High-Res [{H}, {W}]")
    axs[2].axis('off')
    axs[3].imshow(counts_img)
    axs[3].set_title(f"Counts (min {int(counts.min())}, max {int(counts.max())})")
    axs[3].axis('off')
    axs[4].imshow(kmeans_rgb)
    axs[4].set_title(f"K-Means k={np.max(kmeans_labels)+1}")
    axs[4].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ────────────────────────────────── main ─────────────────────────────────── #


def main() -> None:
    model, tokenizer = prepare_model()
    preprocess = make_classification_eval_transform()

    pil_img = download_image(IMAGE_URL)              # keep native resolution
    text_emb = build_text_embeddings(model, tokenizer, CLASS_NAMES)

    fmap_pre, hits = aggregate_features(model, preprocess, pil_img, use_pre=True, return_hits=True)
    fmap_post = aggregate_features(model, preprocess, pil_img, use_pre=False)

    labels_px = kmeans_on_pixels(fmap_pre)
    # Save raw k-means cluster visualization
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_kmeans_clusters(labels_px, OUTPUT_DIR / "2dino_txt_kmeans.png", NUM_CLUSTERS)
    cents = centroids_from_labels(fmap_post, labels_px)
    k2cls = centroid_zero_shot(cents, text_emb)
    pred_map = k2cls[labels_px]                      # [H,W]

    ref_np = save_reference(pil_img, OUTPUT_DIR / "2dino_txt_hr_1.png")
    save_overlay(ref_np, pred_map, CLASS_NAMES, OUTPUT_DIR / "2dino_txt_hr_.png")
    # Save composite visualization
    save_composite_viz(pil_img, fmap_pre, hits, labels_px, OUTPUT_DIR / "2dino_txt_composite.png")


if __name__ == "__main__":
    main()
