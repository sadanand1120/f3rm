# Standard library imports
from typing import Optional

# Third-party imports
import numpy as np
import open_clip
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from spatialmath.base import trexp, trinv, trlog
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize, ToTensor)

# Repo-specific imports
from f3rm.features.clip_extract import CLIPArgs, extract_clip_features


def cluster_xyz(
    xyz: np.ndarray,
    K: Optional[int] = None,
    *,
    max_auto_K: int = 10,
    random_state: int = 0,
) -> tuple[np.ndarray, list[dict]]:
    """
    Cluster an (N, 3) array of 3-D points.

    Parameters
    ----------
    xyz : (N, 3) array_like
    K   : int | None
        • If an int, use exactly that many clusters (K-means under the hood).  
        • If None, the routine selects K automatically via Bayesian-information
          criterion (BIC) on a Gaussian-Mixture model.
    max_auto_K : int
        Upper bound on K explored when K is None (defaults to 10 or √N, whichever is smaller).
    random_state : int
        Reproducibility knob for the underlying sklearn routines.

    Returns
    -------
    labels : (N,) np.ndarray
        Cluster ID for every point.
    stats  : list[dict]
        Per-cluster statistics: ``{'center', 'cov', 'count', 'indices'}``.
    """
    X = np.asarray(xyz, dtype=np.float64)
    N = len(X)

    # ——— choose / fit model ———
    if K is None:
        max_K = min(max_auto_K, int(np.sqrt(N)))
        best_bic, best_gmm = np.inf, None
        for k in range(1, max_K + 1):
            gmm = GaussianMixture(k, covariance_type="full", random_state=random_state).fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm
        labels = best_gmm.predict(X)
        K = best_gmm.n_components
    else:
        km = KMeans(n_clusters=K, n_init="auto", random_state=random_state).fit(X)
        labels = km.labels_

    # ——— assemble stats ———
    stats = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 100:
            continue
        pts = X[idx]
        stats.append(
            dict(
                center=pts.mean(axis=0),
                cov=np.cov(pts, rowvar=False),
                count=len(idx),
                indices=idx,
            )
        )

    return labels, stats


def homo_T_to_exp(T: np.ndarray) -> np.ndarray:
    """4x4 → 6-vector [vx,vy,vz,ωx,ωy,ωz]."""
    return trlog(np.asarray(T, dtype=float), twist=True, check=False)          # returns (6,)


def exp_to_homo_T(xi: np.ndarray) -> np.ndarray:
    """6-vector → 4x4 homogeneous matrix."""
    return trexp(np.asarray(xi, dtype=float))                     # returns (4,4)


def se3_distance(xi1: np.ndarray, xi2: np.ndarray) -> float:
    T1 = exp_to_homo_T(xi1)
    T2 = exp_to_homo_T(xi2)
    T_rel = trinv(T2) @ T1
    return np.linalg.norm(trlog(T_rel, twist=True, check=False))


@torch.inference_mode()
def run_pca(tokens, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(tokens)
    projected_tokens = pca.transform(tokens)
    return projected_tokens


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
def compute_similarity_text2vis(img_patch_descriptors, text_embeddings, has_negatives=False, softmax_temp=1.0):
    """
    Args:
        img_patch_descriptors: (**, dim)
        text_embeddings: (num_texts, dim). First entry is positive, rest (if any) are negatives.
        has_negatives: whether to apply paired softmax
        softmax_temp: temperature for softmax scaling

    Returns:
        sims: (**, 1) similarity score to the positive text(s) (normalized probability)
    """
    # Normalize
    img_patch_descriptors /= img_patch_descriptors.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    raw_sims = img_patch_descriptors @ text_embeddings.T  # (**, num_texts)

    if not has_negatives:
        # Mean over positives (if multiple)
        if raw_sims.shape[-1] > 1:
            raw_sims = raw_sims.mean(dim=-1, keepdim=True)
        return raw_sims

    # Paired softmax setup: split pos/neg
    pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
    pos_sims = pos_sims.expand_as(neg_sims)
    paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)  # (**, num_negatives + 1)

    # Apply temperature-scaled softmax
    probs = (paired_sims / softmax_temp).softmax(dim=-1)[..., :1]  # get prob of positive
    torch.nan_to_num_(probs, nan=0.0)

    return probs  # (**, 1)


if __name__ == "__main__":
    image_path = "/robodata/smodak/repos/f3rm/f3rm/scripts/images/frame_1.png"
    descriptors = extract_clip_features([image_path], device=torch.device("cuda"), verbose=True).cpu().squeeze()  # (h, w, c)
    descriptors_flat = rearrange(descriptors, "h w c -> (h w) c")  # (h * w, c)
    projected_tokens = run_pca(descriptors_flat, n_components=3)  # (h * w, 3)
    img = viz_pca3(projected_tokens, (descriptors.shape[0], descriptors.shape[1]), Image.open(image_path).width, Image.open(image_path).height)
    plt.imshow(img)
    plt.savefig("pca.png")

    model, _, _ = open_clip.create_model_and_transforms(CLIPArgs.model_name, pretrained=CLIPArgs.model_pretrained, device=torch.device("cuda"))
    model.eval()

    text_queries = ["mug", "object"]
    tokenizer = open_clip.get_tokenizer(CLIPArgs.model_name)
    text = tokenizer(text_queries).cuda()
    text_features = model.encode_text(text)
    text_embds = text_features.cpu()
    sims = compute_similarity_text2vis(descriptors, text_embds, has_negatives=True, softmax_temp=1.0).squeeze()
    plt.imshow(sims)
    plt.axis("off")
    plt.title(text_queries[0])
    plt.savefig("similarity.png")
