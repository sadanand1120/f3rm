from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import time
from time import perf_counter

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
import viser
from cuml.cluster import HDBSCAN
from cuml.preprocessing import StandardScaler
from nerfstudio.utils.eval_utils import eval_setup
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KDTree


def as_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def log(msg):
    print(f"[demo] {msg}", flush=True)


def default_cloud_cache_path(load_config: Path, split: str, num_points: int, num_rays_per_batch: int, min_accum: float):
    acc = str(min_accum).replace(".", "p")
    return load_config.parent / f"instance_cloud_{split}_n{num_points}_r{num_rays_per_batch}_acc{acc}.npz"


def preprocess_features(x: cp.ndarray, mode: str) -> cp.ndarray:
    if mode == "raw":
        return x
    if mode == "std":
        return StandardScaler().fit_transform(x)
    if mode == "l2":
        denom = cp.linalg.norm(x, axis=1, keepdims=True)
        return x / cp.clip(denom, 1e-8, None)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def cluster_palette(labels: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    palette = {-1: np.full(3, 0.65, dtype=np.float32)}
    cluster_ids = np.array(sorted(label for label in np.unique(labels) if label >= 0), dtype=np.int64)
    if len(cluster_ids) == 0:
        return palette
    colors = plt.cm.hsv(np.linspace(0.0, 1.0, len(cluster_ids), endpoint=False))[:, :3].astype(np.float32)
    rng = np.random.default_rng(seed)
    colors = colors[rng.permutation(len(colors))]
    for color, label in zip(colors, cluster_ids):
        palette[int(label)] = color
    return palette


def colorize(labels: np.ndarray, palette: dict[int, np.ndarray]) -> np.ndarray:
    image = np.zeros((*labels.shape, 3), dtype=np.float32)
    image[labels == -1] = palette[-1]
    for label, color in palette.items():
        if label >= 0:
            image[labels == label] = color
    return image


def point_colors(labels: np.ndarray, palette: dict[int, np.ndarray]) -> np.ndarray:
    colors = np.repeat(palette[-1][None, :], len(labels), axis=0)
    for label, color in palette.items():
        if label >= 0:
            colors[labels == label] = color
    return colors


def label_pixel_counts(labels: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def nonnegative_unique_labels(labels: np.ndarray) -> np.ndarray:
    return np.array(sorted(int(label) for label in np.unique(labels) if label >= 0), dtype=np.int32)


def label_list_text(labels: np.ndarray) -> str:
    unique_labels = nonnegative_unique_labels(labels)
    return ", ".join(str(int(label)) for label in unique_labels) if len(unique_labels) > 0 else "none"


def sample_embedding_indices(
    labels: np.ndarray,
    pixel_counts: dict[int, int],
    max_points: int,
    seed: int,
) -> np.ndarray:
    candidate_labels = np.array(
        [label for label in sorted(np.unique(labels)) if pixel_counts.get(int(label), 0) > 0],
        dtype=np.int64,
    )
    if len(candidate_labels) == 0:
        return np.zeros(0, dtype=np.int64)

    available_total = int(sum(np.sum(labels == label) for label in candidate_labels))
    if max_points <= 0 or available_total <= max_points:
        return np.sort(np.concatenate([np.flatnonzero(labels == label) for label in candidate_labels], axis=0)).astype(np.int64)

    weights = np.array([pixel_counts[int(label)] for label in candidate_labels], dtype=np.float64)
    weights /= np.clip(weights.sum(), 1.0, None)
    raw_targets = weights * max_points
    quotas = np.floor(raw_targets).astype(np.int64)

    positive = weights > 0.0
    quotas[positive] = np.maximum(quotas[positive], 1)
    available = np.array([np.sum(labels == label) for label in candidate_labels], dtype=np.int64)
    quotas = np.minimum(quotas, available)

    total = int(quotas.sum())
    remainders = raw_targets - np.floor(raw_targets)
    order = np.argsort(-remainders)
    while total < max_points:
        changed = False
        for idx in order:
            if quotas[idx] < available[idx]:
                quotas[idx] += 1
                total += 1
                changed = True
                if total >= max_points:
                    break
        if not changed:
            break

    while total > max_points:
        removable = np.flatnonzero(quotas > 1)
        if len(removable) == 0:
            removable = np.flatnonzero(quotas > 0)
            if len(removable) == 0:
                break
        idx = removable[np.argmin(remainders[removable])]
        quotas[idx] -= 1
        total -= 1

    rng = np.random.default_rng(seed)
    keep = []
    for label, quota in zip(candidate_labels, quotas):
        if quota <= 0:
            continue
        idx = np.flatnonzero(labels == label)
        if len(idx) > quota:
            idx = rng.choice(idx, size=int(quota), replace=False)
        keep.append(np.sort(idx))

    if not keep:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(keep, axis=0)).astype(np.int64)


def compute_pca_projection(
    cluster_features: cp.ndarray,
    labels: np.ndarray,
    pixel_counts: dict[int, int],
    seed: int,
    max_points: int,
):
    fit_mask = labels >= 0
    if not np.any(fit_mask):
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int32)

    fit_idx = np.flatnonzero(fit_mask)
    fit_labels = labels[fit_idx]
    nonneg_pixel_counts = {label: count for label, count in pixel_counts.items() if label >= 0}
    sample_local = sample_embedding_indices(fit_labels, nonneg_pixel_counts, max_points, seed)
    if len(sample_local) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int32)
    sample_idx = fit_idx[sample_local]
    embed_labels = labels[sample_idx].astype(np.int32, copy=False)
    embed_feats = cp.asnumpy(cluster_features[cp.asarray(sample_idx)]).astype(np.float32, copy=False)
    if len(embed_feats) == 1:
        return np.zeros((1, 2), dtype=np.float32), embed_labels

    n_components = min(2, embed_feats.shape[0], embed_feats.shape[1])
    reducer = PCA(n_components=n_components)
    embedding = reducer.fit_transform(embed_feats).astype(np.float32, copy=False)
    if embedding.ndim == 1:
        embedding = embedding[:, None]
    if embedding.shape[1] == 1:
        embedding = np.concatenate([embedding, np.zeros((len(embedding), 1), dtype=np.float32)], axis=1)
    return embedding, embed_labels


def compute_lda_projection(
    cluster_features: cp.ndarray,
    labels: np.ndarray,
    pixel_counts: dict[int, int],
    seed: int,
    max_points: int,
):
    fit_mask = labels >= 0
    if not np.any(fit_mask):
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int32)

    fit_idx = np.flatnonzero(fit_mask)
    fit_labels = labels[fit_idx]
    nonneg_pixel_counts = {label: count for label, count in pixel_counts.items() if label >= 0}
    sample_local = sample_embedding_indices(fit_labels, nonneg_pixel_counts, max_points, seed)
    if len(sample_local) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int32)
    sample_idx = fit_idx[sample_local]
    embed_labels = labels[sample_idx].astype(np.int32, copy=False)
    unique_labels = np.array(sorted(np.unique(embed_labels)), dtype=np.int32)
    if len(unique_labels) < 2:
        return np.zeros((0, 2), dtype=np.float32), embed_labels

    embed_feats = cp.asnumpy(cluster_features[cp.asarray(sample_idx)]).astype(np.float32, copy=False)
    n_components = min(2, len(unique_labels) - 1)
    reducer = LinearDiscriminantAnalysis(n_components=n_components)
    embedding = reducer.fit_transform(embed_feats, embed_labels).astype(np.float32, copy=False)
    if embedding.ndim == 1:
        embedding = embedding[:, None]
    if embedding.shape[1] == 1:
        embedding = np.concatenate([embedding, np.zeros((len(embedding), 1), dtype=np.float32)], axis=1)
    return embedding, embed_labels


def directed_nn_distances(a: cp.ndarray, b: cp.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, dtype=np.float32)
    a_sq = cp.sum(a * a, axis=1, keepdims=True)
    b_sq = cp.sum(b * b, axis=1)
    d2 = cp.maximum(a_sq + b_sq[None, :] - 2.0 * (a @ b.T), 0.0)
    return cp.asnumpy(cp.sqrt(cp.min(d2, axis=1))).astype(np.float32, copy=False)


def closest_cluster_pairs_to_epsilon(
    cluster_features: cp.ndarray,
    labels: np.ndarray,
    epsilon: float,
    seed: int,
    top_k: int = 5,
    sample_size: int = 2048,
):
    cluster_ids = np.array(sorted(label for label in np.unique(labels) if label >= 0), dtype=np.int32)
    if len(cluster_ids) < 2:
        return []

    rng = np.random.default_rng(seed)
    sampled = {}
    for label in cluster_ids:
        idx = np.flatnonzero(labels == label)
        if len(idx) > sample_size:
            idx = rng.choice(idx, size=sample_size, replace=False)
        sampled[int(label)] = cluster_features[cp.asarray(np.sort(idx).astype(np.int64))]

    stats = []
    for i, label_a in enumerate(cluster_ids):
        feats_a = sampled[int(label_a)]
        for label_b in cluster_ids[i + 1 :]:
            feats_b = sampled[int(label_b)]
            dists = np.concatenate(
                [
                    directed_nn_distances(feats_a, feats_b),
                    directed_nn_distances(feats_b, feats_a),
                ],
                axis=0,
            )
            if len(dists) == 0:
                continue
            p1, p5, p10 = np.percentile(dists, [1, 5, 10])
            stats.append(
                {
                    "pair": (int(label_a), int(label_b)),
                    "min": float(dists.min()),
                    "p1": float(p1),
                    "p5": float(p5),
                    "p10": float(p10),
                    "delta_p5_to_eps": float(p5 - epsilon),
                }
            )

    stats.sort(key=lambda x: abs(x["delta_p5_to_eps"]))
    return stats[:top_k]


def plot_feature_embedding(ax, embedding: np.ndarray, labels: np.ndarray, palette: dict[int, np.ndarray], title: str):
    ax.set_xticks([])
    ax.set_yticks([])
    nonneg = labels >= 0
    embedding = embedding[nonneg]
    labels = labels[nonneg]
    if len(embedding) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No labels >= 0", ha="center", va="center", transform=ax.transAxes)
        return

    unique_labels = nonnegative_unique_labels(labels)
    colors = point_colors(labels, palette)
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=4,
        c=colors,
        alpha=0.8,
        linewidths=0,
        rasterized=True,
    )
    n_visible_clusters = len(unique_labels)
    ax.set_title(f"{title} ({n_visible_clusters} visible clusters, {len(labels)} pts)")
    ax.text(
        0.02,
        0.02,
        f"labels: {label_list_text(labels)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    for label in unique_labels:
        pts = embedding[labels == label]
        if len(pts) == 0:
            continue
        center = np.median(pts, axis=0)
        ax.text(
            center[0],
            center[1],
            str(label),
            color="white",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"),
        )


def annotate_clusters(ax, labels: np.ndarray):
    for label in sorted(int(x) for x in np.unique(labels) if x >= 0):
        ys, xs = np.nonzero(labels == label)
        yc = ys.mean()
        xc = xs.mean()
        idx = np.argmin((ys - yc) ** 2 + (xs - xc) ** 2)
        ax.text(
            xs[idx],
            ys[idx],
            str(label),
            color="white",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"),
        )


def render_surface(pipeline, dataset, cameras, image_idx: int, min_accum: float):
    camera_ray_bundle = cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, render_features=False)
    image = dataset.get_data(image_idx)["image"]
    if torch.is_tensor(image):
        image = image.float() / 255.0 if image.dtype == torch.uint8 else image.float()
        image = image.cpu().numpy()

    device = pipeline.model.device
    with torch.no_grad():
        ray_bundle = camera_ray_bundle.to(device)
        depth = outputs["depth"].to(device).unsqueeze(-2)
        ray_samples = ray_bundle.get_ray_samples(depth, depth)
        points = ray_samples.frustums.get_positions().squeeze(-2)

    return {
        "image": image,
        "shape": image.shape[:2],
        "points": as_numpy(points).reshape(-1, 3).astype(np.float32),
        "valid": as_numpy(outputs["accumulation"]).reshape(-1) > min_accum,
    }


def aggregate_voxels(points: np.ndarray, feats: np.ndarray, voxel_size: float):
    if voxel_size <= 0.0:
        return points, feats
    origin = points.min(axis=0, keepdims=True)
    voxel_ids = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_ids, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    pts_sum = np.zeros((len(counts), points.shape[1]), dtype=np.float32)
    feat_sum = np.zeros((len(counts), feats.shape[1]), dtype=np.float32)
    np.add.at(pts_sum, inverse, points)
    np.add.at(feat_sum, inverse, feats)
    return pts_sum / counts[:, None], feat_sum / counts[:, None]


def remove_point_outliers(points: np.ndarray, feats: np.ndarray, std_ratio: float):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
    ind = np.asarray(ind, dtype=np.int64)
    return points[ind], feats[ind]


def export_instance_cloud(pipeline, ray_generator, num_points: int, num_rays_per_batch: int, min_accum: float, seed: int):
    num_images = len(ray_generator.cameras)
    heights = as_numpy(ray_generator.cameras.height).reshape(-1).astype(np.int64)
    widths = as_numpy(ray_generator.cameras.width).reshape(-1).astype(np.int64)
    rng = np.random.default_rng(seed)
    points = []
    feats = []
    total = 0
    iters = 0

    while total < num_points:
        camera_idx = rng.integers(0, num_images, size=num_rays_per_batch, endpoint=False, dtype=np.int64)
        y = (rng.random(num_rays_per_batch) * heights[camera_idx]).astype(np.int64)
        x = (rng.random(num_rays_per_batch) * widths[camera_idx]).astype(np.int64)
        ray_indices = torch.from_numpy(np.stack([camera_idx, y, x], axis=-1)).to(ray_generator.image_coords.device)

        with torch.no_grad():
            ray_bundle = ray_generator(ray_indices)
            outputs = pipeline.model(ray_bundle)
            mask = outputs["accumulation"].reshape(-1) > min_accum
            if not mask.any():
                iters += 1
                continue
            depth = outputs["depth"].unsqueeze(-2)
            ray_samples = ray_bundle.get_ray_samples(depth, depth)
            batch_points = ray_samples.frustums.get_positions().squeeze(-2)[mask]
            batch_feats = pipeline.model.instance_field.get_feature(ray_samples).squeeze(-2)[mask]

        points.append(as_numpy(batch_points).astype(np.float32))
        feats.append(as_numpy(batch_feats).astype(np.float32))
        total += int(mask.sum().item())
        iters += 1
        if iters == 1 or total >= num_points or iters % 10 == 0:
            log(f"sampled {total}/{num_points} valid surface points")

    points = np.concatenate(points, axis=0)
    feats = np.concatenate(feats, axis=0)
    if len(points) > num_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=num_points, replace=False)
        points = points[keep]
        feats = feats[keep]
    return points, feats


def show_cluster_cloud(points: np.ndarray, labels: np.ndarray, palette: dict[int, np.ndarray]):
    colors = (255.0 * point_colors(labels, palette)).astype(np.uint8)
    diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    server = viser.ViserServer()
    share_url = server.request_share_url()
    log(f"viser share url: {share_url}")
    server.scene.add_point_cloud(
        name="/instance_clusters",
        points=points,
        colors=colors,
        point_size=float(diag * 0.001),
    )
    log("showing clustered point cloud in viser; press Ctrl-C to exit")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return


def main():
    t0 = perf_counter()
    p = ArgumentParser()
    p.add_argument("--load-config", type=Path, required=True)
    p.add_argument("--split", choices=("train", "eval"), default="train")
    p.add_argument("--image-idx", type=int, default=0)
    p.add_argument("--min-accum", type=float, default=0.2)
    p.add_argument("--num-points", type=int, default=500_000)
    p.add_argument("--num-rays-per-batch", type=int, default=32768)
    p.add_argument("--cloud-cache", type=Path)
    p.add_argument("--remove-outliers", action=BooleanOptionalAction, default=True)
    p.add_argument("--std-ratio", type=float, default=10.0)
    p.add_argument("--voxel-frac", type=float, default=0.0)
    p.add_argument("--min-size", type=int, default=2048)
    p.add_argument("--min-samples", type=int, default=256)
    p.add_argument("--cluster-selection-epsilon", type=float, default=0.4)  # use 0.105 with bm1_new type config
    p.add_argument("--min-prob", type=float, default=0.0)
    p.add_argument("--preprocess", choices=("raw", "std", "l2"), default="raw")
    p.add_argument("--pca-max-points", type=int, default=100_000)
    p.add_argument("--eval-num-rays-per-chunk", type=int, default=1 << 16)
    p.add_argument("--show-viser-only", action=BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    log(f"loading pipeline from {args.load_config}")
    _, pipeline, _, _ = eval_setup(args.load_config, eval_num_rays_per_chunk=args.eval_num_rays_per_chunk, test_mode="val")
    dataset = pipeline.datamanager.train_dataset if args.split == "train" else pipeline.datamanager.eval_dataset
    ray_generator = pipeline.datamanager.train_ray_generator if args.split == "train" else pipeline.datamanager.eval_ray_generator
    cameras = pipeline.datamanager.train_ray_generator.cameras if args.split == "train" else pipeline.datamanager.eval_ray_generator.cameras
    assert dataset is not None

    cloud_cache = args.cloud_cache or default_cloud_cache_path(
        args.load_config, args.split, args.num_points, args.num_rays_per_batch, args.min_accum
    )
    if cloud_cache.exists():
        log(f"loading cached raw cloud from {cloud_cache}")
        cache = np.load(cloud_cache)
        source_points = cache["points"].astype(np.float32)
        source_feats = cache["feats"].astype(np.float32)
        if len(source_points) > args.num_points:
            rng = np.random.default_rng(args.seed)
            keep = rng.choice(len(source_points), size=args.num_points, replace=False)
            source_points = source_points[keep]
            source_feats = source_feats[keep]
    else:
        log(f"exporting {args.num_points} surface points from random dataset rays")
        source_points, source_feats = export_instance_cloud(
            pipeline, ray_generator, args.num_points, args.num_rays_per_batch, args.min_accum, args.seed
        )
        log(f"saving raw cloud cache to {cloud_cache}")
        np.savez(cloud_cache, points=source_points, feats=source_feats)

    if args.remove_outliers:
        log("removing point-cloud outliers")
        source_points, source_feats = remove_point_outliers(source_points, source_feats, args.std_ratio)

    if args.voxel_frac > 0.0:
        diag = np.linalg.norm(source_points.max(axis=0) - source_points.min(axis=0))
        voxel_size = float(diag * args.voxel_frac)
        log(f"aggregating source samples with voxel size {voxel_size:.6f}")
        source_points, source_feats = aggregate_voxels(source_points, source_feats, voxel_size)

    log(f"clustering {len(source_feats)} exported 3D instance points on GPU")
    cluster_features = preprocess_features(cp.asarray(source_feats), args.preprocess)

    clusterer = HDBSCAN(
        min_cluster_size=args.min_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        metric="euclidean",
        prediction_data=True,
    )
    labels_valid = cp.asnumpy(clusterer.fit_predict(cluster_features))
    if args.min_prob > 0.0:
        probs_valid = cp.asnumpy(clusterer.probabilities_)
        labels_valid[probs_valid < args.min_prob] = -1

    cluster_ids, cluster_counts = np.unique(labels_valid[labels_valid >= 0], return_counts=True)
    close_pairs = closest_cluster_pairs_to_epsilon(
        cluster_features,
        labels_valid,
        args.cluster_selection_epsilon,
        args.seed,
    )
    palette = cluster_palette(labels_valid, args.seed)

    if args.show_viser_only:
        show_cluster_cloud(source_points, labels_valid, palette)
        return

    log(f"rendering target image {args.image_idx}")
    target = render_surface(pipeline, dataset, cameras, args.image_idx, args.min_accum)
    valid = target["valid"]
    assert valid.any(), "No pixels survived the accumulation threshold."

    log("projecting global cluster labels back to the target image")
    tree = KDTree(source_points)
    nn = tree.query(target["points"][valid], k=1, return_distance=False).reshape(-1)

    flat_labels = np.full(len(valid), -2, dtype=np.int32)
    flat_labels[valid] = labels_valid[nn].astype(np.int32)
    label_image = flat_labels.reshape(target["shape"])
    cluster_vis = colorize(label_image, palette)
    visible_pixel_counts = label_pixel_counts(flat_labels[valid].astype(np.int32, copy=False))
    visible_cluster_ids = np.array(sorted(label for label in visible_pixel_counts if label >= 0), dtype=np.int32)
    n_visible_clusters = len(visible_cluster_ids)
    pca_embedding, pca_labels = compute_pca_projection(
        cluster_features,
        labels_valid,
        visible_pixel_counts,
        args.seed,
        args.pca_max_points,
    )
    lda_embedding, lda_labels = compute_lda_projection(
        cluster_features,
        labels_valid,
        visible_pixel_counts,
        args.seed,
        args.pca_max_points,
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    ax0, ax1 = axes[0]
    ax2, ax3 = axes[1]
    ax0.imshow(np.clip(target["image"], 0.0, 1.0))
    ax0.set_title(f"{args.split} image {args.image_idx}")
    ax0.axis("off")
    ax1.imshow(cluster_vis)
    ax1.set_title(f"HDBSCAN labels projected to image ({n_visible_clusters} visible clusters)")
    ax1.axis("off")
    annotate_clusters(ax1, label_image)
    ax1.text(
        0.02,
        0.02,
        f"labels: {label_list_text(visible_cluster_ids)}",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    plot_feature_embedding(ax2, pca_embedding, pca_labels, palette, "PCA of clustered features (labels >= 0)")
    plot_feature_embedding(ax3, lda_embedding, lda_labels, palette, "LDA of clustered features (labels >= 0)")
    fig.tight_layout()

    print(f"image:          {dataset.image_filenames[args.image_idx]}")
    print(f"valid_pixels:   {int(valid.sum())}")
    print(f"clusters:       {n_visible_clusters}")
    print(f"noise_fraction: {np.mean(labels_valid == -1):.3f}")
    print("cluster_points:")
    for label, count in zip(cluster_ids, cluster_counts):
        print(f"  {int(label)}: {int(count)}")
    print("closest_cluster_pairs_to_eps:")
    for stat in close_pairs:
        a, b = stat["pair"]
        print(
            f"  ({a}, {b}): min={stat['min']:.4f} p1={stat['p1']:.4f} "
            f"p5={stat['p5']:.4f} p10={stat['p10']:.4f} delta_p5_to_eps={stat['delta_p5_to_eps']:+.4f}"
        )
    print(f"elapsed_sec:    {perf_counter() - t0:.1f}")

    if args.out:
        log(f"saving plot to {args.out}")
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
    else:
        log("showing plot")
        plt.show()


if __name__ == "__main__":
    main()
