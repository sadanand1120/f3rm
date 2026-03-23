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


def colorize(labels: np.ndarray, seed: int) -> np.ndarray:
    image = np.zeros((*labels.shape, 3), dtype=np.float32)
    image[labels == -1] = 0.65
    cluster_ids = np.array(sorted(label for label in np.unique(labels) if label >= 0), dtype=np.int64)
    if len(cluster_ids) == 0:
        return image
    colors = plt.cm.hsv(np.linspace(0.0, 1.0, len(cluster_ids), endpoint=False))[:, :3].astype(np.float32)
    rng = np.random.default_rng(seed)
    colors = colors[rng.permutation(len(colors))]
    for color, label in zip(colors, cluster_ids):
        image[labels == label] = color
    return image


def point_colors(labels: np.ndarray, seed: int) -> np.ndarray:
    colors = np.full((len(labels), 3), 0.65, dtype=np.float32)
    cluster_ids = np.array(sorted(label for label in np.unique(labels) if label >= 0), dtype=np.int64)
    if len(cluster_ids) == 0:
        return colors
    palette = plt.cm.hsv(np.linspace(0.0, 1.0, len(cluster_ids), endpoint=False))[:, :3].astype(np.float32)
    rng = np.random.default_rng(seed)
    palette = palette[rng.permutation(len(palette))]
    for color, label in zip(palette, cluster_ids):
        colors[labels == label] = color
    return colors


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


def show_cluster_cloud(points: np.ndarray, labels: np.ndarray, seed: int):
    colors = (255.0 * point_colors(labels, seed)).astype(np.uint8)
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
    p.add_argument("--voxel-frac", type=float, default=2e-4)
    p.add_argument("--min-size", type=int, default=2048)
    p.add_argument("--min-samples", type=int, default=256)
    p.add_argument("--cluster-selection-epsilon", type=float, default=0.02)
    p.add_argument("--min-prob", type=float, default=0.0)
    p.add_argument("--preprocess", choices=("raw", "std", "l2"), default="raw")
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

    if args.show_viser_only:
        show_cluster_cloud(source_points, labels_valid, args.seed)
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
    cluster_vis = colorize(label_image, args.seed)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 7))
    ax0.imshow(np.clip(target["image"], 0.0, 1.0))
    ax0.set_title(f"{args.split} image {args.image_idx}")
    ax0.axis("off")
    n_clusters = len(set(labels_valid)) - int(-1 in labels_valid)
    ax1.imshow(cluster_vis)
    ax1.set_title(f"HDBSCAN on exported 3D instance cloud ({n_clusters} clusters)")
    ax1.axis("off")
    annotate_clusters(ax1, label_image)
    fig.tight_layout()

    print(f"image:          {dataset.image_filenames[args.image_idx]}")
    print(f"valid_pixels:   {int(valid.sum())}")
    print(f"clusters:       {n_clusters}")
    print(f"noise_fraction: {np.mean(labels_valid == -1):.3f}")
    print(f"elapsed_sec:    {perf_counter() - t0:.1f}")

    if args.out:
        log(f"saving plot to {args.out}")
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
    else:
        log("showing plot")
        plt.show()


if __name__ == "__main__":
    main()
