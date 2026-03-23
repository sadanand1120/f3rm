from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from cuml.cluster import HDBSCAN
from cuml.preprocessing import StandardScaler
from nerfstudio.utils.eval_utils import eval_setup
from sklearn.neighbors import KDTree


def as_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def log(msg):
    print(f"[demo] {msg}", flush=True)


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
        feats = pipeline.model.instance_field.get_feature(ray_samples).squeeze(-2)

    return {
        "image": image,
        "shape": image.shape[:2],
        "points": as_numpy(points).reshape(-1, 3).astype(np.float32),
        "feats": as_numpy(feats).reshape(-1, feats.shape[-1]).astype(np.float32),
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


def main():
    t0 = perf_counter()
    p = ArgumentParser()
    p.add_argument("--load-config", type=Path, required=True)
    p.add_argument("--split", choices=("train", "eval"), default="train")
    p.add_argument("--image-idx", type=int, default=0)
    p.add_argument("--min-accum", type=float, default=0.2)
    p.add_argument("--num-cluster-images", type=int, default=32)
    p.add_argument("--cluster-pixel-stride", type=int, default=8)
    p.add_argument("--voxel-frac", type=float, default=2e-5)
    p.add_argument("--min-size", type=int, default=2048)
    p.add_argument("--min-samples", type=int, default=256)
    p.add_argument("--cluster-selection-epsilon", type=float, default=0.02)
    p.add_argument("--min-prob", type=float, default=0.0)
    p.add_argument("--preprocess", choices=("raw", "std", "l2"), default="raw")
    p.add_argument("--eval-num-rays-per-chunk", type=int, default=1 << 16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    log(f"loading pipeline from {args.load_config}")
    _, pipeline, _, _ = eval_setup(args.load_config, eval_num_rays_per_chunk=args.eval_num_rays_per_chunk, test_mode="val")
    dataset = pipeline.datamanager.train_dataset if args.split == "train" else pipeline.datamanager.eval_dataset
    cameras = pipeline.datamanager.train_ray_generator.cameras if args.split == "train" else pipeline.datamanager.eval_ray_generator.cameras
    assert dataset is not None

    n_cluster = min(args.num_cluster_images, len(dataset))
    cluster_ids = np.linspace(0, len(dataset) - 1, n_cluster, dtype=np.int32)
    if args.image_idx not in cluster_ids:
        cluster_ids[0] = args.image_idx
    cluster_ids = np.unique(cluster_ids)

    log(f"rendering target image {args.image_idx} and {len(cluster_ids)} source views")
    target = render_surface(pipeline, dataset, cameras, args.image_idx, args.min_accum)
    source_points = []
    source_feats = []
    stride = max(1, args.cluster_pixel_stride)

    for idx in cluster_ids:
        surf = target if int(idx) == args.image_idx else render_surface(pipeline, dataset, cameras, int(idx), args.min_accum)
        h, w = surf["shape"]
        keep = np.zeros((h, w), dtype=bool)
        keep[::stride, ::stride] = True
        keep = keep.reshape(-1) & surf["valid"]
        source_points.append(surf["points"][keep])
        source_feats.append(surf["feats"][keep])

    source_points = np.concatenate(source_points, axis=0)
    source_feats = np.concatenate(source_feats, axis=0)
    valid = target["valid"]
    assert valid.any(), "No pixels survived the accumulation threshold."

    if args.voxel_frac > 0.0:
        diag = np.linalg.norm(source_points.max(axis=0) - source_points.min(axis=0))
        voxel_size = float(diag * args.voxel_frac)
        log(f"aggregating source samples with voxel size {voxel_size:.6f}")
        source_points, source_feats = aggregate_voxels(source_points, source_feats, voxel_size)

    log(f"clustering {len(source_feats)} global 3D source samples on GPU")
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
    ax1.set_title(f"HDBSCAN on global 3D instance samples ({n_clusters} clusters)")
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
