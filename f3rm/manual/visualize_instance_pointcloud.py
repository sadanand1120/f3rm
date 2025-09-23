"""
F3RM Instance Segmentation on Exported Pointcloud Data

PSEUDO-ALGORITHM:
Starting from saved pointcloud data (points.npy, features.npy, centroids.npy, fg_soft.npy):

1. DATA LOADING & PREPROCESSING:
   - Load points, RGB, CLIP features, predicted centroids, soft foreground probs
   - Filter to foreground points only (fg_prob > 0.6)
   - Normalize CLIP features to unit norm for cosine similarity

2. SEMANTIC OVERSEGMENTATION (Semantics First):
   - Apply spherical k-means (K_SEM=64) to CLIP features to create semantic shards
   - Each semantic shard contains points with similar semantic meaning
   - Skip tiny shards (< MIN_SEM_POINTS) to avoid noise

3. GEOMETRY SEEDING WITHIN SEMANTIC SHARDS:
   - Within each semantic shard, apply voxel clustering to predicted centroids
   - Each voxel with >= MIN_CLUSTER_POINTS becomes an instance seed
   - This gives us K instance centers {μ_k} distributed across semantic groups

4. RESTRICTED GEOMETRIC ASSIGNMENT:
   - Assign each point to nearest centroid seed within R_ASSIGN radius
   - CRITICAL: Points can ONLY be assigned to seeds within the SAME semantic shard
   - Remove instances that are too small (< MIN_CLUSTER_POINTS)

5. SEMANTIC PROTOTYPE COMPUTATION:
   - For each instance k, compute semantic prototype:
     g_k = mean(normalize(features_i) for all i where y_i = k)
   - Normalize prototypes to unit norm

6. EM-STYLE REFINEMENT (Restricted to Semantic Shards):
   - Compute soft assignment scores S_ik combining geometry + semantics:
     S_ik = exp(-||centroid_i - μ_k||²/σ_c²) × exp(λ × <f_i, g_k>)
   - Reassign: y_i = argmax_k S_ik (restricted to same semantic shard)
   - This is a product kernel: stay near centroid AND agree semantically

7. VISUALIZATION:
   - Color each instance with distinct color from tab20 colormap
   - Color non-foreground points black
   - Save as instances_colored.ply and show interactive 3D view

KEY INSIGHT: We use SEMANTICS-FIRST approach to prevent cross-semantic merging.
Instead of pure geometry-first clustering, we:
1. Segment by semantic similarity first (CLIP features)
2. Then use geometry (centroid predictions) to find instances within each semantic group
3. This ensures objects of different semantic classes won't be merged into same instance

This approach leverages both the semantic structure of CLIP embeddings and the geometric
precision of centroid predictions, giving more meaningful and stable instance segmentation.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


# Hardcoded inputs (exported directory)
DATA_DIR = Path("exports/betam1_noori_emasched")

# Hyperparams (contracted units) - tuned for fewer, meaningful instances
H_C = 0.1        # increased from 0.03: larger voxel size for fewer clusters
R_ASSIGN = 0.15   # increased from 0.06: larger assignment radius
MIN_CLUSTER_POINTS = 2000  # increased from 300: require larger clusters
FOREGROUND_THRESH = 0.6   # increased from 0.5: stricter foreground filtering
LAMBDA_SEM = 0.5  # increased from 1.5: stronger semantic influence
SIGMA_C = 0.03    # increased from 0.015: softer geometric constraint

# Semantics-first parameters
MIN_SEM_POINTS = 1500      # skip tiny semantic shards
MIN_SEM_CLUSTERS = 2       # minimum number of semantic clusters
MAX_SEM_CLUSTERS = 7     # maximum number of semantic clusters


def _voxel_seed_centers(votes: np.ndarray, h: float, min_points: int) -> np.ndarray:
    grid = np.floor(votes / h).astype(np.int32)
    uniq, inv, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    keep = np.where(counts >= min_points)[0]
    if len(keep) == 0:
        return np.empty((0, 3), dtype=np.float32)
    centers = []
    for k in keep:
        pts = votes[inv == k]
        centers.append(pts.mean(axis=0))
    return np.vstack(centers).astype(np.float32)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def automatic_semantic_clustering(features: np.ndarray, min_clusters: int = 4, max_clusters: int = 16) -> np.ndarray:
    """Automatically cluster CLIP features using MiniBatchKMeans with optimal K selection."""
    # Try different numbers of clusters and find the best one using silhouette score
    best_score = -1
    best_labels = None
    best_k = min_clusters

    print(f"  Testing K from {min_clusters} to {max_clusters}...")

    for k in tqdm(range(min_clusters, max_clusters + 1), desc="Testing K values", position=0, leave=True):
        # Use MiniBatchKMeans for memory efficiency
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=50000,  # Process in batches to save memory
            random_state=42,
            n_init=3
        )

        # Fit and predict
        labels = kmeans.fit_predict(features)

        # Compute silhouette score (higher is better)
        try:
            # Sample a subset for silhouette computation to save memory
            if len(features) > 10000:
                sample_indices = np.random.choice(len(features), 10000, replace=False)
                sample_features = features[sample_indices]
                sample_labels = labels[sample_indices]
                score = silhouette_score(sample_features, sample_labels, metric='cosine')
            else:
                score = silhouette_score(features, labels, metric='cosine')

            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k

            # Clear the line and print silhouette score cleanly
            tqdm.write(f"    K={k}: silhouette={score:.3f}")

        except Exception as e:
            tqdm.write(f"    K={k}: failed ({str(e)})")
            continue

    if best_labels is None:
        print("  Warning: All clustering attempts failed, using fallback")
        # Fallback: use minimum clusters
        kmeans = MiniBatchKMeans(n_clusters=min_clusters, batch_size=10000, random_state=42)
        best_labels = kmeans.fit_predict(features)
        best_k = min_clusters

    print(f"  Best K: {best_k} (silhouette={best_score:.3f})")
    return best_labels


if __name__ == "__main__":
    print(f"Loading data from {DATA_DIR}...")
    meta = json.loads((DATA_DIR / "metadata.json").read_text())
    points = np.load(DATA_DIR / meta['files']['points'])
    rgbs = np.load(DATA_DIR / meta['files']['rgbs'])
    features_path = meta['files']['features']
    feats = np.load(DATA_DIR / features_path)
    raw = meta.get('raw_arrays', {})
    if 'centroids' not in raw:
        print("centroids.npy missing. Re-export with updated exporter.")
        raise SystemExit(0)
    cents = np.load(DATA_DIR / raw['centroids'])
    if 'fg_soft' in raw:
        fg_soft = np.load(DATA_DIR / raw['fg_soft'])
        fg_mask = (fg_soft[..., 0] > FOREGROUND_THRESH)
    else:
        fg_mask = np.ones((len(points),), dtype=bool)
    print(f"Loaded {len(points):,} points, {fg_mask.sum():,} foreground (>{FOREGROUND_THRESH})")

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    m = cents[fg_mask]
    f = feats[fg_mask]
    rgb_valid = rgbs[fg_mask]

    print("Normalizing CLIP features...")
    f = _normalize_rows(f)

    # Move to GPU early for better performance
    f_tensor = torch.from_numpy(f).float().to(device)

    print("Automatic semantic clustering (MiniBatchKMeans)...")
    sem_labels = automatic_semantic_clustering(f, MIN_SEM_CLUSTERS, MAX_SEM_CLUSTERS)
    K_SEM = len(np.unique(sem_labels[sem_labels >= 0]))  # Get actual number of semantic clusters
    print(f"Found {K_SEM} semantic clusters automatically")

    # Show semantic clustering visualization immediately
    print("Showing semantic clustering visualization...")
    sem_colors = np.zeros_like(rgbs)
    sem_colors[~fg_mask] = [0.0, 0.0, 0.0]  # non-foreground = black

    for s in tqdm(np.unique(sem_labels), desc="Semantic coloring"):
        if s < 0:  # Skip noise points
            continue
        idx = np.where(sem_labels == s)[0]
        if idx.size >= MIN_SEM_POINTS:
            fg_indices = np.where(fg_mask)[0]
            full_indices = fg_indices[idx]
            color = np.array(plt.get_cmap("tab20")(s % 20))[:3]
            sem_colors[full_indices] = color

    sem_pcd = o3d.geometry.PointCloud()
    sem_pcd.points = o3d.utility.Vector3dVector(points)
    sem_pcd.colors = o3d.utility.Vector3dVector(sem_colors)
    o3d.visualization.draw_geometries([sem_pcd], window_name=f"Semantic Clustering: {K_SEM} clusters")

    # ===== COMPLETELY SEPARATE: GEOMETRY-ONLY PIPELINE =====
    print("\n" + "=" * 60)
    print("GEOMETRY-ONLY PIPELINE (Independent)")
    print("=" * 60)

    # Fresh copy of data for geometry-only
    m_geom = m.copy()
    fg_mask_geom = fg_mask.copy()

    print("Geometry-only: Voxel clustering on centroids...")
    # Use different hyperparams for geometry-only to avoid contamination
    H_C_GEOM = 0.15        # larger voxels for geometry-only
    MIN_CLUSTER_POINTS_GEOM = 1500  # different threshold

    centers_geom = _voxel_seed_centers(m_geom, H_C_GEOM, MIN_CLUSTER_POINTS_GEOM)
    if centers_geom.size == 0:
        print("Geometry-only: No centers found.")
        centers_geom = np.array([[0, 0, 0]], dtype=np.float32)  # fallback

    print(f"Geometry-only: Found {len(centers_geom)} centers")

    # Assign points to geometry centers (no semantic constraints)
    print("Geometry-only: Assigning points to centers...")
    labels_geom = np.full(m_geom.shape[0], -1, np.int32)

    m_tensor_geom = torch.from_numpy(m_geom).float().to(device)
    centers_tensor_geom = torch.from_numpy(centers_geom).float().to(device)

    # Simple nearest neighbor assignment
    d_geom = torch.cdist(m_tensor_geom, centers_tensor_geom)
    nn_geom = torch.argmin(d_geom, dim=1).cpu().numpy().astype(np.int32)
    nd_geom = torch.min(d_geom, dim=1).values.cpu().numpy().astype(np.float32)

    # Apply assignment radius
    R_ASSIGN_GEOM = 0.15  # different radius for geometry-only
    labels_geom = nn_geom.copy()
    labels_geom[nd_geom > R_ASSIGN_GEOM] = -1

    # Filter by size
    valid_lbls_geom, lbl_counts_geom = np.unique(labels_geom[labels_geom >= 0], return_counts=True)
    keep_geom = set(valid_lbls_geom[lbl_counts_geom >= MIN_CLUSTER_POINTS_GEOM])
    labels_geom = np.array([l if l in keep_geom else -1 for l in labels_geom], dtype=np.int32)

    print(f"Geometry-only: Kept {len(keep_geom)} instances after size filtering")

    # Show geometry-only visualization
    print("Showing geometry-only visualization...")
    geom_colors = np.zeros_like(rgbs)
    geom_colors[~fg_mask_geom] = [0.0, 0.0, 0.0]  # non-foreground = black

    for i, l in enumerate(tqdm(np.unique(labels_geom), desc="Geometry coloring")):
        if l < 0:  # Skip unassigned points
            continue
        idx = np.where(labels_geom == l)[0]
        if idx.size >= MIN_CLUSTER_POINTS_GEOM:
            # Map back to full pointcloud indices
            fg_indices = np.where(fg_mask_geom)[0]
            full_indices = fg_indices[idx]
            color = np.array(plt.get_cmap("tab20")(i % 20))[:3]
            geom_colors[full_indices] = color

    geom_pcd = o3d.geometry.PointCloud()
    geom_pcd.points = o3d.utility.Vector3dVector(points)
    geom_pcd.colors = o3d.utility.Vector3dVector(geom_colors)
    o3d.visualization.draw_geometries([geom_pcd], window_name=f"Geometry-Only Clustering: {len(np.unique(labels_geom[labels_geom >= 0]))} instances")

    # ===== COMPLETELY SEPARATE: SEM+GEOM PIPELINE =====
    print("\n" + "=" * 60)
    print("SEMANTICS + GEOMETRY PIPELINE (Independent)")
    print("=" * 60)

    # Fresh copy of data for sem+geom
    m_semgeom = m.copy()
    fg_mask_semgeom = fg_mask.copy()
    sem_labels_semgeom = sem_labels.copy()

    print("Sem+Geom: Geometry seeding per semantic shard...")
    centers_list, center_owner = [], []

    for s in tqdm(np.unique(sem_labels_semgeom), desc="Geometry seeding"):
        if s < 0:  # Skip noise points
            continue
        idx = np.where(sem_labels_semgeom == s)[0]
        if idx.size < MIN_SEM_POINTS:
            continue
        c_s = _voxel_seed_centers(m_semgeom[idx], H_C, MIN_CLUSTER_POINTS)
        if c_s.size == 0:
            continue
        centers_list.append(c_s)
        center_owner.extend([s] * len(c_s))

    if len(centers_list) == 0:
        print("Sem+Geom: No instance seeds found.")
        exit(0)

    centers = np.vstack(centers_list).astype(np.float32)
    center_owner = np.asarray(center_owner, dtype=np.int32)
    centers_tensor = torch.from_numpy(centers).float().to(device)
    print(f"Sem+Geom: Seeds: {len(centers)} from {len(np.unique(center_owner))} semantic shards")

    print("Sem+Geom: Assign points to seeds within SAME semantic shard...")
    labels = np.full(m_semgeom.shape[0], -1, np.int32)

    # Create tensor for Sem+Geom pipeline
    m_tensor_semgeom = torch.from_numpy(m_semgeom).float().to(device)

    for s in tqdm(np.unique(center_owner), desc="Point assignment"):
        pts = np.where(sem_labels_semgeom == s)[0]
        ctr_idx = np.where(center_owner == s)[0]
        if ctr_idx.size == 0 or pts.size == 0:
            continue
        d = torch.cdist(m_tensor_semgeom[pts], centers_tensor[ctr_idx])  # (P,Cs)
        nd, nn = torch.min(d, dim=1)
        nn = nn.cpu().numpy().astype(np.int32)
        nd = nd.cpu().numpy().astype(np.float32)
        labels[pts] = ctr_idx[nn]
        labels[pts[nd > R_ASSIGN]] = -1  # radius gate

    # drop tiny instances
    valid_lbls, lbl_counts = np.unique(labels[labels >= 0], return_counts=True)
    keep = set(valid_lbls[lbl_counts >= MIN_CLUSTER_POINTS])
    labels = np.array([l if l in keep else -1 for l in labels], dtype=np.int32)
    print(f"Sem+Geom: Kept {len(keep)} instances after size filtering")

    print("Sem+Geom: Compute semantic prototypes per instance...")
    K = centers.shape[0]
    protos = torch.zeros((K, f.shape[1]), dtype=torch.float32, device=device)

    for k in tqdm(range(K), desc="Prototypes"):
        idx = labels == k
        if idx.any():
            # Compute on GPU
            f_k = f_tensor[idx]
            f_k_norm = torch.nn.functional.normalize(f_k, dim=1)
            protos[k] = f_k_norm.mean(dim=0)

    protos = torch.nn.functional.normalize(protos, dim=1)
    print("Sem+Geom: One EM refinement (restricted within semantic shard)...")

    new_labels = labels.copy()
    for s in tqdm(np.unique(center_owner), desc="EM refinement"):
        pts = np.where(sem_labels_semgeom == s)[0]
        ctr_idx = np.where(center_owner == s)[0]
        if ctr_idx.size == 0 or pts.size == 0:
            continue
        d = torch.cdist(m_tensor_semgeom[pts], centers_tensor[ctr_idx])                  # (P,Cs)
        geom_term = torch.exp(-(d * d) / (SIGMA_C ** 2))
        # cosine similarity (stable whether or not NORM_CLIP=True due to proto norm)
        sim = f_tensor[pts] @ protos[ctr_idx].T                           # (P,Cs)
        sem_term = torch.exp(LAMBDA_SEM * sim)
        scores = geom_term * sem_term
        nn = torch.argmax(scores, dim=1).cpu().numpy().astype(np.int32)
        nd = torch.min(d, dim=1).values.cpu().numpy().astype(np.float32)
        new_labels[pts] = ctr_idx[nn]
        new_labels[pts[nd > R_ASSIGN]] = -1

    uniq_lbls = [l for l in np.unique(new_labels) if l >= 0]
    print(f"Sem+Geom: Final: {len(uniq_lbls)} instances")

    # Create final visualization with foreground filtering
    print("Creating final sem+geom visualization...")
    final_labels = np.full(len(points), -1, dtype=np.int32)
    final_labels[fg_mask_semgeom] = new_labels

    # Color assignment: instances get colors, non-foreground gets black
    cmap = plt.get_cmap("tab20")
    instance_colors = np.zeros_like(rgbs)

    # Color instances
    for i, l in enumerate(tqdm(uniq_lbls, desc="Instance coloring")):
        color = np.array(cmap(i % 20))[:3]
        mask = final_labels == l
        instance_colors[mask] = color

    # Color non-foreground points black
    non_fg_mask = ~fg_mask_semgeom
    instance_colors[non_fg_mask] = [0.0, 0.0, 0.0]

    print("Showing final sem+geom instance segmentation visualization...")
    # Then show the final instance segmentation (semantics + geometry)
    instance_pcd = o3d.geometry.PointCloud()
    instance_pcd.points = o3d.utility.Vector3dVector(points)
    instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors)
    o3d.visualization.draw_geometries([instance_pcd], window_name=f"Final Sem+Geom: {len(uniq_lbls)} instances")

    # New visualization: overlay centroid predictions using sem+geom instance colors
    print("Showing centroid predictions overlaid with instance colors...")
    # Background RGB pointcloud with reduced intensity to simulate opacity, and downsampled for clarity
    bg_intensity = 0.1
    ds_ratio = 0.05
    bg_colors_full = (rgbs * bg_intensity).astype(np.float64)
    n_pts = points.shape[0]
    if n_pts > 0:
        keep = max(1, int(n_pts * ds_ratio))
        idx_bg = np.random.choice(n_pts, keep, replace=False)
        bg_points = points[idx_bg].astype(np.float64)
        bg_colors = bg_colors_full[idx_bg]
    else:
        bg_points = points.astype(np.float64)
        bg_colors = bg_colors_full
    background_pcd = o3d.geometry.PointCloud()
    background_pcd.points = o3d.utility.Vector3dVector(bg_points)
    background_pcd.colors = o3d.utility.Vector3dVector(bg_colors)

    # Centroid predictions colored by sem+geom instance colors
    valid_idx = np.where(final_labels >= 0)[0]
    if valid_idx.size > 0:
        cent_points = cents[valid_idx].astype(np.float64)
        cent_colors = instance_colors[valid_idx].astype(np.float64)
        centroids_pcd = o3d.geometry.PointCloud()
        centroids_pcd.points = o3d.utility.Vector3dVector(cent_points)
        centroids_pcd.colors = o3d.utility.Vector3dVector(cent_colors)

        # Add a small sphere per instance center, colored as the instance
        # Radius = 80th percentile of distances of that instance's predicted centroids (like debug_segpc)
        spheres_mesh = o3d.geometry.TriangleMesh()
        for l in uniq_lbls:
            inst_mask = (final_labels == l)
            if not np.any(inst_mask):
                continue
            inst_centroids = cents[inst_mask]
            if inst_centroids.size == 0:
                continue
            center_point = inst_centroids.mean(axis=0)
            if inst_centroids.shape[0] >= 2:
                dists = np.linalg.norm(inst_centroids - center_point[None, :], axis=1)
                radius = float(np.percentile(dists, 80.0))
            else:
                radius = 0.01
            radius = max(radius, 0.01)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            inst_color = instance_colors[inst_mask][0].astype(np.float64)
            sphere.paint_uniform_color(inst_color)
            sphere.translate(center_point.astype(np.float64))
            spheres_mesh += sphere

        o3d.visualization.draw_geometries(
            # [background_pcd, centroids_pcd, spheres_mesh],
            [background_pcd, spheres_mesh],
            window_name=f"Centroid Predictions (overlay) — {len(uniq_lbls)} instances",
            width=1200,
            height=800,
            left=50,
            top=50
        )
    else:
        # Fallback: show only background if no valid instances
        o3d.visualization.draw_geometries(
            [background_pcd],
            window_name="Centroid Predictions (overlay) — 0 instances",
            width=1200,
            height=800,
            left=50,
            top=50
        )
