# F3RM Feature Pointcloud Tools

This directory contains tools for exporting, visualizing, and analyzing F3RM feature pointclouds with open-vocabulary semantic similarity capabilities.

## Overview

The tools consist of four main components:

1. **`export_feature_pointcloud.py`** - Export RGB, features, and feature_PCA pointclouds with bounding box filtering
2. **`align_pointcloud.py`** - Interactive alignment tool to orient pointclouds with coordinate axes
3. **`visualize_feature_pointcloud.py`** - Simple visualization with RGB/PCA/semantic modes and coordinate guides
4. **`semantic_similarity_utils.py`** - Advanced semantic analysis utilities (reuses visualizer code)

## Quick Start

### 1. Export Feature Pointcloud

Export a pointcloud from your trained F3RM model with bounding box filtering to remove outliers:

```bash
python export_feature_pointcloud.py \
    --config outputs/sittingroom/f3rm/2025-04-07_123618/config.yml \
    --output-dir exports/sitting_pcd_features/ \
    --num-points 1000000 \
    --bbox-min -1 -1 -1 \
    --bbox-max 1 1 1
```

**Bounding Box Filtering**: The default bbox of `[-1,-1,-1]` to `[1,1,1]` removes outlier points that are too far away, keeping only the core scene content.

### 2. Align Pointcloud (Optional but Recommended)

Interactively align your pointcloud with the coordinate system for better visualization:

```bash
python align_pointcloud.py --data-dir exports/sitting_pcd_features/
```

**Interactive Controls:**
- **Mouse**: Normal Open3D viewing (rotate, zoom, pan)
- **X/1**: Rotate ±10° around X-axis (Red)
- **Y/2**: Rotate ±10° around Y-axis (Green)  
- **Z/3**: Rotate ±10° around Z-axis (Blue)
- **W/T**: Move forward/back (±Y)
- **A/D**: Move left/right (±X)
- **F/V**: Move up/down (±Z)
- **R**: Reset transform
- **S**: Save transform and exit
- **Q**: Quit without saving

**Goal**: Align so the floor is on the XY plane and objects are properly oriented with the coordinate axes.

### 3. Visualize and Analyze

Visualize with direct control (following opt.py approach):

```bash
# RGB mode with coordinate guides
python visualize_feature_pointcloud.py --data-dir exports/sitting_pcd_features/ --mode rgb

# PCA mode without guides
python visualize_feature_pointcloud.py --data-dir exports/sitting_pcd_features/ --mode pca --no-guides

# Semantic mode with direct threshold control (like opt.py)
python visualize_feature_pointcloud.py \
    --data-dir exports/sitting_pcd_features/ \
    --mode semantic \
    --query "magazine" \
    --threshold 0.502 \
    --softmax-temp 1.0 \
    --negative-queries "object"
```

**Coordinate Guides**: Shows origin (0,0,0), axes (Red=X, Green=Y, Blue=Z), bounding box wireframe, and grid lines for spatial reference.

## Complete Workflow

The recommended workflow is:

```bash
# Step 1: Export pointcloud with features
python export_feature_pointcloud.py \
    --config outputs/scene/f3rm/config.yml \
    --output-dir exports/scene_pcd \
    --num-points 1000000

# Step 2: Align pointcloud with coordinate system
python align_pointcloud.py --data-dir exports/scene_pcd

# Step 3: Visualize and analyze
python visualize_feature_pointcloud.py --data-dir exports/scene_pcd --mode rgb
python visualize_feature_pointcloud.py --data-dir exports/scene_pcd --mode semantic --query "chair"
```

This creates:
- `pointcloud_rgb.ply` - Standard RGB pointcloud (aligned)
- `pointcloud_feature_pca.ply` - PCA-projected feature visualization (aligned)
- `features_float16.npy` - Compressed feature vectors (unchanged)
- `points.npy` - Point coordinates (aligned)
- `pca_params.pkl` - PCA parameters for consistency
- `metadata.json` - Export metadata with alignment transform

### 4. Advanced Semantic Analysis

Use the semantic similarity utilities for programmatic analysis:

```python
from f3rm.semantic_similarity_utils import SemanticPointcloudAnalyzer
import numpy as np

# Load exported data (automatically uses aligned data if available)
points = np.load("exports/sitting_pcd_features/points.npy")
features = np.load("exports/sitting_pcd_features/features_float16.npy")

# Create analyzer
analyzer = SemanticPointcloudAnalyzer(features, points)

# Query for objects (same as opt.py)
similarities = analyzer.query_similarity("magazine", negatives=["object"], softmax_temp=1.0)
magazine_mask = similarities > 0.502  # Same threshold as opt.py

# Find object instances
clusters = analyzer.find_object_instances("magazine", threshold=0.502, softmax_temp=1.0)
for i, cluster in enumerate(clusters):
    center = points[cluster].mean(axis=0)
    print(f"Magazine {i+1}: {len(cluster)} points at {center}")
```

## Detailed Usage

### Export Options

The export script provides bounding box filtering to remove outliers:

```bash
python export_feature_pointcloud.py \
    --config path/to/config.yml \
    --output-dir path/to/output/ \
    --num-points 50000000 \           # Number of points to sample
    --bbox-min -1 -1 -1 \             # Bounding box minimum (filters outliers)
    --bbox-max 1 1 1 \                # Bounding box maximum  
    --no-bbox \                       # Disable bounding box filtering
    --no-compress                     # Use float32 instead of float16
```

**Why Bounding Box Filtering?**
- Removes outlier points that are far from the main scene
- Default `[-1,-1,-1]` to `[1,1,1]` works well for indoor scenes
- Use `--no-bbox` to disable if you need the full point range
- Adjust `--bbox-min`/`--bbox-max` for different scene scales

### Semantic Similarity (Following opt.py)

The semantic tools implement the exact same approach as `opt.py`:

```python
# Same method as opt.py
text_queries = ["magazine", "object"]  # positive, then negatives
similarities = compute_similarity_text2vis(features, text_features, 
                                         has_negatives=True, softmax_temp=1.0)
magazine_mask = (similarities > 0.502)  # Same threshold
```

**Direct Control (No Adaptive Logic):**
- `--threshold 0.502` - Same default threshold as opt.py
- `--softmax-temp 1.0` - Same default temperature as opt.py  
- `--negative-queries "object"` - Same default negative as opt.py
- No adaptive threshold computation - you control the exact values

### Coordinate Guides

The visualizer shows helpful spatial references:

- **Origin**: Coordinate frame at (0,0,0) with colored axes
- **Axes**: Red=X, Green=Y, Blue=Z directions
- **Bounding Box**: Gray wireframe showing data extent
- **Grid**: Light gray grid lines on XY plane for scale reference
- **Control**: Use `--no-guides` to hide all reference elements

### Advanced Analysis Features

The semantic analyzer provides:

```python
# Multi-query comparison
queries = ["chair", "table", "book", "magazine"]
results = analyzer.compare_queries(queries, threshold=0.502, softmax_temp=1.0)

# Spatial analysis  
spatial_info = analyzer.spatial_analysis("chair", threshold=0.502, softmax_temp=1.0)
print(f"Chair density: {spatial_info['density']:.3f} points/volume")

# Export comprehensive analysis
analyzer.export_semantic_analysis(queries, "analysis.json", threshold=0.502, softmax_temp=1.0)
```

## Integration with Existing F3RM Code

These tools are designed to integrate seamlessly with existing F3RM workflows:

- **Consistent with `opt.py`**: Uses the same `compute_similarity_text2vis` function and thresholds
- **Same CLIP model**: Uses `CLIPArgs.model_name` and `CLIPArgs.model_pretrained`  
- **Compatible with `render_feature_video.py`**: Uses the same PCA projection for consistency
- **Memory efficient**: Chunked processing with bounding box filtering
- **GPU accelerated**: Automatic GPU usage when available
- **No code duplication**: semantic_similarity_utils.py reuses code from visualize_feature_pointcloud.py

## File Structure

After export, the directory structure will be:

```
exports/sitting_pcd_features/
├── pointcloud_rgb.ply           # RGB visualization
├── pointcloud_feature_pca.ply   # PCA visualization  
├── features_float16.npy         # Compressed features
├── points.npy                   # Point coordinates
├── pca_params.pkl              # PCA parameters
├── metadata.json               # Export metadata
└── semantic_query_*.ply        # Generated by semantic queries
```

## Example Workflows

### Basic Visualization Workflow

```bash
# 1. Export with bounding box filtering
python export_feature_pointcloud.py \
    --config outputs/scene/f3rm/config.yml \
    --output-dir exports/scene_pcd \
    --bbox-min -1 -1 -1 --bbox-max 1 1 1

# 2. Visualize RGB with guides
python visualize_feature_pointcloud.py --data-dir exports/scene_pcd --mode rgb

# 3. Visualize PCA features  
python visualize_feature_pointcloud.py --data-dir exports/scene_pcd --mode pca

# 4. Semantic query (following opt.py approach)
python visualize_feature_pointcloud.py \
    --data-dir exports/scene_pcd --mode semantic \
    --query "magazine" --threshold 0.502 --softmax-temp 1.0
```

### Advanced Analysis Workflow

```python
# Load data
import numpy as np
from f3rm.semantic_similarity_utils import SemanticPointcloudAnalyzer

points = np.load("exports/scene_pcd/points.npy")
features = np.load("exports/scene_pcd/features_float16.npy")
analyzer = SemanticPointcloudAnalyzer(features, points)

# Find object instances (same as opt.py)
magazine_clusters = analyzer.find_object_instances("magazine", threshold=0.502, softmax_temp=1.0)
chair_clusters = analyzer.find_object_instances("chair", threshold=0.502, softmax_temp=1.0)

# Compare multiple objects
results = analyzer.compare_queries(["chair", "table", "magazine"], threshold=0.502, softmax_temp=1.0)

# Analyze spatial distribution
spatial_info = analyzer.spatial_analysis("chair", threshold=0.502, softmax_temp=1.0)

# Export comprehensive analysis
analyzer.export_semantic_analysis(["chair", "table", "magazine"], "analysis.json", 
                                 threshold=0.502, softmax_temp=1.0)
```

## Performance Considerations

- **Bounding Box**: Reduces point count and improves performance
- **Memory Usage**: Features compressed to `float16` by default (50% reduction)
- **GPU Memory**: Chunked processing prevents OOM errors
- **Caching**: Semantic queries are cached for repeated analysis
- **Code Reuse**: No duplication between visualizer and semantic utils

## Comparison with ns-export pointcloud

Unlike the standard `ns-export pointcloud`, these tools provide:

✅ **Feature extraction** in addition to RGB  
✅ **Bounding box filtering** to remove outliers  
✅ **Consistent PCA projection** across all points  
✅ **Memory optimization** with float16 compression  
✅ **Open-vocabulary semantic queries** following opt.py approach  
✅ **Direct threshold control** (no adaptive logic)  
✅ **Coordinate guides** for spatial understanding  
✅ **Advanced analysis** with clustering and spatial statistics  
✅ **No code duplication** between components  

The standard export only provides RGB pointclouds, while these tools extract the full semantic feature representation with proper filtering and analysis capabilities.

## Troubleshooting

**Common Issues:**

1. **Out of Memory**: Reduce `--num-points` or use smaller bounding box
2. **No Points**: Expand bounding box with `--bbox-min`/`--bbox-max` or use `--no-bbox`
3. **CLIP Loading**: Ensure you have `open_clip` installed and GPU access
4. **Visualization Issues**: Ensure Open3D is installed with GUI support
5. **Import Errors**: Make sure f3rm package is in Python path

**Performance Tips:**

- Use bounding box filtering to focus on regions of interest
- Start with fewer points for initial testing  
- Use direct threshold control instead of adaptive logic
- Cache semantic queries for repeated analysis
- Use compressed features unless full precision is needed

---

These tools extend F3RM's capabilities from video rendering to comprehensive pointcloud analysis with open-vocabulary semantic understanding, following the same principled approach demonstrated in `opt.py` while avoiding code duplication and providing spatial filtering capabilities. 