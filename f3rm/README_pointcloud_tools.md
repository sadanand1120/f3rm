# F3RM Feature Pointcloud Tools

This directory contains tools for exporting, visualizing, and analyzing F3RM feature pointclouds with open-vocabulary semantic similarity capabilities.

## Overview

The tools consist of three main components:

1. **`export_feature_pointcloud.py`** - Export RGB, features, and feature_PCA pointclouds
2. **`visualize_feature_pointcloud.py`** - Interactive visualization with RGB/PCA switching  
3. **`semantic_similarity_utils.py`** - Open-vocabulary semantic similarity utilities

## Quick Start

### 1. Export Feature Pointcloud

First, export a pointcloud from your trained F3RM model:

```bash
python export_feature_pointcloud.py \
    --config outputs/sittingroom/f3rm/2025-04-07_123618/config.yml \
    --output-dir exports/sitting_pcd_features/ \
    --num-points 1000000 \
    --bbox-min -1 -1 -1 \
    --bbox-max 1 1 1
```

This creates:
- `pointcloud_rgb.ply` - Standard RGB pointcloud
- `pointcloud_feature_pca.ply` - PCA-projected feature visualization  
- `features_float16.npy` - Compressed feature vectors
- `points.npy` - Point coordinates
- `pca_params.pkl` - PCA parameters for consistency
- `metadata.json` - Export metadata

### 2. Interactive Visualization

Launch the interactive visualizer:

```bash
python visualize_feature_pointcloud.py \
    --data-dir exports/sitting_pcd_features/
```

**Controls:**
- `R` - Switch to RGB mode
- `P` - Switch to PCA mode  
- `S` - Perform semantic similarity query
- `Q` - Quit

### 3. Semantic Analysis

Use the semantic similarity utilities following the `opt.py` approach:

```python
from f3rm.semantic_similarity_utils import SemanticPointcloudAnalyzer
import numpy as np

# Load exported data
points = np.load("exports/sitting_pcd_features/points.npy")
features = np.load("exports/sitting_pcd_features/features_float16.npy")

# Create analyzer
analyzer = SemanticPointcloudAnalyzer(features, points)

# Query for objects (same as opt.py)
similarities = analyzer.query_similarity("magazine", negatives=["object"])
magazine_mask = similarities > 0.502  # Same threshold as opt.py

# Find object instances
clusters = analyzer.find_object_instances("magazine", threshold=0.502)
for i, cluster in enumerate(clusters):
    center = points[cluster].mean(axis=0)
    print(f"Magazine {i+1}: {len(cluster)} points at {center}")
```

## Detailed Usage

### Export Options

The export script provides several optimization options:

```bash
python export_feature_pointcloud.py \
    --config path/to/config.yml \
    --output-dir path/to/output/ \
    --num-points 50000000 \           # Number of points to sample
    --bbox-min -1 -1 -1 \             # Bounding box minimum  
    --bbox-max 1 1 1 \                # Bounding box maximum
    --no-bbox \                       # Disable bounding box filtering
    --no-compress                     # Use float32 instead of float16
```

**Data Type Optimization:**
- Features are saved as `float16` by default (50% size reduction)
- Use `--no-compress` for full `float32` precision if needed
- RGB and PCA pointclouds use standard PLY format

### Semantic Similarity Features

The semantic utilities implement the same approach as `opt.py`:

```python
# Same method as opt.py
text_queries = ["magazine", "object"]
text_features = model.encode_text(tokenizer(text_queries))
sims = compute_similarity_text2vis(features, text_features, has_negatives=True, softmax_temp=1.0)
magazine_mask = (sims > 0.502)
```

**Key capabilities:**
- **Object Detection**: Find instances of specific objects
- **Multi-Query Comparison**: Compare different semantic queries
- **Spatial Analysis**: Analyze spatial distribution of matches
- **Clustering**: Group semantically similar regions
- **Export Analysis**: Save comprehensive analysis to JSON

### Interactive Visualization

The visualizer provides:

- **Mode Switching**: Toggle between RGB and feature_PCA visualization
- **Semantic Queries**: Real-time open-vocabulary queries  
- **Threshold Control**: Adjust similarity thresholds interactively
- **Cluster Analysis**: Automatic clustering of similar regions
- **Statistics**: Display query statistics and cluster information

### Example Workflow

Complete workflow for semantic analysis:

```python
# 1. Export pointcloud
python export_feature_pointcloud.py --config config.yml --output-dir data/

# 2. Load and analyze
from f3rm.semantic_similarity_utils import SemanticPointcloudAnalyzer
analyzer = SemanticPointcloudAnalyzer(features, points)

# 3. Find objects
magazine_clusters = analyzer.find_object_instances("magazine", threshold=0.502)
chair_clusters = analyzer.find_object_instances("chair", threshold=0.5)

# 4. Compare queries  
results = analyzer.compare_queries(["chair", "table", "book", "magazine"])

# 5. Spatial analysis
spatial_info = analyzer.spatial_analysis("chair", threshold=0.5)
print(f"Chair density: {spatial_info['density']:.3f} points/volume")

# 6. Export comprehensive analysis
analyzer.export_semantic_analysis(
    ["chair", "table", "book", "magazine"], 
    "semantic_analysis.json"
)

# 7. Interactive visualization
python visualize_feature_pointcloud.py --data-dir data/
```

## Integration with Existing F3RM Code

These tools are designed to integrate seamlessly with existing F3RM workflows:

- **Consistent with `opt.py`**: Uses the same `compute_similarity_text2vis` function and approach
- **Same CLIP model**: Uses `CLIPArgs.model_name` and `CLIPArgs.model_pretrained`  
- **Compatible with `render_feature_video.py`**: Uses the same PCA projection for consistency
- **Memory efficient**: Chunked processing for large pointclouds
- **GPU accelerated**: Automatic GPU usage when available

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
└── semantic_query_*.ply        # Generated by queries
```

## Performance Considerations

- **Memory Usage**: Features are compressed to `float16` by default
- **GPU Memory**: Chunked processing prevents OOM errors
- **Caching**: Semantic queries are cached for repeated analysis
- **Parallel Processing**: Uses efficient PyTorch operations

## Comparison with ns-export pointcloud

Unlike the standard `ns-export pointcloud`, these tools provide:

✅ **Feature extraction** in addition to RGB  
✅ **Consistent PCA projection** across all points  
✅ **Memory optimization** with float16 compression  
✅ **Open-vocabulary semantic queries**  
✅ **Interactive visualization** with mode switching  
✅ **Spatial clustering** and analysis  
✅ **Export compatibility** with analysis tools  

The standard export only provides RGB pointclouds, while these tools extract the full semantic feature representation for analysis.

## Troubleshooting

**Common Issues:**

1. **Out of Memory**: Reduce `--num-points` or use smaller `chunk_size`
2. **CLIP Loading**: Ensure you have `open_clip` installed and GPU access
3. **File Not Found**: Check paths and ensure export completed successfully  
4. **Visualization Issues**: Ensure Open3D is installed with GUI support

**Performance Tips:**

- Use bounding box filtering to focus on regions of interest
- Start with fewer points for initial testing
- Cache semantic queries for repeated analysis
- Use compressed features unless full precision is needed

---

These tools extend F3RM's capabilities from video rendering to comprehensive pointcloud analysis with open-vocabulary semantic understanding, following the same principled approach demonstrated in `opt.py`. 