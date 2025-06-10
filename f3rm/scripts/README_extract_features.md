# Standalone Feature Extraction

The `extract_features_standalone.py` script allows you to extract and cache features independently of the training pipeline. This is useful for:

- Pre-processing features before training to save time
- Extracting features for multiple feature types without re-running training
- Running feature extraction on different hardware than training

## Architecture & No Circular Imports

To avoid code duplication and circular import issues, the architecture is designed as follows:

1. **Standalone script is the source of truth**: All feature extraction logic, `LazyFeatures` class, and utility functions are defined in `extract_features_standalone.py`

2. **Feature datamanager imports from standalone**: `feature_datamanager.py` imports and uses the shared functions from the standalone script

3. **No reverse imports**: The standalone script never imports from `feature_datamanager.py`, preventing circular dependencies

4. **All imports at the top**: No embedded imports within functions to ensure clean dependency resolution

### Key Components in Standalone Script:
- `LazyFeatures` class - Memory-mapped feature access
- `extract_features_for_dataset()` - Main extraction logic
- `feature_saver()` / `feature_loader()` - Caching utilities  
- `_async_to_cuda()` - CUDA async utilities

## Usage

```bash
# Extract CLIP features
python f3rm/scripts/extract_features_standalone.py \
    --data datasets/f3rm/custom/scene001 \
    --feature-type CLIP

# Extract DINO features with custom shard size
python f3rm/scripts/extract_features_standalone.py \
    --data datasets/f3rm/custom/scene001 \
    --feature-type DINO \
    --shard-size 32

# Extract both DINO and CLIP for DINOCLIP training
python f3rm/scripts/extract_features_standalone.py \
    --data datasets/f3rm/custom/scene001 \
    --feature-type DINOCLIP

# Force re-extraction even if cache exists
python f3rm/scripts/extract_features_standalone.py \
    --data datasets/f3rm/custom/scene001 \
    --feature-type CLIP \
    --force
```

## Arguments

- `--data`: Path to the dataset directory (must contain `transforms.json`)
- `--feature-type`: Feature type to extract (`CLIP`, `DINO`, `ROBOPOINTproj`, `ROBOPOINTnoproj`, `DINOCLIP`)
- `--shard-size`: Number of images per shard (default: 64)
- `--device`: Device to use (`auto`, `cuda`, `cpu`, `cuda:0`, etc.)
- `--force`: Force re-extraction even if cache exists

## Output

The script creates feature shards in the exact same format and location as expected by the training pipeline:

```
datasets/f3rm/custom/scene001/
├── features/
│   ├── clip/
│   │   ├── chunk_0000.npy
│   │   ├── chunk_0001.npy
│   │   ├── ...
│   │   └── meta.pt
│   └── dino/
│       ├── chunk_0000.npy
│       ├── chunk_0001.npy
│       ├── ...
│       └── meta.pt
├── images/
├── transforms.json
└── ...
```

## Integration with Training

Once features are extracted, training will automatically use the cached features:

```bash
# This will use the pre-extracted CLIP features
ns-train f3rm \
    --data datasets/f3rm/custom/scene001 \
    --pipeline.datamanager.feature-type CLIP
```

The training pipeline checks for cached features and will skip extraction if they match the expected configuration. 