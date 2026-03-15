# Legacy Training Reference (Head-by-Head)

This document captures a legacy snapshot of the training behavior that was analyzed earlier.
It is intentionally based on prior findings and is not re-derived from current code.

## Scope

This covers the major supervised outputs ("heads") and how each one was trained:

- `rgb`
- `feature`
- `foreground`
- `centroid`
- `centroid_spread`
- `orientany_rx`
- `orientany_rz`
- `orientany_foreground`

It also includes inherited/base losses that were active in the same training objective.

## High-Level Execution Flow

Per train step, the flow was:

1. Sample rays from datamanager.
2. (Pipeline) Optionally refresh full-image caches for centroid/spread/orientany targets before model forward.
3. Run NeRF/proposal sampling to get per-sample weights along each ray.
4. Render each head to per-ray predictions by weighted integration over samples.
5. Build loss dict = base Nerfacto losses + F3RM custom head losses.

Important detail: custom head rendering used detached NeRF weights, so these head losses did not backprop through the density-weight computation path.

## Data / Target Loading Pipeline

### Direct per-ray batch targets

These came from feature loaders and were indexed per ray:

- `batch["feature"]`: CLIP/DINO feature tensor
- `batch["foreground"]`: 2-class foreground one-hot-like target
- `batch["orientany"]`: 9D target per pixel
  - channels `0:3` = `R_x`
  - channels `3:6` = `R_z`
  - channel `6` = confidence
  - channels `7:9` = foreground one-hot

Indexing logic:

- Ray indices were `(camera_idx, y, x)`.
- For each target type, `y`/`x` were rescaled to that target map resolution and gathered from per-camera tensors.

Loader behavior (legacy):

- If cached feature files existed, they were loaded.
- Otherwise, extraction ran and then batch loaders served data.
- `ORIENTANY_` maps were reconstructed from per-pixel foreground+instance-id and per-instance vector/confidence metadata into full `HxWx9`.

### Cache-generated targets (full-image -> per-ray sampling)

`centroid`, `centroid_spread`, and cached orientany vectors used a full-image cache path:

1. For cameras in the current train batch, render full images (depth, centroid preds, spread preds, orientany preds).
2. Build pseudo-GT for centroid/spread from SAM2 masks + rendered geometry.
3. Optionally EMA-blend pseudo-GT with current model predictions (scheduled).
4. Store CPU caches per camera.
5. During ray loss computation, pick per-ray GT from cache using `(cam, y, x)`.

Cold-start behavior:

- Cache supervision was disabled for an initial step window.
- During that phase, centroid and spread losses were effectively off.
- OrientAny still had direct batch supervision fallback.

## Head Definitions and Losses

## 1) RGB head (`rgb`)

### What it predicts

- Standard rendered RGB from the base NeRF field.

### Target path

- Uses `batch["image"]` from datamanager.
- Background/alpha blending is applied for loss computation.

### Loss

- `rgb_loss = MSE(pred_rgb, gt_rgb)` (mean squared error after blending).

## 2) Feature head (`feature`)

### What it predicts

- Per-ray distilled semantic feature embedding (e.g., CLIP-space).

### Target path

- `batch["feature"]`, gathered per ray from feature maps.
- No extra normalization step in the loss path was used in this snapshot.

### Loss

- `feature_loss = feat_loss_weight * MSE(pred_feature, gt_feature)`.

Legacy default weight was `1e-3`.

## 3) Foreground head (`foreground_logits`)

### What it predicts

- Binary foreground logits (2 classes) for the main foreground branch.

### Target path

- `batch["foreground"]` (2-channel one-hot-like target), gathered per ray.

### Loss

- Convert GT to class index via `argmax`.
- `foreground_loss = foreground_loss_weight * CrossEntropy(pred_logits, gt_class_idx)`.
- Supervised on all sampled rays.

Legacy configured weight was `1e-3`.

## 4) Centroid head (`centroid`)

### What it predicts

- 3D world-space centroid vector per ray (rendered from sample-wise centroid predictions).

### Target path (pseudo-GT)

- Built from SAM2-derived instance masks + rendered depth/ray geometry.
- For each instance:
  - Reproject pixel to world point.
  - Compute mean world point = instance centroid.
  - Assign that centroid to pixels in the instance.
- Optional filtering:
  - minimum instance area percent
  - minimum accumulation threshold
- Optional EMA blending with model predictions (per-segment mean blend, scheduled).
- Cache is built per camera, then per-ray values are gathered by `(cam,y,x)`.
- Only valid-mask pixels contribute to centroid MSE.

### Loss

- `centroid_loss = centroid_loss_weight * MSE(pred_centroid[valid], gt_centroid[valid])`.

Legacy configured weight was `1e-3`.

## 5) Centroid Spread head (`centroid_spread`)

### What it predicts

- 4 channels (despite some naming implying "2ch"):
  - channel 0: centroid error regression signal
  - channel 1: foreground probability logit (binary)
  - channels 2:4: 2-class softmax logits for foreground classification

### Target path

- Derived during centroid-cache construction:
  - `target[...,0]` = L2 distance between current full-image centroid prediction and centroid pseudo-GT
    - optionally EMA-blended with predicted spread ch0 under blend schedule
  - `target[...,1]` = foreground-valid mask (`1` foreground, `0` background)
- Per-ray target sampled from cached full-image spread GT.

### Losses

All scaled by `centroid_loss_weight`:

- `centroid_spread_error_loss = MSE(pred[:,0:1], target[:,0:1])`
- `centroid_spread_prob_loss = BCEWithLogits(pred[:,1:2], target[:,1:2])`
- `centroid_spread_prob_soft_loss = CrossEntropy(pred[:,2:4], (target[:,1] > 0.5))`

Legacy configured weight was `1e-3` (same multiplier for all spread terms).

## 6) OrientAny Foreground head (`orientany_foreground_logits`)

### What it predicts

- Binary foreground logits specific to the OrientAny branch.

### Target path

- Foreground target came directly from `batch["orientany"][:,7:9]`.
- This foreground target was not taken from orientany cache; it stayed direct-from-batch.

### Loss

- `orientany_foreground_loss = orientany_loss_weight * CrossEntropy(pred_logits, gt_class_idx)`.
- Supervised on all rays.

Legacy configured weight was `1e-4`.

## 7) OrientAny vector heads (`orientany_rx`, `orientany_rz`)

### What they predict

- Two orientation vectors per ray:
  - `R_x` (3D)
  - `R_z` (3D)

### Model input mode (legacy config)

- Legacy run used centroid-conditioned encoding for OrientAny (`orientany_use_xyz_encoding=False`), i.e. orientany heads used encoded centroid predictions rather than direct xyz encoding.

### Target path

Vector/confidence target used:

- cached orientany target (`7D`) after cache enable, or
- direct `batch["orientany"]` fallback before cache enable.

Supervision mask:

- `foreground` AND `confidence > 0.85`.
- Foreground was taken from orientany batch channels `7:9`.
- Confidence came from orientany target channel `6`.

Optional cache blending behavior:

- In cache build, `R_x` and `R_z` GT could be EMA-blended with per-segment mean predictions.
- Blending applied on foreground+high-confidence regions.

### Losses

- `orientany_rx_loss = orientany_loss_weight * (1 - mean_cosine_similarity(pred_rx[mask], gt_rx[mask]))`
- `orientany_rz_loss = orientany_loss_weight * (1 - mean_cosine_similarity(pred_rz[mask], gt_rz[mask]))`

Legacy configured weight was `1e-4`.

### Optional orthogonality regularizer

If enabled:

- `orientany_perp_loss = (orientany_loss_weight / 4) * mean((dot(norm(rx), norm(rz)))^2)` on the same supervision mask.

Legacy config enabled this regularizer.

## Inherited Base Losses (co-trained)

In addition to custom heads, base Nerfacto losses were part of the objective:

- `rgb_loss` (MSE, described above)
- `interlevel_loss` (proposal consistency term, scaled)
- `distortion_loss` (scaled)
- if normal prediction enabled:
  - `orientation_loss` (scaled)
  - `pred_normal_loss` (scaled)
- camera optimizer loss term(s) if camera optimization active

## Legacy Weight Snapshot

From the analyzed legacy configuration:

- `feat_loss_weight = 1e-3`
- `foreground_loss_weight = 1e-3`
- `centroid_loss_weight = 1e-3`
- `orientany_loss_weight = 1e-4`
- `predict_normals = True`
- `enable_orientany_perp_loss = True`
- `orientany_use_xyz_encoding = False`
- cache cold-start skip around `6000` steps

## Practical Interpretation

The training objective was a multi-task sum:

- base NeRF photometric + proposal/distortion (+ optional normals/camera opt)
- feature distillation (`feature`)
- binary foreground classification (`foreground`)
- pseudo-GT geometric centroid regression (`centroid`)
- centroid consistency/probability auxiliary terms (`centroid_spread`)
- orientany vector + orientany-foreground supervision (`orientany_*`)

The centroid/spread/orientany-vector pseudo-targets were strongly shaped by the cache refresh policy, SAM2 segmentation quality, confidence thresholding, and blend schedule.
