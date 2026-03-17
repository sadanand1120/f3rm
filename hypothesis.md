# Active Hypotheses

- None for `poster`. The final sweep exhausted the remaining loader, cache-format, and training-hyperparameter frontiers for this experiment.

# Final Conclusions

- Full experiment history now lives in `results.tsv`. This file is intentionally reduced to the final validated state instead of keeping the entire search diary.
- Final validated winner on `2026-03-16`: [summary.json](/robodata/smodak/repos/f3rm/benchmark_runs/measure/20260316_192659_measure_324d4ab/summary.json) with `wall_time_s=189.22`, `train_total_time_s=174.59`, `startup_overhead_s=14.62`, `train_iter_time_s=20.91ms`, `train_rays_per_sec=392955.62`, `peak_gpu_mem_mb=6094.32`, `eval_all_psnr=24.79`, `eval_all_ssim=0.9191`, `eval_all_lpips=0.2642`, and `max_quality_regression_pct=9.52`.
- Winning stack in [f3rm_config.py](/robodata/smodak/repos/f3rm/f3rm/f3rm_config.py): `camera_optimizer=off`, `predict_normals=False`, `num_proposal_iterations=1`, `num_proposal_samples_per_ray=(128, 96)`, `num_nerf_samples_per_ray=24`, `distortion_loss_mult=0.0025`, `feat_use_pe=False`, `feat_num_levels=10`, `feat_features_per_level=4`, `feat_train_ray_ratio=0.5`, `foreground_train_ray_ratio=0.25`, `images_on_gpu=True`, `pin_cpu_feature_cache=False`, `cpu_feature_cache_images=32`, `gpu_feature_cache_images=0`, plus `uint8` foreground label caches.
- Feature-side loading and cache hypotheses are exhausted on this scene. The validated keeps are the repeat-window cache path, cache-hit startup cleanup, `images_on_gpu=True` with `gpu_feature_cache_images=0`, `pin_cpu_feature_cache=False`, `cpu_feature_cache_images=32`, and `uint8` foreground labels. Split foreground cache policies, CPU-backed CLIP loading, and more aggressive CPU-cache cuts all lost.
- GPU-fill tuning is exhausted on this stack. Larger `train_num_rays_per_batch` and larger `eval_num_rays_per_chunk` both regressed despite the available VRAM headroom.
- Sampling is exhausted on this stack. `num_nerf_samples_per_ray=24` is the stable floor, `23` and `22` both lost in full measure, and `num_proposal_samples_per_ray=(128, 96)` is the only proposal-stage trim that improved the measured winner. Larger proposal counts, `proposal_initial_sampler="uniform"`, and re-expanding final NeRF samples all regressed.
- Base-width and auxiliary-head trims are exhausted. Safe TCNN widths like `32` still regressed, `48` falls off the `FullyFusedMLP` fast path, and smaller appearance or foreground/base head widths did not produce end-to-end wins.
- Earlier trainer and pipeline hypotheses are closed rather than active. The current code already incorporates the non-viewer import stubs, timing-write throttling, repeat-window batch feature reuse, and direct eval render path, and later benchmarks subsume the need to isolate those ideas any further.

# Out Of Scope Issues For Human

- Bulk-convert other scenes' `foreground_` caches from `(H, W, 2)` float16 one-hot maps to `(H, W)` `uint8` label maps now that the runtime supports both layouts. On `poster`, that was a simple cache rewrite rather than a re-extraction and reduced the foreground cache footprint by about `4x`.
- The dominant feature artifact is still the CLIP cache, not the foreground cache. On `poster`, `features/clip` is about `13G`, so the next substantial feature-format frontier is CLIP compression or quantization. That would require either re-extraction or a cache-conversion path plus a fresh quality sweep; I did not change it in this pass.
- If you want scene-wide cache migrations to be more turnkey later, add a small repo utility for cache conversion and metadata backfill. I only converted the local `poster` foreground cache in this pass because that was enough to validate the runtime path and benchmark the tradeoff.
