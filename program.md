# F3RM Autoresearch Program

## Mission

This experiment targets end-to-end F3RM training latency for the poster workloads.

Primary objective:
- minimize official `measure` end-to-end wall time, defined conceptually as a combination of `feature extraction, feature loading during training, and the rest of training`

Success criteria:
- `end_to_end_wall_s` must go down on the fixed `measure` profile
- extraction and training must still do the same conceptual work: CLIP feature extraction with aggregation, then RGB + feature-field training
- final unweighted metrics should not degrade by roughly more than 10-15% unless the speed win is clearly worth it
- code must stay succinct, readable, and extensible for adding future extractors

Secondary constraints:
- prioritize wins in extraction scheduling and feature loading before spending effort on the core trainer
- extraction must run on GPUs `6,7` and show overlapping activity on both GPUs
- training must run on GPU `6` only
- no cheating with pre-existing feature caches: the benchmark deletes `features/clip` before extraction and again after the run
- official `measure` runs should stay within roughly 2 hours on the provided hardware; `smoke` runs should stay comfortably under 20 minutes

## Runtime Context

All runtime-dependent commands must execute inside:
- container: `fresh`
- conda env: `f3rm`
- repo path: `/robodata/smodak/repos/f3rm`

Execution policy:
- host-side read-only inspection is fine for repo files
- project execution, training, extraction, and package inspection must run in the container
- do not edit Nerfstudio site-packages directly
- if a Nerfstudio patch is unavoidable, stage it in `nerfstudio_changes/` for manual copy instead

## Benchmark Harness

All official evaluation goes through the fixed harness:
- benchmark script: `benchmark_f3rm.py`
- smoke command: `python benchmark_f3rm.py --profile smoke`
- measure command: `python benchmark_f3rm.py --profile measure`

Harness guarantees:
- it re-enters `fresh` + `f3rm` automatically when launched from the host
- it deletes `datasets/.../features/clip` before extraction and again after the run
- it runs extraction and training as subprocesses
- it streams subprocess stdout/stderr live to the terminal while also saving raw logs
- it writes raw logs under `benchmark_runs/<profile>/<run_tag>/`
- it saves `extract_visual_check/visual_check_*.png`: 4 side-by-side `original image | PCA(feature)` plots built from the saved `.npy` files
- it monitors extraction GPU overlap and training GPU memory
- it prints a stable JSON summary block between `=== F3RM_BENCHMARK_SUMMARY_BEGIN ===` and `=== F3RM_BENCHMARK_SUMMARY_END ===`
- `measure` is the only authoritative profile for keep/discard decisions

Do not use ad hoc training commands for official comparisons once the harness exists.

## In-Scope Files

Read these before changing anything:
- `program.md`: this file, the experiment contract
- `benchmark_f3rm.py`: fixed measurement contract
- `f3rm/features/extract_features_standalone.py`: benchmarked extraction entrypoint and cache contract
- `f3rm/features/clip_extract.py`: CLIP worker creation, warmup, and scheduling
- `f3rm/features/utils.py`: multi-worker dispatch and per-image feature cache loading
- `f3rm/feature_datamanager.py`: feature-window load path during training
- `f3rm/pipeline.py`: training-step timing boundaries
- `f3rm/trainer.py`: final metric emission
- `f3rm/f3rm_config.py`: schedule, logging, and model/training defaults
- `f3rm/train_schedule.py`: derived schedule math

Mutable files:
- `f3rm/**`
- `hypothesis.md`

Fixed files once the loop starts:
- `benchmark_f3rm.py`
- `program.md`
- `analysis.py`

Auxiliary backlog:
- `hypothesis.md`

Keep `hypothesis.md` split into:
- `# Active Hypotheses`
- `# Out Of Scope Issues For Human`

## Setup

Before the experiment loop:

1. Create a run tag and branch: `autoresearch/<tag>`.
2. Re-read `program.md`, `benchmark_f3rm.py`, and `hypothesis.md`.
3. Verify the runtime assumptions:
   - container `fresh` is running
   - conda env `f3rm` activates successfully
   - datasets exist at `datasets/f3rm/test/poster_smoke` and `datasets/f3rm/test/poster2`
   - `ns-train` resolves inside the container
4. Run `python benchmark_f3rm.py --profile smoke`.
5. Visually inspect the 4 extraction plots under `benchmark_runs/smoke/<run_tag>/extract_visual_check/` and confirm (1) feature PCA looks good, and (2) the original images correspond one to one to the PCA feature visualizations with no obvious image-feature misalignment.
6. Create `results.tsv` if it does not exist yet, with this exact header:

```tsv
commit	end_to_end_wall_s	extract_wall_s	train_wall_s	extract_worker_init_s	extract_worker_warmup_s	extract_batch_compute_s	extract_per_image_write_s	extract_parallel_ok	peak_extract_gpu_mem_mb	peak_train_gpu_mem_mb	train_feature_cache_load_avg_s	train_feature_window_fetch_avg_s	train_feature_window_stack_avg_s	train_batch_load_avg_s	train_model_forward_avg_s	final_eval_all_psnr	final_eval_all_ssim	final_eval_all_lpips	final_train_feature_error	status	description
```

7. Run `python benchmark_f3rm.py --profile measure`.
8. Visually inspect the 4 extraction plots under `benchmark_runs/measure/<run_tag>/extract_visual_check/` before accepting the baseline.
9. Parse the summary JSON from stdout or `benchmark_runs/.../summary.json`.
10. Append one `measure` row to `results.tsv` for the baseline.
11. Refresh `hypothesis.md` and then START the loop (do not wait for user approval).

`results.tsv` must contain `measure` runs only. Never append `smoke` runs.

## Experiment Contract

Allowed changes:
- extraction scheduling and worker lifecycle inside `f3rm/features/`
- feature cache layout and loading logic inside `f3rm/features/` and `f3rm/feature_datamanager.py`
- trainer/config tuning inside `f3rm/` that preserves the benchmark’s training intent
- disabling non-essential accessories when they are not part of the true workload

Disallowed changes:
- changing the benchmark definition after the loop starts
- skipping feature extraction or reusing stale cache state
- collapsing the standalone extractor entrypoint into an extractor-specific implementation file (since I want it extensible for future extractors)
- changing the benchmark to stop training RGB or feature learning
- editing Nerfstudio site-packages directly

Guiding principles:
- prioritize the low-hanging fruit first
- preserve clean abstractions for future extractor additions
- remove complexity when it does not buy measurable speed
- treat the visual extraction check as a real correctness gate, not a decorative artifact
- record deferred ideas in `hypothesis.md` instead of letting them vanish
- prune abandoned scratch branches occasionally; the kept history should stay readable

## Result Format

Append one tab-separated `measure` row per experiment:

```tsv
commit	end_to_end_wall_s	extract_wall_s	train_wall_s	extract_worker_init_s	extract_worker_warmup_s	extract_batch_compute_s	extract_per_image_write_s	extract_parallel_ok	peak_extract_gpu_mem_mb	peak_train_gpu_mem_mb	train_feature_cache_load_avg_s	train_feature_window_fetch_avg_s	train_feature_window_stack_avg_s	train_batch_load_avg_s	train_model_forward_avg_s	final_eval_all_psnr	final_eval_all_ssim	final_eval_all_lpips	final_train_feature_error	status	description
```

Field notes:
- `extract_parallel_ok` is `1` for valid overlap on GPUs `6,7`, else `0`
- `peak_extract_gpu_mem_mb` is the higher of the monitored extraction GPUs
- `peak_train_gpu_mem_mb` is the peak memory on the single training GPU
- timing columns come from the benchmark summary and final timing scalars
- `final_eval_all_*` and `final_train_feature_error` come from `Final Metrics/*`
- `status` is one of `keep`, `discard`, or `crash`

## Keep/Discard Rule

Keep a change only if:
- the `measure` run completed successfully
- `extract_parallel_ok == 1`
- the saved extraction visual checks still look correct and image-feature alignment is plausible
- `end_to_end_wall_s` improved enough to justify the code change
- the final metrics stay within an acceptable tradeoff window

Discard a change if:
- the benchmark failed or extraction lost valid multi-GPU overlap
- the extraction visual check shows degradation or misalignment
- end-to-end time regressed
- the code got noticeably more complex for a tiny or noisy win
- final quality regressed too much for the speed gain

## Loop

Once setup is complete:

1. Inspect the git state.
2. Pick one hypothesis from `hypothesis.md`.
3. Edit only mutable files.
4. Commit the change.
5. Run `smoke` if the change is risky.
6. Inspect the 4 saved `smoke` extraction visual checks whenever the change touches extraction, cache layout, or image-feature indexing.
7. Run `measure`.
8. Inspect the 4 saved `measure` extraction visual checks before trusting the result.
9. Parse the benchmark summary and append one `measure` row to `results.tsv`.
10. Update `hypothesis.md`.
11. Keep the commit only if it wins by the rule above.
12. Continue immediately with the next idea.
