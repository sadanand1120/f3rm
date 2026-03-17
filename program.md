# F3RM Autoresearch Program

## Mission

This experiment is to improve end-to-end training time for F3RM on the fixed
`poster` benchmark scene.

Primary objective:
- minimize `wall_time_s` from the official `measure` benchmark profile

Success criterion:
- `wall_time_s` goes down on the official benchmark
- the final unweighted metrics from [trainer.py](/robodata/smodak/repos/f3rm/f3rm/trainer.py) do not regress by more than roughly `10% .. 15%`, unless a larger tradeoff is clearly justified
- the code stays succinct, readable, and justified by measured gains

Secondary constraints:
- keep the real training intent intact: RGB + feature + foreground training
- do not materially increase peak GPU memory without a clear speed win
- avoid benchmark-specific hacks, hidden shortcuts, or metric-shaping tricks
- prefer code compression or deletion over additive complexity unless the extra code clearly pays for itself

## Runtime Contract

Every runtime-dependent command must run inside the F3RM container/env workflow.
Do not run training, benchmark, debugging, or Nerfstudio package inspection on
the host.

- container: `fresh`
- conda env: `f3rm`
- env path: `/opt/miniconda3/envs/f3rm`
- repo path: `/robodata/smodak/repos/f3rm`
- GPU contract: set `CUDA_VISIBLE_DEVICES=1`
- Nerfstudio package for read-only inspection:
  `/opt/miniconda3/envs/f3rm/lib/python3.11/site-packages/nerfstudio`
- benchmark seed contract: `PYTHONHASHSEED=0`, `--machine.seed 0`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Canonical wrapper:

```bash
docker exec fresh bash -lc '
  source /opt/miniconda3/etc/profile.d/conda.sh &&
  conda activate f3rm &&
  export CUDA_VISIBLE_DEVICES=1 &&
  cd /robodata/smodak/repos/f3rm &&
  python benchmark_train_time.py --profile smoke
'
```

Use that shape for every runtime-dependent command. Host-side inspection with
`rg`, `sed`, `ls`, and similar tools is fine for repo-mounted files.

## Benchmark Harness

All official evaluation must go through the fixed benchmark script:

- benchmark script: `benchmark_train_time.py`
- smoke profile:

```bash
python benchmark_train_time.py --profile smoke
```

- measure profile:

```bash
python benchmark_train_time.py --profile measure
```

- measure profile with a baseline quality reference:

```bash
python benchmark_train_time.py \
  --profile measure \
  --baseline-summary benchmark_runs/measure/<baseline_run>/summary.json
```

The harness is intentionally close to the current user command:

```bash
ns-train f3rm \
  --machine.num-devices 1 \
  --machine.seed 0 \
  --vis tensorboard \
  --data datasets/f3rm/test/poster \
  --output-dir benchmark_outputs \
  --experiment-name poster-train-time
```

The official `measure` profile should stay almost identical to that command:

- same method, dataset, vis mode, seed, and single-device setup
- same config-owned training/eval/logging knobs
- only bookkeeping changes: output dir, experiment name, and `--timestamp`

The `smoke` profile is the only place where the harness adds scale-down flags:

- `--max-num-iterations 1024`
- `--logging.profiler none`
- `--steps-per-save 0`
- final-only eval at step `1023`
- `--pipeline.steps-per-train-image-viz 0`

That keeps smoke near a `1/5 .. 1/6` runtime slice while leaving the official
`measure` run conceptually identical to the real training command.

Train/model hyperparameters must stay owned by
[f3rm/f3rm_config.py](/robodata/smodak/repos/f3rm/f3rm/f3rm_config.py). The
harness must not be used to encode performance wins. If a speedup comes from
changing training behavior, put it in config or in the runtime code under
`f3rm/`.

## Coverage Invariant

For the official `measure` profile, the main training-coverage invariant is the
approximate per-pixel train-ray visitation budget:

- `X = (max_num_iterations * train_num_rays_per_batch) / total_train_pixels`

For the current poster baseline:

- train split: `215` images
- train image size: `1904 x 1050`
- `total_train_pixels = 429,828,000`
- baseline config:
  `max_num_iterations=8100`, `train_num_rays_per_batch=8192`,
  `train_num_images_to_sample_from=32`, `train_num_times_to_repeat_images=512`
- baseline `X ≈ 0.1544`

Interpretation:

- over the whole run, the baseline samples about `0.1544` train rays per train
  pixel on average
- you may tune `max_num_iterations`, `train_num_rays_per_batch`,
  `train_num_images_to_sample_from`, and `train_num_times_to_repeat_images` in
  [f3rm/f3rm_config.py](/robodata/smodak/repos/f3rm/f3rm/f3rm_config.py)
- when you do, keep `X` roughly constant for the official `measure` profile
- helper sanity check for the image-window schedule:
  `W = ceil(max_num_iterations / train_num_times_to_repeat_images) * train_num_images_to_sample_from / train_image_count`
- baseline `W ≈ 2.3814`, which is the average number of selected-window
  appearances per train image
- when you change the image-window knobs, keep `W` in the same ballpark so the
  train-image windows still cover the train split smoothly instead of starving
  some images

Benchmark harness requirements:

- it stays fixed during the experiment loop
- it runs `ns-train` as a subprocess inside the current container/env
- it writes raw combined stdout/stderr to
  `benchmark_runs/<profile>/<run_id>/train.log`
- it writes `summary.json` next to the log
- for every official `measure` run, it also writes a ready-to-paste
  `results_row.tsv` next to `summary.json`
- it prints a stable summary block at the end
- it exits non-zero on failure
- the `measure` profile is the only source of truth for keep/discard decisions

Do not evaluate changes with ad hoc `ns-train` commands once this benchmark
contract is in place.

## In-Scope Files

Read these before changing code:

- `program.md`: fixed experiment contract
- `f3rm/f3rm_config.py`: default training/config surface
- `f3rm/trainer.py`: final metrics, train loop, and non-finite safety path
- `f3rm/pipeline.py`: train/eval structure and timing hooks
- `f3rm/feature_datamanager.py`: batch sampling, feature loading, cache path
- `f3rm/model.py`: feature/foreground metrics, losses, eval rendering
- `f3rm/feature_field.py`: feature-head architecture and model-compression ideas

Mutable files:
- `f3rm/**/*.py`
- `hypothesis.md`

Read-only reference files:
- `/opt/miniconda3/envs/f3rm/lib/python3.11/site-packages/nerfstudio/**/*.py`
- sibling repos only when imported by F3RM code, such as `sam2` or `sam3`

Fixed files:
- `benchmark_train_time.py`
- `program.md`
- `analysis.py`

Auxiliary notes file:
- `hypothesis.md`: a living backlog of active hypotheses plus a separate
  `Out Of Scope Issues For Human` section

Do not edit Nerfstudio site-packages directly. If a Nerfstudio patch becomes
unavoidable, stage it under `nerfstudio_changes/` and stop for human review.

## Setup

Before starting the experiment loop:

1. Create a run tag and branch, for example `autoresearch/poster-train-time`.
2. Verify the runtime:
   - `fresh` is running
   - `conda activate f3rm` succeeds
   - current working directory is `/robodata/smodak/repos/f3rm`
   - commands are being run through the `docker exec fresh ... conda activate f3rm ...` wrapper
3. Verify the fixed assets exist:
   - `datasets/f3rm/test/poster/transforms.json`
   - cached `clip` features with `226` per-image files
   - cached `foreground_` features with `226` per-image files
4. Review `hypothesis.md`. It should contain these section headers:

```md
# Active Hypotheses

# Out Of Scope Issues For Human
```

5. Refresh `# Active Hypotheses` based on anything newly learned during setup.
6. Run the smoke profile once.
7. Create `results.tsv` if it does not exist. This file is measure-only and must
   never contain smoke rows. Use this header:

```tsv
commit	profile	wall_time_s	train_total_time_s	startup_overhead_s	train_iter_time_s	train_rays_per_sec	train_batch_load_s	train_feature_cache_load_s	train_feature_gather_s	train_feature_populate_s	train_model_forward_s	train_metrics_loss_s	train_image_viz_s	eval_batch_load_s	eval_feature_cache_load_s	eval_feature_gather_s	eval_feature_populate_s	eval_image_s	eval_all_images_s	peak_gpu_mem_mb	pixel_visit_x	pixel_visit_x_pct_diff	window_coverage_w	window_coverage_w_pct_diff	train_psnr	train_feature_error	train_foreground_acc	eval_all_psnr	eval_all_ssim	eval_all_lpips	max_quality_regression_pct	status	description	summary_path
```

8. Run the measure profile once on the current baseline without
   `--baseline-summary`.
9. Record that baseline row in `results.tsv` with
   `max_quality_regression_pct=0.0`.
10. Use that baseline `summary.json` for all later measure runs, then begin the
    loop.

## Experiment Contract

Each experiment should be one coherent idea.

Allowed changes:
- training-loop changes inside `f3rm/`
- dataloading or cache-path changes inside `f3rm/`
- model or hyperparameter changes inside `f3rm/`
- import cleanup or startup simplification if it measurably lowers benchmark wall time
- deleting code when it preserves the benchmark contract and improves speed or clarity

Disallowed changes:
- changing `benchmark_train_time.py`
- changing the dataset, seed, iteration count, or summary schema
- removing RGB, feature, or foreground supervision from the training objective
- bypassing final eval metrics or changing what the harness parses
- moving benchmarked execution outside the container/env contract
- hiding speedups inside harness-only CLI overrides that should really live in config

Guiding principles:

- prefer low-hanging fruit first, then progressively more invasive changes
- keep the code compressed and direct
- analyze both the repo and Nerfstudio internals before changing behavior
- first check whether the current config is under-utilizing the available GPU,
  then tune knobs that can use more of the device without breaking quality
- make phase timings explain the speedup, not just the final wall-clock number
- feature extraction code matters for this benchmark mainly through cache-hit
  startup/import overhead; rewriting async extraction logic is only justified if
  timing data proves it still affects training startup materially
- lazy imports and import cleanup are valid hypotheses if they reduce
  `startup_overhead_s` measurably
- use `hypothesis.md` as a living backlog, not as a dump of stale notes

## Result Format

Append one tab-separated row to `results.tsv` for every official `measure` run:

```tsv
commit	profile	wall_time_s	train_total_time_s	startup_overhead_s	train_iter_time_s	train_rays_per_sec	train_batch_load_s	train_feature_cache_load_s	train_feature_gather_s	train_feature_populate_s	train_model_forward_s	train_metrics_loss_s	train_image_viz_s	eval_batch_load_s	eval_feature_cache_load_s	eval_feature_gather_s	eval_feature_populate_s	eval_image_s	eval_all_images_s	peak_gpu_mem_mb	pixel_visit_x	pixel_visit_x_pct_diff	window_coverage_w	window_coverage_w_pct_diff	train_psnr	train_feature_error	train_foreground_acc	eval_all_psnr	eval_all_ssim	eval_all_lpips	max_quality_regression_pct	status	description	summary_path
```

Field meanings:

- `commit`: short git hash
- `profile`: always `measure` for official comparisons
- `wall_time_s`: full subprocess wall-clock from the harness
- `train_total_time_s`: Nerfstudio train-loop wall-clock from tensorboard events
- `startup_overhead_s`: `wall_time_s - train_total_time_s`
- timing columns: last reported benchmark timing averages or totals from the harness summary
- `train_image_viz_s`: optional timing; blank when train image viz is disabled
- `peak_gpu_mem_mb`: max logged GPU memory sample
- `pixel_visit_x`: measured train-ray-per-train-pixel budget from the saved run config
- `pixel_visit_x_pct_diff`: signed percent difference from the poster baseline
  `X ≈ 0.1544`
- `window_coverage_w`: measured selected-window coverage from the saved run
  config
- `window_coverage_w_pct_diff`: signed percent difference from the poster
  baseline `W ≈ 2.3814`
- `train_*` / `eval_all_*`: representative final unweighted metrics pulled from
  `summary.json`
- `max_quality_regression_pct`: worst quality regression against the baseline
  `summary.json`
- interpret `max_quality_regression_pct` in bands:
  - `<= 10.0`: comfortable
  - `10.0 .. 15.0`: judgment band
  - `> 15.0`: too much degradation
- `status`: `keep`, `discard`, or `crash`
- `description`: one-line summary of the idea
- `summary_path`: path to the benchmark `summary.json`

`results.tsv` is for actual benchmark outcomes only. Do not mix speculative
notes into it.
Do not append smoke runs to `results.tsv`.
For readability, keep numeric cells in `results.tsv` to two decimal places.
The harness emits `benchmark_runs/measure/<run_id>/results_row.tsv` in that
format, with blank `status` and `description` cells for successful runs so you
can review the run first and then paste or append the row into the main
`results.tsv`. Keep full precision in `summary.json`; only the TSV view is
rounded.

## Hypotheses File

Maintain `hypothesis.md` as a lightweight living backlog with these two
sections:

- `# Active Hypotheses`
- `# Out Of Scope Issues For Human`

Keep entries short and concrete:

- remove ideas once they have been tested
- add new follow-ups when experiments create them
- do not let stale ideas accumulate

## Keep/Discard Rule

Keep a change only if:

- the `measure` run completed successfully
- `wall_time_s` improved
- `max_quality_regression_pct <= 10.0`, or it lands in the `10.0 .. 15.0`
  judgment band and the speedup clearly justifies the loss
- the code complexity cost is justified by the gain

Discard a change if:

- the benchmark failed or the summary is incomplete
- `wall_time_s` did not improve enough to justify the change
- `max_quality_regression_pct > 15.0`
- the speedup comes from breaking the fixed benchmark contract

## Loop

Once setup is complete:

1. Inspect the current git state.
2. Pick one speed hypothesis.
3. Edit only mutable files.
4. Commit the change.
5. Run `python benchmark_train_time.py --profile smoke` if the change is risky or possibly buggy or crash-prone.
6. Run the official measure command with `--baseline-summary`.
7. Read `benchmark_runs/measure/<run_id>/summary.json`.
8. Append the flattened measure row to `results.tsv`. Never append smoke rows.
9. Update `hypothesis.md`.
10. Keep only accepted commits in commit history.
11. Do not leave behind stray processes.
12. Move directly to the next idea.
