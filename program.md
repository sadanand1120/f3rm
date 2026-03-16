# F3RM Autoresearch Program

## Mission

This experiment is to improve end-to-end training time for F3RM on the fixed
`poster` benchmark scene.

Primary objective:
- minimize `wall_time_s` from the official `measure` benchmark profile

Success criterion:
- `wall_time_s` goes down on the official benchmark
- `max_quality_regression_pct <= 10.0` is the comfortable zone
- `10.0 < max_quality_regression_pct <= 15.0` is a judgment band: keep it only if the wall-time win clearly justifies the quality loss
- the code stays concise, readable, and justified by measured gains

Secondary constraints:
- do not materially increase peak GPU memory without a clear speed win
- preserve the real training intent: RGB + feature + foreground training for the fixed iteration budget
- avoid benchmark-specific hacks, hidden shortcuts, or metric-shaping tricks

## Runtime Contract

**NON-NEGOTIABLE: every runtime-dependent command must run inside thecontainer/env workflow. Do not run training, benchmarking, debugging, nerfstudio package inspection, or other environment-sensitive commands on the host.**

- container: `fresh`
- conda env: `f3rm`
- env path: `/opt/miniconda3/envs/f3rm`
- repo path in runtime (and on host, as its mounted): `/robodata/smodak/repos/f3rm`
- GPU contract: set `CUDA_VISIBLE_DEVICES=1` for benchmarked runs
- Nerfstudio package for read-only inspection: `/opt/miniconda3/envs/f3rm/lib/python3.11/site-packages/nerfstudio`
- canonical order: `docker exec fresh` -> `conda activate f3rm` -> `cd /robodata/smodak/repos/f3rm` -> run command

The benchmark harness itself is meant to be run inside the container, not as a
host-side Docker wrapper.

Canonical runtime entry:

```bash
docker exec fresh bash -lc '
  source /opt/miniconda3/etc/profile.d/conda.sh &&
  conda activate f3rm &&
  export CUDA_VISIBLE_DEVICES=1 &&
  cd /robodata/smodak/repos/f3rm &&
  python benchmark_train_time.py --profile smoke
'
```

Use that wrapper shape for all runtime-dependent commands. If a command depends
on installed packages, imports, CUDA, or the training environment, it belongs
inside `fresh` with `f3rm` activated.

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

The benchmark harness is authoritative because it fixes:

- dataset: `datasets/f3rm/test/poster`
- seed: `0`
- single-device training: `--machine.num-devices 1`
- single visible GPU: `CUDA_VISIBLE_DEVICES=1`
- measure iteration budget: `6943`
- eval policy: final-only eval at step `6942`
- save policy: no periodic checkpoints
- vis/logging policy: `tensorboard`, profiler off, train image viz off
- train image sampling policy: `train_num_images_to_sample_from=27`, `train_num_times_to_repeat_images=439`
- eval image sampling policy: use all `11` eval images

Why `27` train images per cache window:
- the `poster` split is `215` train images and `11` eval images
- the measure benchmark budget is `6943` steps
- with cache refresh every `439` steps, `ceil(6943 / 439) = 16` train cache windows
- `27 * 16 = 432`, which is the nearest whole-number schedule to covering the `215`-image train split twice

Benchmark harness requirements:

- it stays fixed during the experiment loop
- it runs `ns-train` as a subprocess inside the current container/env
- it writes raw combined stdout/stderr to `benchmark_runs/<profile>/<run_id>/train.log`
- it writes `summary.json` next to the log
- it prints a stable summary block at the end
- it exits non-zero on failure
- the `measure` profile is the only source of truth for keep/discard decisions

Do not evaluate changes with ad hoc `ns-train` commands once the benchmark
contract is in place.

## In-Scope Files

Read these before changing code:

- `program.md`: fixed experiment contract and keep/discard rule
- `benchmark_train_time.py`: fixed benchmark contract and summary schema
- `f3rm/f3rm_config.py`: default F3RM training config and likely first hyperparameter surface
- `f3rm/trainer.py`: final metric logging and train/eval loop behavior
- `f3rm/pipeline.py`: train/eval phase structure and full-image eval behavior
- `f3rm/feature_datamanager.py`: batch sampling, feature loading, and timing hooks

Before editing any additional file, read that file first. In practice,
`f3rm/model.py` is the next file to read when a change touches losses or
quality metrics.

Mutable files:
- `f3rm/**/*.py`

Read-only reference files:
- `/opt/miniconda3/envs/f3rm/lib/python3.11/site-packages/nerfstudio/**/*.py`
- sibling repos only when imported from F3RM code, such as `sam2` or `sam3`

Fixed files:
- `benchmark_train_time.py`
- `program.md`
- `analysis.py`

Auxiliary notes file:
- `hypothesis.md`: a living backlog of active hypotheses plus a separate `Out Of Scope Issues For Human` section

Do not create `hypothesis.md` during this contract-edit pass unless the human
explicitly asks. The experiment loop still must not start until that file
exists and has been reviewed.

Do not edit Nerfstudio site-packages directly. **If a Nerfstudio patch becomes
unavoidable, stage it under `nerfstudio_changes/` instead and HAND OFF CONTROL TO THE USER, i.e., STOP the experiment after this run.**

## Setup

Before starting the loop:

1. Create a run tag and branch, for example `autoresearch/poster-train-time`.
2. Verify the runtime:
   - `fresh` is running
   - `conda activate f3rm` succeeds
   - current working directory is `/robodata/smodak/repos/f3rm`
   - the command is being run through the `docker exec fresh ... conda activate f3rm ...` wrapper above
3. Verify the fixed assets exist:
   - `datasets/f3rm/test/poster/transforms.json`
   - cached `clip` features with `226` per-image files
   - cached `foreground_` features with `226` per-image files
4. Review `hypothesis.md` once it exists. It should contain these section headers:

```md
# Active Hypotheses

# Out Of Scope Issues For Human
```

5. If `hypothesis.md` does not exist yet because the human deferred it during artifact generation, stop here and wait for that file before running any baseline or starting the loop.
6. Refresh `# Active Hypotheses` based on anything new learned during setup validation. Remove stale ideas before running any baseline.
7. Run the smoke profile once.
8. Create `results.tsv` if it does not exist, with this header:

```tsv
commit	profile	wall_time_s	train_total_time_s	startup_overhead_s	train_iter_time_s	train_rays_per_sec	train_batch_load_s	train_feature_cache_load_s	train_feature_gather_s	train_feature_populate_s	train_model_forward_s	train_metrics_loss_s	eval_batch_load_s	eval_feature_cache_load_s	eval_feature_gather_s	eval_feature_populate_s	eval_image_s	eval_all_images_s	peak_gpu_mem_mb	train_psnr	train_feature_error	train_foreground_acc	eval_all_psnr	eval_all_ssim	eval_all_lpips	max_quality_regression_pct	status	description	summary_path
```

9. Run the measure profile once on the current baseline without
   `--baseline-summary`.
10. Record that baseline row in `results.tsv` with `max_quality_regression_pct=0.0`.
11. Use that baseline `summary.json` for all later measure runs, then begin the loop.

## Experiment Contract

Each experiment should be one coherent idea.

Allowed changes (not exhaustive):
- training-loop changes inside `f3rm/`
- dataloading or cache-path changes inside `f3rm/`
- model or hyperparameter changes inside `f3rm/`
- import cleanup or startup simplification if it measurably lowers benchmark wall time
- deleting code when it preserves the benchmark contract and improves speed or clarity

Disallowed changes (not exhaustive):
- changing `benchmark_train_time.py`
- changing the fixed dataset, seed, iteration count, or summary schema
- removing RGB, feature, or foreground supervision from the training objective
- bypassing final eval metrics or changing what the harness parses
- moving benchmarked execution outside the container/env contract

Guiding principles:

- prefer low-hanging fruit first, then progressively more invasive changes
- keep the code compressed and direct where possible
- make phase timings explain the speedup, not just the final wall-clock number
- the official benchmark already uses cached CLIP and foreground features, so feature extraction code matters mainly for cache-hit startup/import overhead unless the cache path is invalidated
- use `hypothesis.md` as a living backlog, not as a dump of stale notes
- when you notice promising ideas you cannot pursue yet, record them in `# Active Hypotheses`
- when you discover an important change that is likely valuable but outside your allowed scope, record it under `# Out Of Scope Issues For Human`

## Result Format

Append one tab-separated row to `results.tsv` for every official `measure` run:

```tsv
commit	profile	wall_time_s	train_total_time_s	startup_overhead_s	train_iter_time_s	train_rays_per_sec	train_batch_load_s	train_feature_cache_load_s	train_feature_gather_s	train_feature_populate_s	train_model_forward_s	train_metrics_loss_s	eval_batch_load_s	eval_feature_cache_load_s	eval_feature_gather_s	eval_feature_populate_s	eval_image_s	eval_all_images_s	peak_gpu_mem_mb	train_psnr	train_feature_error	train_foreground_acc	eval_all_psnr	eval_all_ssim	eval_all_lpips	max_quality_regression_pct	status	description	summary_path
```

Field meanings:

- `commit`: short git hash
- `profile`: always `measure` for official comparisons
- `wall_time_s`: full subprocess wall-clock from the harness
- `train_total_time_s`: Nerfstudio train-loop wall-clock from tensorboard events
- `startup_overhead_s`: `wall_time_s - train_total_time_s`
- timing columns: last reported benchmark timing averages or totals from the harness summary
- `peak_gpu_mem_mb`: max logged GPU memory sample
- `train_*` / `eval_all_*`: representative final unweighted metrics pulled from `summary.json`
- `max_quality_regression_pct`: worst quality regression against the baseline `summary.json`
- interpret `max_quality_regression_pct` in bands:
  - `<= 10.0`: comfortable
  - `10.0 .. 15.0`: judgment band
  - `> 15.0`: too much degradation
- `status`: `keep`, `discard`, or `crash`
- `description`: one-line summary of the idea
- `summary_path`: path to the benchmark `summary.json`

`results.tsv` is for actual benchmark outcomes only. Do not mix speculative
notes into it.

## Hypotheses File

Maintain `hypothesis.md` as a lightweight living backlog with these two
sections:

- `# Active Hypotheses`
- `# Out Of Scope Issues For Human`

`# Active Hypotheses` is for:

- follow-up speed hypotheses worth testing later
- ideas generated during the initial repo scan or later setup work
- ideas blocked by sequencing, time, or local priority
- new follow-ups created by recent experiment outcomes

`# Out Of Scope Issues For Human` is for:

- changes that would likely help but violate the current scope boundary
- architectural or workflow issues that need human judgment
- bugs or code smells that matter but are not part of the current experiment contract

Keep entries short and concrete. Update the file continuously:

- remove ideas once they have been tested
- add new follow-ups when experiments create them
- do not let stale, already-resolved ideas accumulate

A good entry usually includes:

- the date or commit context, if relevant
- the idea or issue
- why it matters
- why it was deferred, if relevant

## Keep/Discard Rule

Keep a change only if:

- the `measure` run completed successfully
- `wall_time_s` improved
- `max_quality_regression_pct <= 10.0`, or it lands in the `10.0 .. 15.0` judgment band and the speedup clearly justifies the loss
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
3. Edit only `f3rm/**/*.py`.
4. Commit the change.
5. Run `python benchmark_train_time.py --profile smoke` if the change is risky.
6. Run the official measure command with `--baseline-summary`.
7. Read `benchmark_runs/measure/<run_id>/summary.json`.
8. Append the flattened measure row to `results.tsv`.
9. Update `hypothesis.md`:
   - remove ideas that were just tested
   - add new follow-up hypotheses created by the result
   - add any clearly promising but off-scope changes under `# Out Of Scope Issues For Human`
10. Keep only accepted commits in history.
11. Do NOT leave behind any stray processes. Terminate all appropriately.
12. Move directly to the next idea.
