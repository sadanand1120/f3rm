# F3RM Compression Program

## Mission

This experiment targets code compression inside `f3rm/` while preserving the real F3RM workload.

Primary objective:
- minimize `f3rm_loc`, defined as the total number of non-empty, non-comment Python lines under `f3rm/`

Hard constraints:
- the workload must still do the same conceptual work: standalone feature extraction followed by RGB + feature-field training
- you should not sacrifice code readability and maintainability for code size
- keep the standalone extractor entrypoint separate from extractor-specific implementations so new extractors can be added cleanly later
- the 4 saved visual checks must still look plausible
- derived end-to-end time (`extract_wall_s + train_wall_s`) must not regress by more than 10%; aim to stay within 5%
- final unweighted trainer metrics must not regress by more than 10% each; aim to stay within 5%
  - `final_eval_all_psnr`: higher is better
  - `final_eval_all_ssim`: higher is better
  - `final_eval_all_lpips`: lower is better
  - `final_train_feature_error`: lower is better

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
- smoke command: `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 && python benchmark_f3rm.py --profile smoke`
- measure command: `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 && python benchmark_f3rm.py --profile measure`

Harness guarantees:
- it re-enters `fresh` + `f3rm` automatically when launched from the host
- it exports `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` before running the benchmark process
- it deletes `datasets/.../features/clip` before extraction and again after the run
- it runs extraction and training as subprocesses
- it streams subprocess stdout/stderr live to the terminal while also saving raw logs
- it writes raw logs under `benchmark_runs/<profile>/<run_tag>/`
- it saves 4 visual checks under `benchmark_runs/<profile>/<run_tag>/visual_check/`
- each visual check is a single row with `original rgb | render rgb | render feature`
- it writes `summary.json` and prints the same summary between `=== F3RM_BENCHMARK_SUMMARY_BEGIN ===` and `=== F3RM_BENCHMARK_SUMMARY_END ===`
- `measure` is the only authoritative profile for keep/discard decisions

Only two global times are recorded:
- `extract_wall_s`
- `train_wall_s`

Derived end-to-end time is always computed as their sum.

## In-Scope Files

Read these before changing anything:
- `program.md`
- `benchmark_f3rm.py`
- `analysis.py`
- `f3rm/features/extract_features_standalone.py`
- `f3rm/features/clip_extract.py`
- `f3rm/features/utils.py`
- `f3rm/feature_datamanager.py`
- `f3rm/pipeline.py`
- `f3rm/trainer.py`
- `f3rm/f3rm_config.py`
- `f3rm/train_schedule.py`

Mutable files:
- `f3rm/**`

Fixed files once the loop starts:
- `program.md`
- `analysis.py`
- `benchmark_f3rm.py`

## Setup

Before the experiment loop:

1. Re-read `program.md`, `benchmark_f3rm.py`, and `analysis.py`.
2. Verify the runtime assumptions:
   - container `fresh` is running
   - conda env `f3rm` activates successfully
   - datasets exist at `datasets/f3rm/test/poster_smoke` and `datasets/f3rm/test/poster2`
   - `ns-train` resolves inside the container
3. Run `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 && python benchmark_f3rm.py --profile smoke`.
4. Visually inspect the 4 saved smoke checks under `benchmark_runs/smoke/<run_tag>/visual_check/` and confirm (1) feature PCA looks good, and (2) the original images correspond one to one to the PCA feature visualizations with no obvious image-feature misalignment.
5. Create `results.tsv` if it does not exist yet, with this exact header:

```tsv
revision	f3rm_loc	extract_wall_s	train_wall_s	final_eval_all_psnr	final_eval_all_ssim	final_eval_all_lpips	final_train_feature_error	status	description
```

6. Run `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 && python benchmark_f3rm.py --profile measure`.
7. Visually inspect the 4 saved measure checks under `benchmark_runs/measure/<run_tag>/visual_check/` and confirm (1) feature PCA looks good, and (2) the original images correspond one to one to the PCA feature visualizations with no obvious image-feature misalignment.
8. Parse the summary JSON from stdout or `benchmark_runs/.../summary.json`.
9. Append one `measure` row to `results.tsv` for the baseline.

`results.tsv` must contain `measure` runs only. Never append `smoke` runs.

## Experiment Contract

Allowed changes:
- code compression inside `f3rm/`
- redundancy removal and responsibility cleanup between `f3rm/` files
- feature extraction and feature loading refactors that preserve the benchmarked workload
- removing benchmark-driven instrumentation from `f3rm/` when it is no longer required by this contract

Disallowed changes:
- changing the benchmark definition after the loop starts
- skipping feature extraction or reusing stale cache state
- collapsing the standalone extractor entrypoint into an extractor-specific implementation file
- changing the benchmark to stop training RGB or feature learning
- removing the saved visual-check gate
- editing Nerfstudio site-packages directly

Guiding principles:
- reduce code first by deleting unnecessary structure, not by compressing readability away
- preserve clear separation of concerns between files
- keep the standalone extractor entrypoint generic and extensible
- prefer simpler data flow over benchmark-specific hooks
- treat the saved visual checks as a real correctness gate

## Result Format

Append one tab-separated `measure` row per experiment:

```tsv
revision	f3rm_loc	extract_wall_s	train_wall_s	final_eval_all_psnr	final_eval_all_ssim	final_eval_all_lpips	final_train_feature_error	status	description
```

Field notes:
- `revision` can be a commit hash, branch+dirty label, or any stable experiment identifier
- `f3rm_loc` is the non-empty, non-comment line count across `f3rm/**/*.py`
- `extract_wall_s` and `train_wall_s` are the only recorded times
- end-to-end time is always derived as `extract_wall_s + train_wall_s`
- `final_eval_all_*` and `final_train_feature_error` come from `Final Metrics/*`
- `status` is one of `keep`, `discard`, or `crash`

## Keep/Discard Rule

Keep a change only if:
- the `measure` run completed successfully
- the saved visual checks still look correct
- `f3rm_loc` improved enough to justify the change
- derived end-to-end time stayed within the allowed regression window
- each final metric stayed within the allowed regression window

Discard a change if:
- the benchmark failed
- the visual checks look wrong
- code size did not improve meaningfully
- end-to-end time regressed too far
- any final metric regressed too far
- the code got harder to understand even if it got shorter

## Loop

Once setup is complete:

1. Inspect the git state.
2. Pick one compression or cleanup idea.
3. Edit only mutable files.
4. Run `smoke` if the change is risky.
5. Inspect the 4 saved `smoke` visual checks whenever the change touches extraction, rendering, or image-feature indexing.
6. Run `measure`.
7. Visually inspect the 4 saved `measure` visual checks and confirm (1) feature PCA looks good, and (2) the original images correspond one to one to the PCA feature visualizations with no obvious image-feature misalignment.
8. Parse the benchmark summary and append one `measure` row to `results.tsv`.
9. Keep the change only if it wins by the rule above. Only accepted commits should remain in git history.
10. Continue immediately with the next idea.
