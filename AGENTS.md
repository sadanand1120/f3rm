# F3RM Agent Notes

## Runtime Context
- Run runtime-dependent project execution inside the container, in the conda environment. Do not run project scripts, training, debugging commands, or environment-sensitive checks on the host.
- Container name: `fresh`
- Conda environment: `f3rm`
- Conda env path: `/opt/miniconda3/envs/f3rm`
- Repo path inside workflow: `/robodata/smodak/repos/f3rm`
- Nerfstudio package path inside container (runtime inspection source): `/opt/miniconda3/envs/f3rm/lib/python3.11/site-packages/nerfstudio`

## Edit Boundaries
- Primary code focus: `f3rm/` and `f3rm/features/`.
- Do not directly edit Nerfstudio site-packages.
- If a Nerfstudio edit is unavoidable and cannot be overridden cleanly, place patched files in `nerfstudio_changes/` for manual copy (user will copy them to the correct location).

## Command Execution Policy
- For project execution commands, execute in container + env context:
  1. `docker exec` into `fresh`
  2. use/activate `f3rm` environment
  3. `cd /robodata/smodak/repos/f3rm`
  4. run command
- For any command that uses GPU, set `CUDA_VISIBLE_DEVICES='1'` before running the command.
- Playwright/Node/npm/npx checks and browser automation commands are runtime-dependent and must be executed inside `fresh` with the `f3rm` environment. Do not treat host `node`/`npm`/`npx` availability as authoritative.
- For repo-mounted source files, prefer host-side edits and host-side read-only inspection by default. Use container-side editing only when a runtime-dependent workflow explicitly requires it.
- Non-execution inspection on repo-mounted files can be done on the host. This includes simple file lookups, `rg`, `grep`, `sed`, `ls`, and similar read-only code inspection under `/robodata/smodak/`.
- Inspect installed packages from inside the container, since package code and imports depend on the container environment rather than the host-mounted repo alone.

## W&B Debugging Notes
- Prefer local artifact analysis from run directories and `config.yml` paths.
- Track canonical metric groups (`Train*`, `Eval*`, `Final Metrics/*`).
- Flag NaNs/Infs, jumps, flatlines, and step gaps explicitly.

## F3RM Abstract
- F3RM keeps standard Nerfacto RGB/density training as the geometric/base path.
- In parallel, it trains a feature field that predicts per-ray CLIP features.
- The datamanager injects per-ray CLIP supervision from precomputed feature maps, and training jointly optimizes base Nerfacto losses plus CLIP feature distillation loss.
