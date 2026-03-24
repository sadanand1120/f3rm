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
- For checkpoint-loading runtime commands (`ns-viewer`, eval/demo scripts, anything that calls Nerfstudio checkpoint loading), set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in the command rather than patching code around PyTorch 2.6 `weights_only` behavior.
- For browser automation and Playwright usage, use the host conda environment `browser`. Do not use `fresh`/`f3rm` for browser-driving commands.
- Exact browser flow that worked for Viser inspection:
  1. Launch the app/server inside `fresh` + `f3rm`.
  2. From the host, run `source /home/smodak/anaconda3/etc/profile.d/conda.sh && conda activate browser`.
  3. Prefer Python Playwright with bundled Chromium, not `playwright-cli open`, since the CLI defaulted to the Chrome channel and was brittle here.
  4. Canonical pattern:
     `xvfb-run -a python - <<'PY'`
     `from playwright.sync_api import sync_playwright`
     `with sync_playwright() as p:`
     `    browser = p.chromium.launch(headless=False)`
     `    page = browser.new_page(viewport={'width': 1400, 'height': 900})`
     `    page.goto('<VISER_SHARE_URL>', wait_until='domcontentloaded', timeout=120000)`
     `    page.wait_for_timeout(8000)`
     `    page.mouse.move(700, 450); page.mouse.down(); page.mouse.move(950, 500, steps=20); page.mouse.up()`
     `    page.screenshot(path='/tmp/viser.png', full_page=True)`
     `    browser.close()`
     `PY`
  5. Ensure the `browser` env has `playwright` installed and `python -m playwright install chromium` has been run.
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
