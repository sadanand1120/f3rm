---
name: f3rm-container-env
description: Use the F3RM runtime context for execution commands that depend on the container/env (container `fresh`, conda env `f3rm`, repo `/robodata/smodak/repos/f3rm`). Host-side read-only repo inspection is fine for simple lookups and search.
---

# F3RM Container + Env Execution

Use this workflow for execution commands that need the project runtime. Do not use it for simple host-side inspection of repo-mounted files such as `rg`, `grep`, `sed`, or `ls`.

1. Enter container: `docker exec fresh ...`
2. Ensure env context is `f3rm` (path: `/opt/miniconda3/envs/f3rm`)
3. Change to repo: `cd /robodata/smodak/repos/f3rm`
4. Run the requested command

Canonical command wrapper:

```bash
docker exec fresh bash -lc '
  source /opt/miniconda3/etc/profile.d/conda.sh &&
  conda activate f3rm &&
  cd /robodata/smodak/repos/f3rm &&
  <COMMAND>
'
```

If `conda activate` is unavailable, fall back to:

```bash
docker exec fresh bash -lc '
  cd /robodata/smodak/repos/f3rm &&
  /opt/miniconda3/envs/f3rm/bin/python -V
'
```

## Search Command Guardrails
- Do not assume `rg` is installed in `fresh`.
- Before using `rg`, check availability with `command -v rg`.
- If `rg` is unavailable:
  - file listing: `find <dir> -type f`
  - content search: `grep -RIn "<pattern>" <dir>`
- Prefer adding excludes to `grep` when scanning large trees:
  - `--exclude-dir=.git --exclude-dir=wandb --exclude-dir=outputs`

## Troubleshooting
- If container fails: verify `fresh` is running.
- If env fails: verify `/opt/miniconda3/envs/f3rm` exists in container.
- If imports mismatch: print `which python`, `python -V`, and package versions from inside container before proceeding.
