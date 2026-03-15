---
name: f3rm-wandb-analyzer
description: Analyze local W&B run artifacts from config.yml or run files; plot grouped metric curves and report instability diagnostics (NaN/Inf, non-finite fields, jumps, flatlines, and step gaps).
---

# F3RM W&B Local Analyzer

Use this skill when the user asks to debug or compare W&B training behavior from local run artifacts.

## Scope
- Local-artifact analysis only (no live W&B API dependency).
- Works from one or more `config.yml` paths and/or explicit `run-*.wandb` files.
- Produces grouped plots plus JSON/Markdown anomaly summaries.

## Script
- Use `scripts/wandb_logs_analyzer.py`.

Example:

```bash
python .codex/skills/f3rm-wandb-analyzer/scripts/wandb_logs_analyzer.py \
  --config-path /robodata/smodak/repos/f3rm/testdeter_outputs/lang2fix5_tcnn/f3rm/2026-02-11_131452/config.yml \
  --output-dir /tmp/wandb_analysis
```

## Required Checks in Analysis
- Canonical key groups:
  - train metrics/losses
  - eval metrics/losses
- Explicitly call out:
  - NaN/Inf counts
  - abrupt jumps
  - flatlines
  - step gaps

See `references/metric_groups.md` for grouping defaults and watchlist details.
