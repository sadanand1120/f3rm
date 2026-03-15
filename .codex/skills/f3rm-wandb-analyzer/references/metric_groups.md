# Metric Group Defaults

## Train Group Prefixes
- `Train Metrics Dict/`
- `Train Loss Dict/`
- `train/`

## Eval Group Prefixes
- `Eval Metrics Dict/`
- `Eval Loss Dict/`
- `eval/`

## What to Watch
- Any non-zero NaN/Inf counts.
- Sudden metric jumps (especially PSNR/features).
- Long flatlines in losses/metrics where learning is expected.
- Step gaps or missing segments in logged series.
