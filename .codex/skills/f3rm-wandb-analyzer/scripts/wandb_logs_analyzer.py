#!/usr/bin/env python3
"""Generic local W&B logs analyzer for plotting and instability diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


TRAIN_PREFIXES = ["Train Metrics Dict/", "Train Loss Dict/", "train/"]
EVAL_PREFIXES = ["Eval Metrics Dict/", "Eval Loss Dict/", "eval/"]
NONFINITE_PREFIXES = [
    "train/nonfinite",
    "train/nonfinite_grad",
    "train/nonfinite_grad_groups",
    "train/grad_scaler",
    "grad_scaler",
    "nonfinite",
]


@dataclass
class MetricDiagnostics:
    metric: str
    points: int
    nonfinite_count: int
    gap_count: int
    jump_count: int
    flatline: bool
    anomaly_score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze local W&B logs from config.yml and/or run-*.wandb files.")
    p.add_argument("--config-path", action="append", default=[], help="Path to a model config.yml (repeatable).")
    p.add_argument("--run-file", action="append", default=[], help="Explicit path to run-*.wandb file (repeatable).")
    p.add_argument("--output-dir", type=str, default="wandb_analysis", help="Output directory.")
    p.add_argument("--include-regex", type=str, default=None, help="Only include metrics matching this regex.")
    p.add_argument("--exclude-regex", type=str, default=None, help="Exclude metrics matching this regex.")
    p.add_argument(
        "--groups",
        nargs="+",
        default=["train", "eval", "nonfinite", "other", "custom"],
        choices=["train", "eval", "nonfinite", "other", "custom"],
        help="Groups to include in plots/summaries.",
    )
    p.add_argument("--custom-prefix", action="append", default=[], help="Custom metric key prefix (repeatable).")
    p.add_argument("--cols", type=int, default=3, help="Subplot columns.")
    p.add_argument("--dpi", type=int, default=140, help="Plot DPI.")
    p.add_argument("--max-metrics", type=int, default=120, help="Max metrics to plot in 'all metrics' figure.")
    p.add_argument("--top-anomalies", type=int, default=20, help="Max anomalies in summary.")
    return p.parse_args()


def _import_wandb_readers():
    try:
        from wandb.sdk.internal.datastore import DataStore
        import wandb.proto.wandb_internal_pb2 as pb
    except Exception as exc:
        raise RuntimeError(
            "Failed to import wandb internals. Run in env where wandb is installed."
        ) from exc
    return DataStore, pb


def resolve_run_file_from_config(config_path: Path) -> Path:
    run_dir = config_path.resolve().parent
    wandb_root = run_dir / "wandb"
    if not wandb_root.exists():
        raise FileNotFoundError(f"Missing wandb directory: {wandb_root}")
    candidates = sorted(wandb_root.glob("run-*/run-*.wandb"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run-*.wandb found under: {wandb_root}")
    return candidates[-1]


def collect_run_files(config_paths: List[str], run_files: List[str]) -> List[Path]:
    out: List[Path] = []
    for cp in config_paths:
        out.append(resolve_run_file_from_config(Path(cp)))
    for rf in run_files:
        rp = Path(rf)
        if not rp.exists():
            raise FileNotFoundError(f"Run file not found: {rp}")
        out.append(rp.resolve())
    dedup = []
    seen = set()
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            dedup.append(p)
    if not dedup:
        raise ValueError("Provide at least one --config-path or --run-file")
    return dedup


def iter_history_items(run_file: Path) -> Iterable[Tuple[int, str, object]]:
    DataStore, pb = _import_wandb_readers()
    ds = DataStore()
    ds.open_for_scan(str(run_file))

    while True:
        data = ds.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue

        step = record.history.step.num if record.history.HasField("step") else None
        items: List[Tuple[str, object]] = []
        for item in record.history.item:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            try:
                value = json.loads(item.value_json)
            except Exception:
                value = item.value_json
            items.append((key, value))
            if key == "_step" and isinstance(value, int):
                step = value

        if step is None:
            continue
        for key, value in items:
            yield int(step), key, value


def extract_series(run_file: Path) -> Dict[str, List[Tuple[int, float]]]:
    accum: Dict[str, Dict[int, float]] = {}
    for step, key, value in iter_history_items(run_file):
        if not isinstance(value, (int, float, bool)):
            continue
        accum.setdefault(key, {})
        accum[key][step] = float(value)
    return {k: sorted(v.items(), key=lambda x: x[0]) for k, v in accum.items()}


def filter_metrics(
    series: Dict[str, List[Tuple[int, float]]],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
) -> Dict[str, List[Tuple[int, float]]]:
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    out = {}
    for k, v in series.items():
        if inc and not inc.search(k):
            continue
        if exc and exc.search(k):
            continue
        out[k] = v
    return out


def classify_metric(metric: str, custom_prefixes: List[str]) -> str:
    if any(metric.startswith(p) for p in NONFINITE_PREFIXES):
        return "nonfinite"
    if any(metric.startswith(p) for p in TRAIN_PREFIXES):
        return "train"
    if any(metric.startswith(p) for p in EVAL_PREFIXES):
        return "eval"
    if any(metric.startswith(p) for p in custom_prefixes):
        return "custom"
    return "other"


def split_groups(
    series: Dict[str, List[Tuple[int, float]]],
    enabled_groups: List[str],
    custom_prefixes: List[str],
) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    groups: Dict[str, Dict[str, List[Tuple[int, float]]]] = {g: {} for g in enabled_groups}
    for metric, points in series.items():
        g = classify_metric(metric, custom_prefixes)
        if g in groups:
            groups[g][metric] = points
    return groups


def metric_diagnostics(metric: str, points: List[Tuple[int, float]]) -> MetricDiagnostics:
    steps = np.array([p[0] for p in points], dtype=np.int64)
    vals = np.array([p[1] for p in points], dtype=np.float64)
    finite_mask = np.isfinite(vals)
    nonfinite_count = int((~finite_mask).sum())

    step_diffs = np.diff(steps) if len(steps) > 1 else np.array([], dtype=np.int64)
    pos_diffs = step_diffs[step_diffs > 0]
    typical = np.median(pos_diffs) if len(pos_diffs) > 0 else 1.0
    gap_count = int((step_diffs > (2.0 * typical)).sum()) if len(step_diffs) else 0

    finite_vals = vals[finite_mask]
    if len(finite_vals) > 1:
        deltas = np.diff(finite_vals)
        med = np.median(deltas)
        mad = np.median(np.abs(deltas - med))
        jump_thresh = max(1e-8, 8.0 * mad)
        jump_count = int(np.sum(np.abs(deltas) > jump_thresh))
        flatline = bool(np.nanstd(finite_vals) < 1e-8 or (np.nanmax(finite_vals) - np.nanmin(finite_vals)) < 1e-8)
    else:
        jump_count = 0
        flatline = False

    score = nonfinite_count * 100.0 + jump_count * 5.0 + gap_count * 2.0 + (10.0 if flatline and len(points) >= 20 else 0.0)
    return MetricDiagnostics(
        metric=metric,
        points=len(points),
        nonfinite_count=nonfinite_count,
        gap_count=gap_count,
        jump_count=jump_count,
        flatline=flatline,
        anomaly_score=score,
    )


def plot_group(
    series: Dict[str, List[Tuple[int, float]]],
    title: str,
    output_path: Path,
    cols: int,
    dpi: int,
    max_metrics: Optional[int] = None,
) -> int:
    if not series:
        return 0
    names = sorted(series.keys())
    if max_metrics is not None:
        names = names[:max_metrics]

    n = len(names)
    cols = max(1, cols)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.2 * rows), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for i, name in enumerate(names):
        ax = flat[i]
        pts = series[name]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, linewidth=1.6)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)

    for j in range(n, len(flat)):
        flat[j].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return n


def run_id_for_path(run_file: Path) -> str:
    return run_file.parent.name


def write_summary(
    out_dir: Path,
    run_file: Path,
    diagnostics: List[MetricDiagnostics],
    groups: Dict[str, Dict[str, List[Tuple[int, float]]]],
    top_k: int,
) -> None:
    diagnostics_sorted = sorted(diagnostics, key=lambda d: d.anomaly_score, reverse=True)
    top = diagnostics_sorted[:top_k]

    nonfinite_metrics = [d.metric for d in diagnostics if d.nonfinite_count > 0]
    warnings = []
    if nonfinite_metrics:
        warnings.append(f"Detected non-finite values in {len(nonfinite_metrics)} metric(s).")
    if "nonfinite" in groups and len(groups["nonfinite"]) > 0:
        warnings.append(f"Non-finite watchlist group present with {len(groups['nonfinite'])} metric(s).")

    summary_obj = {
        "run_file": str(run_file),
        "num_metrics": len(diagnostics),
        "num_nonfinite_metrics": len(nonfinite_metrics),
        "warnings": warnings,
        "groups": {g: len(m) for g, m in groups.items()},
        "top_anomalies": [d.__dict__ for d in top],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")

    lines = [
        f"# W&B Local Analysis: {run_id_for_path(run_file)}",
        "",
        f"- run file: `{run_file}`",
        f"- total metrics: **{len(diagnostics)}**",
        f"- metrics with non-finite values: **{len(nonfinite_metrics)}**",
        "",
        "## Group Counts",
    ]
    for g, n in summary_obj["groups"].items():
        lines.append(f"- `{g}`: {n}")
    lines.append("")
    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("## Top Anomalies")
    for d in top:
        lines.append(
            f"- `{d.metric}` | score={d.anomaly_score:.1f} | nonfinite={d.nonfinite_count} | "
            f"jumps={d.jump_count} | gaps={d.gap_count} | flatline={d.flatline}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_one_run(
    run_file: Path,
    output_dir: Path,
    include_regex: Optional[str],
    exclude_regex: Optional[str],
    enabled_groups: List[str],
    custom_prefixes: List[str],
    cols: int,
    dpi: int,
    max_metrics: int,
    top_anomalies: int,
) -> None:
    raw_series = extract_series(run_file)
    series = filter_metrics(raw_series, include_regex=include_regex, exclude_regex=exclude_regex)
    groups = split_groups(series, enabled_groups=enabled_groups, custom_prefixes=custom_prefixes)

    diags = [metric_diagnostics(name, points) for name, points in series.items()]

    run_out = output_dir / run_id_for_path(run_file)
    run_out.mkdir(parents=True, exist_ok=True)

    plot_group(
        series=series,
        title=f"All Metrics ({run_id_for_path(run_file)})",
        output_path=run_out / "all_metrics.png",
        cols=cols,
        dpi=dpi,
        max_metrics=max_metrics,
    )
    for group_name, group_series in groups.items():
        if not group_series:
            continue
        plot_group(
            series=group_series,
            title=f"{group_name.capitalize()} Metrics ({run_id_for_path(run_file)})",
            output_path=run_out / f"group_{group_name}.png",
            cols=cols,
            dpi=dpi,
        )

    write_summary(
        out_dir=run_out,
        run_file=run_file,
        diagnostics=diags,
        groups=groups,
        top_k=top_anomalies,
    )


def main() -> None:
    args = parse_args()
    run_files = collect_run_files(config_paths=args.config_path, run_files=args.run_file)
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for run_file in run_files:
        analyze_one_run(
            run_file=run_file,
            output_dir=out_root,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            enabled_groups=args.groups,
            custom_prefixes=args.custom_prefix,
            cols=args.cols,
            dpi=args.dpi,
            max_metrics=args.max_metrics,
            top_anomalies=args.top_anomalies,
        )
        print(f"[ok] analyzed: {run_file}")

    print(f"[ok] output dir: {out_root}")


if __name__ == "__main__":
    main()
