#!/usr/bin/env python3
"""Plot Eval Metrics Dict curves from a local W&B run given a model config.yml path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all 'Eval Metrics Dict/*' curves from a local W&B run."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the model config.yml (e.g. .../f3rm/<timestamp>/config.yml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to <run_dir>/eval_metrics_dict_curves.png",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of subplot columns in the combined figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )
    return parser.parse_args()


def resolve_run_file(config_path: Path) -> Path:
    run_dir = config_path.resolve().parent
    wandb_root = run_dir / "wandb"
    if not wandb_root.exists():
        raise FileNotFoundError(f"Missing wandb directory: {wandb_root}")

    candidates = sorted(wandb_root.glob("run-*/run-*.wandb"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run-*.wandb file found under: {wandb_root}")
    return candidates[-1]


def _import_wandb_readers():
    try:
        from wandb.sdk.internal.datastore import DataStore
        import wandb.proto.wandb_internal_pb2 as pb
    except Exception as exc:
        raise RuntimeError(
            "Failed to import wandb internals. Run this script in the environment where wandb is installed."
        ) from exc
    return DataStore, pb


def _history_items(run_file: Path) -> Iterable[Tuple[int, str, object]]:
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
            key = "/".join(item.nested_key) if len(item.nested_key) > 0 else item.key
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


def extract_metric_series(
    run_file: Path, prefix: str = "Eval Metrics Dict/"
) -> Dict[str, List[Tuple[int, float]]]:
    series: Dict[str, Dict[int, float]] = {}
    for step, key, value in _history_items(run_file):
        if not key.startswith(prefix):
            continue
        if not isinstance(value, (int, float, bool)):
            continue
        metric_name = key[len(prefix) :]
        if metric_name not in series:
            series[metric_name] = {}
        series[metric_name][step] = float(value)

    out: Dict[str, List[Tuple[int, float]]] = {}
    for metric_name, step_to_value in series.items():
        out[metric_name] = sorted(step_to_value.items(), key=lambda x: x[0])
    return out


def plot_metric_series(
    metric_series: Dict[str, List[Tuple[int, float]]], output_path: Path, cols: int = 3, dpi: int = 140
) -> None:
    if not metric_series:
        raise ValueError("No 'Eval Metrics Dict/*' scalar series found in W&B history.")

    metric_names = sorted(metric_series.keys())
    n = len(metric_names)
    cols = max(1, cols)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.2 * rows), squeeze=False)
    flat_axes = [ax for row_axes in axes for ax in row_axes]

    for idx, metric_name in enumerate(metric_names):
        ax = flat_axes[idx]
        points = metric_series[metric_name]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, linewidth=1.8)
        ax.set_title(metric_name)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle("Eval Metrics Dict Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config_path does not exist: {config_path}")

    run_file = resolve_run_file(config_path)
    series = extract_metric_series(run_file, prefix="Eval Metrics Dict/")

    output_path = Path(args.output) if args.output else config_path.resolve().parent / "eval_metrics_dict_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_metric_series(series, output_path=output_path, cols=args.cols, dpi=args.dpi)

    print(f"Run file: {run_file}")
    print(f"Num metrics: {len(series)}")
    print(f"Saved plot: {output_path}")

    if args.show:
        # Re-open minimally for interactive mode.
        img = plt.imread(output_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
