"""
Passive analysis for F3RM training-speed experiments.

This script reads a tab-separated results log, prints summaries, and saves a
non-interactive plot. It does not mutate experimental state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULTS_PATH = Path("results.tsv")
DEFAULT_PLOT_PATH = Path("progress.png")

COMMIT_COLUMN = "commit"
PROFILE_COLUMN = "profile"
METRIC_COLUMN = "wall_time_s"
QUALITY_COLUMN = "max_quality_regression_pct"
STATUS_COLUMN = "status"
DESCRIPTION_COLUMN = "description"
SUMMARY_PATH_COLUMN = "summary_path"
OFFICIAL_PROFILE = "measure"
KEEP_LABEL = "KEEP"
DISCARD_LABEL = "DISCARD"
CRASH_LABEL = "CRASH"
QUALITY_CLEAR_PCT = 10.0
QUALITY_REVIEW_PCT = 15.0

TIMING_COLUMNS = [
    "train_total_time_s",
    "startup_overhead_s",
    "train_iter_time_s",
    "train_rays_per_sec",
    "train_batch_load_s",
    "train_feature_cache_load_s",
    "train_feature_gather_s",
    "train_feature_populate_s",
    "train_model_forward_s",
    "train_metrics_loss_s",
    "train_image_viz_s",
    "eval_batch_load_s",
    "eval_feature_cache_load_s",
    "eval_feature_gather_s",
    "eval_feature_populate_s",
    "eval_image_s",
    "eval_all_images_s",
]

NUMERIC_COLUMNS = [
    METRIC_COLUMN,
    QUALITY_COLUMN,
    "peak_gpu_mem_mb",
    "pixel_visit_x",
    "pixel_visit_x_pct_diff",
    "window_coverage_w",
    "window_coverage_w_pct_diff",
    "train_psnr",
    "train_feature_error",
    "train_foreground_acc",
    "eval_all_psnr",
    "eval_all_ssim",
    "eval_all_lpips",
    *TIMING_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passive analysis for F3RM benchmark runs.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Results TSV to analyze.")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH, help="Output plot path.")
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Results file not found: {path}\n"
            "Pass --results <path> explicitly if you want to analyze a non-default results file."
        )
    df = pd.read_csv(path, sep="\t")
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if PROFILE_COLUMN in df.columns:
        df[PROFILE_COLUMN] = df[PROFILE_COLUMN].astype(str).str.strip().str.lower()
    df[STATUS_COLUMN] = df[STATUS_COLUMN].astype(str).str.strip().str.upper()
    return df


def select_official_runs(df: pd.DataFrame) -> pd.DataFrame:
    if PROFILE_COLUMN not in df.columns:
        return df.copy().reset_index(drop=True)
    return df[df[PROFILE_COLUMN] == OFFICIAL_PROFILE].copy().reset_index(drop=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    baseline = float(df.loc[0, METRIC_COLUMN])
    quality = df[QUALITY_COLUMN]
    quality_known = quality.notna()
    df["quality_clear"] = quality_known & (quality <= QUALITY_CLEAR_PCT)
    df["quality_reviewable"] = quality_known & (quality <= QUALITY_REVIEW_PCT)
    df["quality_band"] = "unknown"
    df.loc[quality_known, "quality_band"] = "reject"
    df.loc[quality_known & (quality <= QUALITY_REVIEW_PCT), "quality_band"] = "judgment"
    df.loc[quality_known & (quality <= QUALITY_CLEAR_PCT), "quality_band"] = "clear"
    df["completed"] = df[STATUS_COLUMN] != CRASH_LABEL
    df["valid"] = df["completed"] & df["quality_reviewable"] & df[METRIC_COLUMN].notna()
    df["speedup_pct"] = 0.0 if baseline == 0 else (baseline - df[METRIC_COLUMN]) / baseline * 100.0
    return df


def print_header(df: pd.DataFrame, results_path: Path) -> None:
    print(f"Results path: {results_path}")
    print(f"Official measure runs: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if not df.empty:
        print(df.head(10).to_string(index=False))


def print_outcomes(df: pd.DataFrame) -> None:
    counts = df[STATUS_COLUMN].value_counts()
    print("\nExperiment outcomes:")
    print(counts.to_string())

    quality_clear = int(df["quality_clear"].sum()) if "quality_clear" in df else 0
    quality_review = int((df["quality_band"] == "judgment").sum()) if "quality_band" in df else 0
    quality_reject = int((df["quality_band"] == "reject").sum()) if "quality_band" in df else 0
    quality_unknown = int((df["quality_band"] == "unknown").sum()) if "quality_band" in df else 0
    print(f"\nWithin 10% comfort zone: {quality_clear}/{len(df)}")
    print(f"In 10-15% judgment band: {quality_review}/{len(df)}")
    print(f"Over 15% quality degradation: {quality_reject}/{len(df)}")
    print(f"Missing quality summary: {quality_unknown}/{len(df)}")


def print_fastest_valid_runs(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["valid"]].copy().sort_values(METRIC_COLUMN)
    if valid.empty:
        print("\nNo completed runs within the 15% review window.")
        return valid

    baseline = float(df.loc[0, METRIC_COLUMN])
    best = valid.iloc[0]
    print("\nFastest review-eligible run:")
    print(
        f"  {best[COMMIT_COLUMN]}  wall_time_s={best[METRIC_COLUMN]:.3f}  "
        f"speedup={best['speedup_pct']:.2f}%  quality_regression={best[QUALITY_COLUMN]:.2f}%  "
        f"band={best['quality_band']}"
    )
    print(f"  {best[DESCRIPTION_COLUMN]}")
    if SUMMARY_PATH_COLUMN in best:
        print(f"  summary={best[SUMMARY_PATH_COLUMN]}")

    print("\nTop review-eligible runs:")
    for _, row in valid.head(10).iterrows():
        print(
            f"  {row[COMMIT_COLUMN]}  wall_time_s={row[METRIC_COLUMN]:.3f}  "
            f"speedup={row['speedup_pct']:.2f}%  quality_regression={row[QUALITY_COLUMN]:.2f}%  "
            f"band={row['quality_band']}  {row[DESCRIPTION_COLUMN]}"
        )

    clear = valid[valid["quality_band"] == "clear"]
    if not clear.empty:
        best_clear = clear.iloc[0]
        print("\nFastest run in the <=10% comfort zone:")
        print(
            f"  {best_clear[COMMIT_COLUMN]}  wall_time_s={best_clear[METRIC_COLUMN]:.3f}  "
            f"speedup={best_clear['speedup_pct']:.2f}%  quality_regression={best_clear[QUALITY_COLUMN]:.2f}%"
        )
        print(f"  {best_clear[DESCRIPTION_COLUMN]}")

    print(f"\nBaseline wall_time_s: {baseline:.3f}")
    return valid


def print_kept_runs(df: pd.DataFrame) -> None:
    kept = df[df[STATUS_COLUMN] == KEEP_LABEL].copy()
    print(f"\nKept runs: {len(kept)}")
    for idx, row in kept.iterrows():
        print(
            f"  #{idx:03d}  {row[COMMIT_COLUMN]}  wall_time_s={row[METRIC_COLUMN]:.3f}  "
            f"quality_regression={row[QUALITY_COLUMN]:.2f}%  band={row['quality_band']}  "
            f"{row[DESCRIPTION_COLUMN]}"
        )


def print_best_phase_breakdown(valid: pd.DataFrame) -> None:
    if valid.empty:
        return
    best = valid.iloc[0]
    train_iter = best.get("train_iter_time_s")
    print("\nBest-run timing breakdown:")
    print(
        f"  wall_time_s={best.get('wall_time_s'):.3f}  "
        f"train_total_time_s={best.get('train_total_time_s'):.3f}  "
        f"startup_overhead_s={best.get('startup_overhead_s'):.3f}"
    )

    phase_rows = []
    for column in TIMING_COLUMNS:
        if column in {"train_total_time_s", "startup_overhead_s"}:
            continue
        value = best.get(column)
        if pd.isna(value):
            continue
        share = None
        if column.startswith("train_") and column not in {"train_total_time_s", "train_rays_per_sec"}:
            if pd.notna(train_iter) and float(train_iter) > 0:
                share = float(value) / float(train_iter) * 100.0
        phase_rows.append((column, float(value), share))

    phase_rows.sort(key=lambda item: item[1], reverse=True)
    for column, value, share in phase_rows[:10]:
        if share is None:
            print(f"  {column}={value:.6f}")
        else:
            print(f"  {column}={value:.6f} ({share:.1f}% of train_iter_time_s)")


def plot_progress(df: pd.DataFrame, plot_path: Path) -> None:
    if df.empty:
        print("No official runs available; skipping plot.")
        return

    fig, (ax_time, ax_tradeoff) = plt.subplots(2, 1, figsize=(14, 11), height_ratios=[1.5, 1.0])
    baseline = float(df.loc[0, METRIC_COLUMN])

    non_crash = df[df[STATUS_COLUMN] != CRASH_LABEL].copy()
    crash = df[df[STATUS_COLUMN] == CRASH_LABEL].copy()
    discarded = non_crash[non_crash[STATUS_COLUMN] == DISCARD_LABEL]
    kept = non_crash[non_crash[STATUS_COLUMN] == KEEP_LABEL]
    valid = non_crash[non_crash["valid"]].copy()

    ax_time.scatter(discarded.index, discarded[METRIC_COLUMN], c="#c7c7c7", s=22, alpha=0.8, label="Discarded")
    ax_time.scatter(kept.index, kept[METRIC_COLUMN], c="#2b8c56", s=44, label="Kept")
    ax_time.scatter(crash.index, [baseline] * len(crash), c="#b2182b", marker="x", s=60, label="Crash")
    ax_time.axhline(baseline, color="#4c78a8", linestyle="--", linewidth=1.2, label="Baseline")

    if not valid.empty:
        frontier = valid[METRIC_COLUMN].cummin()
        ax_time.step(valid.index, frontier, where="post", color="#0b6e4f", linewidth=2.0, label="Running best")
        best = valid.loc[valid[METRIC_COLUMN].idxmin()]
        ax_time.annotate(
            f"best {best[METRIC_COLUMN]:.3f}s",
            (best.name, best[METRIC_COLUMN]),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=9,
        )

    ax_time.set_title("Official Measure Progress")
    ax_time.set_xlabel("Experiment #")
    ax_time.set_ylabel("wall_time_s")
    ax_time.grid(True, alpha=0.2)
    ax_time.legend(loc="best")

    tradeoff = non_crash.copy()
    colors = tradeoff["quality_band"].map(
        {"clear": "#2b8c56", "judgment": "#f28e2b", "reject": "#b2182b", "unknown": "#7f7f7f"}
    )
    ax_tradeoff.scatter(
        tradeoff[METRIC_COLUMN],
        tradeoff[QUALITY_COLUMN].fillna(0.0),
        c=colors.tolist(),
        s=36,
        alpha=0.9,
    )
    ax_tradeoff.axhline(QUALITY_CLEAR_PCT, color="#4c78a8", linestyle="--", linewidth=1.2, label="10% comfort")
    ax_tradeoff.axhline(QUALITY_REVIEW_PCT, color="#b2182b", linestyle="--", linewidth=1.2, label="15% ceiling")
    ax_tradeoff.axvline(baseline, color="#4c78a8", linestyle="--", linewidth=1.2, label="Baseline wall time")
    ax_tradeoff.set_title("Speed vs Quality Regression")
    ax_tradeoff.set_xlabel("wall_time_s")
    ax_tradeoff.set_ylabel("max_quality_regression_pct")
    ax_tradeoff.grid(True, alpha=0.2)
    ax_tradeoff.legend(loc="best")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {plot_path}")


def main() -> None:
    args = parse_args()
    df = add_derived_columns(select_official_runs(load_results(args.results)))
    print_header(df, args.results)
    print_outcomes(df)
    valid = print_fastest_valid_runs(df)
    print_kept_runs(df)
    print_best_phase_breakdown(valid)
    plot_progress(df, args.plot)


if __name__ == "__main__":
    main()
