"""
Passive analysis for F3RM autoresearch runs.

This script consumes the stable `results.tsv` schema defined by `program.md`
and analyzes the official `measure` rows. If the run also maintains a
`hypothesis.md` file, treat that as a living qualitative backlog for active
hypotheses and out-of-scope findings; this script should stay focused on the
structured benchmark outcomes in `results.tsv`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results.tsv")
PLOT_PATH = Path("progress.png")

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

NUMERIC_COLUMNS = [
    "wall_time_s",
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
    "eval_batch_load_s",
    "eval_feature_cache_load_s",
    "eval_feature_gather_s",
    "eval_feature_populate_s",
    "eval_image_s",
    "eval_all_images_s",
    "peak_gpu_mem_mb",
    "train_psnr",
    "train_feature_error",
    "train_foreground_acc",
    "eval_all_psnr",
    "eval_all_ssim",
    "eval_all_lpips",
    "max_quality_regression_pct",
]


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df[PROFILE_COLUMN] = df[PROFILE_COLUMN].astype(str).str.strip().str.lower()
    df[STATUS_COLUMN] = df[STATUS_COLUMN].astype(str).str.strip().str.upper()
    return df


def select_official_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[PROFILE_COLUMN] == OFFICIAL_PROFILE].copy().reset_index(drop=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    baseline = float(df.loc[0, METRIC_COLUMN])
    quality = df[QUALITY_COLUMN].fillna(0.0)
    df["quality_clear"] = quality <= QUALITY_CLEAR_PCT
    df["quality_reviewable"] = quality <= QUALITY_REVIEW_PCT
    df["quality_band"] = "reject"
    df.loc[quality <= QUALITY_REVIEW_PCT, "quality_band"] = "judgment"
    df.loc[quality <= QUALITY_CLEAR_PCT, "quality_band"] = "clear"
    df["completed"] = df[STATUS_COLUMN] != CRASH_LABEL
    df["valid"] = df["completed"] & df["quality_reviewable"] & df[METRIC_COLUMN].notna()
    df["speedup_pct"] = 0.0 if baseline == 0 else (baseline - df[METRIC_COLUMN]) / baseline * 100.0
    return df


def print_header(df: pd.DataFrame) -> None:
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
    print(f"\nWithin 10% comfort zone: {quality_clear}/{len(df)}")
    print(f"In 10-15% judgment band: {quality_review}/{len(df)}")
    print(f"Over 15% quality degradation: {quality_reject}/{len(df)}")


def print_fastest_valid_runs(df: pd.DataFrame) -> None:
    valid = df[df["valid"]].copy().sort_values(METRIC_COLUMN)
    if valid.empty:
        print("\nNo completed runs within the 15% quality review window.")
        return

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


def print_kept_runs(df: pd.DataFrame) -> None:
    kept = df[df[STATUS_COLUMN] == KEEP_LABEL].copy()
    print(f"\nKept runs: {len(kept)}")
    for idx, row in kept.iterrows():
        print(
            f"  #{idx:03d}  {row[COMMIT_COLUMN]}  wall_time_s={row[METRIC_COLUMN]:.3f}  "
            f"quality_regression={row[QUALITY_COLUMN]:.2f}%  band={row['quality_band']}  {row[DESCRIPTION_COLUMN]}"
        )


def plot_progress(df: pd.DataFrame) -> None:
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
    colors = tradeoff["quality_band"].map({"clear": "#2b8c56", "judgment": "#f28e2b", "reject": "#b2182b"})
    ax_tradeoff.scatter(
        tradeoff[METRIC_COLUMN],
        tradeoff[QUALITY_COLUMN].fillna(0.0),
        c=colors,
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
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {PLOT_PATH}")


def main() -> None:
    df = add_derived_columns(select_official_runs(load_results(RESULTS_PATH)))
    print_header(df)
    print_outcomes(df)
    print_fastest_valid_runs(df)
    print_kept_runs(df)
    plot_progress(df)


if __name__ == "__main__":
    main()
