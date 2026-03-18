"""
Passive analysis for F3RM autoresearch runs.

Expected input:
- measure-only rows in results.tsv
- the stable schema defined in program.md
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results.tsv")
PROGRESS_PATH = Path("progress.png")
PHASE_PATH = Path("phase_breakdown.png")

STATUS_COLUMN = "status"
DESCRIPTION_COLUMN = "description"
COMMIT_COLUMN = "commit"
PRIMARY_METRIC = "end_to_end_wall_s"
KEEP_LABEL = "keep"
DISCARD_LABEL = "discard"
CRASH_LABEL = "crash"

NUMERIC_COLUMNS = [
    "end_to_end_wall_s",
    "extract_wall_s",
    "train_wall_s",
    "extract_worker_init_s",
    "extract_worker_warmup_s",
    "extract_batch_compute_s",
    "extract_per_image_write_s",
    "extract_parallel_ok",
    "peak_extract_gpu_mem_mb",
    "peak_train_gpu_mem_mb",
    "train_feature_cache_load_avg_s",
    "train_feature_window_fetch_avg_s",
    "train_feature_window_stack_avg_s",
    "train_batch_load_avg_s",
    "train_model_forward_avg_s",
    "final_eval_all_psnr",
    "final_eval_all_ssim",
    "final_eval_all_lpips",
    "final_train_feature_error",
]


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df[STATUS_COLUMN] = df[STATUS_COLUMN].astype(str).str.strip().str.lower()
    return df


def print_header(df: pd.DataFrame) -> None:
    print(f"Total measure runs: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(10).to_string())


def print_outcomes(df: pd.DataFrame) -> None:
    print("\nRun outcomes:")
    print(df[STATUS_COLUMN].value_counts().to_string())
    if "extract_parallel_ok" in df.columns:
        invalid = (df["extract_parallel_ok"] != 1).sum()
        print(f"\nRuns with invalid extraction overlap: {int(invalid)}")


def print_best(df: pd.DataFrame) -> None:
    valid = df[df[STATUS_COLUMN] != CRASH_LABEL].copy()
    if valid.empty:
        print("\nNo non-crash runs available.")
        return

    baseline = valid.iloc[0]
    best = valid.loc[valid[PRIMARY_METRIC].idxmin()]
    delta = baseline[PRIMARY_METRIC] - best[PRIMARY_METRIC]
    pct = 0.0 if baseline[PRIMARY_METRIC] == 0 else delta / baseline[PRIMARY_METRIC] * 100.0

    print(f"\nBaseline {PRIMARY_METRIC}: {baseline[PRIMARY_METRIC]:.3f}s")
    print(f"Best {PRIMARY_METRIC}:     {best[PRIMARY_METRIC]:.3f}s")
    print(f"Improvement:               {delta:.3f}s ({pct:.2f}%)")
    print(f"Best commit:               {best.get(COMMIT_COLUMN, 'n/a')}")
    print(f"Best description:          {best.get(DESCRIPTION_COLUMN, '')}")

    for column in [
        "extract_wall_s",
        "train_wall_s",
        "train_feature_cache_load_avg_s",
        "train_batch_load_avg_s",
        "train_model_forward_avg_s",
        "final_eval_all_psnr",
        "final_eval_all_ssim",
        "final_eval_all_lpips",
        "final_train_feature_error",
    ]:
        if column not in df.columns:
            continue
        before = baseline[column]
        after = best[column]
        if pd.isna(before) or pd.isna(after):
            continue
        print(f"{column}: {before:.6f} -> {after:.6f}")


def plot_progress(df: pd.DataFrame) -> None:
    valid = df[df[STATUS_COLUMN] != CRASH_LABEL].copy().reset_index(drop=True)
    if valid.empty:
        print("\nNo non-crash runs available; skipping progress plot.")
        return

    kept = valid[valid[STATUS_COLUMN] == KEEP_LABEL]
    discarded = valid[valid[STATUS_COLUMN] == DISCARD_LABEL]
    frontier = kept[PRIMARY_METRIC].cummin() if not kept.empty else pd.Series(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(discarded.index, discarded[PRIMARY_METRIC], c="#bdbdbd", s=20, alpha=0.7, label="discard")
    ax.scatter(kept.index, kept[PRIMARY_METRIC], c="#2e8b57", s=44, label="keep")
    if not kept.empty:
        ax.step(kept.index, frontier, where="post", color="#1f6d43", linewidth=2, alpha=0.8, label="running best")

    ax.set_xlabel("Experiment #")
    ax.set_ylabel(PRIMARY_METRIC)
    ax.set_title("F3RM End-to-End Progress")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PROGRESS_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {PROGRESS_PATH}")


def plot_phase_breakdown(df: pd.DataFrame) -> None:
    valid = df[df[STATUS_COLUMN] != CRASH_LABEL].copy().reset_index(drop=True)
    if valid.empty:
        print("No non-crash runs available; skipping phase breakdown plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].bar(valid.index, valid["extract_wall_s"], color="#4c78a8", label="extract")
    axes[0].bar(valid.index, valid["train_wall_s"], bottom=valid["extract_wall_s"], color="#f58518", label="train")
    axes[0].set_ylabel("wall time (s)")
    axes[0].set_title("Phase Breakdown")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.2)

    for column, color in [
        ("train_feature_cache_load_avg_s", "#54a24b"),
        ("train_batch_load_avg_s", "#e45756"),
        ("train_model_forward_avg_s", "#72b7b2"),
    ]:
        if column in valid.columns:
            axes[1].plot(valid.index, valid[column], marker="o", linewidth=1.5, label=column, color=color)
    axes[1].set_xlabel("Experiment #")
    axes[1].set_ylabel("avg seconds")
    axes[1].set_title("Training Timing Averages")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(PHASE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PHASE_PATH}")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} does not exist yet. Create it during setup as described in program.md.")
    df = load_results(RESULTS_PATH)
    print_header(df)
    print_outcomes(df)
    print_best(df)
    plot_progress(df)
    plot_phase_breakdown(df)


if __name__ == "__main__":
    main()
