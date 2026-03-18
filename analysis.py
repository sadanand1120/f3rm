"""
Passive analysis for compression-oriented F3RM autoresearch runs.

Expected input:
- measure-only rows in results.tsv
- the reduced schema defined in program.md
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results.tsv")
PLOT_PATH = Path("compression_progress.png")

STATUS_COLUMN = "status"
DESCRIPTION_COLUMN = "description"
REVISION_COLUMN = "revision"
LOC_COLUMN = "f3rm_loc"
CRASH_LABEL = "crash"

TIME_COLUMNS = ["extract_wall_s", "train_wall_s"]
METRIC_COLUMNS = [
    "final_eval_all_psnr",
    "final_eval_all_ssim",
    "final_eval_all_lpips",
    "final_train_feature_error",
]
NUMERIC_COLUMNS = [LOC_COLUMN, *TIME_COLUMNS, *METRIC_COLUMNS]
HIGHER_IS_BETTER = {"final_eval_all_psnr", "final_eval_all_ssim"}
TARGET_MAX_DEGRADE = 0.05
HARD_MAX_DEGRADE = 0.10


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df[STATUS_COLUMN] = df[STATUS_COLUMN].astype(str).str.strip().str.lower()
    df["end_to_end_wall_s"] = df["extract_wall_s"] + df["train_wall_s"]
    return df


def print_header(df: pd.DataFrame) -> None:
    print(f"Total measure runs: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(10).to_string())


def print_outcomes(df: pd.DataFrame) -> None:
    print("\nRun outcomes:")
    print(df[STATUS_COLUMN].value_counts().to_string())


def within_guardrail(value: float, baseline: float, max_degrade: float, higher_is_better: bool) -> bool:
    if pd.isna(value) or pd.isna(baseline):
        return False
    if higher_is_better:
        return value >= baseline * (1.0 - max_degrade)
    return value <= baseline * (1.0 + max_degrade)


def add_constraint_flags(valid: pd.DataFrame, baseline: pd.Series, max_degrade: float, column: str) -> None:
    valid[column] = valid["end_to_end_wall_s"].apply(
        lambda value: within_guardrail(value, baseline["end_to_end_wall_s"], max_degrade, higher_is_better=False)
    )
    for metric in METRIC_COLUMNS:
        valid[column] &= valid[metric].apply(
            lambda value, metric_name=metric: within_guardrail(
                value,
                baseline[metric_name],
                max_degrade,
                higher_is_better=metric_name in HIGHER_IS_BETTER,
            )
        )


def print_candidate(label: str, row: pd.Series, baseline: pd.Series) -> None:
    end_to_end = row["end_to_end_wall_s"]
    base_end = baseline["end_to_end_wall_s"]
    end_pct = 0.0 if base_end == 0 else (end_to_end - base_end) / base_end * 100.0
    print(f"\n{label}")
    print(f"Revision:          {row.get(REVISION_COLUMN, 'n/a')}")
    print(f"Description:       {row.get(DESCRIPTION_COLUMN, '')}")
    print(f"LOC:               {row[LOC_COLUMN]:.0f} (baseline {baseline[LOC_COLUMN]:.0f})")
    print(f"Extract wall:      {row['extract_wall_s']:.3f}s")
    print(f"Train wall:        {row['train_wall_s']:.3f}s")
    print(f"End-to-end:        {end_to_end:.3f}s ({end_pct:+.2f}%)")
    for metric in METRIC_COLUMNS:
        base_value = baseline[metric]
        value = row[metric]
        if pd.isna(base_value) or pd.isna(value):
            continue
        pct = 0.0 if base_value == 0 else (value - base_value) / base_value * 100.0
        print(f"{metric}: {value:.6f} ({pct:+.2f}% vs baseline)")


def print_best(df: pd.DataFrame) -> None:
    valid = df[df[STATUS_COLUMN] != CRASH_LABEL].copy().reset_index(drop=True)
    if valid.empty:
        print("\nNo non-crash runs available.")
        return

    baseline = valid.iloc[0].copy()
    add_constraint_flags(valid, baseline, TARGET_MAX_DEGRADE, "target_ok")
    add_constraint_flags(valid, baseline, HARD_MAX_DEGRADE, "hard_ok")

    print(f"\nBaseline LOC:          {baseline[LOC_COLUMN]:.0f}")
    print(f"Baseline end-to-end:   {baseline['end_to_end_wall_s']:.3f}s")
    print(f"Within 5% guardrails:  {int(valid['target_ok'].sum())}")
    print(f"Within 10% guardrails: {int(valid['hard_ok'].sum())}")

    smallest = valid.sort_values([LOC_COLUMN, "end_to_end_wall_s"]).iloc[0]
    print_candidate("Smallest code footprint", smallest, baseline)

    hard_ok = valid[valid["hard_ok"]]
    if hard_ok.empty:
        print("\nNo runs satisfy the 10% hard guardrails.")
        return

    print_candidate(
        "Best compression within 10% guardrails",
        hard_ok.sort_values([LOC_COLUMN, "end_to_end_wall_s"]).iloc[0],
        baseline,
    )

    target_ok = valid[valid["target_ok"]]
    if not target_ok.empty:
        print_candidate(
            "Best compression within 5% guardrails",
            target_ok.sort_values([LOC_COLUMN, "end_to_end_wall_s"]).iloc[0],
            baseline,
        )


def plot_progress(df: pd.DataFrame) -> None:
    valid = df[df[STATUS_COLUMN] != CRASH_LABEL].copy().reset_index(drop=True)
    if valid.empty:
        print("\nNo non-crash runs available; skipping plot.")
        return

    baseline_end = valid.iloc[0]["end_to_end_wall_s"]
    color_map = {"keep": "#2e8b57", "discard": "#bdbdbd", "crash": "#d62728"}
    colors = [color_map.get(status, "#4c78a8") for status in valid[STATUS_COLUMN]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].scatter(valid.index, valid[LOC_COLUMN], c=colors, s=40)
    axes[0].step(valid.index, valid[LOC_COLUMN].cummin(), where="post", color="#1f6d43", linewidth=2)
    axes[0].set_ylabel("f3rm_loc")
    axes[0].set_title("Code Compression Progress")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(valid.index, valid["end_to_end_wall_s"], marker="o", linewidth=1.8, color="#4c78a8")
    axes[1].axhline(baseline_end, color="#999999", linestyle="--", linewidth=1)
    axes[1].axhline(baseline_end * (1.0 + TARGET_MAX_DEGRADE), color="#54a24b", linestyle=":", linewidth=1)
    axes[1].axhline(baseline_end * (1.0 + HARD_MAX_DEGRADE), color="#e45756", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Experiment #")
    axes[1].set_ylabel("extract + train (s)")
    axes[1].set_title("Derived End-to-End Time")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {PLOT_PATH}")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} does not exist yet. Create it during setup as described in program.md.")
    df = load_results(RESULTS_PATH)
    print_header(df)
    print_outcomes(df)
    print_best(df)
    plot_progress(df)


if __name__ == "__main__":
    main()
