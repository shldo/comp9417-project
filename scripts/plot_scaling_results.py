from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INPUT_CSV_PATH = PROJECT_ROOT / "results" / "metrics" / "scaling_experiment_results.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
ROC_AUC_FIGURE_PATH = FIGURES_DIR / "test_roc_auc_vs_sample_size.png"
TRAINING_TIME_FIGURE_PATH = FIGURES_DIR / "training_time_vs_sample_size_log.png"
SUMMARY_CSV_PATH = PROJECT_ROOT / "results" / "metrics" / "scaling_experiment_summary.csv"

SUMMARY_COLUMNS = [
    "model",
    "sample_size",
    "test_accuracy",
    "test_roc_auc",
    "training_time_seconds",
    "inference_time_per_sample",
]


def load_scaling_results(csv_path: Path) -> pd.DataFrame:
    """Load the scaling experiment results from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Scaling experiment CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.sort_values(by=["model", "sample_size"]).reset_index(drop=True)
    return df


def save_summary_csv(df: pd.DataFrame, output_path: Path) -> Path:
    """Save the concise results table used in the report body."""
    summary_df = df.loc[:, SUMMARY_COLUMNS].copy()
    summary_df.to_csv(output_path, index=False)
    return output_path


def plot_metric_vs_sample_size(
    df: pd.DataFrame,
    y_column: str,
    y_label: str,
    title: str,
    output_path: Path,
    log_y: bool = False,
) -> Path:
    """Create and save a single line chart for one metric."""
    plt.figure(figsize=(8, 5), dpi=200)

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name].sort_values(by="sample_size")
        plt.plot(
            model_df["sample_size"],
            model_df[y_column],
            marker="o",
            label=model_name,
        )

    plt.title(title)
    plt.xlabel("Sample Size")
    plt.ylabel(y_label)
    if log_y:
        plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scaling_results(INPUT_CSV_PATH)

    saved_roc_auc_path = plot_metric_vs_sample_size(
        df=df,
        y_column="test_roc_auc",
        y_label="Test ROC-AUC",
        title="Test ROC-AUC vs Sample Size",
        output_path=ROC_AUC_FIGURE_PATH,
    )
    saved_training_time_path = plot_metric_vs_sample_size(
        df=df,
        y_column="training_time_seconds",
        y_label="Training Time (Seconds)",
        title="Training Time vs Sample Size (Log Scale)",
        output_path=TRAINING_TIME_FIGURE_PATH,
        log_y=True,
    )
    saved_summary_path = save_summary_csv(df, SUMMARY_CSV_PATH)

    print(f"Read {len(df)} results from: {INPUT_CSV_PATH.resolve()}")
    print(f"Saved figure: {saved_roc_auc_path.resolve()}")
    print(f"Saved figure: {saved_training_time_path.resolve()}")
    print(f"Saved summary CSV: {saved_summary_path.resolve()}")


if __name__ == "__main__":
    main()
