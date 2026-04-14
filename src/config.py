from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

DEFAULT_DATASET_NAME = "sleep_health_dataset.csv"
TARGET_COLUMN = "felt_rested"
DROP_COLUMNS = [
    "person_id",
    "sleep_disorder_risk",
    "cognitive_performance_score",
]


def ensure_project_dirs() -> None:
    """Create the standard project directories if they do not exist."""
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
