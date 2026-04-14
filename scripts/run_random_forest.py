from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATASET_NAME, METRICS_DIR, RAW_DATA_DIR
from src.train_random_forest import DEFAULT_SAMPLE_SIZE, run_random_forest_experiment


def main() -> None:
    csv_path = RAW_DATA_DIR / DEFAULT_DATASET_NAME
    metrics_path = METRICS_DIR / "random_forest_metrics.json"

    try:
        result = run_random_forest_experiment(
            csv_path=csv_path,
            sample_size=DEFAULT_SAMPLE_SIZE,
            metrics_output_path=metrics_path,
        )
    except ModuleNotFoundError as exc:
        print(f"Cannot run Random Forest experiment: {exc}")
        return

    print(f"Sample size used: {result['sample_size']}")
    print(f"Raw data feature shape: {tuple(result['raw_feature_shape'])}")
    print(
        "Preprocessed feature shapes: "
        f"train={tuple(result['preprocessed_feature_shapes']['train'])}, "
        f"val={tuple(result['preprocessed_feature_shapes']['val'])}, "
        f"test={tuple(result['preprocessed_feature_shapes']['test'])}"
    )
    print(f"Best params: {result['best_params']}")
    print(f"Validation metrics: {result['validation_metrics']}")
    print(f"Test metrics: {result['test_metrics']}")
    print(f"Saved metrics to: {result['metrics_output_path']}")


if __name__ == "__main__":
    main()
