from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATASET_NAME, METRICS_DIR, RAW_DATA_DIR, ensure_project_dirs
from src.data_loading import get_feature_target_data, load_raw_data
from src.preprocessing import (
    build_preprocessor,
    detect_feature_types,
    fit_transform_features,
    transform_features,
)
from src.splitting import make_train_val_test_split
from src.train_xrfm import (
    build_categorical_info,
    evaluate_binary_classifier,
    train_single_xrfm_model,
)


SAMPLE_SIZE = 10000
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42

JSON_OUTPUT_PATH = METRICS_DIR / "xrfm_tuning_10000.json"
CSV_OUTPUT_PATH = METRICS_DIR / "xrfm_tuning_10000.csv"


XRFM_CONFIGS = [
    {
        "config_id": "A",
        "notes": "Current baseline",
        "params": {
            "tuning_metric": "auc",
            "max_leaf_size": 10000,
            "n_trees": 1,
            "n_tree_iters": 0,
            "rfm_params": {
                "model": {
                    "kernel": "l2_high_dim",
                    "bandwidth": 10.0,
                    "exponent": 1.0,
                    "diag": False,
                    "bandwidth_mode": "constant",
                },
                "fit": {
                    "reg": 1e-3,
                    "iters": 3,
                    "verbose": False,
                    "early_stop_rfm": True,
                },
            },
        },
    },
    {
        "config_id": "B",
        "notes": "Allow splitting with smaller max_leaf_size",
        "params": {
            "tuning_metric": "auc",
            "max_leaf_size": 3000,
            "n_trees": 1,
            "n_tree_iters": 0,
            "rfm_params": {
                "model": {
                    "kernel": "l2_high_dim",
                    "bandwidth": 10.0,
                    "exponent": 1.0,
                    "diag": False,
                    "bandwidth_mode": "constant",
                },
                "fit": {
                    "reg": 1e-3,
                    "iters": 3,
                    "verbose": False,
                    "early_stop_rfm": True,
                },
            },
        },
    },
    {
        "config_id": "C",
        "notes": "Allow splitting and train leaf RFM longer",
        "params": {
            "tuning_metric": "auc",
            "max_leaf_size": 3000,
            "n_trees": 1,
            "n_tree_iters": 0,
            "rfm_params": {
                "model": {
                    "kernel": "l2_high_dim",
                    "bandwidth": 10.0,
                    "exponent": 1.0,
                    "diag": False,
                    "bandwidth_mode": "constant",
                },
                "fit": {
                    "reg": 1e-3,
                    "iters": 5,
                    "verbose": False,
                    "early_stop_rfm": True,
                },
            },
        },
    },
    {
        "config_id": "D",
        "notes": "Allow splitting and use one tree iteration",
        "params": {
            "tuning_metric": "auc",
            "max_leaf_size": 3000,
            "n_trees": 1,
            "n_tree_iters": 1,
            "rfm_params": {
                "model": {
                    "kernel": "l2_high_dim",
                    "bandwidth": 10.0,
                    "exponent": 1.0,
                    "diag": False,
                    "bandwidth_mode": "constant",
                },
                "fit": {
                    "reg": 1e-3,
                    "iters": 3,
                    "verbose": False,
                    "early_stop_rfm": True,
                },
            },
        },
    },
    {
        "config_id": "E",
        "notes": "Allow splitting and average across two trees",
        "params": {
            "tuning_metric": "auc",
            "max_leaf_size": 3000,
            "n_trees": 2,
            "n_tree_iters": 0,
            "rfm_params": {
                "model": {
                    "kernel": "l2_high_dim",
                    "bandwidth": 10.0,
                    "exponent": 1.0,
                    "diag": False,
                    "bandwidth_mode": "constant",
                },
                "fit": {
                    "reg": 1e-3,
                    "iters": 3,
                    "verbose": False,
                    "early_stop_rfm": True,
                },
            },
        },
    },
]


def prepare_data(sample_size: int) -> dict[str, Any]:
    """Load, split, and preprocess the data once for a fair comparison."""
    csv_path = RAW_DATA_DIR / DEFAULT_DATASET_NAME
    df = load_raw_data(csv_path).head(sample_size).copy()
    X, y, _ = get_feature_target_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(
        X=X,
        y=y,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=True,
    )

    numeric_cols, categorical_cols = detect_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train_processed = fit_transform_features(preprocessor, X_train)
    X_val_processed = transform_features(preprocessor, X_val)
    X_test_processed = transform_features(preprocessor, X_test)
    categorical_info = build_categorical_info(preprocessor, numeric_cols, categorical_cols)

    return {
        "csv_path": csv_path.resolve(),
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "raw_feature_shape": [int(X.shape[0]), int(X.shape[1])],
        "preprocessed_feature_shapes": {
            "train": [int(X_train_processed.shape[0]), int(X_train_processed.shape[1])],
            "val": [int(X_val_processed.shape[0]), int(X_val_processed.shape[1])],
            "test": [int(X_test_processed.shape[0]), int(X_test_processed.shape[1])],
        },
        "categorical_info": categorical_info,
    }


def evaluate_config(config: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    """Train and evaluate one xRFM configuration."""
    params = copy.deepcopy(config["params"])
    try:
        model, training_time_seconds = train_single_xrfm_model(
            X_train=prepared["X_train"],
            y_train=prepared["y_train"],
            X_val=prepared["X_val"],
            y_val=prepared["y_val"],
            params=params,
            categorical_info=prepared["categorical_info"],
            random_state=RANDOM_STATE,
        )

        validation_metrics, _ = evaluate_binary_classifier(
            model,
            prepared["X_val"],
            prepared["y_val"],
        )
        test_metrics, test_inference_time = evaluate_binary_classifier(
            model,
            prepared["X_test"],
            prepared["y_test"],
        )

        return {
            "config_id": config["config_id"],
            "notes": config["notes"],
            "status": "success",
            "error_type": None,
            "error_message": None,
            "sample_size": SAMPLE_SIZE,
            "params": params,
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_roc_auc": float(validation_metrics["roc_auc"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_roc_auc": float(test_metrics["roc_auc"]),
            "training_time_seconds": float(training_time_seconds),
            "inference_time_per_sample": float(test_inference_time),
        }
    except Exception as exc:
        return {
            "config_id": config["config_id"],
            "notes": config["notes"],
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "sample_size": SAMPLE_SIZE,
            "params": params,
            "validation_accuracy": None,
            "validation_roc_auc": None,
            "test_accuracy": None,
            "test_roc_auc": None,
            "training_time_seconds": None,
            "inference_time_per_sample": None,
        }


def save_results(results: list[dict[str, Any]], prepared: dict[str, Any]) -> None:
    """Save tuning results to JSON and CSV."""
    ensure_project_dirs()

    json_payload = {
        "dataset_path": str(prepared["csv_path"]),
        "sample_size": SAMPLE_SIZE,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "random_state": RANDOM_STATE,
        "raw_feature_shape": prepared["raw_feature_shape"],
        "preprocessed_feature_shapes": prepared["preprocessed_feature_shapes"],
        "results_sorted": results,
    }
    JSON_OUTPUT_PATH.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    fieldnames = [
        "config_id",
        "notes",
        "status",
        "error_type",
        "error_message",
        "sample_size",
        "validation_accuracy",
        "validation_roc_auc",
        "test_accuracy",
        "test_roc_auc",
        "training_time_seconds",
        "inference_time_per_sample",
    ]
    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    prepared = prepare_data(SAMPLE_SIZE)
    results = []

    print(f"xRFM tuning on sample_size={SAMPLE_SIZE}")
    print(f"Raw feature shape: {tuple(prepared['raw_feature_shape'])}")
    print(
        "Preprocessed feature shapes: "
        f"train={tuple(prepared['preprocessed_feature_shapes']['train'])}, "
        f"val={tuple(prepared['preprocessed_feature_shapes']['val'])}, "
        f"test={tuple(prepared['preprocessed_feature_shapes']['test'])}"
    )

    for config in XRFM_CONFIGS:
        print()
        print(f"Running config {config['config_id']}: {config['notes']}")
        result = evaluate_config(config, prepared)
        results.append(result)
        if result["status"] == "success":
            print(
                "Validation: "
                f"accuracy={result['validation_accuracy']:.4f}, "
                f"roc_auc={result['validation_roc_auc']:.4f}"
            )
            print(
                "Test: "
                f"accuracy={result['test_accuracy']:.4f}, "
                f"roc_auc={result['test_roc_auc']:.4f}"
            )
            print(
                "Timing: "
                f"train={result['training_time_seconds']:.4f}s, "
                f"inference/sample={result['inference_time_per_sample']:.6e}s"
            )
        else:
            print(
                f"Failed: {result['error_type']} | {result['error_message'] or '(no message)'}"
            )

    results_sorted = sorted(
        results,
        key=lambda row: (
            row["status"] == "success",
            row["validation_roc_auc"] if row["validation_roc_auc"] is not None else float("-inf"),
            row["validation_accuracy"] if row["validation_accuracy"] is not None else float("-inf"),
        ),
        reverse=True,
    )
    save_results(results_sorted, prepared)

    best = results_sorted[0]
    print()
    print("Best config by validation ROC-AUC:")
    if best["status"] == "success":
        print(
            f"{best['config_id']} | "
            f"val_acc={best['validation_accuracy']:.4f}, "
            f"val_auc={best['validation_roc_auc']:.4f}, "
            f"test_acc={best['test_accuracy']:.4f}, "
            f"test_auc={best['test_roc_auc']:.4f}"
        )
    else:
        print(f"{best['config_id']} failed: {best['error_type']} | {best['error_message']}")
    print(f"Saved JSON: {JSON_OUTPUT_PATH.resolve()}")
    print(f"Saved CSV: {CSV_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
