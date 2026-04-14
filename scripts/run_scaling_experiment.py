from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedShuffleSplit

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
from src.train_random_forest import (
    DEFAULT_PARAM_GRID as DEFAULT_RF_PARAM_GRID,
    evaluate_binary_classifier as evaluate_random_forest,
    train_single_random_forest_model,
)
from src.train_xgboost import (
    DEFAULT_PARAM_GRID as DEFAULT_XGB_PARAM_GRID,
    evaluate_binary_classifier as evaluate_xgboost,
    train_single_xgboost_model,
)
from src.train_xrfm import (
    DEFAULT_USE_ONE_HOT_LABELS,
    DEFAULT_XRFM_PARAMS,
    build_categorical_info,
    evaluate_binary_classifier as evaluate_xrfm,
    train_single_xrfm_model,
)


SAMPLE_SIZES = [2000, 5000, 10000, 20000]
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42

SCALING_RUNS_DIR = METRICS_DIR / "scaling_runs"
JSON_OUTPUT_PATH = METRICS_DIR / "scaling_experiment_results.json"
CSV_OUTPUT_PATH = METRICS_DIR / "scaling_experiment_results.csv"

# Freeze the formal comparison configuration for each model.
XGBOOST_FORMAL_PARAMS = copy.deepcopy(DEFAULT_XGB_PARAM_GRID[0])
RANDOM_FOREST_FORMAL_PARAMS = copy.deepcopy(DEFAULT_RF_PARAM_GRID[0])
XRFM_FORMAL_PARAMS = copy.deepcopy(DEFAULT_XRFM_PARAMS)


def prepare_full_split(csv_path: Path) -> dict[str, Any]:
    """
    Load the full dataset and create one fixed train/validation/test split.

    The split is created once on the full 100000-row dataset and reused across
    every model and every sample size.
    """
    df = load_raw_data(csv_path)
    X_full, y_full, feature_columns = get_feature_target_data(df)

    X_train_full, X_val_fixed, X_test_fixed, y_train_full, y_val_fixed, y_test_fixed = (
        make_train_val_test_split(
            X=X_full,
            y=y_full,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=True,
        )
    )

    return {
        "dataset_path": str(csv_path.resolve()),
        "feature_column_count": len(feature_columns),
        "full_feature_shape": [int(X_full.shape[0]), int(X_full.shape[1])],
        "fixed_split_shapes": {
            "train": [int(X_train_full.shape[0]), int(X_train_full.shape[1])],
            "val": [int(X_val_fixed.shape[0]), int(X_val_fixed.shape[1])],
            "test": [int(X_test_fixed.shape[0]), int(X_test_fixed.shape[1])],
        },
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_val_fixed": X_val_fixed,
        "y_val_fixed": y_val_fixed,
        "X_test_fixed": X_test_fixed,
        "y_test_fixed": y_test_fixed,
    }


def subsample_fixed_train_split(
    X_train_full,
    y_train_full,
    sample_size: int,
    random_state: int,
):
    """
    Draw a stratified subsample from the fixed full training split only.

    Validation and test sets are never touched here, which keeps them unchanged
    for every sample size and every model.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")
    if sample_size > len(X_train_full):
        raise ValueError("sample_size cannot be larger than the fixed training split size.")
    if sample_size == len(X_train_full):
        return X_train_full.copy(), y_train_full.copy()

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        random_state=random_state,
    )
    train_indices, _ = next(splitter.split(X_train_full, y_train_full))

    X_train_subsample = X_train_full.iloc[train_indices].copy()
    y_train_subsample = y_train_full.iloc[train_indices].copy()
    return X_train_subsample, y_train_subsample


def prepare_processed_data_for_sample_size(
    prepared_full_split: dict[str, Any],
    sample_size: int,
) -> dict[str, Any]:
    """
    Prepare training/validation/test matrices for one sample size.

    The fixed validation and test splits are reused exactly as they are, while
    the preprocessor is fit only on the subsampled training set.
    """
    X_train_subsample, y_train_subsample = subsample_fixed_train_split(
        X_train_full=prepared_full_split["X_train_full"],
        y_train_full=prepared_full_split["y_train_full"],
        sample_size=sample_size,
        random_state=RANDOM_STATE,
    )

    X_val_fixed = prepared_full_split["X_val_fixed"]
    y_val_fixed = prepared_full_split["y_val_fixed"]
    X_test_fixed = prepared_full_split["X_test_fixed"]
    y_test_fixed = prepared_full_split["y_test_fixed"]

    numeric_cols, categorical_cols = detect_feature_types(X_train_subsample)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train_processed = fit_transform_features(preprocessor, X_train_subsample)
    X_val_processed = transform_features(preprocessor, X_val_fixed)
    X_test_processed = transform_features(preprocessor, X_test_fixed)

    return {
        "sample_size": sample_size,
        "X_train_subsample": X_train_subsample,
        "y_train_subsample": y_train_subsample,
        "X_val_fixed": X_val_fixed,
        "y_val_fixed": y_val_fixed,
        "X_test_fixed": X_test_fixed,
        "y_test_fixed": y_test_fixed,
        "preprocessor": preprocessor,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "X_train_processed": X_train_processed,
        "X_val_processed": X_val_processed,
        "X_test_processed": X_test_processed,
        "raw_subsample_shape": [int(X_train_subsample.shape[0]), int(X_train_subsample.shape[1])],
        "preprocessed_feature_shapes": {
            "train": [int(X_train_processed.shape[0]), int(X_train_processed.shape[1])],
            "val": [int(X_val_processed.shape[0]), int(X_val_processed.shape[1])],
            "test": [int(X_test_processed.shape[0]), int(X_test_processed.shape[1])],
        },
    }


def save_single_run_result(
    model_name: str,
    sample_bundle: dict[str, Any],
    params: dict[str, Any],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    training_time_seconds: float,
    inference_time_per_sample: float,
    output_path: Path,
) -> Path:
    """Save one model/sample-size run result to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model_name,
        "sample_size": sample_bundle["sample_size"],
        "raw_subsample_shape": sample_bundle["raw_subsample_shape"],
        "fixed_val_shape": [
            int(sample_bundle["X_val_fixed"].shape[0]),
            int(sample_bundle["X_val_fixed"].shape[1]),
        ],
        "fixed_test_shape": [
            int(sample_bundle["X_test_fixed"].shape[0]),
            int(sample_bundle["X_test_fixed"].shape[1]),
        ],
        "preprocessed_feature_shapes": sample_bundle["preprocessed_feature_shapes"],
        "params": params,
        "validation_metrics": validation_metrics,
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "roc_auc": float(test_metrics["roc_auc"]),
            "training_time_seconds": float(training_time_seconds),
            "inference_time_per_sample": float(inference_time_per_sample),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def run_xgboost_scaling_experiment(sample_bundle: dict[str, Any]) -> dict[str, Any]:
    """Run XGBoost on one fixed-train subsample."""
    model, training_time_seconds = train_single_xgboost_model(
        X_train=sample_bundle["X_train_processed"],
        y_train=sample_bundle["y_train_subsample"],
        params=copy.deepcopy(XGBOOST_FORMAL_PARAMS),
        random_state=RANDOM_STATE,
    )
    validation_metrics, _ = evaluate_xgboost(
        model,
        sample_bundle["X_val_processed"],
        sample_bundle["y_val_fixed"],
    )
    test_metrics, inference_time_per_sample = evaluate_xgboost(
        model,
        sample_bundle["X_test_processed"],
        sample_bundle["y_test_fixed"],
    )

    output_path = save_single_run_result(
        model_name="XGBoost",
        sample_bundle=sample_bundle,
        params=copy.deepcopy(XGBOOST_FORMAL_PARAMS),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        training_time_seconds=training_time_seconds,
        inference_time_per_sample=inference_time_per_sample,
        output_path=SCALING_RUNS_DIR / f"xgboost_{sample_bundle['sample_size']}.json",
    )

    return {
        "model": "XGBoost",
        "sample_size": sample_bundle["sample_size"],
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "training_time_seconds": float(training_time_seconds),
        "inference_time_per_sample": float(inference_time_per_sample),
        "metrics_output_path": str(output_path.resolve()),
    }


def run_xrfm_scaling_experiment(sample_bundle: dict[str, Any]) -> dict[str, Any]:
    """Run xRFM on one fixed-train subsample."""
    categorical_info = build_categorical_info(
        preprocessor=sample_bundle["preprocessor"],
        numeric_cols=sample_bundle["numeric_cols"],
        categorical_cols=sample_bundle["categorical_cols"],
    )

    model, training_time_seconds = train_single_xrfm_model(
        X_train=sample_bundle["X_train_processed"],
        y_train=sample_bundle["y_train_subsample"],
        X_val=sample_bundle["X_val_processed"],
        y_val=sample_bundle["y_val_fixed"],
        params=copy.deepcopy(XRFM_FORMAL_PARAMS),
        categorical_info=categorical_info,
        random_state=RANDOM_STATE,
        use_one_hot_labels=DEFAULT_USE_ONE_HOT_LABELS,
    )
    validation_metrics, _ = evaluate_xrfm(
        model,
        sample_bundle["X_val_processed"],
        sample_bundle["y_val_fixed"],
    )
    test_metrics, inference_time_per_sample = evaluate_xrfm(
        model,
        sample_bundle["X_test_processed"],
        sample_bundle["y_test_fixed"],
    )

    output_path = save_single_run_result(
        model_name="xRFM",
        sample_bundle=sample_bundle,
        params=copy.deepcopy(XRFM_FORMAL_PARAMS),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        training_time_seconds=training_time_seconds,
        inference_time_per_sample=inference_time_per_sample,
        output_path=SCALING_RUNS_DIR / f"xrfm_{sample_bundle['sample_size']}.json",
    )

    return {
        "model": "xRFM",
        "sample_size": sample_bundle["sample_size"],
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "training_time_seconds": float(training_time_seconds),
        "inference_time_per_sample": float(inference_time_per_sample),
        "metrics_output_path": str(output_path.resolve()),
    }


def run_random_forest_scaling_experiment(sample_bundle: dict[str, Any]) -> dict[str, Any]:
    """Run Random Forest on one fixed-train subsample."""
    model, training_time_seconds = train_single_random_forest_model(
        X_train=sample_bundle["X_train_processed"],
        y_train=sample_bundle["y_train_subsample"],
        params=copy.deepcopy(RANDOM_FOREST_FORMAL_PARAMS),
        random_state=RANDOM_STATE,
    )
    validation_metrics, _ = evaluate_random_forest(
        model,
        sample_bundle["X_val_processed"],
        sample_bundle["y_val_fixed"],
    )
    test_metrics, inference_time_per_sample = evaluate_random_forest(
        model,
        sample_bundle["X_test_processed"],
        sample_bundle["y_test_fixed"],
    )

    output_path = save_single_run_result(
        model_name="Random Forest",
        sample_bundle=sample_bundle,
        params=copy.deepcopy(RANDOM_FOREST_FORMAL_PARAMS),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        training_time_seconds=training_time_seconds,
        inference_time_per_sample=inference_time_per_sample,
        output_path=SCALING_RUNS_DIR / f"random_forest_{sample_bundle['sample_size']}.json",
    )

    return {
        "model": "Random Forest",
        "sample_size": sample_bundle["sample_size"],
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "training_time_seconds": float(training_time_seconds),
        "inference_time_per_sample": float(inference_time_per_sample),
        "metrics_output_path": str(output_path.resolve()),
    }


def save_scaling_results(results: list[dict[str, Any]], prepared_full_split: dict[str, Any]) -> None:
    """Save the strict scaling experiment results to JSON and CSV."""
    ensure_project_dirs()
    SCALING_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "dataset_path": prepared_full_split["dataset_path"],
        "task": "binary classification",
        "target_column": "felt_rested",
        "sample_sizes": SAMPLE_SIZES,
        "random_state": RANDOM_STATE,
        "split_protocol": (
            "One fixed full train/validation/test split is created on the full dataset first. "
            "Only the fixed training split is stratified-subsampled for each sample size. "
            "Validation and test splits remain unchanged across all sample sizes and models."
        ),
        "full_feature_shape": prepared_full_split["full_feature_shape"],
        "fixed_split_shapes": prepared_full_split["fixed_split_shapes"],
        "xgboost_formal_params": XGBOOST_FORMAL_PARAMS,
        "xrfm_formal_params": XRFM_FORMAL_PARAMS,
        "xrfm_use_one_hot_labels": DEFAULT_USE_ONE_HOT_LABELS,
        "random_forest_formal_params": RANDOM_FOREST_FORMAL_PARAMS,
        "results": results,
    }
    JSON_OUTPUT_PATH.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    fieldnames = [
        "model",
        "sample_size",
        "validation_accuracy",
        "validation_roc_auc",
        "test_accuracy",
        "test_roc_auc",
        "training_time_seconds",
        "inference_time_per_sample",
        "metrics_output_path",
    ]
    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main() -> None:
    ensure_project_dirs()
    SCALING_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RAW_DATA_DIR / DEFAULT_DATASET_NAME
    prepared_full_split = prepare_full_split(csv_path)

    print("Scaling experiment")
    print(f"Dataset: {csv_path.resolve()}")
    print(f"Full feature shape: {tuple(prepared_full_split['full_feature_shape'])}")
    print(
        "Fixed split shapes: "
        f"train={tuple(prepared_full_split['fixed_split_shapes']['train'])}, "
        f"val={tuple(prepared_full_split['fixed_split_shapes']['val'])}, "
        f"test={tuple(prepared_full_split['fixed_split_shapes']['test'])}"
    )
    print(f"Sample sizes: {SAMPLE_SIZES}")

    results: list[dict[str, Any]] = []

    for sample_size in SAMPLE_SIZES:
        sample_bundle = prepare_processed_data_for_sample_size(
            prepared_full_split=prepared_full_split,
            sample_size=sample_size,
        )

        print()
        print(f"Running XGBoost with train subsample size={sample_size}")
        xgboost_result = run_xgboost_scaling_experiment(sample_bundle)
        results.append(xgboost_result)
        print(
            f"Val AUC={xgboost_result['validation_roc_auc']:.4f}, "
            f"Test AUC={xgboost_result['test_roc_auc']:.4f}, "
            f"Train={xgboost_result['training_time_seconds']:.4f}s"
        )

        print(f"Running xRFM with train subsample size={sample_size}")
        xrfm_result = run_xrfm_scaling_experiment(sample_bundle)
        results.append(xrfm_result)
        print(
            f"Val AUC={xrfm_result['validation_roc_auc']:.4f}, "
            f"Test AUC={xrfm_result['test_roc_auc']:.4f}, "
            f"Train={xrfm_result['training_time_seconds']:.4f}s"
        )

        print(f"Running Random Forest with train subsample size={sample_size}")
        random_forest_result = run_random_forest_scaling_experiment(sample_bundle)
        results.append(random_forest_result)
        print(
            f"Val AUC={random_forest_result['validation_roc_auc']:.4f}, "
            f"Test AUC={random_forest_result['test_roc_auc']:.4f}, "
            f"Train={random_forest_result['training_time_seconds']:.4f}s"
        )

    save_scaling_results(results, prepared_full_split)

    print()
    print(f"Saved JSON: {JSON_OUTPUT_PATH.resolve()}")
    print(f"Saved CSV: {CSV_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
