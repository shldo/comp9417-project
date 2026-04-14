from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DROP_COLUMNS, METRICS_DIR, TARGET_COLUMN, ensure_project_dirs
from src.data_loading import get_feature_target_data, load_raw_data
from src.metrics import compute_classification_metrics
from src.preprocessing import (
    build_preprocessor,
    detect_feature_types,
    fit_transform_features,
    transform_features,
)
from src.splitting import make_train_val_test_split
from src.timing import measure_inference_time_per_sample, measure_training_time

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


DEFAULT_PARAM_GRID = [
    {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
    {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
    {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
]


DEFAULT_SAMPLE_SIZE = 2000


def _get_class_distribution(y: pd.Series) -> dict[str, int]:
    """Return class counts with JSON-friendly string keys."""
    counts = y.value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _apply_sample_size(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Optionally keep only the first N rows for a lightweight training run."""
    if sample_size is None:
        return X, y
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer or None.")
    return X.iloc[:sample_size].copy(), y.iloc[:sample_size].copy()


def _raise_if_xgboost_missing() -> None:
    if XGBClassifier is None:
        raise ModuleNotFoundError(
            "xgboost is not installed in the current environment. "
            "Please install it before running XGBoost training."
        ) from XGBOOST_IMPORT_ERROR


def create_xgb_classifier(params: dict[str, Any], random_state: int) -> Any:
    """Create a simple XGBClassifier for binary classification."""
    _raise_if_xgboost_missing()
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
        **params,
    )


def train_single_xgboost_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    params: dict[str, Any],
    random_state: int,
) -> tuple[Any, float]:
    """Train one XGBoost model and return the fitted model and elapsed time."""
    model = create_xgb_classifier(params=params, random_state=random_state)
    fitted_model, training_time_seconds = measure_training_time(
        model.fit,
        X_train,
        y_train,
    )
    return fitted_model, training_time_seconds


def evaluate_binary_classifier(model: Any, X: np.ndarray, y: pd.Series) -> tuple[dict[str, float], float]:
    """Evaluate a binary classifier and measure average inference time per sample."""
    y_pred = model.predict(X)
    y_prob, inference_time_per_sample = measure_inference_time_per_sample(
        model.predict_proba,
        X,
    )
    metrics = compute_classification_metrics(y, y_pred, y_prob)
    metrics["inference_time_per_sample"] = float(inference_time_per_sample)
    return metrics, inference_time_per_sample


def select_best_xgboost_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    param_grid: list[dict[str, Any]],
    random_state: int,
) -> dict[str, Any]:
    """Train candidate models and choose the best one using validation ROC-AUC."""
    best_result: dict[str, Any] | None = None

    for params in param_grid:
        model, training_time_seconds = train_single_xgboost_model(
            X_train=X_train,
            y_train=y_train,
            params=params,
            random_state=random_state,
        )
        validation_metrics, _ = evaluate_binary_classifier(model, X_val, y_val)

        current_result = {
            "model": model,
            "params": params,
            "training_time_seconds": float(training_time_seconds),
            "validation_metrics": validation_metrics,
        }

        if best_result is None:
            best_result = current_result
            continue

        current_key = (
            current_result["validation_metrics"]["roc_auc"],
            current_result["validation_metrics"]["accuracy"],
        )
        best_key = (
            best_result["validation_metrics"]["roc_auc"],
            best_result["validation_metrics"]["accuracy"],
        )
        if current_key > best_key:
            best_result = current_result

    if best_result is None:
        raise ValueError("param_grid must contain at least one parameter set.")

    return best_result


def save_xgboost_metrics(metrics_payload: dict[str, Any], output_path: str | Path) -> Path:
    """Save XGBoost metrics to a JSON file."""
    ensure_project_dirs()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return path


def run_xgboost_experiment(
    csv_path: str | Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    target_col: str = TARGET_COLUMN,
    drop_cols: list[str] | tuple[str, ...] = tuple(DROP_COLUMNS),
    param_grid: list[dict[str, Any]] | None = None,
    sample_size: int | None = DEFAULT_SAMPLE_SIZE,
    metrics_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run a complete XGBoost experiment with train/validation/test split.

    The best model is chosen using validation metrics, and the test set is used
    only for the final evaluation.
    """
    _raise_if_xgboost_missing()
    ensure_project_dirs()

    df = load_raw_data(csv_path)
    X, y, feature_columns = get_feature_target_data(
        df=df,
        target_col=target_col,
        drop_cols=drop_cols,
    )
    X, y = _apply_sample_size(X, y, sample_size)

    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=True,
    )

    numeric_cols, categorical_cols = detect_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train_processed = fit_transform_features(preprocessor, X_train)
    X_val_processed = transform_features(preprocessor, X_val)
    X_test_processed = transform_features(preprocessor, X_test)

    search_space = param_grid or DEFAULT_PARAM_GRID
    best_result = select_best_xgboost_model(
        X_train=X_train_processed,
        y_train=y_train,
        X_val=X_val_processed,
        y_val=y_val,
        param_grid=search_space,
        random_state=random_state,
    )

    test_metrics, test_inference_time = evaluate_binary_classifier(
        best_result["model"],
        X_test_processed,
        y_test,
    )

    final_payload = {
        "dataset_path": str(Path(csv_path).resolve()),
        "sample_size": sample_size,
        "target_column": target_col,
        "dropped_columns": list(drop_cols),
        "class_distribution": {
            "full": _get_class_distribution(y),
            "train": _get_class_distribution(y_train),
            "val": _get_class_distribution(y_val),
            "test": _get_class_distribution(y_test),
        },
        "feature_column_count": len(feature_columns),
        "raw_feature_shape": [int(X.shape[0]), int(X.shape[1])],
        "split_shapes": {
            "train": [int(X_train.shape[0]), int(X_train.shape[1])],
            "val": [int(X_val.shape[0]), int(X_val.shape[1])],
            "test": [int(X_test.shape[0]), int(X_test.shape[1])],
        },
        "preprocessed_feature_shapes": {
            "train": [int(X_train_processed.shape[0]), int(X_train_processed.shape[1])],
            "val": [int(X_val_processed.shape[0]), int(X_val_processed.shape[1])],
            "test": [int(X_test_processed.shape[0]), int(X_test_processed.shape[1])],
        },
        "feature_types": {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
        },
        "best_params": best_result["params"],
        "validation_metrics": best_result["validation_metrics"],
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "roc_auc": float(test_metrics["roc_auc"]),
            # This is the fit time for the final selected best model only.
            # It does not include the total elapsed time of the whole parameter search.
            "training_time_seconds": float(best_result["training_time_seconds"]),
            "inference_time_per_sample": float(test_inference_time),
        },
    }

    output_path = metrics_output_path or (METRICS_DIR / "xgboost_metrics.json")
    saved_path = save_xgboost_metrics(final_payload, output_path)
    final_payload["metrics_output_path"] = str(saved_path.resolve())

    return final_payload
