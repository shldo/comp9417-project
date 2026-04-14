from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

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
    from xrfm import xRFM
except ModuleNotFoundError as exc:
    xRFM = None
    XRFM_IMPORT_ERROR = exc
else:
    XRFM_IMPORT_ERROR = None


DEFAULT_XRFM_PARAMS = {
    "tuning_metric": "auc",
    "max_leaf_size": 3000,
    "n_trees": 2,
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
}

DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_USE_ONE_HOT_LABELS = True


def _raise_if_xrfm_missing() -> None:
    if xRFM is None:
        raise ModuleNotFoundError(
            "xrfm is not installed in the current environment. "
            "Please install it before running xRFM training."
        ) from XRFM_IMPORT_ERROR


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


def build_categorical_info(
    preprocessor,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, Any]:
    """
    Build xRFM categorical metadata from a fitted sklearn ColumnTransformer.

    This assumes numeric features appear first and one-hot categorical features
    appear after them, matching the current preprocessing pipeline.
    """
    categorical_transformer = preprocessor.named_transformers_["categorical"]
    n_numeric = len(numeric_cols)
    categorical_indices = []
    categorical_vectors = []
    start_index = n_numeric

    if categorical_cols:
        for categories in categorical_transformer.categories_:
            category_count = len(categories)
            indices = torch.arange(start_index, start_index + category_count, dtype=torch.long)
            categorical_indices.append(indices)
            categorical_vectors.append(torch.eye(category_count, dtype=torch.float32))
            start_index += category_count

    return {
        "numerical_indices": torch.arange(0, n_numeric, dtype=torch.long),
        "categorical_indices": categorical_indices,
        "categorical_vectors": categorical_vectors,
    }


def prepare_xrfm_targets(
    y_train: pd.Series,
    y_val: pd.Series,
    use_one_hot_labels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare xRFM target arrays.

    For xrfm 0.4.3, configurations with n_tree_iters > 0 require classification
    targets in a 2D one-hot form in the internal score_tree path.
    """
    if not use_one_hot_labels:
        return y_train.to_numpy(), y_val.to_numpy()

    y_train_encoded = np.eye(2, dtype=np.float32)[y_train.to_numpy()]
    y_val_encoded = np.eye(2, dtype=np.float32)[y_val.to_numpy()]
    return y_train_encoded, y_val_encoded


def create_xrfm_classifier(
    params: dict[str, Any],
    categorical_info: dict[str, Any],
    random_state: int,
) -> Any:
    """Create an xRFM model for binary classification."""
    _raise_if_xrfm_missing()
    return xRFM(
        device=torch.device("cpu"),
        classification_mode="zero_one",
        categorical_info=categorical_info,
        random_state=random_state,
        **params,
    )


def train_single_xrfm_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    params: dict[str, Any],
    categorical_info: dict[str, Any],
    random_state: int,
    use_one_hot_labels: bool = False,
) -> tuple[Any, float]:
    """Train one xRFM model and return the fitted model and elapsed time."""
    model = create_xrfm_classifier(
        params=params,
        categorical_info=categorical_info,
        random_state=random_state,
    )
    y_train_input, y_val_input = prepare_xrfm_targets(
        y_train=y_train,
        y_val=y_val,
        use_one_hot_labels=use_one_hot_labels,
    )
    fitted_model, training_time_seconds = measure_training_time(
        model.fit,
        X_train,
        y_train_input,
        X_val,
        y_val_input,
    )
    return fitted_model, training_time_seconds


def evaluate_binary_classifier(
    model: Any,
    X: np.ndarray,
    y: pd.Series,
) -> tuple[dict[str, float], float]:
    """Evaluate a fitted xRFM binary classifier."""
    y_pred = model.predict(X)
    y_prob, inference_time_per_sample = measure_inference_time_per_sample(
        model.predict_proba,
        X,
    )
    metrics = compute_classification_metrics(y, y_pred, y_prob)
    metrics["inference_time_per_sample"] = float(inference_time_per_sample)
    return metrics, inference_time_per_sample


def save_xrfm_metrics(metrics_payload: dict[str, Any], output_path: str | Path) -> Path:
    """Save xRFM metrics to a JSON file."""
    ensure_project_dirs()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return path


def run_xrfm_experiment(
    csv_path: str | Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    target_col: str = TARGET_COLUMN,
    drop_cols: list[str] | tuple[str, ...] = tuple(DROP_COLUMNS),
    xrfm_params: dict[str, Any] | None = None,
    sample_size: int | None = DEFAULT_SAMPLE_SIZE,
    use_one_hot_labels: bool = DEFAULT_USE_ONE_HOT_LABELS,
    metrics_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run a lightweight single-model xRFM experiment with train/validation/test split.

    This version intentionally trains only one xRFM configuration.
    """
    _raise_if_xrfm_missing()
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

    categorical_info = build_categorical_info(preprocessor, numeric_cols, categorical_cols)
    params = xrfm_params or DEFAULT_XRFM_PARAMS

    model, training_time_seconds = train_single_xrfm_model(
        X_train=X_train_processed,
        y_train=y_train,
        X_val=X_val_processed,
        y_val=y_val,
        params=params,
        categorical_info=categorical_info,
        random_state=random_state,
        use_one_hot_labels=use_one_hot_labels,
    )

    validation_metrics, _ = evaluate_binary_classifier(model, X_val_processed, y_val)
    test_metrics, test_inference_time = evaluate_binary_classifier(
        model,
        X_test_processed,
        y_test,
    )

    final_payload = {
        "dataset_path": str(Path(csv_path).resolve()),
        "sample_size": sample_size,
        "use_one_hot_labels": use_one_hot_labels,
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
        "xrfm_params": params,
        "validation_metrics": validation_metrics,
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "roc_auc": float(test_metrics["roc_auc"]),
            "training_time_seconds": float(training_time_seconds),
            "inference_time_per_sample": float(test_inference_time),
        },
    }

    output_path = metrics_output_path or (METRICS_DIR / "xrfm_metrics.json")
    saved_path = save_xrfm_metrics(final_payload, output_path)
    final_payload["metrics_output_path"] = str(saved_path.resolve())

    return final_payload
