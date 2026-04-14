from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATASET_NAME, RAW_DATA_DIR
from src.data_loading import get_feature_target_data, load_raw_data
from src.preprocessing import (
    build_preprocessor,
    detect_feature_types,
    fit_transform_features,
    transform_features,
)
from src.splitting import make_train_val_test_split


def build_categorical_info(preprocessor, numeric_cols: list[str]) -> dict[str, object]:
    """
    Build minimal categorical metadata for xRFM from the fitted preprocessor.

    This matches the current preprocessing layout:
    numeric columns first, then one-hot encoded categorical columns.
    """
    categorical_transformer = preprocessor.named_transformers_["categorical"]
    n_numeric = len(numeric_cols)
    categorical_indices = []
    categorical_vectors = []
    start_index = n_numeric

    for categories in categorical_transformer.categories_:
        category_count = len(categories)
        categorical_indices.append(
            torch.arange(start_index, start_index + category_count, dtype=torch.long)
        )
        categorical_vectors.append(torch.eye(category_count, dtype=torch.float32))
        start_index += category_count

    return {
        "numerical_indices": torch.arange(0, n_numeric, dtype=torch.long),
        "categorical_indices": categorical_indices,
        "categorical_vectors": categorical_vectors,
    }


def main() -> None:
    try:
        import xrfm
        from xrfm import xRFM
    except Exception as exc:
        print("xrfm import failed")
        print(f"error_type: {type(exc).__name__}")
        print(f"error_message: {exc}")
        return

    csv_path = RAW_DATA_DIR / DEFAULT_DATASET_NAME
    sample_size = 2000

    print("xRFM smoke test")
    print(f"package_version: {getattr(xrfm, '__version__', 'unknown')}")
    print(f"class_name: {xRFM.__name__}")
    print(f"init_signature: {inspect.signature(xRFM.__init__)}")
    print(f"fit_signature: {inspect.signature(xRFM.fit)}")
    print(f"predict_signature: {inspect.signature(xRFM.predict)}")
    print(f"supports_predict_proba: {hasattr(xRFM, 'predict_proba')}")
    if hasattr(xRFM, "predict_proba"):
        print(f"predict_proba_signature: {inspect.signature(xRFM.predict_proba)}")

    df = load_raw_data(csv_path).head(sample_size).copy()
    X, y, _ = get_feature_target_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(
        X,
        y,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        stratify=True,
    )

    numeric_cols, categorical_cols = detect_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_processed = fit_transform_features(preprocessor, X_train)
    X_val_processed = transform_features(preprocessor, X_val)
    X_test_processed = transform_features(preprocessor, X_test)

    init_params = {
        "device": torch.device("cpu"),
        "classification_mode": "zero_one",
        "tuning_metric": "auc",
        "max_leaf_size": 10000,
        "n_trees": 1,
        "n_tree_iters": 0,
        "random_state": 42,
        "categorical_info": build_categorical_info(preprocessor, numeric_cols),
    }

    print(f"subset_size: {len(df)}")
    print(f"raw_feature_shape: {X.shape}")
    print(f"preprocessed_train_shape: {X_train_processed.shape}")
    print("init_params_used:")
    for key, value in init_params.items():
        if key == "categorical_info":
            print(
                "  categorical_info: "
                f"numerical={len(value['numerical_indices'])}, "
                f"categorical_groups={len(value['categorical_indices'])}"
            )
        else:
            print(f"  {key}: {value}")

    model = xRFM(**init_params)

    train_start = time.perf_counter()
    success = False
    fit_error = None
    try:
        model.fit(X_train_processed, y_train.to_numpy(), X_val_processed, y_val.to_numpy())
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed) if hasattr(model, "predict_proba") else None
        success = True
    except Exception as exc:
        fit_error = exc
    training_time_seconds = time.perf_counter() - train_start

    print(f"small_sample_run_success: {success}")
    print(f"training_time_seconds_approx: {training_time_seconds:.4f}")

    if success:
        print(f"predict_output_shape: {getattr(y_pred, 'shape', None)}")
        print(f"predict_proba_output_shape: {getattr(y_prob, 'shape', None)}")
        print(f"test_subset_size: {len(y_test)}")
    else:
        print(f"error_type: {type(fit_error).__name__}")
        print(f"error_message: {fit_error}")


if __name__ == "__main__":
    main()
