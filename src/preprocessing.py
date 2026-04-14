from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical columns from a feature DataFrame."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for tabular preprocessing.

    Numeric columns are standard-scaled.
    Categorical columns are one-hot encoded.
    The output is forced to be a dense numeric matrix.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def fit_transform_features(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> np.ndarray:
    """Fit the preprocessor on training data and transform it into a numeric matrix."""
    X_train_processed = preprocessor.fit_transform(X_train)
    return np.asarray(X_train_processed, dtype=np.float32)


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Transform feature data into a numeric matrix using a fitted preprocessor."""
    X_processed = preprocessor.transform(X)
    return np.asarray(X_processed, dtype=np.float32)
