from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DROP_COLUMNS, TARGET_COLUMN


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load a raw CSV file into a pandas DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def get_feature_target_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    drop_cols: list[str] | tuple[str, ...] = tuple(DROP_COLUMNS),
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Split a DataFrame into features and target.

    Returns:
    - X: feature DataFrame
    - y: target Series
    - feature column names: list[str]
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    columns_to_drop = [target_col, *drop_cols]
    X = df.drop(columns=columns_to_drop, errors="ignore").copy()
    y = df[target_col].copy()
    feature_columns = X.columns.tolist()

    return X, y, feature_columns
