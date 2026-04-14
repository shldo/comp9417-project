from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def make_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    The split is done in two stages:
    1. train+val vs test
    2. train vs val from the remaining data
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    stratify_labels = y if stratify else None

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    val_ratio_within_train_val = val_size / (1.0 - test_size)
    stratify_train_val = y_train_val if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_within_train_val,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
