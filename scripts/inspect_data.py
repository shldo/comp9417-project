from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATASET_NAME, DROP_COLUMNS, TARGET_COLUMN, ensure_project_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a tabular CSV dataset for schema, types, and missing values."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(DEFAULT_DATASET_NAME),
        help="Path to the CSV dataset.",
    )
    return parser.parse_args()


def classify_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS], errors="ignore")
    numeric_columns = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = feature_df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return numeric_columns, categorical_columns


def print_column_report(df: pd.DataFrame) -> None:
    print("Columns")
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        unique_count = int(df[column].nunique(dropna=True))
        print(
            f"- {column}: dtype={df[column].dtype}, "
            f"missing={missing_count}, unique={unique_count}"
        )


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)

    numeric_columns, categorical_columns = classify_feature_columns(df)

    print(f"Dataset path: {args.csv.resolve()}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target column present: {TARGET_COLUMN in df.columns}")
    print("Drop columns present:")
    for column in DROP_COLUMNS:
        print(f"- {column}: {column in df.columns}")

    print()
    print_column_report(df)

    print()
    print(f"Detected numeric feature columns ({len(numeric_columns)}):")
    for column in numeric_columns:
        print(f"- {column}")

    print()
    print(f"Detected categorical feature columns ({len(categorical_columns)}):")
    for column in categorical_columns:
        print(f"- {column}")

    if TARGET_COLUMN in df.columns:
        print()
        print("Target summary:")
        print(df[TARGET_COLUMN].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
