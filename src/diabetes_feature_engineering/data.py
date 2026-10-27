"""Dataset loading and input validation."""

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)
TARGET_COLUMN = "Outcome"
DATASET_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)


def validate_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a dataset with the columns required by this project."""
    if dataframe.empty:
        raise ValueError("The dataset is empty.")
    if dataframe.columns.has_duplicates:
        raise ValueError("The dataset contains duplicate column names.")

    missing_columns = [column for column in DATASET_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}.")

    selected = dataframe.loc[:, DATASET_COLUMNS].copy()
    non_numeric = [
        column
        for column in DATASET_COLUMNS
        if not pd.api.types.is_numeric_dtype(selected[column])
    ]
    if non_numeric:
        columns = ", ".join(non_numeric)
        raise ValueError(f"Expected numeric values in columns: {columns}.")
    if selected[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains missing values.")

    target_values = set(selected[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Target column '{TARGET_COLUMN}' must contain only 0 and 1.")

    return selected


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and validate its schema and target values."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Add the CSV or pass its path with --data."
        )

    return validate_dataset(pd.read_csv(dataset_path))

