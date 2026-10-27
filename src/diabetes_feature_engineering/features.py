"""Leakage-safe, reusable feature engineering."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from diabetes_feature_engineering.data import FEATURE_COLUMNS

NUMERIC_FEATURES = (
    *FEATURE_COLUMNS,
    "GlucoseInsulin",
    "GlucosePregnancies",
)
CATEGORICAL_FEATURES = (
    "AgeCategory",
    "BMICategory",
    "GlucoseCategory",
    "AgeBMICategory",
    "AgeGlucoseCategory",
    "InsulinCategory",
)


class DiabetesFeatureEngineer(TransformerMixin, BaseEstimator):
    """Convert implausible zeros and create domain-inspired model features."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DiabetesFeatureEngineer":
        self._validate_input(X)
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        self._validate_input(X)

        features = X.loc[:, FEATURE_COLUMNS].copy()
        for column in ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"):
            features[column] = features[column].replace(0, np.nan)

        age_category = pd.cut(
            features["Age"],
            bins=[-np.inf, 21, 50, np.inf],
            labels=["young", "adult", "senior"],
            right=False,
        )
        bmi_category = pd.cut(
            features["BMI"],
            bins=[-np.inf, 18.5, 25, 30, np.inf],
            labels=["underweight", "healthy", "overweight", "obese"],
            right=False,
        )
        glucose_category = pd.cut(
            features["Glucose"],
            bins=[-np.inf, 70, 100, 126, np.inf],
            labels=["low", "normal", "prediabetes", "diabetes"],
            right=False,
        )

        features["GlucoseInsulin"] = features["Glucose"] * features["Insulin"]
        features["GlucosePregnancies"] = (
            features["Glucose"] * features["Pregnancies"]
        )
        features["AgeCategory"] = age_category.astype("object")
        features["BMICategory"] = bmi_category.astype("object")
        features["GlucoseCategory"] = glucose_category.astype("object")
        features["AgeBMICategory"] = self._combine_categories(
            age_category, bmi_category
        )
        features["AgeGlucoseCategory"] = self._combine_categories(
            age_category, glucose_category
        )
        features["InsulinCategory"] = np.where(
            features["Insulin"].between(16, 166),
            "normal",
            "abnormal",
        )
        features.loc[features["Insulin"].isna(), "InsulinCategory"] = np.nan

        return features.loc[:, (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return np.asarray((*NUMERIC_FEATURES, *CATEGORICAL_FEATURES), dtype=object)

    @staticmethod
    def _combine_categories(
        first: pd.Series, second: pd.Series
    ) -> pd.Series:
        combined = first.astype("object") + "_" + second.astype("object")
        return combined.where(first.notna() & second.notna(), np.nan)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Feature engineering expects a pandas DataFrame.")
        if X.columns.has_duplicates:
            raise ValueError("Input data contains duplicate column names.")

        missing_columns = [column for column in FEATURE_COLUMNS if column not in X.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Input data is missing required feature columns: {missing}.")

        non_numeric = [
            column
            for column in FEATURE_COLUMNS
            if not pd.api.types.is_numeric_dtype(X[column])
        ]
        if non_numeric:
            columns = ", ".join(non_numeric)
            raise ValueError(f"Expected numeric values in feature columns: {columns}.")

