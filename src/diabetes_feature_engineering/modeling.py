"""Model pipeline construction, training, and evaluation."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from diabetes_feature_engineering.data import TARGET_COLUMN, validate_dataset
from diabetes_feature_engineering.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    DiabetesFeatureEngineer,
)


def build_model_pipeline(random_state: int = 46) -> Pipeline:
    """Build the complete preprocessing and random-forest classification pipeline."""
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, list(NUMERIC_FEATURES)),
            ("categorical", categorical_preprocessor, list(CATEGORICAL_FEATURES)),
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineering", DiabetesFeatureEngineer()),
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate(
    dataframe: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 17,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit on a stratified training split and return the fitted model and hold-out metrics."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    dataset = validate_dataset(dataframe)
    X = dataset.drop(columns=TARGET_COLUMN)
    y = dataset[TARGET_COLUMN]
    if y.nunique() != 2:
        raise ValueError("Training requires at least one sample from each target class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_model_pipeline().fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    return model, metrics
