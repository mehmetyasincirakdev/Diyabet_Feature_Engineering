import numpy as np
import pandas as pd
import pytest

from diabetes_feature_engineering.data import DATASET_COLUMNS
from diabetes_feature_engineering.modeling import train_and_evaluate


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    size = 60
    index = np.arange(size)
    return pd.DataFrame(
        {
            "Pregnancies": index % 6,
            "Glucose": 75 + (index * 7) % 120,
            "BloodPressure": 55 + index % 40,
            "SkinThickness": 15 + index % 30,
            "Insulin": 30 + (index * 11) % 180,
            "BMI": 18 + (index % 20) * 0.8,
            "DiabetesPedigreeFunction": 0.1 + (index % 10) * 0.08,
            "Age": 21 + index % 55,
            "Outcome": index % 2,
        },
        columns=DATASET_COLUMNS,
    )


def test_train_and_evaluate_returns_valid_metrics(sample_dataset: pd.DataFrame) -> None:
    model, metrics = train_and_evaluate(sample_dataset, random_state=3)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= score <= 1 for score in metrics.values())
    assert len(model.predict(sample_dataset.drop(columns="Outcome"))) > 0


def test_train_and_evaluate_rejects_invalid_test_size(
    sample_dataset: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="test_size"):
        train_and_evaluate(sample_dataset, test_size=1)

