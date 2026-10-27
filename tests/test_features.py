import pandas as pd
import pytest

from diabetes_feature_engineering.data import FEATURE_COLUMNS
from diabetes_feature_engineering.features import DiabetesFeatureEngineer


@pytest.fixture
def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pregnancies": [0, 2, 1],
            "Glucose": [0, 110, 150],
            "BloodPressure": [70, 0, 80],
            "SkinThickness": [20, 25, 30],
            "Insulin": [80, 0, 170],
            "BMI": [18.5, 27, 32],
            "DiabetesPedigreeFunction": [0.2, 0.4, 0.6],
            "Age": [20, 35, 60],
        },
        columns=FEATURE_COLUMNS,
    )


def test_engineer_handles_zeros_and_creates_expected_categories(
    sample_features: pd.DataFrame,
) -> None:
    original = sample_features.copy(deep=True)

    transformed = DiabetesFeatureEngineer().fit_transform(sample_features)

    assert pd.isna(transformed.loc[0, "Glucose"])
    assert pd.isna(transformed.loc[1, "BloodPressure"])
    assert transformed.loc[0, "Pregnancies"] == 0
    assert transformed.loc[0, "BMICategory"] == "healthy"
    assert transformed.loc[1, "AgeBMICategory"] == "adult_overweight"
    assert transformed.loc[2, "GlucoseCategory"] == "diabetes"
    pd.testing.assert_frame_equal(sample_features, original)


def test_engineer_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required feature columns"):
        DiabetesFeatureEngineer().fit(pd.DataFrame({"Age": [30]}))

