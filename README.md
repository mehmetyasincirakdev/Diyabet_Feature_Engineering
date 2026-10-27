# Diabetes Feature Engineering

An end-to-end, reproducible binary-classification example using the Pima Indians
Diabetes dataset. The project demonstrates safe preprocessing, feature
engineering, model evaluation, and optional model serialization.

> This is an educational example, not a medical diagnostic tool.

## Project structure

```text
.
├── data/                              # Local dataset; CSV files are not committed
├── src/
│   └── diabetes_feature_engineering/
│       ├── __init__.py
│       ├── cli.py                     # Command-line entry point
│       ├── data.py                    # Dataset loading and validation
│       ├── features.py                # Reusable feature transformation
│       └── modeling.py                # Pipeline, training, and evaluation
├── tests/
├── pyproject.toml
└── README.md
```

## Dataset

The program expects a CSV at `data/diabetes.csv` with these columns:

`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`,
`DiabetesPedigreeFunction`, `Age`, and binary target `Outcome` (0 or 1).

Obtain the dataset from a source whose terms permit your intended use, then place
it at `data/diabetes.csv`. The dataset is intentionally not bundled with this
repository. Pass `--data` to use a different local CSV path.

## Setup

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run

From the repository root:

```powershell
diabetes-feature-engineering
```

Choose a different dataset or save the fitted pipeline:

```powershell
diabetes-feature-engineering --data data\diabetes.csv --model-output models\diabetes-model.joblib
```

The command reports accuracy, precision, recall, F1, and ROC AUC on a
stratified hold-out set. Preprocessing is fitted on training data only, through
the same scikit-learn pipeline as the classifier.

## Tests and lint

```powershell
python -m pytest
ruff check .
```

## Method notes

- Zero values in glucose, blood pressure, skin thickness, insulin, and BMI are
  treated as missing; zero pregnancies remains valid.
- Missing numeric values are median-imputed and scaled. Engineered categories
  are imputed and one-hot encoded.
- The train/test split is stratified and made before fitting any imputer,
  encoder, scaler, or model.
- The random forest and split use fixed random seeds for repeatability.

