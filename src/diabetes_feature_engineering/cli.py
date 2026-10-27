"""Command-line interface for model training and evaluation."""

import argparse
from pathlib import Path

import joblib

from diabetes_feature_engineering.data import load_dataset
from diabetes_feature_engineering.modeling import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Engineer diabetes features and evaluate a random forest classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/diabetes.csv"),
        help="Path to the input CSV (default: data/diabetes.csv).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of the dataset reserved for evaluation (default: 0.3).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=17,
        help="Random seed for the train/test split (default: 17).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Optional path at which to save the fitted pipeline.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.data)
        model, metrics = train_and_evaluate(
            dataset,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Dataset rows: {len(dataset)}")
    print(f"Train/test split: {1 - args.test_size:.0%}/{args.test_size:.0%}")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.3f}")

    if args.model_output:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_output)
        print(f"Saved model: {args.model_output}")


if __name__ == "__main__":
    main()

