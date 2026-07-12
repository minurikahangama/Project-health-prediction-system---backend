"""Repeatably evaluate the XGBoost health-score training recipe.

The repository currently has only synthetic targets. This script therefore
measures reproducibility of the synthetic training recipe, not real-world
prediction accuracy. Run it from the backend root:

    venv\\Scripts\\python.exe app/ml/training/evaluate_xgboost.py
"""
import argparse
import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def make_synthetic_dataset(samples: int, seed: int):
    """Create the documented five-feature synthetic health-score dataset."""
    rng = np.random.RandomState(seed)
    features = rng.rand(samples, 5)
    targets = (
        (features[:, 0] * 30)
        - (features[:, 1] * 15)
        + (features[:, 2] * 40)
        - (features[:, 3] * 20)
        - (features[:, 4] * 15)
        + 45
    )
    return features, np.clip(targets, 0, 100)


def evaluate(samples: int, seed: int, test_size: float) -> dict:
    """Fit on a deterministic training split and return held-out metrics."""
    features, targets = make_synthetic_dataset(samples, seed)
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=test_size, random_state=seed
    )
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "dataset": "synthetic",
        "samples": samples,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "seed": seed,
        "test_size": test_size,
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the XGBoost health scorer.")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output", help="Optional path for the JSON metrics report.")
    args = parser.parse_args()
    if args.samples < 2 or not 0 < args.test_size < 1:
        parser.error("--samples must be at least 2 and --test-size must be between 0 and 1")

    report = evaluate(args.samples, args.seed, args.test_size)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
