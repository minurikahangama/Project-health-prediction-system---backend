"""Validate the deployed XGBoost artifact against its held-out data and regression cases.

Run from the backend root with ``python -m app.ml.training.validate_model``.
The reported metrics are regression metrics: RMSE, MAE and R².  They are not
classification metrics and should be replaced with results from real labelled
project history when that data becomes available.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.ml.features import FEATURE_NAMES
from app.ml.preprocessor import FeaturePreprocessor
from app.ml.scorer import _get_model

HERE = Path(__file__).resolve().parent


def _predict(rows: list[dict[str, float]]) -> np.ndarray:
    model = _get_model()
    if model is None:
        raise RuntimeError("The deployed XGBoost model could not be loaded")
    return np.clip(model.predict(FeaturePreprocessor.load().transform_matrix(rows)), 0.0, 100.0)


def validate() -> dict:
    dataset = pd.read_csv(HERE / "accurate_fusion_dataset.csv")
    _, held_out = train_test_split(dataset, test_size=.2, random_state=42)
    actual = held_out["health_score"].to_numpy(dtype=float)
    predicted = _predict(held_out.loc[:, FEATURE_NAMES].to_dict("records"))

    regression_rows = list(csv.DictReader((HERE / "prediction_regression_cases.csv").open(encoding="utf-8")))
    regression_predictions = _predict([
        {name: float(row[name]) for name in FEATURE_NAMES} for row in regression_rows
    ])
    regression = []
    for row, prediction in zip(regression_rows, regression_predictions):
        expected = float(row["expected_prediction"])
        difference = round(float(prediction) - expected, 2)
        regression.append({"case": row["case"], "expected": expected,
                           "predicted": round(float(prediction), 2), "difference": difference,
                           "status": "PASS" if abs(difference) <= .01 else "FAIL"})

    return {
        "model_metrics": {
            "rmse": round(float(mean_squared_error(actual, predicted) ** .5), 4),
            "mae": round(float(mean_absolute_error(actual, predicted)), 4),
            "r2": round(float(r2_score(actual, predicted)), 4),
            "held_out_rows": len(held_out),
        },
        "regression_cases": regression,
        "regression_passed": all(case["status"] == "PASS" for case in regression),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
