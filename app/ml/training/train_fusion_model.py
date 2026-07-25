"""Train the fusion model using the production feature contract.

Run from this directory after generating snapshots with ``generate_dataset``.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.ml.features import FEATURE_NAMES
from app.ml.preprocessor import FeaturePreprocessor, clip_health

HERE = Path(__file__).resolve().parent
DATASET = HERE / "accurate_fusion_dataset.csv"
MODEL = HERE / "phps_xgboost_fusion.json"


def train_phps_model():
    dataset = pd.read_csv(DATASET)
    required = set(FEATURE_NAMES) | {"health_score"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset does not use the production feature contract: {sorted(missing)}")
    raw_features = dataset.loc[:, FEATURE_NAMES].to_dict("records")
    preprocessor = FeaturePreprocessor().fit(raw_features)
    x = preprocessor.transform_matrix(raw_features)
    y = dataset["health_score"].clip(0.0, 100.0).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    # Enforce the domain relationships at model level.  Counts are left
    # unconstrained because their effect is learned from labelled history.
    # Feature order is the shared ``FEATURE_NAMES`` contract.
    monotone_constraints = (1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 1, 1, -1)
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=.05,
                             subsample=.85, colsample_bytree=.9, random_state=42,
                             monotone_constraints=monotone_constraints)
    model.fit(x_train, y_train)
    predicted = np.clip(model.predict(x_test), 0.0, 100.0)
    report = {"rmse": float(mean_squared_error(y_test, predicted) ** .5),
              "mae": float(mean_absolute_error(y_test, predicted)),
              "r2": float(r2_score(y_test, predicted)), "features": list(FEATURE_NAMES)}
    model.save_model(MODEL)
    preprocessor.save()
    (HERE / "phps_xgboost_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    train_phps_model()
