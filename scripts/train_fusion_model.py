"""Train the production seven-feature PHPS XGBoost fusion model.

Input CSV must contain the raw fields accepted by ``prepare_advanced_features``.
Targets are deliberately recomputed, never trusted from a legacy score column.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import xgboost as xgb
from app.services.feature_processor import FEATURE_NAMES, prepare_advanced_features, feature_vector


def train(dataset_path: str | Path) -> Path:
    rows = pd.read_csv(dataset_path).fillna(0).to_dict("records")
    processed = [prepare_advanced_features(row) for row in rows]
    x = [feature_vector(item) for item in processed]
    y = [item["ground_truth_score"] for item in processed]
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=.05, objective="reg:squarederror")
    model.fit(x, y)
    destination = Path(__file__).resolve().parents[1] / "app" / "ml" / "artifacts" / "phps_xgboost_fusion.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    print(train(parser.parse_args().dataset))
