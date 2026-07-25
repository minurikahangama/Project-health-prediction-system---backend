"""Seven-feature XGBoost inference with a deterministic safety fallback."""
from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
from app.services.feature_processor import FEATURE_NAMES, feature_vector

logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "phps_xgboost_fusion.json"
_model = None
_load_attempted = False


def _get_model():
    global _model, _load_attempted
    if not _load_attempted:
        _load_attempted = True
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(MODEL_PATH)
            _model = model
        except Exception as exc:
            logger.warning("PHPS fusion model unavailable; deterministic fallback will be used: %s", exc)
    return _model


def predict(processed: dict) -> tuple[float, str]:
    """Predict score, never allowing model/load/input errors to escape."""
    try:
        model = _get_model()
        if model is None:
            raise RuntimeError("model unavailable")
        value = float(model.predict(np.asarray([feature_vector(processed)], dtype=float))[0])
        if not np.isfinite(value):
            raise ValueError("non-finite prediction")
        return max(0.0, min(100.0, round(value, 2))), "XGBoost ML Fusion Model"
    except Exception as exc:
        logger.info("Using deterministic PHPS score: %s", exc)
        return float(processed["ground_truth_score"]), "Deterministic Fallback Engine"
