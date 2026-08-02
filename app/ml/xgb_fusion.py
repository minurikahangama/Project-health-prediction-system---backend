"""Seven-feature XGBoost inference with a deterministic safety fallback."""
from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
from app.services.feature_processor import FEATURE_NAMES, feature_vector

logger = logging.getLogger(__name__)
# Keep the deployable model with the application.  Older checkouts stored it
# alongside the training material, so retain that location as a read-only
# compatibility fallback.
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "phps_xgboost_fusion.json"
LEGACY_MODEL_PATH = Path(__file__).resolve().parent / "training" / "phps_xgboost_fusion.json"
_model = None
_load_attempted = False


def _get_model():
    global _model, _load_attempted
    if not _load_attempted:
        _load_attempted = True
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH)
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


def predict_with_shap(values: dict[str, float], feature_names: tuple[str, ...]) -> tuple[float, dict, dict | None]:
    """Return one XGBoost prediction and the exact TreeSHAP values behind it.

    No attribution is synthesised when the model is unavailable: callers can
    surface that limitation explicitly instead of presenting a plausible but
    false explanation.
    """
    model = _get_model()
    if model is None:
        raise RuntimeError("The XGBoost model is unavailable; SHAP cannot be generated.")
    vector = np.asarray([[float(values[name]) for name in feature_names]], dtype=float)
    expected_features = getattr(model, "n_features_in_", vector.shape[1])
    if expected_features != vector.shape[1]:
        raise RuntimeError(f"Model expects {expected_features} features, received {vector.shape[1]}.")
    from app.ml.explainability import generate_shap_explanation
    prediction = float(model.predict(vector)[0])
    if not np.isfinite(prediction):
        raise RuntimeError("XGBoost returned a non-finite prediction.")
    explanation = generate_shap_explanation(model, vector[0], feature_names)
    reconstructed = float(explanation["base_value"]) + sum(item["shap_impact"] for item in explanation["waterfall"])
    if not np.isclose(reconstructed, prediction, atol=0.01):
        raise RuntimeError(f"TreeSHAP consistency check failed ({reconstructed} != {prediction}).")
    explanation["prediction"] = round(prediction, 6)
    return max(0.0, min(100.0, round(prediction, 2))), explanation, {name: float(values[name]) for name in feature_names}
