"""Top-down, seven-feature local explanations for PHPS health predictions."""
from __future__ import annotations
import numpy as np


def generate_shap_explanation(model, feature_vector, feature_names):
    """Build a serialisable TreeSHAP waterfall in the production feature order."""
    import shap
    vector = np.asarray(feature_vector, dtype=float)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(vector))
    expected = np.asarray(explainer.expected_value).reshape(-1)[0]
    waterfall = [{
        "feature": name, "raw_value": float(value), "shap_impact": round(float(impact), 6),
        "direction": "positive" if impact >= 0 else "negative",
    } for name, value, impact in zip(feature_names, vector[0], shap_values[0])]
    return {"base_value": round(float(expected), 6), "waterfall": waterfall,
            "feature_importance": sorted(waterfall, key=lambda item: abs(item["shap_impact"]), reverse=True)}


def deterministic_explanation(processed: dict) -> dict:
    """An exact 100-point waterfall when the optional trained model is absent."""
    analysis, features = processed["score_analysis"], processed["features"]
    impacts = {
        "norm_decayed_sentiment": analysis["sentiment_pts"] - 30.0,
        "effective_velocity": analysis["velocity_pts"] - 40.0,
        "overdue_ratio": analysis["overdue_pts"] - 20.0,
        "bug_ratio": analysis["bug_pts"] - 10.0,
        "urgency_flag": -analysis["urgency_penalty"],
        "is_divergent": -analysis["divergence_penalty"],
        "days_since_last_transcript": 0.0,
    }
    waterfall = [{"feature": name, "raw_value": float(features[name]), "shap_impact": round(impacts[name], 2),
                  "direction": "positive" if impacts[name] >= 0 else "negative"} for name in features]
    return {"base_value": 100.0, "waterfall": waterfall,
            "feature_importance": sorted(waterfall, key=lambda item: abs(item["shap_impact"]), reverse=True)}
