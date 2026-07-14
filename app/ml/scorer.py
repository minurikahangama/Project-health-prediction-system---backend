"""
Health score fusion — Layer 4 of the PHPS ML pipeline.

compute_health_score() fuses 5 input signals into:
  - health_score    [0, 100]     (calibrated, monotonic signal fusion)
  - divergence_flag {0, 1}       (FR-36: Jira healthy but sentiment declining)
  - rag_status      GREEN|AMBER|RED  (FR-37: per-project thresholds)

The checked-in XGBoost artifact was trained on synthetic random data, so it is
not evidence that a real project's health can be predicted accurately.  Live
scoring therefore uses the documented weighted formula.  The artifact can be
enabled only for controlled experiments with ``PHPS_USE_XGBOOST_MODEL=true``.
"""
import os
import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)

# ── XGBoost model loading ─────────────────────────────────────────────────────

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "inference", "xgb_model.json")

_model = None


def _normalise_tone(tone_score: float) -> float:
    """Convert the public sentiment range [-1, 1] to the model's [0, 1]."""
    return max(0.0, min(1.0, (float(tone_score) + 1.0) / 2.0))

def _get_model():
    """Load the experimental XGBoost model only when explicitly enabled."""
    global _model
    if os.getenv("PHPS_USE_XGBOOST_MODEL", "").strip().lower() not in {"1", "true", "yes"}:
        return None
    if _model is None:
        if os.path.isfile(MODEL_PATH):
            try:
                import xgboost as xgb
                # train_xgboost.py creates an XGBRegressor whose target is a
                # health score from 0 to 100.  Loading it as a classifier
                # raises "Expecting: classifier, got: regressor", which made
                # the application silently use the rule-based fallback.
                m = xgb.XGBRegressor()
                m.load_model(MODEL_PATH)
                _model = m
                logger.info(f"Loaded XGBoost model from: {MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to load XGBoost model: {e}")
        else:
            logger.warning(
                f"XGBoost model not found at {MODEL_PATH}. "
                "Using rule-based fallback. "
                "Train with: backend/app/ml/training/train_xgboost.py"
            )
    return _model


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_score(
    tone: float,
    urgency: int,
    velocity: float,
    overdue: float,
    bug_ratio: float,
) -> float:
    """
    Weighted rule-based health score.
    Used when XGBoost model has not been trained yet.

    Weights:
      Tone score   → 30 points  (communication sentiment)
      Velocity     → 40 points  (delivery performance — highest weight)
      Overdue rate → 20 points  (inverted — low overdue = good)
      Bug ratio    → 10 points  (inverted — low bugs = good)

    Urgency flag applies a 10-point penalty.
    """
    # Clamp all externally supplied values before calculation.  This makes
    # the score stable even if a connector returns malformed percentages.
    tone = max(-1.0, min(1.0, float(tone)))
    velocity = max(0.0, min(1.0, float(velocity)))
    overdue = max(0.0, min(1.0, float(overdue)))
    bug_ratio = max(0.0, min(1.0, float(bug_ratio)))
    urgency = 1 if urgency else 0

    tone_component     = ((tone + 1.0) / 2.0) * 30.0
    velocity_component = velocity * 40.0
    overdue_component  = (1.0 - overdue) * 20.0
    bug_component      = (1.0 - bug_ratio) * 10.0
    urgency_penalty    = urgency * 10.0

    raw = (
        tone_component
        + velocity_component
        + overdue_component
        + bug_component
        - urgency_penalty
    )
    return max(0.0, min(100.0, raw))


def explain_health_score(
    *,
    health_score: float,
    tone_score: float,
    urgency_flag: int,
    velocity_percent: float,
    overdue_rate: float,
    bug_ratio: float,
) -> Dict[str, float]:
    """Return additive, read-only attributions for an already-made prediction.

    This never participates in model inference.  It uses the documented PHPS
    factor weights and normalises the positive components to the score already
    returned by ``compute_health_score``.  Consequently the displayed values
    always add up to that unchanged final score, including with XGBoost.
    """
    sentiment = _normalise_tone(tone_score) * 30.0
    velocity = max(0.0, min(1.0, float(velocity_percent))) * 40.0
    overdue = (1.0 - max(0.0, min(1.0, float(overdue_rate)))) * 20.0
    bugs = (1.0 - max(0.0, min(1.0, float(bug_ratio)))) * 10.0
    urgency = -10.0 if urgency_flag else 0.0
    positive_total = sentiment + velocity + overdue + bugs

    # Keep the urgency penalty visible, then scale only positive factors to
    # reconcile their sum with the already-predicted score.
    target_positive = max(0.0, float(health_score) - urgency)
    scale = target_positive / positive_total if positive_total else 0.0
    values = {
        "communication_sentiment": sentiment * scale,
        "velocity": velocity * scale,
        "overdue_rate": overdue * scale,
        "bug_ratio": bugs * scale,
        "urgency_penalty": urgency,
    }
    values["delivery_metrics"] = values["velocity"] + values["overdue_rate"] + values["bug_ratio"]

    # Round for the API while preserving the additive total exactly.
    for key in values:
        values[key] = round(values[key], 2)
    values["delivery_metrics"] = round(
        values["velocity"] + values["overdue_rate"] + values["bug_ratio"], 2
    )
    residual = round(float(health_score) - (
        values["communication_sentiment"] + values["delivery_metrics"] + values["urgency_penalty"]
    ), 2)
    values["delivery_metrics"] = round(values["delivery_metrics"] + residual, 2)
    values["velocity"] = round(values["velocity"] + residual, 2)
    return values


# ── Main function ─────────────────────────────────────────────────────────────

def compute_health_score(
    tone_score:       float,
    urgency_flag:     int,
    velocity_percent: float,
    overdue_rate:     float,
    bug_ratio:        float,
    green_threshold:  float = 70.0,
    red_threshold:    float = 40.0,
) -> Dict:
    """
    Fuse all ML signals into a health score, RAG status, and divergence flag.

    Args:
        tone_score:       RoBERTa output [-1.0, +1.0]
        urgency_flag:     Keyword urgency detector {0, 1}
        velocity_percent: Jira sprint velocity [0.0, 1.0]
        overdue_rate:     Jira overdue rate [0.0, 1.0]
        bug_ratio:        Jira bug ratio [0.0, 1.0]
        green_threshold:  Project-specific green boundary (default 70)
        red_threshold:    Project-specific red boundary (default 40)

    Returns dict with:
        health_score    float  [0, 100]
        rag_status      str    "GREEN" | "AMBER" | "RED"
        divergence_flag int    0 or 1

    FR-35: Signal fusion.
    FR-36: Divergence detection.
    FR-37: RAG classification using project thresholds.
    """
    model = _get_model()

    if model is not None:
        # The trained regressor expects all five inputs in the [0, 1] range.
        # score_tone() exposes sentiment as [-1, 1], so normalise it before
        # inference to match the feature range used during training.
        features = np.array([[
            _normalise_tone(tone_score),
            max(0.0, min(1.0, float(urgency_flag))),
            max(0.0, min(1.0, float(velocity_percent))),
            max(0.0, min(1.0, float(overdue_rate))),
            max(0.0, min(1.0, float(bug_ratio))),
        ]])
        score = round(float(model.predict(features)[0]), 2)
    else:
        # Production path: each favourable communication/delivery change can
        # only raise the score; each risk change can only lower it.  This is
        # deliberately auditable until a model is validated against labelled
        # historical project outcomes.
        score = round(
            _rule_based_score(
                tone_score, urgency_flag,
                velocity_percent, overdue_rate, bug_ratio,
            ),
            2,
        )

    # Clamp to [0, 100]
    score = max(0.0, min(100.0, score))

    # ── FR-36: Divergence detection ──────────────────────────────────────────
    # Divergence = Jira metrics look healthy BUT sentiment is declining
    # This is the novel research contribution of PHPS.
    #
    # Condition: velocity > 60% AND overdue < 30% AND tone < -0.3
    # Interpretation: "Team is delivering on time (Jira looks fine) but
    # communication is increasingly negative — early warning of hidden risk"
    divergence_flag = 1 if (
        velocity_percent > 0.6
        and overdue_rate  < 0.3
        and tone_score    < -0.3
    ) else 0

    # ── FR-37: RAG classification using project-specific thresholds ──────────
    if score >= green_threshold:
        rag_status = "GREEN"
    elif score < red_threshold:
        rag_status = "RED"
    else:
        rag_status = "AMBER"

    return {
        "health_score":    score,
        "rag_status":      rag_status,
        "divergence_flag": divergence_flag,
    }
