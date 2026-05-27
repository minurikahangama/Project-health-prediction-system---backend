"""
Health score fusion — Layer 4 of the PHPS ML pipeline.

compute_health_score() fuses 5 input signals into:
  - health_score    [0, 100]     (XGBoost or rule-based fallback)
  - divergence_flag {0, 1}       (FR-36: Jira healthy but sentiment declining)
  - rag_status      GREEN|AMBER|RED  (FR-37: per-project thresholds)

Model loading:
  - Looks for trained model at app/ml/inference/xgb_model.json
  - Falls back to a weighted rule-based formula while the model is not yet trained
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

def _get_model():
    """Lazy-load XGBoost model on first call."""
    global _model
    if _model is None:
        if os.path.isfile(MODEL_PATH):
            try:
                import xgboost as xgb
                m = xgb.XGBClassifier()
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
    tone_component     = ((tone + 1.0) / 2.0) * 30.0    # normalise [-1,1] → [0,1] → [0,30]
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
        # XGBoost model available — use trained classifier
        features    = np.array([[tone_score, urgency_flag,
                                  velocity_percent, overdue_rate, bug_ratio]])
        # predict_proba returns [[prob_class_0, prob_class_1]]
        # class 1 = Healthy → use as health score
        prob_healthy = float(model.predict_proba(features)[0][1])
        score        = round(prob_healthy * 100.0, 2)
    else:
        # Fallback to rule-based scoring
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
