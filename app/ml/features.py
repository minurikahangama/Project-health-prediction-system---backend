"""The single feature contract for PHPS training and production inference."""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np


FEATURE_NAMES = (
    "overall_sentiment", "urgency_count", "velocity", "overdue_rate",
    "bug_ratio", "open_issues", "open_bugs", "email_count",
    "transcript_count", "days_to_deadline", "sentiment_trend",
    "velocity_trend", "bug_trend",
)


def _number(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_feature_dict(*, overall_sentiment, urgency_count, velocity, overdue_rate,
                       bug_ratio, open_issues, open_bugs, email_count,
                       transcript_count, deadline=None, sentiment_trend=0.0,
                       velocity_trend=0.0, bug_trend=0.0) -> dict[str, float]:
    """Build every model feature once, with training-compatible raw scales.

    Trend inputs must be recomputed from historical evidence by the caller;
    prior health predictions are never accepted as features.
    """
    now = datetime.now(timezone.utc)
    if deadline is not None:
        deadline = deadline.replace(tzinfo=timezone.utc) if deadline.tzinfo is None else deadline
        days_to_deadline = max(0.0, (deadline - now).total_seconds() / 86400)
    else:
        days_to_deadline = 0.0
    sentiment = max(-1.0, min(1.0, _number(overall_sentiment)))
    velocity = max(0.0, min(1.0, _number(velocity)))
    overdue = max(0.0, min(1.0, _number(overdue_rate)))
    bugs = max(0.0, min(1.0, _number(bug_ratio)))
    return {
        "overall_sentiment": sentiment,
        "urgency_count": max(0.0, _number(urgency_count)),
        "velocity": velocity, "overdue_rate": overdue, "bug_ratio": bugs,
        "open_issues": max(0.0, _number(open_issues)), "open_bugs": max(0.0, _number(open_bugs)),
        "email_count": max(0.0, _number(email_count)), "transcript_count": max(0.0, _number(transcript_count)),
        "days_to_deadline": days_to_deadline,
        "sentiment_trend": max(-1.0, min(1.0, _number(sentiment_trend))),
        "velocity_trend": max(-1.0, min(1.0, _number(velocity_trend))),
        "bug_trend": max(-1.0, min(1.0, _number(bug_trend))),
    }


def vector_from_features(features: dict[str, float]) -> np.ndarray:
    """Return the only ordered vector accepted by the fusion model."""
    return np.array([[float(features[name]) for name in FEATURE_NAMES]], dtype=float)
