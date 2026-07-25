"""Canonical seven-feature, baseline-first PHPS feature processor."""
from __future__ import annotations

from math import exp
import os
from typing import Any
from app.services.health_policy import baseline_score

FEATURE_NAMES = (
    "norm_decayed_sentiment", "effective_velocity", "overdue_ratio",
    "bug_ratio", "urgency_flag", "is_divergent", "days_since_last_transcript",
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any, low: float, high: float, default: float = 0.0) -> float:
    return max(low, min(high, _number(value, default)))


def prepare_advanced_features(payload: dict[str, Any] | None = None, **values: Any) -> dict[str, Any]:
    """Normalise raw Jira/communication evidence and calculate the audit score.

    ``payload`` may contain the documented raw names. Keyword values override
    it so this is convenient both for ingestion code and model training.
    """
    data = {**(payload or {}), **values}
    has_communication_data = bool(data.get("has_communication_data", data.get("tone_score") is not None))
    has_jira_data = bool(data.get("has_jira_data", True))
    # Missing communication is unknown, not positive evidence.  In
    # particular it must never mint the old +15/+30 sentiment credit.
    tone = _clamp(data.get("tone_score"), -1.0, 1.0, 0.0)
    days = max(0, int(_number(data.get("days_since_last_transcript"), 0)))
    committed = max(0.0, _number(data.get("committed_story_pts")))
    completed = max(0.0, _number(data.get("completed_story_pts")))
    is_active = bool(data.get("is_sprint_active", False))
    historical = _clamp(data.get("historical_3sprint_avg_velocity"), 0.0, 1.0, 0.85)
    is_brand_new = bool(data.get("is_brand_new", False))
    effective_velocity = min(completed / committed, 1.0) if is_active and committed > 0 else (1.0 if is_brand_new else historical)

    total_open = max(0.0, _number(data.get("total_open_issues", data.get("open_issue_count"))))
    overdue = max(0.0, _number(data.get("overdue_issues", data.get("overdue_issue_count"))))
    bugs = max(0.0, _number(data.get("open_bugs", data.get("open_bug_count"))))
    overdue_ratio = min(overdue / total_open, 1.0) if total_open else 0.0
    bug_ratio = min(bugs / total_open, 1.0) if total_open else 0.0

    decay = exp(-0.15 * max(0, days - 3))
    effective_tone = tone * decay
    normalised_sentiment = _clamp((effective_tone + 1.0) / 2.0, 0.0, 1.0)
    urgency_count = max(0, int(_number(data.get("urgency_keywords_count", data.get("urgency_count", data.get("urgency_flag", 0))))))
    urgency_flag = 1 if urgency_count else 0
    # Divergence is a disagreement between the two independent evidence
    # streams, not a generic risk flag.  Keep the machine key and UI copy in
    # the API contract so clients never reverse this logic themselves.
    if tone > 0.10 and (overdue_ratio > 0.15 or effective_velocity < 0.70):
        divergence_key = "DELIVERY_METRICS_LAGGING"
        divergence_label = "Delivery metrics are lagging behind positive communication sentiment."
    elif effective_velocity >= 0.90 and overdue_ratio <= 0.10 and tone < -0.10:
        divergence_key = "COMMUNICATION_DEGRADED"
        divergence_label = "High delivery output, but communication sentiment has dropped."
    else:
        divergence_key = "NONE_DETECTED"
        divergence_label = "Signal alignment normal."
    is_divergent = int(divergence_key != "NONE_DETECTED")

    # Health is a top-down score: a project begins at 100 and only current
    # risks deduct points.  Do not turn missing data, neutral tone, or a
    # healthy delivery metric into a partial "earned" allocation.
    overdue_penalty = overdue_ratio * 40.0 if has_jira_data else 0.0
    bug_penalty = bug_ratio * 30.0 if has_jira_data else 0.0
    # Velocity is already the completion ratio for the current sprint.  A
    # project at or above its time-proportional expectation (1.0) has no
    # velocity deduction.
    velocity_penalty = max(0.0, 1.0 - effective_velocity) * 30.0 if has_jira_data else 0.0
    sentiment_penalty = max(0.0, -effective_tone) * 20.0 if has_communication_data else 0.0
    urgency_penalty = min(urgency_count * 5.0, 10.0) if has_communication_data else 0.0
    divergence_penalty = float(os.getenv("PHPS_DIVERGENCE_DEDUCTION", "0")) if is_divergent else 0.0
    delivery_deduction = overdue_penalty + bug_penalty + velocity_penalty + divergence_penalty
    communication_deduction = sentiment_penalty + urgency_penalty
    baseline = baseline_score()
    ground_truth = round(max(0.0, min(baseline, baseline - delivery_deduction - communication_deduction)), 2)
    def deduction_delta(value: float) -> float:
        """Format zero as 0.0 rather than the misleading -0.0."""
        return round(-value, 2) if value else 0.0

    return {
        "features": {
            "norm_decayed_sentiment": normalised_sentiment, "effective_velocity": effective_velocity,
            "overdue_ratio": overdue_ratio, "bug_ratio": bug_ratio, "urgency_flag": urgency_flag,
            "is_divergent": is_divergent, "days_since_last_transcript": days,
            "divergence_key": divergence_key,
        },
        "ground_truth_score": ground_truth,
        "score_analysis": {
            "baseline": baseline,
            "sprint_velocity_penalty": deduction_delta(velocity_penalty),
            "overdue_rate_penalty": deduction_delta(overdue_penalty),
            "bug_ratio_penalty": deduction_delta(bug_penalty),
            "divergence_warning_penalty": deduction_delta(divergence_penalty),
            "model_calibration_penalty": 0.0,
            "transcript_sentiment_penalty": deduction_delta(sentiment_penalty + urgency_penalty),
            "delivery_deduction": deduction_delta(delivery_deduction),
            "communication_deduction": deduction_delta(communication_deduction),
            # Legacy aliases now contain deduction deltas, never allocations.
            "sentiment_pts": deduction_delta(sentiment_penalty + urgency_penalty),
            "velocity_pts": deduction_delta(velocity_penalty),
            "overdue_pts": deduction_delta(overdue_penalty),
            "bug_pts": deduction_delta(bug_penalty),
            "urgency_penalty": deduction_delta(urgency_penalty),
            "divergence_penalty": deduction_delta(divergence_penalty),
        },
        "raw_metrics_summary": {"total_open_issues": int(total_open), "overdue_issues": int(overdue), "open_bugs": int(bugs), "completed_story_pts": completed, "committed_story_pts": committed, "tone_score": tone},
        "flags": {"is_divergent_warning": bool(is_divergent), "divergence_key": divergence_key,
                  "divergence_label": divergence_label, "is_data_stale": days > 3,
                  "is_sprint_active": is_active, "is_kickoff_state": not has_jira_data,
                  "has_communication_data": has_communication_data, "has_jira_data": has_jira_data},
    }


def feature_vector(processed: dict[str, Any]) -> list[float]:
    return [float(processed["features"][name]) for name in FEATURE_NAMES]
