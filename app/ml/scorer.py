"""Public scoring facade for the PHPS seven-feature fusion pipeline."""
from __future__ import annotations
from typing import Dict
from app.services.feature_processor import prepare_advanced_features
from app.services.health_policy import baseline_score
from app.services.health_scoring import delivery_deductions, health_score, status_for_score


def explain_health_score(*, health_score: float, tone_score: float, urgency_flag: int,
                         velocity_percent: float, overdue_rate: float, bug_ratio: float,
                         has_communication_data: bool = True, has_jira_data: bool = True) -> Dict[str, float]:
    """Return top-down deduction deltas for a saved or simulated score."""
    result = prepare_advanced_features(
        tone_score=tone_score, urgency_keywords_count=urgency_flag,
        committed_story_pts=1, completed_story_pts=velocity_percent, is_sprint_active=True,
        total_open_issues=100, overdue_issues=overdue_rate * 100, open_bugs=bug_ratio * 100,
        has_communication_data=has_communication_data, has_jira_data=has_jira_data,
    )
    analysis = result["score_analysis"].copy()
    urgency_penalty = round(-min(max(0, urgency_flag) * 5.0, 10.0), 2)
    values = {
        "baseline": analysis["baseline"],
        # Keep urgency separately so existing source-level displays can split
        # it among urgent messages without counting it twice.
        "communication_sentiment": round(analysis["communication_deduction"] - urgency_penalty, 2),
        "velocity": analysis["sprint_velocity_penalty"],
        "overdue_rate": analysis["overdue_rate_penalty"],
        "bug_ratio": analysis["bug_ratio_penalty"],
        "urgency_penalty": urgency_penalty,
        "delivery_metrics": analysis["delivery_deduction"],
        "delivery_deduction": analysis["delivery_deduction"],
        "communication_deduction": analysis["communication_deduction"],
        "sprint_velocity_penalty": analysis["sprint_velocity_penalty"],
        "overdue_rate_penalty": analysis["overdue_rate_penalty"],
        "bug_ratio_penalty": analysis["bug_ratio_penalty"],
        "transcript_sentiment_penalty": analysis["transcript_sentiment_penalty"],
        "divergence_warning_penalty": analysis["divergence_warning_penalty"],
        "model_calibration_penalty": analysis["model_calibration_penalty"],
    }
    values["final_health_score"] = round(values["baseline"] + values["delivery_deduction"] + values["communication_deduction"], 2)
    return values


def compute_health_score(tone_score: float, urgency_flag: int, velocity_percent: float,
                         overdue_rate: float, bug_ratio: float, green_threshold: float = 70.0,
                         red_threshold: float = 40.0, *, urgency_count: int | None = None,
                         open_issues: int = 0, open_bugs: int = 0, committed_story_pts: float | None = None,
                         completed_story_pts: float | None = None, is_sprint_active: bool | None = None,
                         historical_3sprint_avg_velocity: float = 0.85,
                         days_since_last_transcript: int = 0, has_communication_data: bool | None = None,
                         has_jira_data: bool = True, **_ignored) -> Dict:
    """Return the deterministic top-down health score.

    The first five positional arguments are retained for older integrations;
    production callers should also supply the sprint totals and state.
    """
    communication_items = _ignored.get("communication_items")
    metrics = _ignored.get("jira_metrics")
    if metrics is not None:
        # The central service works from exact Jira counts/sprint timing and
        # per-item communication observations.  Legacy scalar arguments stay
        # supported for simulations and old integrations.
        items = communication_items or [_Communication(tone_score, urgency_flag)]
        result = health_score(metrics, items, green_threshold, red_threshold)
        delivery, communication = result["delivery"], result["communication"]
        score = result["health_score"]
        return {
            "health_score": score, "ground_truth_score": score, "rag_status": result["rag_status"],
            "divergence_flag": result["divergence_flag"], "prediction_source": "Deterministic Project Health Scoring Engine",
            "prediction_version": "PHPS Deduction Engine v2", "feature_vector": dict(metrics), "shap_values": None,
            "score_analysis": {"baseline": 100.0, "sprint_velocity_penalty": -round(delivery["velocity"], 2),
                "overdue_rate_penalty": -round(delivery["overdue"], 2), "bug_ratio_penalty": -round(delivery["bug"], 2),
                "delivery_deduction": -round(delivery["total"], 2), "communication_deduction": -round(communication["total"], 2),
                "transcript_sentiment_penalty": -round(communication["total"], 2), "divergence_warning_penalty": 0.0,
                "model_calibration_penalty": 0.0}, "flags": {"divergence_key": result["divergence_key"], "divergence_label": result["divergence_label"]}, "divergence_key": result["divergence_key"],
            "divergence_label": result["divergence_label"], "raw_metrics_summary": dict(metrics), "baseline": 100.0,
            "delivery_deduction": -round(delivery["total"], 2), "communication_deduction": -round(communication["total"], 2),
            "final_health_score": score, "breakdown": {"sprint_velocity_penalty": -round(delivery["velocity"], 2),
                "overdue_rate_penalty": -round(delivery["overdue"], 2), "bug_ratio_penalty": -round(delivery["bug"], 2),
                "transcript_sentiment_penalty": -round(communication["total"], 2)}}
    active = bool(is_sprint_active) if is_sprint_active is not None else True
    committed = committed_story_pts if committed_story_pts is not None else (1.0 if active else 0.0)
    completed = completed_story_pts if completed_story_pts is not None else max(0.0, velocity_percent) * committed
    # Older callers passed ratios without issue counts. Preserve that API by
    # using a synthetic denominator only in that legacy shape; real zero-open
    # boards still receive the clean-board baseline.
    denominator = open_issues or (100 if (overdue_rate or bug_ratio) and not open_issues else 0)
    processed = prepare_advanced_features(
        tone_score=tone_score, urgency_keywords_count=urgency_flag if urgency_count is None else urgency_count,
        committed_story_pts=committed, completed_story_pts=completed, is_sprint_active=active,
        historical_3sprint_avg_velocity=historical_3sprint_avg_velocity,
        total_open_issues=denominator, overdue_issues=max(0.0, overdue_rate) * denominator,
        open_bugs=open_bugs if open_bugs else max(0.0, bug_ratio) * denominator,
        days_since_last_transcript=days_since_last_transcript,
        has_communication_data=has_communication_data if has_communication_data is not None else True,
        has_jira_data=has_jira_data,
    )
    # The trained additive model is intentionally not used here: it can only
    # produce a partial bucket allocation and therefore violates the 100-point
    # baseline contract.
    score, source = processed["ground_truth_score"], "Top-Down Deduction Engine"
    rag = "GREEN" if score >= green_threshold else "RED" if score < red_threshold else "AMBER"
    return {
        "health_score": score, "ground_truth_score": processed["ground_truth_score"],
        "rag_status": rag, "divergence_flag": processed["features"]["is_divergent"],
        "prediction_source": source,
        "prediction_version": "PHPS Top-Down Deduction Engine v1",
        "feature_vector": processed["features"], "shap_values": None,
        "score_analysis": processed["score_analysis"], "flags": processed["flags"],
        "divergence_key": processed["flags"]["divergence_key"],
        "divergence_label": processed["flags"]["divergence_label"],
        "raw_metrics_summary": processed["raw_metrics_summary"],
        "baseline": baseline_score(),
        "delivery_deduction": processed["score_analysis"]["delivery_deduction"],
        "communication_deduction": processed["score_analysis"]["communication_deduction"],
        "final_health_score": score,
        "breakdown": {
            key: processed["score_analysis"][key] for key in (
                "sprint_velocity_penalty", "overdue_rate_penalty",
                "bug_ratio_penalty", "transcript_sentiment_penalty",
            )
        },
    }


class _Communication:
    """Compatibility observation for scalar/simulation score calls."""
    def __init__(self, tone_score: float, urgency_flag: int):
        self.tone_score, self.urgency_flag = tone_score, urgency_flag
