"""Snapshot-backed TreeSHAP response for the explainability dashboard."""
from __future__ import annotations

from datetime import datetime, timezone


LABELS = {
    "overall_sentiment": "Communication sentiment", "urgency_count": "Urgency signals",
    "velocity": "Sprint velocity", "overdue_rate": "Overdue rate", "bug_ratio": "Bug ratio",
    "open_issues": "Open issues", "open_bugs": "Open bugs", "email_count": "Email count",
    "transcript_count": "Transcript count", "days_to_deadline": "Days to deadline",
    "sentiment_trend": "Sentiment trend", "velocity_trend": "Velocity trend", "bug_trend": "Bug trend",
}
COMMUNICATION = {"overall_sentiment", "urgency_count", "email_count", "transcript_count", "sentiment_trend"}


def _date(value):
    return value.isoformat() if value else None


class ExplainabilityService:
    @staticmethod
    def build(project, score, history=()):
        """Expose only values traceable to a persisted prediction snapshot."""
        saved = getattr(score, "shap_explanation", None) or {}
        values = getattr(score, "feature_vector", None) or {}
        shap = saved.get("waterfall") if isinstance(saved, dict) else None
        if not shap and isinstance(getattr(score, "shap_explanation", None), dict):
            # Earlier snapshot rows stored the feature->SHAP map directly.
            shap = [{"feature": key, "shap_impact": value, "raw_value": values.get(key)}
                    for key, value in score.shap_explanation.items()]
        available = bool(shap) and getattr(score, "prediction_source", "").lower().find("xgboost") >= 0
        contributions = sorted([
            {"feature": item["feature"], "label": LABELS.get(item["feature"], item["feature"].replace("_", " ").title()),
             "shap_value": round(float(item.get("shap_impact", 0)), 4),
             "feature_value": item.get("raw_value", values.get(item["feature"])),
             "direction": "positive" if float(item.get("shap_impact", 0)) > 0 else "negative" if float(item.get("shap_impact", 0)) < 0 else "neutral"}
            for item in (shap or [])
        ], key=lambda item: abs(item["shap_value"]), reverse=True)
        impacts = {item["feature"]: item["shap_value"] for item in contributions}
        total_abs = sum(abs(value) for value in impacts.values())
        comm_abs = sum(abs(value) for feature, value in impacts.items() if feature in COMMUNICATION)
        delivery_abs = total_abs - comm_abs
        previous = history[-2] if len(history) > 1 else None
        previous_values = getattr(previous, "feature_vector", None) or {}
        changes = [{"feature": item["feature"], "label": item["label"], "previous": previous_values[item["feature"]],
                    "current": values[item["feature"]], "delta": round(float(values[item["feature"]]) - float(previous_values[item["feature"]]), 4)}
                   for item in contributions if item["feature"] in values and item["feature"] in previous_values
                   and float(values[item["feature"]]) != float(previous_values[item["feature"]])]
        positives = [item for item in contributions if item["shap_value"] > 0][:5]
        risks = [item for item in contributions if item["shap_value"] < 0][:5]
        supporting = ", ".join(item["label"].lower() for item in positives[:3])
        reducing = ", ".join(item["label"].lower() for item in risks[:3])
        narrative = (f"The XGBoost prediction is supported most by {supporting}. " if supporting else "No positive TreeSHAP contributors were found. ") + (f"It is reduced most by {reducing}." if reducing else "No negative TreeSHAP contributors were found.")
        trend = [{"timestamp": _date(row.recorded_at), "health_score": row.health_score,
                  "features": getattr(row, "feature_vector", None) or {}}
                 for row in history]
        jira = getattr(project, "jira_metrics_snapshot", None) or {}
        email_count = values.get("email_count")
        transcript_count = values.get("transcript_count")
        issue_count = jira.get("open_issue_count", values.get("open_issues"))
        active_sprints = jira.get("active_sprint_count")
        completed_sprints = jira.get("completed_sprint_count")
        # Coverage is a reproducible property of the actual evidence in this
        # snapshot.  It is not model certainty, so it is explicitly labelled
        # as data confidence and never invented when source counts are absent.
        coverage_values = [value for value in (issue_count, email_count, transcript_count, completed_sprints)
                           if isinstance(value, (int, float))]
        evidence_total = sum(max(0.0, float(value)) for value in coverage_values)
        confidence = saved.get("confidence") if isinstance(saved, dict) else None
        if confidence is None:
            confidence = round(100 * (1 - 1 / (1 + evidence_total)), 2) if coverage_values else None
        lineage = {
            "jira_issues": issue_count, "active_sprints": active_sprints, "completed_sprints": completed_sprints,
            "emails": email_count, "meeting_transcripts": transcript_count,
            "commit_events": jira.get("commit_count"), "words_analysed": jira.get("words_analysed"),
            "analysis_window_days": jira.get("analysis_window_days"),
        }
        if isinstance(saved, dict):
            lineage.update({key: value for key, value in (saved.get("lineage") or {}).items() if value is not None})
        return {
            "project_id": str(project.id), "current_health_score": float(score.health_score),
            "previous_health_score": float(previous.health_score) if previous else None,
            "health_change": round(float(score.health_score) - float(previous.health_score), 2) if previous else None,
            "prediction_timestamp": _date(getattr(score, "recorded_at", None)), "project_status": score.rag_status,
            "prediction_source": getattr(score, "prediction_source", None), "model_version": saved.get("model_version") if isinstance(saved, dict) else None, "shap_available": available,
            "shap_method": "tree_shap" if available else "unavailable", "base_prediction": saved.get("base_value") if isinstance(saved, dict) else None,
            "feature_vector": values, "feature_contributions": contributions, "waterfall": shap or [],
            "positive_drivers": positives, "risk_drivers": risks, "root_causes": risks,
            "human_readable_explanation": narrative if available else "TreeSHAP is unavailable for this prediction; no explanation or confidence has been inferred.",
            "communication_influence": round(comm_abs / total_abs * 100, 2) if total_abs else None,
            "delivery_influence": round(delivery_abs / total_abs * 100, 2) if total_abs else None,
            "changes_since_previous": changes, "timeline": trend,
            "prediction_confidence": confidence if available else None,
            "confidence_level": ("High" if confidence is not None and confidence >= 80 else "Medium" if confidence is not None and confidence >= 50 else "Low" if confidence is not None else "Unavailable"),
            "confidence_reason": "Calculated from the count of source records included in this prediction." if confidence is not None else "Source coverage is not available for this snapshot.",
            "data_lineage": lineage,
            "rag_status": score.rag_status, "divergence_flag": score.divergence_flag,
        }
