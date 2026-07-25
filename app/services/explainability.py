"""Explainability API adapter for the top-down PHPS health engine."""
from __future__ import annotations
from datetime import datetime, timezone

from app.ml.explainability import deterministic_explanation
from app.services.feature_processor import FEATURE_NAMES, prepare_advanced_features


class ExplainabilityService:
    @staticmethod
    def build(project, score, email_count=0, transcript_count=0, health_observation_count=0):
        is_initiation = not getattr(project, "last_jira_synced_at", None) and not (email_count or transcript_count)
        has_jira_data = bool(getattr(project, "last_jira_synced_at", None))
        saved = getattr(score, "feature_vector", None)
        if isinstance(saved, dict) and all(name in saved for name in FEATURE_NAMES):
            processed = {"features": {name: float(saved[name]) for name in FEATURE_NAMES},
                         "score_analysis": getattr(score, "score_analysis", None)}
        else:
            # Historical score rows predate the seven-feature snapshot. Build
            # the closest auditable representation from their stored signals.
            processed = prepare_advanced_features(
                tone_score=score.tone_score, urgency_keywords_count=score.urgency_flag,
                committed_story_pts=1, completed_story_pts=score.velocity_percent, is_sprint_active=True,
                total_open_issues=getattr(score, "open_issues", 0),
                overdue_issues=score.overdue_rate * getattr(score, "open_issues", 0),
                open_bugs=getattr(score, "open_bugs", 0),
            )
        if is_initiation:
            processed = prepare_advanced_features(is_brand_new=True, has_jira_data=False, has_communication_data=False)
        # A score snapshot only contains feature values; derive its exact
        # top-down analysis for deterministic explanations.
        if not processed.get("score_analysis"):
            f = processed["features"]
            processed["score_analysis"] = {
                "sentiment_pts": round(f["norm_decayed_sentiment"] * 30, 2),
                "velocity_pts": round(f["effective_velocity"] * 40, 2),
                "overdue_pts": round((1 - f["overdue_ratio"]) * 20, 2),
                "bug_pts": round((1 - f["bug_ratio"]) * 10, 2),
                "urgency_penalty": 5.0 * f["urgency_flag"],
                "divergence_penalty": 15.0 * f["is_divergent"],
            }
        # The explainability screen is a top-down deduction ledger. TreeSHAP
        # uses the model-training expected value, which is not the product's
        # 100-point baseline, so it cannot be displayed as this waterfall.
        explanation, method = deterministic_explanation(processed), "top_down_deduction"
        if is_initiation:
            explanation["waterfall"] = [{"feature": name, "raw_value": float(processed["features"][name]), "shap_impact": 0.0, "direction": "neutral"} for name in FEATURE_NAMES]
            explanation["feature_importance"] = explanation["waterfall"]
        displayed_score = 100.0 if is_initiation else round(float(score.health_score), 2)
        explained_score = round(100.0 + sum(item["shap_impact"] for item in explanation["waterfall"]), 2)
        calibration = round(displayed_score - explained_score, 2)
        if calibration:
            explanation["waterfall"].append({"feature": "model_calibration", "raw_value": calibration, "shap_impact": calibration, "direction": "positive" if calibration > 0 else "negative"})
            explanation["feature_importance"] = sorted(explanation["waterfall"], key=lambda item: abs(item["shap_impact"]), reverse=True)
        labels = {
            "norm_decayed_sentiment": "Decayed sentiment", "effective_velocity": "Effective velocity",
            "overdue_ratio": "Overdue issues", "bug_ratio": "Bug load",
            "urgency_flag": "Urgency keywords", "is_divergent": "Divergence warning",
            "days_since_last_transcript": "Transcript recency", "model_calibration": "Model calibration",
        }
        feature_contributions = [{
            "feature": item["feature"], "label": labels[item["feature"]],
            "shap_value": item["shap_impact"], "feature_value": item["raw_value"],
            "reason": f"{labels[item['feature']]} {'increased' if item['shap_impact'] >= 0 else 'deducted from'} project health.",
        } for item in explanation["feature_importance"]]
        analysis = processed["score_analysis"]
        impacts = {item["feature"]: item["shap_impact"] for item in explanation["waterfall"]}
        communication = round(sum(impacts.get(name, 0.0) for name in ("norm_decayed_sentiment", "urgency_flag", "days_since_last_transcript")), 2)
        delivery = round(sum(impacts.get(name, 0.0) for name in ("effective_velocity", "overdue_ratio", "bug_ratio")), 2)
        source_count = email_count + transcript_count
        email_contribution = round(communication * email_count / source_count, 2) if source_count else 0.0
        transcript_contribution = round(communication - email_contribution, 2) if source_count else 0.0
        synced_at = max((value for value in (getattr(project, "last_jira_synced_at", None), getattr(project, "last_email_synced_at", None)) if value), default=None)
        synced_days = min(30.0, max(0.0, (datetime.now(timezone.utc) - (synced_at.replace(tzinfo=timezone.utc) if synced_at and synced_at.tzinfo is None else synced_at)).total_seconds() / 86400 + 1)) if synced_at else 0.0
        ingestion_score = (50.0 if has_jira_data else 0.0) + min(50.0, source_count * 10.0)
        confidence = 0 if is_initiation else round(min(100.0, (synced_days / 30.0) * ingestion_score + min(20.0, health_observation_count * 2.0)))
        return {
            "project_id": str(project.id), "current_health_score": displayed_score,
            "prediction_source": getattr(score, "prediction_source", None) or "Deterministic Fallback Engine",
            "shap_method": method, **explanation,
            "score_analysis": processed["score_analysis"], "feature_vector": processed["features"],
            # Compatibility fields consumed by the existing dashboard page.
            "base_prediction": explanation["base_value"], "shap_values": {item["feature"]: item["shap_impact"] for item in explanation["waterfall"]},
            "feature_contributions": feature_contributions,
            "prediction_confidence": confidence, "confidence_level": "Awaiting Sync Data" if is_initiation else ("High" if confidence >= 80 else "Medium" if confidence >= 60 else "Low"),
            "human_readable_explanation": "Project is in Initiation Baseline state (100/100). Sync Jira issues and team communications to generate live SHAP feature attributions." if is_initiation else "This explanation starts from the 100-point health baseline; every displayed impact reconciles exactly to the final health score.",
            "is_initiation": is_initiation,
            "communication_breakdown": {"contribution": communication, "emails": email_contribution, "meeting_transcripts": transcript_contribution},
            "delivery_breakdown": {"contribution": delivery, "sprint_velocity": impacts.get("effective_velocity", 0.0), "overdue_rate": impacts.get("overdue_ratio", 0.0), "bug_ratio": impacts.get("bug_ratio", 0.0)},
            "rag_status": score.rag_status, "divergence_flag": score.divergence_flag,
        }
