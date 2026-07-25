"""Read-only health insight and simulation payloads."""
from app.ml.scorer import compute_health_score, explain_health_score


class AIInsightsService:
    @staticmethod
    def insights(project, score, email_count=0, transcript_count=0):
        c = explain_health_score(health_score=score.health_score, tone_score=score.tone_score, urgency_flag=score.urgency_flag, velocity_percent=score.velocity_percent, overdue_rate=score.overdue_rate, bug_ratio=score.bug_ratio)
        communication = c["communication_sentiment"] + c["urgency_penalty"]
        count = email_count + transcript_count
        risks = []
        if score.velocity_percent < .6: risks.append("sprint velocity is below target")
        if score.overdue_rate > .2: risks.append("overdue work is increasing")
        if score.bug_ratio > .15: risks.append("bug volume needs attention")
        if score.tone_score < 0: risks.append("communication sentiment needs attention")
        return {"current_health_score": score.health_score, "communication_sentiment": round(communication, 2), "delivery_metrics": round(c["delivery_metrics"], 2), "sprint_velocity_contribution": round(c["velocity"], 2), "overdue_rate_contribution": round(c["overdue_rate"], 2), "bug_ratio_contribution": round(c["bug_ratio"], 2), "urgency_penalty": round(c["urgency_penalty"], 2), "email_contribution": round(communication * email_count / count, 2) if count else 0, "transcript_contribution": round(communication * transcript_count / count, 2) if count else 0, "email_count": email_count, "transcript_count": transcript_count, "tone_score": score.tone_score, "urgency_flag": score.urgency_flag, "velocity_percent": score.velocity_percent, "overdue_rate": score.overdue_rate, "bug_ratio": score.bug_ratio, "explanation": ("Current delivery and communication signals are healthy." if not risks else "Project health is affected because " + ", ".join(risks) + "."), "confidence_score": min(96, 55 + count * 4 + (10 if project.jira_url else 0)), "rag_status": score.rag_status, "divergence_flag": score.divergence_flag}

    @staticmethod
    def simulate(project, score, v, email_count=0, transcript_count=0):
        p = compute_health_score(
            tone_score=float(v["communication_sentiment"]), urgency_flag=int(v["urgency_flag"]),
            velocity_percent=float(v["sprint_velocity"]), overdue_rate=float(v["overdue_rate"]),
            bug_ratio=float(v["bug_ratio"]), green_threshold=project.green_threshold,
            red_threshold=project.red_threshold, open_issues=getattr(score, "open_issues", 0),
            open_bugs=getattr(score, "open_bugs", 0), email_count=email_count,
            transcript_count=transcript_count, deadline=project.deadline,
        )
        return {"current_score": score.health_score, "predicted_score": p["health_score"], "difference": round(p["health_score"] - score.health_score, 2), "rag_status": p["rag_status"], "divergence_flag": p["divergence_flag"], "prediction_source": p["prediction_source"], "feature_vector": p["feature_vector"], "shap_values": p["shap_values"], "impact_analysis": {k: float(v[k]) for k in ("sprint_velocity", "bug_ratio", "overdue_rate", "communication_sentiment")}}
