"""Data-grounded PHPS project assistant.

This service deliberately has no project-specific answer catalogue.  Each
response is assembled from a fresh, immutable context built from the newest
Jira snapshot, persisted communication observations, XGBoost and TreeSHAP.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from app.ml.scorer import compute_health_score
from app.services.ai_insights import AIInsightsService
from app.services.absence_simulator import AbsenceSimulator
from app.services.decision_support import DecisionSupportService
from app.services.team_capacity import TeamCapacityService
from app.services.identity_resolver import IdentityResolver


class AIProjectAssistant:
    @staticmethod
    def collect_project_context(project, score, db) -> dict:
        if not project.jira_capacity_snapshot:
            raise ValueError("Jira has not been synced yet; a current sprint snapshot is required.")
        analysis = TeamCapacityService.analyse(project.jira_capacity_snapshot)
        resolver = IdentityResolver.for_project(db, project.id)
        for member in analysis["team_capacity"]:
            member["developer"] = resolver.resolve(member.get("jira_account_id") or member["developer"])
        for item in analysis["_items"]:
            item["assignee"] = resolver.resolve(item.get("jira_account_id") or item["assignee"])
        from app.models.models import ProcessedEmail, TranscriptUpload
        emails = db.query(ProcessedEmail).filter_by(project_id=project.id).count()
        transcripts = db.query(TranscriptUpload).filter_by(project_id=project.id).count()
        prediction = compute_health_score(
            score.tone_score, score.urgency_flag, score.velocity_percent,
            score.overdue_rate, score.bug_ratio, project.green_threshold,
            project.red_threshold, open_issues=score.open_issues,
            open_bugs=score.open_bugs, email_count=emails,
            transcript_count=transcripts, deadline=project.deadline,
        )
        planner = DecisionSupportService.build(
            project, score, email_count=emails, transcript_count=transcripts,
            capacity_analysis=analysis,
        )
        deadline = project.deadline.replace(tzinfo=timezone.utc) if project.deadline.tzinfo is None else project.deadline
        days_to_deadline = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds() / 86400)
        dashboard = TeamCapacityService.dashboard(project.jira_capacity_snapshot, project, score, emails, transcripts)
        shap = prediction["shap_values"] or {}
        feature_importance = sorted(
            ({"feature": key, "value": value} for key, value in shap.items() if key != "base_value"),
            key=lambda row: abs(row["value"]), reverse=True,
        )
        return {
            "health_score": prediction["health_score"], "persisted_health_score": score.health_score,
            "prediction_confidence": min(100, round(55 + emails * 3 + transcripts * 4 + (15 if analysis["team_capacity"] else 0), 2)),
            "overall_sentiment": score.tone_score, "urgency": score.urgency_flag,
            "sprint_velocity": score.velocity_percent, "sprint_progress": analysis["burndown"]["sprint_completion_rate"],
            "burndown": analysis["burndown"], "burndown_deviation": analysis["burndown"]["burndown_deviation"],
            "overdue_rate": score.overdue_rate, "bug_ratio": score.bug_ratio,
            "open_issues": score.open_issues, "open_bugs": score.open_bugs,
            "blocked_issues": analysis["dependency_analysis"]["blocked_issues"],
            "critical_dependencies": analysis["dependency_analysis"]["critical_path"],
            "team_capacity": analysis["team_capacity"], "developer_workloads": analysis["team_capacity"],
            "unavailable_members": [], "remaining_story_points": analysis["burndown"]["remaining_story_points"],
            "remaining_hours": analysis["burndown"]["remaining_hours"], "days_to_deadline": round(days_to_deadline, 2),
            "deadline_confidence": dashboard["deadline_confidence"], "delivery_risk": dashboard["delivery_risk"],
            "shap_values": shap, "feature_importance": feature_importance,
            "ai_action_plan": planner["recommendations"], "recovery_plan": planner["recovery_plan"],
            "prediction_source": prediction["prediction_source"], "analysis": analysis,
            "email_count": emails, "transcript_count": transcripts,
        }

    @staticmethod
    def explain_prediction(context: dict) -> dict:
        return {"health_score": context["health_score"], "prediction_confidence": context["prediction_confidence"],
                "shap_values": context["shap_values"], "feature_importance": context["feature_importance"],
                "prediction_source": context["prediction_source"]}

    @staticmethod
    def estimate_deadline(context: dict) -> dict:
        burndown = context["burndown"]
        return {"deadline_confidence": context["deadline_confidence"], "projected_finish": burndown["projected_completion_date"],
                "burndown_deviation": burndown["burndown_deviation"], "remaining_story_points": burndown["remaining_story_points"]}

    @staticmethod
    def estimate_capacity(context: dict) -> dict:
        members = context["team_capacity"]
        overloaded_names = set(context["analysis"]["workload_distribution"]["overloaded_members"])
        return {"overloaded": [member for member in members if member["developer"] in overloaded_names],
                "available": [member for member in members if member.get("is_active_in_sprint") and member.get("capacity_percentage", 0) < 80],
                "developers": members}

    @classmethod
    def run_simulation(cls, question: str, context: dict, project, score) -> dict | None:
        text = question.casefold()
        unavailable = re.search(r"unavailable(?:\s+for)?\s+(\d+(?:\.\d+)?)\s+days?", text)
        if unavailable:
            days = float(unavailable.group(1))
            developer = next((row["developer"] for row in context["team_capacity"] if row["developer"].casefold() in text), None)
            if not developer:
                return {"missing": "No current Jira assignee was named in the question.", "type": "unavailability"}
            # Absence questions use the same task-reallocation simulation as
            # the Resource Simulator, not generic metric optimisation.
            result = AbsenceSimulator.simulate(context["analysis"], developer, days, project, score)
            return {"type": "unavailability", "input": {"developer": developer, "unavailable_days": days}, **result}

        values = {"sprint_velocity": score.velocity_percent, "bug_ratio": score.bug_ratio,
                  "overdue_rate": score.overdue_rate, "communication_sentiment": score.tone_score,
                  "urgency_flag": score.urgency_flag}
        percentage = re.search(r"(velocity|overdue(?:\s+issues?)?|bug(?:\s+ratio)?)[^\d]*(\d+(?:\.\d+)?)\s*%", text)
        if percentage:
            metric, percent = percentage.group(1), float(percentage.group(2)) / 100
            if metric == "velocity": values["sprint_velocity"] = percent
            elif metric.startswith("overdue"): values["overdue_rate"] = percent
            else: values["bug_ratio"] = percent
            simulated = AIInsightsService.simulate(project, score, values, context["email_count"], context["transcript_count"])
            return {"type": "metric", "input": values, "current_health": context["health_score"],
                    "simulated_health": simulated["predicted_score"], "health_difference": round(simulated["predicted_score"] - context["health_score"], 2),
                    "shap_values": simulated["shap_values"], "prediction_source": simulated["prediction_source"]}

        # These requests name an intervention but not a target.  Derive the
        # counterfactual strictly from the current Jira graph; the live model
        # still makes the prediction.
        blocked_fraction = len(context["blocked_issues"]) / max(1, context["open_issues"])
        if "communication" in text and any(word in text for word in ("improve", "better", "increase")):
            values["communication_sentiment"] = min(1.0, values["communication_sentiment"] + 0.25)
        elif any(word in text for word in ("blocker", "blockers", "capacity", "resolve")):
            values["sprint_velocity"] = min(1.0, values["sprint_velocity"] + (1 - values["sprint_velocity"]) * blocked_fraction)
        else:
            return None
        simulated = AIInsightsService.simulate(project, score, values, context["email_count"], context["transcript_count"])
        return {"type": "derived_metric", "input": values, "current_health": context["health_score"],
                "simulated_health": simulated["predicted_score"],
                "health_difference": round(simulated["predicted_score"] - context["health_score"], 2),
                "shap_values": simulated["shap_values"], "prediction_source": simulated["prediction_source"]}
        return None

    @staticmethod
    def generate_recommendation(context: dict) -> dict | None:
        return context["ai_action_plan"][0] if context["ai_action_plan"] else None

    @classmethod
    def answer_question(cls, question: str, project, score, db) -> dict:
        context = cls.collect_project_context(project, score, db)
        simulation = cls.run_simulation(question, context, project, score) if "what if" in question.casefold() or "unavailable" in question.casefold() else None
        capacity = cls.estimate_capacity(context)
        deadline = cls.estimate_deadline(context)
        top = context["feature_importance"][0] if context["feature_importance"] else None
        summary_values = {"health_score": context["health_score"], "deadline_confidence": context["deadline_confidence"],
                          "remaining_story_points": context["remaining_story_points"], "blocked_issue_count": len(context["blocked_issues"])}
        if simulation and "missing" not in simulation:
            summary_values.update({"simulated_health": simulation.get("simulated_health", simulation.get("predicted_health", simulation.get("simulated_health_score"))),
                                   "health_difference": simulation.get("health_difference"), "simulation_type": simulation["type"]})
        # The answer selects live evidence based on the question rather than
        # returning a canned response.  The full context remains in the
        # structured sections so clients can render/stream it without having
        # to parse prose.
        text = question.casefold()
        if "trimming low-priority" in text or "backlog scope" in text:
            low_priority = [item for item in context["analysis"]["_items"] if item.get("priority", "").casefold() in ("lowest", "low")]
            scope_points = round(sum(item.get("points", 0) for item in low_priority), 2)
            scope_hours = round(sum(item.get("remaining_seconds", 0) for item in low_priority) / 3600, 2)
            focus, focus_value = "low_priority_scope", {"issue_count": len(low_priority), "story_points": scope_points, "remaining_hours": scope_hours,
                                                           "issues": [item["key"] for item in low_priority]}
            narrative = f"The active snapshot has {len(low_priority)} low-priority issues containing {scope_points:.2f} story points and {scope_hours:.2f} remaining hours. Removing that scope reduces the current remaining backlog of {context['remaining_story_points']:.2f} story points by {scope_points:.2f}."
        elif "historical" in text or "burndown velocity" in text:
            current_velocity = context["sprint_velocity"] * 100
            historical_velocity = context["burndown"].get("historical_velocity_trend")
            focus, focus_value = "burndown_velocity", {"current_velocity_percent": current_velocity, "historical_velocity": historical_velocity,
                                                         "burndown_deviation": context["burndown_deviation"]}
            historical_text = f"{historical_velocity:.2f}" if isinstance(historical_velocity, (int, float)) else "not available from synced historical sprints"
            narrative = f"Current sprint velocity is {current_velocity:.2f}%; historical velocity is {historical_text}. The live burndown deviation is {context['burndown_deviation']} story points."
        elif "communications indicate" in text or "technical debt" in text:
            focus, focus_value = "communication_delivery_risk", {"email_count": context["email_count"], "transcript_count": context["transcript_count"],
                                                                   "sentiment": context["overall_sentiment"], "urgency": context["urgency"], "bug_ratio": context["bug_ratio"]}
            narrative = f"The current communication evidence includes {context['email_count']} emails and {context['transcript_count']} transcripts, with sentiment {context['overall_sentiment']:.2f} and urgency flag {context['urgency']}. Delivery risk is corroborated by a {context['bug_ratio']:.2%} bug ratio."
        elif "reassign blocked tasks" in text or "reallocating the most delayed" in text:
            blocked_points = context["analysis"]["dependency_analysis"]["blocked_story_points"]
            available = [member["developer"] for member in capacity["available"]]
            focus, focus_value = "task_reallocation", {"blocked_story_points": blocked_points, "blocked_issues": context["blocked_issues"], "available_developers": available}
            narrative = f"There are {blocked_points:.2f} blocked story points across {len(context['blocked_issues'])} issues. Available active-sprint capacity is held by {', '.join(available) if available else 'no developers'}, so reallocation can only proceed after dependency ownership is cleared."
        elif any(term in text for term in ("overload", "spare capacity", "more work", "who")):
            focus = "capacity"
            focus_value = {
                "overloaded": [row["developer"] for row in capacity["overloaded"]],
                "available": [row["developer"] for row in capacity["available"]],
            }
            narrative = f"Capacity analysis finds {len(focus_value['overloaded'])} overloaded and {len(focus_value['available'])} available active-sprint developers."
        elif any(term in text for term in ("deadline", "release", "schedule", "delivery")):
            focus, focus_value = "deadline", deadline
            narrative = f"The current deadline confidence is {deadline['deadline_confidence']:.2f}% with {context['days_to_deadline']:.2f} days remaining and {context['remaining_story_points']:.2f} story points still open."
        elif any(term in text for term in ("block", "dependenc", "sprint")):
            focus = "sprint_dependencies"
            focus_value = {"blocked_issues": context["blocked_issues"], "critical_dependencies": context["critical_dependencies"], "burndown_deviation": context["burndown_deviation"]}
            narrative = f"The active sprint has {len(context['blocked_issues'])} blocked issues; its burndown deviation is {context['burndown_deviation']} story points."
        else:
            focus, focus_value = "prediction", cls.explain_prediction(context)
            contributor = f"{top['feature']} ({top['value']:+.2f})" if top else "no TreeSHAP attribution available"
            narrative = f"Current XGBoost health is {context['health_score']:.2f}/100 with {context['prediction_confidence']:.2f}% context confidence; the largest model contributor is {contributor}."
        if simulation and "missing" not in simulation:
            simulated_health = simulation.get("simulated_health", simulation.get("predicted_health", simulation.get("simulated_health_score")))
            difference = simulation.get("health_difference", round(simulated_health - context["health_score"], 2))
            narrative += f" The requested counterfactual predicts {simulated_health:.2f}/100 ({difference:+.2f}) from the same current context."
        recommendation = cls.generate_recommendation(context)
        # Absence questions must be answered from the calculated workload and
        # dependency impact, never by falling through to a generic metric
        # optimiser such as "reduce overdue rate".
        if simulation and simulation.get("type") == "unavailability" and "missing" not in simulation:
            replacement = simulation.get("recommended_replacement") or "Unassigned / Offload Scope"
            blocked_points = simulation.get("blocked_story_points", 0)
            affected = simulation.get("affected_developers", [])
            focus, focus_value = "developer_unavailability", {
                "developer": simulation["input"]["developer"], "unavailable_days": simulation["input"]["unavailable_days"],
                "affected_developers": affected, "blocked_story_points": blocked_points,
            }
            action = f"Reassign {blocked_points} SP from {simulation['input']['developer']} to {replacement}."
            narrative = (f"Predicted Health: {simulation['simulated_health_score']}\n\n"
                         f"Health Change: {simulation.get('health_score_delta', 0)}\n\n"
                         f"Expected Deadline Shift: {simulation['deadline_impact_days']}\n\n"
                         f"Blocked Story Points: {blocked_points}\n\n"
                         f"Affected Downstream Developers: {', '.join(affected) if affected else 'None'}\n\n"
                         f"Recommendation: {action}")
            recommendation = {"action": action,
                              "expected_deadline_improvement_days": 0}
        return {"question": question, "answer": narrative, "summary": summary_values,
                "evidence": {"metrics": {key: context[key] for key in ("overall_sentiment", "urgency", "sprint_velocity", "sprint_progress", "overdue_rate", "bug_ratio", "open_issues", "open_bugs", "remaining_hours")},
                             "burndown": context["burndown"], "dependencies": {"blocked_issues": context["blocked_issues"], "critical_path": context["critical_dependencies"]},
                             "capacity": capacity},
                "reasoning": {"focus": focus, "focus_evidence": focus_value, "dominant_feature": top, "prediction": cls.explain_prediction(context), "deadline": deadline},
                "impact": simulation or {"delivery_risk": context["delivery_risk"]},
                "recommendation": recommendation, "context_generated_at": datetime.now(timezone.utc).isoformat()}
