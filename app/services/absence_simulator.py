"""Stateless capacity-loss and reassignment simulator for a Jira sprint snapshot."""
from __future__ import annotations
from app.ml.scorer import compute_health_score


class AbsenceSimulator:
    @staticmethod
    def simulate(analysis: dict, developer_id: str, unavailable_days: float, project, score) -> dict:
        members, items = analysis["team_capacity"], analysis["_items"]
        selected = next((m for m in members if developer_id in {m["jira_account_id"], m["developer"]}), None)
        if selected is None:
            raise ValueError("Developer is not assigned to active sprint work")
        lost_hours = max(0.0, float(unavailable_days)) * 7.0
        orphaned = [item for item in items if item["assignee"] == selected["developer"] and item["status"].lower() in {"in progress", "to do", "todo"}]
        graph = analysis["dependency_analysis"].get("graph", {})
        downstream = {child for item in orphaned for child in graph.get(item["key"], [])}
        item_by_key = {item["key"]: item for item in items}
        blocked_developers = sorted({item_by_key[key]["assignee"] for key in downstream if key in item_by_key})
        candidates = [dict(member) for member in members if member["developer"] != selected["developer"] and member["capacity_percentage"] < 75]
        candidates.sort(key=lambda member: member["available_sprint_hours"] - member["estimated_remaining_hours"], reverse=True)
        recommendations = []
        for item in sorted(orphaned, key=lambda row: row["remaining_seconds"], reverse=True):
            if not candidates: break
            candidate = candidates[0]
            hours = item["remaining_seconds"] / 3600
            candidate["estimated_remaining_hours"] += hours
            candidates.sort(key=lambda member: member["available_sprint_hours"] - member["estimated_remaining_hours"], reverse=True)
            after = candidate["estimated_remaining_hours"] / candidate["available_sprint_hours"] * 100 if candidate["available_sprint_hours"] else 100
            recommendations.append({"issue_key": item["key"], "summary": item["summary"], "story_points": item["points"],
                                    "estimated_hours": round(hours, 2), "current_assignee": selected["developer"],
                                    "recommended_assignee": candidate["developer"], "candidate_capacity_after": f"{after:.1f}%"})
        original_confidence = max(0.0, min(100.0, 100.0 - sum(m["capacity_percentage"] > 100 for m in members) * 10))
        lost_fraction = lost_hours / max(1.0, selected["available_sprint_hours"])
        simulated = compute_health_score(score.tone_score, score.urgency_flag,
                                         max(0.0, score.velocity_percent * (1 - lost_fraction)), score.overdue_rate, score.bug_ratio,
                                         project.green_threshold, project.red_threshold, open_issues=score.open_issues, open_bugs=score.open_bugs)
        deadline_delay = round(float(unavailable_days) * .5 + sum(item["points"] for item in orphaned) * .2, 1)
        recovery = max(15, min(95, int(100 - sum(item["points"] for item in orphaned) * 4)))
        simulated_confidence = max(10, 85 - int(deadline_delay * 6))
        impacted_points = round(sum(item["points"] for item in orphaned), 2)
        replacement = recommendations[0]["recommended_assignee"] if recommendations else "Unassigned / Offload Scope"
        impact = {"original_health_score": score.health_score, "simulated_health_score": simulated["health_score"],
                  "original_deadline_confidence": round(original_confidence, 2), "simulated_deadline_confidence": round(max(0, original_confidence - lost_fraction * 100), 2),
                  "impacted_story_points": impacted_points, "blocked_downstream_developers": blocked_developers}
        return {"status": "success", "is_simulated": True,
                "simulation_params": {"developer_id": selected["jira_account_id"], "developer_name": selected["developer"], "unavailable_days": unavailable_days},
                "impact_summary": impact, "recommendations": recommendations,
                "simulated_recovery_probability": recovery,
                # Flat aliases let existing state handlers update their metric
                # cards without decoding a nested simulation payload.
                "simulated_health_score": simulated["health_score"], "predicted_health": simulated["health_score"],
                "health_score_delta": f"{simulated['health_score'] - score.health_score:.1f}",
                "deadline_impact_days": f"+{deadline_delay} days", "deadline_delay_days": deadline_delay,
                "deadline_confidence": f"{simulated_confidence}%", "blocked_story_points": impacted_points,
                "affected_developers": blocked_developers or ["None"], "recovery_probability": recovery,
                "recommended_replacement": replacement,
                "recommendation_text": f"Reassign {impacted_points} SP from {selected['developer']} to {replacement} to mitigate a {deadline_delay}-day sprint delay."}
