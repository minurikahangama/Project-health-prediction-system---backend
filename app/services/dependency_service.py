"""Dependency DAG and critical-blocker extraction for active Jira sprint work."""
from __future__ import annotations


class DependencyService:
    @staticmethod
    def analyse(capacity_analysis: dict) -> dict:
        items = {item["key"]: item for item in capacity_analysis.get("_items", [])}
        graph = capacity_analysis["dependency_analysis"].get("graph", {})
        blockers = [items[key] for key, children in graph.items()
                    if children and key in items and items[key]["remaining_seconds"] > 0]
        waiting = [items[key] for blocker in blockers for key in graph.get(blocker["key"], []) if key in items]
        return {"dag": graph,
                "critical_path_blockers": [{"issue_key": item["key"], "assignee": item["assignee"],
                                            "remaining_hours": round(item["remaining_seconds"] / 3600, 2),
                                            "downstream_issue_count": len(graph.get(item["key"], []))} for item in blockers],
                "blocked_story_points": round(sum(item["points"] for item in blockers), 2),
                "waiting_time_hours": round(sum(item["remaining_seconds"] for item in waiting) / 3600, 2)}
