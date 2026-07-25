"""Developer allocation facade for Team Capacity & Delivery Intelligence."""
from __future__ import annotations
from app.services.team_capacity import TeamCapacityService


class CapacityService:
    HOURS_PER_DAY = 7.0

    @classmethod
    def analyse(cls, jira_snapshot: dict) -> dict:
        """Return live per-developer allocation using the 7h/day contract."""
        result = TeamCapacityService.analyse(jira_snapshot)
        for member in result["team_capacity"]:
            available = member["available_sprint_hours"]
            remaining = member["estimated_remaining_hours"]
            allocation = round(remaining / available * 100, 2) if available else (100.0 if remaining else 0.0)
            member.update({"available_hours": available, "remaining_estimated_work_hours": remaining,
                           "allocation_ratio": allocation,
                           "risk": "High" if allocation > 100 else "Medium" if allocation >= 50 else "Low",
                           "ui_color": "Red" if allocation > 100 else "Orange/Yellow" if allocation >= 50 else "Green"})
        return result
