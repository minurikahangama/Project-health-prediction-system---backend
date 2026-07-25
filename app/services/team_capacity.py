"""Live Jira team-capacity and dependency analytics.

All measures are calculated per request from the current Jira sprint snapshot.
No capacity, availability or dependency result is stored in PHPS.
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import sqrt

from app.ml.scorer import compute_health_score


def _number(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _working_days(start: datetime, end: datetime) -> int:
    """Inclusive weekday capacity between two UTC dates."""
    if end.date() < start.date():
        return 0
    return sum(1 for offset in range((end.date() - start.date()).days + 1)
               if (start.date() + timedelta(days=offset)).weekday() < 5)


class TeamCapacityService:
    """Transforms a live Jira snapshot into capacity and risk evidence."""

    @classmethod
    def validate_snapshot(cls, snapshot: dict) -> dict:
        """Validate Jira-derived dashboard inputs before a sync is published."""
        errors = []
        if not snapshot.get("story_points_field"):
            errors.append("Jira snapshot is missing the configured story-points field")
        issues = snapshot.get("issues")
        if not isinstance(issues, list):
            errors.append("Jira snapshot issues must be a list")
            issues = []
        keys = [row.get("key") for row in issues]
        if any(not key for key in keys):
            errors.append("One or more Jira sprint issues has no key")
        if len(keys) != len(set(keys)):
            errors.append("Jira sprint issue keys are not unique")
        try:
            analysis = cls.analyse(snapshot)
            for member in analysis["team_capacity"]:
                if member["estimated_remaining_hours"] < 0 or member["time_spent_hours"] < 0:
                    errors.append(f"Invalid worklog estimate for {member['developer']}")
            for source, target in analysis["dependency_analysis"]["dependency_chains"]:
                if source not in {item["key"] for item in analysis["_items"]} or target not in {item["key"] for item in analysis["_items"]}:
                    errors.append(f"Dependency edge {source}->{target} does not match active Jira issues")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"Could not calculate Jira capacity metrics: {exc}")
        return {"valid": not errors, "errors": errors}

    @staticmethod
    def _issue(issue: dict, points_field: str, role_field: str | None = None) -> dict:
        fields = issue.get("fields", {})
        tracking = fields.get("timetracking") or {}
        assignee = fields.get("assignee") or {}
        status = fields.get("status") or {}
        return {
            "key": issue.get("key", str(issue.get("id", "unknown"))),
            "summary": fields.get("summary") or issue.get("key", str(issue.get("id", "unknown"))),
            "assignee": assignee.get("displayName") or assignee.get("accountId") or "Unassigned",
            "jira_account_id": assignee.get("accountId") or assignee.get("displayName") or "Unassigned",
            "email": assignee.get("emailAddress"),
            "avatar_url": (assignee.get("avatarUrls") or {}).get("48x48") or (assignee.get("avatarUrls") or {}).get("24x24"),
            "points": _number(fields.get(points_field)),
            "remaining_seconds": _number(tracking.get("remainingEstimateSeconds")),
            "original_seconds": _number(tracking.get("originalEstimateSeconds")),
            "spent_seconds": _number(tracking.get("timeSpentSeconds")),
            "priority": (fields.get("priority") or {}).get("name") or "Unspecified",
            "issue_type": (fields.get("issuetype") or {}).get("name") or "Unspecified",
            "status": status.get("name") or "Unspecified",
            "done": ((status.get("statusCategory") or {}).get("key") == "done"),
            "due_date": fields.get("duedate"),
            "created_date": fields.get("created"),
            "resolution_date": fields.get("resolutiondate"),
            "parent": (fields.get("parent") or {}).get("key"),
            "epic": (fields.get("epic") or {}).get("key") or fields.get("customfield_10014"),
            "links": fields.get("issuelinks") or [],
            "reporter": ((fields.get("reporter") or {}).get("displayName")),
            "labels": fields.get("labels") or [],
            "components": [component.get("name") for component in fields.get("components") or []],
            "comment_count": _number((fields.get("comment") or {}).get("total")),
            "subtask_count": len(fields.get("subtasks") or []),
            "role": fields.get(role_field) if role_field else None,
        }

    @staticmethod
    def _block_edges(items: list[dict]) -> list[tuple[str, str]]:
        """Return predecessor -> dependent edges from Jira link metadata."""
        known = {item["key"] for item in items}
        edges = set()
        for item in items:
            for link in item["links"]:
                typ = link.get("type") or {}
                inward = str(typ.get("inward", "")).lower()
                outward = str(typ.get("outward", "")).lower()
                if any(term in outward for term in ("blocks", "depends on")) and (target := link.get("outwardIssue", {}).get("key")) in known:
                    edges.add((item["key"], target))
                if any(term in inward for term in ("blocks", "depends on")) and (source := link.get("inwardIssue", {}).get("key")) in known:
                    edges.add((source, item["key"]))
            # Jira's parent field is a dependency relation even when an
            # instance has no explicit issue-link record for it.
            if item["parent"] in known:
                edges.add((item["parent"], item["key"]))
        return sorted(edges)

    @staticmethod
    def _delivery_risk(analysis: dict, score, project) -> dict:
        """Calculate a 0-100 risk score from normalised current evidence."""
        now = datetime.now(timezone.utc)
        dependency, burndown = analysis["dependency_analysis"], analysis["burndown"]
        members, distribution = analysis["team_capacity"], analysis["workload_distribution"]
        active_count = max(1, len(analysis["_items"]))
        project_deadline = getattr(project, "deadline", None)
        deadline = (project_deadline.replace(tzinfo=timezone.utc) if project_deadline and project_deadline.tzinfo is None else project_deadline)
        days = max(0.0, (deadline - now).total_seconds() / 86400) if deadline else 0.0
        overdue = sum(1 for item in analysis["_items"] if _parse_date(item["due_date"]) and _parse_date(item["due_date"]) < now)
        remaining_hours = burndown["remaining_hours"] or 0.0
        available_hours = sum(member["available_sprint_hours"] for member in members)
        critical_fraction = len(dependency["critical_path"]) / active_count
        components = {
            "sprint_velocity": 1 - max(0.0, min(1.0, float(score.velocity_percent))),
            "burndown_deviation": max(0.0, float(burndown["burndown_deviation"] or 0.0)) / max(1.0, float(burndown["remaining_story_points"] or 0.0)),
            "overdue_rate": max(0.0, min(1.0, float(score.overdue_rate))),
            "bug_ratio": max(0.0, min(1.0, float(score.bug_ratio))),
            "blocked_story_points": min(1.0, dependency["blocked_story_points"] / max(1.0, float(burndown["remaining_story_points"] or 0.0))),
            "critical_dependencies": min(1.0, critical_fraction),
            "team_capacity": min(1.0, len(distribution["overloaded_members"]) / max(1, len(members))),
            "remaining_work": min(1.0, remaining_hours / max(1.0, available_hours)),
            "deadline_pressure": min(1.0, remaining_hours / max(1.0, days * 8 * max(1, len(members)))),
            "communication_sentiment": 1 - max(0.0, min(1.0, (float(score.tone_score) + 1) / 2)),
            "urgency": float(bool(score.urgency_flag)),
        }
        ordered = sorted(components.items(), key=lambda item: item[1], reverse=True)
        value = round(sum(components.values()) / len(components) * 100, 2)
        return {"score": value,
                "level": "Critical" if value >= 80 else "High" if value >= 60 else "Medium" if value >= 30 else "Low",
                "top_contributors": [{"factor": name, "normalised_risk": round(amount, 4)} for name, amount in ordered[:5]],
                "components": {name: round(amount, 4) for name, amount in components.items()},
                "overdue_issues": overdue, "blocked_issues": len(dependency["blocked_issues"]),
                "overloaded_developers": len(distribution["overloaded_members"])}

    @staticmethod
    def _cycles(graph: dict[str, list[str]]) -> list[list[str]]:
        cycles, path, active, done = [], [], set(), set()
        def visit(node):
            if node in active:
                cycles.append(path[path.index(node):] + [node])
                return
            if node in done:
                return
            active.add(node); path.append(node)
            for child in graph[node]: visit(child)
            path.pop(); active.remove(node); done.add(node)
        for node in graph: visit(node)
        return cycles

    @staticmethod
    def _critical_path(graph: dict[str, list[str]], items: dict[str, dict]) -> list[str]:
        """Longest dependency chain by Jira remaining estimate, for a DAG."""
        memo, visiting = {}, set()
        def longest(node):
            if node in memo: return memo[node]
            if node in visiting: return (0.0, [])
            visiting.add(node)
            children = [longest(child) for child in graph[node]] or [(0.0, [])]
            child_weight, child_path = max(children, key=lambda row: row[0])
            visiting.remove(node)
            result = (items[node]["remaining_seconds"] + child_weight, [node] + child_path)
            memo[node] = result
            return result
        return max((longest(node) for node in graph), default=(0.0, []), key=lambda row: row[0])[1]

    @classmethod
    def analyse(cls, snapshot: dict, overload_threshold: float | None = None) -> dict:
        role_field = snapshot.get("developer_role_field")
        sprint_items = [cls._issue(row, snapshot["story_points_field"], role_field) for row in snapshot["issues"]]
        # Finished work remains in the active-sprint scope solely for the
        # burndown baseline; capacity and dependency risks use active work.
        items = [item for item in sprint_items if not item["done"]]
        by_key = {item["key"]: item for item in items}
        edges = cls._block_edges(items)
        graph = {key: [] for key in by_key}
        for source, target in edges: graph[source].append(target)
        blocked_keys = {target for _, target in edges}
        for item in items: item["blocked"] = item["key"] in blocked_keys
        sprint = snapshot.get("active_sprint", {})
        start, end = _parse_date(sprint.get("start_date")), _parse_date(sprint.get("end_date"))
        now = datetime.now(timezone.utc)

        team = defaultdict(lambda: {"story_points": 0.0, "remaining_seconds": 0.0, "remaining_tasks": 0, "high_priority_tasks": 0, "blocked_tasks": 0, "jira_account_id": "", "email": None, "avatar_url": None})
        for item in items:
            row = team[item["assignee"]]
            row["jira_account_id"] = item["jira_account_id"]
            row["email"] = item["email"] or row["email"]
            row["avatar_url"] = item["avatar_url"] or row["avatar_url"]
            row["story_points"] += item["points"]
            row["remaining_seconds"] += item["remaining_seconds"]
            row["remaining_tasks"] += 1
            row["high_priority_tasks"] += int(item["priority"].lower() in {"highest", "high"})
            row["blocked_tasks"] += int(item["blocked"])
        workloads = [row["remaining_seconds"] for row in team.values()]
        average_workload = sum(workloads) / len(workloads) if workloads else 0.0
        threshold = float(overload_threshold if overload_threshold is not None else os.getenv("PHPS_CAPACITY_OVERLOAD_THRESHOLD", "90"))
        remaining_work_days = _working_days(max(now, start), end) if start and end else 0
        # Capacity intelligence uses the documented seven productive hours
        # per remaining sprint day (not a nominal eight-hour calendar day).
        hours_per_day = _number(os.getenv("PHPS_CAPACITY_HOURS_PER_DAY", "7")) or 7.0
        available_hours = remaining_work_days * hours_per_day
        members = []
        for developer, row in sorted(team.items()):
            workload = (row["remaining_seconds"] / average_workload * 100) if average_workload else 0.0
            remaining_hours = row["remaining_seconds"] / 3600
            capacity_pct = (remaining_hours / available_hours * 100) if available_hours else (100.0 if remaining_hours else 0.0)
            risk = "High" if capacity_pct > 100 else "Medium" if capacity_pct >= 50 else "Low"
            recommendation = ("Reassign work to an available teammate" if capacity_pct >= 100
                              else "Resolve blocked work before assigning more" if row["blocked_tasks"]
                              else "Capacity is within the remaining sprint window")
            members.append({
                "developer": developer, "current_story_points": round(row["story_points"], 2),
                "role": None, "assigned_issues": row["remaining_tasks"],
                "is_active_in_sprint": True,
                "jira_account_id": row["jira_account_id"], "email": row["email"], "avatar_url": row["avatar_url"],
                "assigned_story_points": round(row["story_points"], 2),
                "completed_story_points": 0.0,
                "remaining_story_points": round(row["story_points"], 2),
                "estimated_remaining_hours": round(remaining_hours, 2),
                "original_estimate_hours": round(row["remaining_seconds"] / 3600, 2),
                "time_spent_hours": 0.0,
                "remaining_estimate_hours": round(remaining_hours, 2),
                "available_sprint_hours": round(available_hours, 2),
                "capacity_percentage": round(capacity_pct, 2), "risk": risk,
                "status": "Overloaded" if risk == "High" else "Balanced" if risk == "Medium" else "Underutilized / Available",
                "recommendation": recommendation,
                "remaining_tasks": row["remaining_tasks"], "high_priority_tasks": row["high_priority_tasks"],
                "blocked_tasks": row["blocked_tasks"], "workload_percentage": round(workload, 2),
            })
        capacities = [member["capacity_percentage"] for member in members]
        mean = sum(capacities) / len(capacities) if capacities else 0.0
        deviation = sqrt(sum((value - mean) ** 2 for value in capacities) / len(capacities)) if capacities else 0.0
        # A member may be imbalanced relative to the team before their
        # absolute remaining-sprint capacity is exhausted; expose both cases.
        overloaded = [member["developer"] for member in members if member["capacity_percentage"] > threshold or member["workload_percentage"] > threshold]
        underutilized = [member["developer"] for member in members if member["workload_percentage"] < mean - deviation]

        total_points = sum(item["points"] for item in sprint_items)
        completed_points = sum(item["points"] for item in sprint_items if item["done"])
        remaining_points = total_points - completed_points
        total_seconds = sum(item["original_seconds"] or item["remaining_seconds"] for item in sprint_items)
        remaining_seconds = sum(item["remaining_seconds"] for item in items)
        duration = (end - start).total_seconds() if start and end and end > start else None
        elapsed = min(max((now - start).total_seconds(), 0), duration) if duration and start else None
        ideal_remaining = total_points * (1 - elapsed / duration) if duration else None
        actual_remaining = remaining_points
        completion_rate = completed_points / total_points if total_points else 0.0
        projected_completion = None
        if start and elapsed and completed_points > 0:
            point_rate = completed_points / elapsed
            projected_completion = now + timedelta(seconds=remaining_points / point_rate) if point_rate else None

        historical_sprints = defaultdict(float)
        sprint_field = os.getenv("JIRA_SPRINT_FIELD", "customfield_10020")
        for raw in snapshot.get("historical_completed_issues", []):
            item = cls._issue(raw, snapshot["story_points_field"])
            value = raw.get("fields", {}).get(sprint_field) or raw.get("fields", {}).get("sprint")
            for sprint in value if isinstance(value, list) else [value]:
                if isinstance(sprint, dict) and sprint.get("state", "").upper() == "CLOSED":
                    historical_sprints[str(sprint.get("id") or sprint.get("name"))] += item["points"]
        velocity_trend = (sum(historical_sprints.values()) / len(historical_sprints)) if historical_sprints else None
        blocked = [by_key[key] for key in blocked_keys]
        affected = sorted({item["assignee"] for item in blocked})
        # Completed estimates are deliberately counted per developer too so
        # every table column is sourced from the same Jira sprint snapshot.
        completed_by_developer = defaultdict(lambda: {"points": 0.0, "spent": 0.0, "original": 0.0})
        for item in sprint_items:
            if item["done"]:
                done = completed_by_developer[item["assignee"]]
                done["points"] += item["points"]; done["spent"] += item["spent_seconds"]; done["original"] += item["original_seconds"]
        for member in members:
            done = completed_by_developer[member["developer"]]
            member["completed_story_points"] = round(done["points"], 2)
            member["assigned_story_points"] = round(member["current_story_points"] + done["points"], 2)
            member["time_spent_hours"] = round((done["spent"] + (member["original_estimate_hours"] * 3600 - member["estimated_remaining_hours"] * 3600)) / 3600, 2)
            owned = [item for item in sprint_items if item["assignee"] == member["developer"]]
            roles = [item["role"] for item in owned if item["role"]]
            member["role"] = str(roles[0]) if roles else None
            completed = [item for item in owned if item["done"]]
            lead_days = [( _parse_date(item["resolution_date"]) - _parse_date(item["created_date"])).total_seconds() / 86400 for item in completed if _parse_date(item["resolution_date"]) and _parse_date(item["created_date"])]
            member["velocity"] = round(done["points"], 2)
            member["average_completion_rate"] = round(len(completed) / len(owned), 4) if owned else 0.0
            member["average_lead_time_days"] = round(sum(lead_days) / len(lead_days), 2) if lead_days else None
            member["average_cycle_time_days"] = member["average_lead_time_days"]
            member["blocked_work"] = member["blocked_tasks"]
            member["critical_tasks"] = sum(1 for item in owned if item["priority"].casefold() in {"highest", "high"})
            member["late_tasks"] = sum(1 for item in owned if _parse_date(item["due_date"]) and _parse_date(item["due_date"]) < now and not item["done"])
            member["upcoming_due_tasks"] = sum(1 for item in owned if _parse_date(item["due_date"]) and now <= _parse_date(item["due_date"]) <= now + timedelta(days=3))
            member["dependency_count"] = sum(1 for source, target in edges if source in {item["key"] for item in owned} or target in {item["key"] for item in owned})

        # The active sprint drives health, burndown and capacity.  The board
        # roster is broader: Jira users with assigned backlog/future/completed
        # work must remain visible in the developer table rather than silently
        # disappearing just because they have no active-sprint assignment.
        project_items = [cls._issue(row, snapshot["story_points_field"], role_field)
                         for row in snapshot.get("all_issues", [])]
        present = {member["developer"] for member in members}
        project_by_developer = defaultdict(list)
        for item in project_items:
            project_by_developer[item["assignee"]].append(item)
        for developer, owned in sorted(project_by_developer.items()):
            if developer in present:
                continue
            done = [item for item in owned if item["done"]]
            unresolved = [item for item in owned if not item["done"]]
            roles = [item["role"] for item in owned if item["role"]]
            members.append({
                "developer": developer,
                "jira_account_id": owned[0]["jira_account_id"], "email": owned[0]["email"], "avatar_url": owned[0]["avatar_url"],
                "role": str(roles[0]) if roles else None, "is_active_in_sprint": False,
                "assigned_issues": len(owned), "current_story_points": 0.0, "assigned_story_points": round(sum(item["points"] for item in owned), 2),
                "completed_story_points": round(sum(item["points"] for item in done), 2), "remaining_story_points": 0.0,
                "estimated_remaining_hours": 0.0, "original_estimate_hours": round(sum(item["original_seconds"] for item in owned) / 3600, 2),
                "time_spent_hours": round(sum(item["spent_seconds"] for item in owned) / 3600, 2), "remaining_estimate_hours": 0.0,
                "available_sprint_hours": round(available_hours, 2), "capacity_percentage": 0.0, "workload_percentage": 0.0,
                "remaining_tasks": 0, "high_priority_tasks": 0, "blocked_tasks": 0, "blocked_work": 0,
                "velocity": round(sum(item["points"] for item in done), 2), "average_completion_rate": round(len(done) / len(owned), 4) if owned else 0.0,
                "average_lead_time_days": None, "average_cycle_time_days": None,
                "critical_tasks": sum(1 for item in unresolved if item["priority"].casefold() in {"highest", "high"}),
                "late_tasks": sum(1 for item in unresolved if _parse_date(item["due_date"]) and _parse_date(item["due_date"]) < now),
                "upcoming_due_tasks": sum(1 for item in unresolved if _parse_date(item["due_date"]) and now <= _parse_date(item["due_date"]) <= now + timedelta(days=3)),
                "dependency_count": 0, "risk": "Low", "status": "No active sprint work",
                "recommendation": "No active-sprint work is assigned; Jira board history is shown for roster visibility.",
            })

        no_active_sprint = not sprint.get("name")
        trajectory = []
        if not no_active_sprint and start and end:
            total_days = max(1.0, (end - start).total_seconds() / 86400)
            elapsed_ratio = min(1.0, max(0.0, (now - start).total_seconds() / 86400 / total_days))
            trajectory = [
                {"date": start.isoformat(), "ideal": round(total_points, 2), "actual": round(total_points, 2), "predicted": round(total_points, 2)},
                {"date": now.isoformat(), "ideal": round(ideal_remaining, 2), "actual": round(actual_remaining, 2), "predicted": round(max(0.0, actual_remaining - (velocity_trend or 0) * elapsed_ratio), 2)},
                {"date": end.isoformat(), "ideal": 0.0, "actual": None, "predicted": round(max(0.0, remaining_points - (velocity_trend or 0) * max(0.0, 1 - elapsed_ratio)), 2)},
            ]
        burndown = {"status": "No active sprint found" if no_active_sprint else "ok",
            "ideal_remaining_story_points": round(ideal_remaining, 2) if ideal_remaining is not None else None, "actual_remaining_story_points": round(actual_remaining, 2) if not no_active_sprint else None, "predicted_remaining_story_points": round(max(0, remaining_points - (velocity_trend or 0) * (elapsed / duration if duration else 0)), 2) if not no_active_sprint else None, "burndown_deviation": round(actual_remaining - ideal_remaining, 2) if ideal_remaining is not None else None, "sprint_completion_rate": round(completion_rate, 4) if not no_active_sprint else None, "remaining_story_points": round(remaining_points, 2) if not no_active_sprint else None, "remaining_hours": round(remaining_seconds / 3600, 2) if not no_active_sprint else None, "projected_completion_date": projected_completion.isoformat() if projected_completion else None, "historical_velocity_trend": round(velocity_trend, 2) if velocity_trend is not None else None,
            "expected_remaining_days": round(max(0, (end - now).total_seconds() / 86400), 2) if end and not no_active_sprint else None, "trajectory": trajectory}
        return {
            "active_sprint": sprint, "team_capacity": members,
            "workload_distribution": {"average_team_capacity": round(mean, 2), "standard_deviation": round(deviation, 2), "overload_threshold": threshold, "overloaded_members": overloaded, "underutilized_members": underutilized},
            "dependency_analysis": {"blocked_issues": sorted(blocked_keys), "affected_developers": affected, "blocked_story_points": round(sum(item["points"] for item in blocked), 2), "blocked_estimated_hours": round(sum(item["remaining_seconds"] for item in blocked) / 3600, 2), "dependency_chains": [[source, target] for source, target in edges], "graph": {key: list(value) for key, value in graph.items()}, "circular_dependencies": cls._cycles(graph), "critical_path": cls._critical_path(graph, by_key), "blocking_developers": sorted({by_key[source]["assignee"] for source, _ in edges}), "blocking_epics": sorted({by_key[source]["epic"] for source, _ in edges if by_key[source]["epic"]}), "blocking_sprint": sprint.get("name")},
            "burndown": burndown,
            "_items": items, "_total_remaining_seconds": remaining_seconds,
        }

    @classmethod
    def dashboard(cls, snapshot: dict, project, score, email_count: int = 0, transcript_count: int = 0) -> dict:
        """One frontend-ready payload built solely from the synced Jira snapshot and model output."""
        analysis = cls.analyse(snapshot)
        dependency, burndown, members = analysis["dependency_analysis"], analysis["burndown"], analysis["team_capacity"]
        risk = cls._delivery_risk(analysis, score, project)
        confidence = round(100 - risk["score"], 2)
        return {
            "project_id": project.id, "generated_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_at": snapshot.get("fetched_at"), "active_sprint": analysis["active_sprint"],
            "executive_summary": {"health_score": score.health_score, "deadline_confidence": confidence, "delivery_risk": risk["score"], "remaining_story_points": burndown["remaining_story_points"], "remaining_hours": burndown["remaining_hours"], "risk_signal_count": sum(value > 0 for value in risk["components"].values())},
            "health_score": {"value": score.health_score, "rag_status": score.rag_status, "prediction_source": "xgboost", "tone_score": score.tone_score, "urgency_flag": score.urgency_flag},
            "deadline_confidence": confidence,
            "delivery_risk": risk,
            "capacity_metrics": analysis["workload_distribution"], "developer_capacity": members,
            "dependency_graph": dependency, "burndown": burndown,
            # Compatibility aliases keep existing dashboard clients data-only
            # while they migrate to the descriptive section names above.
            "team_capacity": members, "workload_distribution": analysis["workload_distribution"],
            "dependency_analysis": dependency,
            "ai_insights": {"sentiment": score.tone_score, "communications_analysed": email_count + transcript_count, "critical_path": dependency["critical_path"], "overloaded_developers": analysis["workload_distribution"]["overloaded_members"]},
            "chat_context": {"health": score.health_score, "velocity": score.velocity_percent, "overdue_rate": score.overdue_rate, "blocked_story_points": dependency["blocked_story_points"], "critical_path": dependency["critical_path"], "developer_capacity": [{"developer": m["developer"], "capacity_percentage": m["capacity_percentage"], "risk": m["risk"]} for m in members]},
        }

    @classmethod
    def simulate_unavailability(cls, analysis: dict, developer: str, unavailable_days: float, project, score) -> dict:
        members = analysis["team_capacity"]
        selected = next((member for member in members if member["developer"] == developer), None)
        if not selected:
            raise ValueError("Developer is not assigned to active Jira work")
        sprint = analysis["active_sprint"]
        start, end = _parse_date(sprint.get("start_date")), _parse_date(sprint.get("end_date"))
        if not start or not end or end <= start:
            raise ValueError("Active sprint dates are required for an availability simulation")
        sprint_days = (end - start).total_seconds() / 86400
        unavailable_fraction = min(1.0, unavailable_days / sprint_days)
        total_seconds = analysis["_total_remaining_seconds"]
        removed_seconds = selected["estimated_remaining_hours"] * 3600 * unavailable_fraction
        remaining_capacity = max(0.0, total_seconds - removed_seconds)
        velocity = max(0.0, min(1.0, float(score.velocity_percent) * (remaining_capacity / total_seconds if total_seconds else 1.0)))
        projected = _parse_date(analysis["burndown"]["projected_completion_date"])
        delayed = projected + timedelta(days=unavailable_days) if projected else None
        deadline_delay = max(0.0, (delayed - end).total_seconds() / 86400) if delayed else None
        predicted = compute_health_score(score.tone_score, score.urgency_flag, velocity, score.overdue_rate, score.bug_ratio, project.green_threshold, project.red_threshold, open_issues=score.open_issues, open_bugs=score.open_bugs, deadline=project.deadline)
        blocked = [item for item in analysis["_items"] if item["blocked"] and item["assignee"] == developer]
        risk = predicted["rag_status"]
        selected_items = [item for item in analysis["_items"] if item["assignee"] == developer]
        delayed_points = sum(item["points"] for item in selected_items) * unavailable_fraction
        graph = analysis["dependency_analysis"].get("graph", {})
        downstream, pending = set(), [item["key"] for item in selected_items]
        while pending:
            key = pending.pop()
            for child in graph.get(key, []):
                if child not in downstream:
                    downstream.add(child); pending.append(child)
        by_key = {item["key"]: item for item in analysis["_items"]}
        affected_keys = sorted(downstream | {item["key"] for item in blocked})
        affected_developers = sorted({by_key[key]["assignee"] for key in affected_keys if key in by_key})
        replacements = sorted((member for member in members if member["developer"] != developer and member["is_active_in_sprint"]), key=lambda member: member["capacity_percentage"])
        replacement = next((member for member in replacements if member["capacity_percentage"] < 80), None)
        simulated_risk = min(100.0, max(0.0, 100 - predicted["health_score"] + (deadline_delay or 0) * 5))
        return {"developer": developer, "unavailable_days": unavailable_days, "remaining_capacity_hours": round(remaining_capacity / 3600, 2), "current_capacity_percentage": selected.get("capacity_percentage", 0.0), "new_capacity_percentage": round(selected.get("capacity_percentage", 0.0) + unavailable_fraction * 100, 2), "velocity": round(velocity, 4), "projected_sprint_completion": delayed.isoformat() if delayed else None, "sprint_delay_days": round(deadline_delay, 2) if deadline_delay is not None else None, "deadline_delay_days": round(deadline_delay, 2) if deadline_delay is not None else None, "expected_delay_days": round(deadline_delay, 2) if deadline_delay is not None else None, "predicted_health": predicted["health_score"], "health_prediction_source": predicted["prediction_source"], "shap_values": predicted["shap_values"], "deadline_confidence": round(100 - simulated_risk, 2), "delivery_risk": round(simulated_risk, 2), "recovery_probability": round(max(0, 100 - (deadline_delay or 0) * 10), 2), "blocked_story_points": round(sum(item["points"] for item in blocked), 2), "delayed_story_points": round(delayed_points, 2), "tasks_to_move": [item["key"] for item in selected_items], "redistribution": {"suggested_replacement": replacement["developer"] if replacement else None, "work_to_move_hours": round(removed_seconds / 3600, 2)}, "dependency_impact": affected_keys, "affected_developers": affected_developers, "risk_level": risk}
