"""Jira Cloud delivery-signal extraction.

The project setting must be a Jira Cloud project URL, for example
``https://example.atlassian.net/jira/software/projects/PHPS``.  Only issues
in that project are queried; a failed Jira request raises an error instead of
quietly publishing placeholder health data.
"""
import logging
import os
import re
from datetime import date, datetime, timezone
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import httpx

from app.utils.encryption import decrypt_token

logger = logging.getLogger(__name__)
_PROJECT_URL = re.compile(r"/projects/([A-Z][A-Z0-9_]+)(?:/|$)", re.IGNORECASE)


class JiraSignalError(RuntimeError):
    """Raised when Jira metrics cannot be fetched safely."""


def _cloud_base_and_project_key(jira_project_url: str) -> Tuple[str, str]:
    """Turn a Jira Cloud project URL into its site base URL and project key."""
    parsed = urlparse(jira_project_url.strip())
    if parsed.scheme != "https" or not parsed.netloc.endswith(".atlassian.net"):
        raise JiraSignalError("Jira Cloud project URL must use https://<site>.atlassian.net/.../projects/<KEY>")
    match = _PROJECT_URL.search(parsed.path)
    if not match:
        raise JiraSignalError("Jira URL must include the project key, for example .../projects/PHPS")
    return f"{parsed.scheme}://{parsed.netloc}", match.group(1).upper()


def _is_done(issue: dict) -> bool:
    category = (issue.get("fields", {}).get("status", {}).get("statusCategory", {}) or {})
    return category.get("key") == "done"


def _story_points(issue: dict, field_id: str) -> float:
    value = issue.get("fields", {}).get(field_id)
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _work_units(issues: List[dict], field_id: str) -> tuple[float, bool]:
    """Return story-point work, falling back to issue count when unestimated.

    Jira does not require a story-point field.  Treating an unestimated sprint
    as a neutral 50% velocity hid real delivery progress; issue count is the
    only project data available in that case and keeps the ratio auditable.
    """
    points = [_story_points(issue, field_id) for issue in issues]
    if any(points):
        return sum(points), True
    return float(len(issues)), False


def _sprint_window(issues: List[dict], sprint_field: str) -> tuple[datetime | None, datetime | None]:
    for issue in issues:
        raw = issue.get("fields", {}).get(sprint_field) or issue.get("fields", {}).get("sprint")
        for sprint in raw if isinstance(raw, list) else [raw]:
            if not isinstance(sprint, dict) or str(sprint.get("state", "ACTIVE")).upper() != "ACTIVE":
                continue
            try:
                start = datetime.fromisoformat(sprint["startDate"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(sprint["endDate"].replace("Z", "+00:00"))
                return start, end
            except (KeyError, TypeError, ValueError):
                continue
    return None, None


def _search_all(client: httpx.Client, base: str, auth: tuple, jql: str, fields: List[str]) -> List[dict]:
    """Page through Jira Cloud's current enhanced JQL search endpoint."""
    issues: List[dict] = []
    next_page_token = None
    while True:
        body = {"jql": jql, "fields": fields, "maxResults": 100}
        if next_page_token:
            body["nextPageToken"] = next_page_token
        response = client.post(
            f"{base}/rest/api/3/search/jql",
            auth=auth,
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
        page = payload.get("issues", [])
        issues.extend(page)
        next_page_token = payload.get("nextPageToken")
        if next_page_token:
            continue
        if payload.get("isLast") or not page:
            return issues
        # The total check keeps the helper compatible with test doubles and
        # older Jira responses while production uses nextPageToken above.
        if "total" in payload and len(issues) >= payload["total"]:
            return issues
        # No continuation marker means this is the only page.
        return issues


def fetch_jira_signals(
    jira_url: str,
    encrypted_jira_token: str,
    jira_email: str,
) -> Dict[str, float]:
    """Fetch normalised project-scoped delivery signals from Jira Cloud.

    ``JIRA_STORY_POINTS_FIELD`` can override the common Jira Cloud default
    ``customfield_10016`` when a site uses another story-points field.
    """
    if not jira_url or not encrypted_jira_token or not jira_email:
        raise JiraSignalError("Jira URL, account email, and API token are required")

    base, project_key = _cloud_base_and_project_key(jira_url)
    raw_token = None
    raw_token = decrypt_token(encrypted_jira_token)
    story_points_field = os.getenv("JIRA_STORY_POINTS_FIELD", "customfield_10016")
    sprint_field = os.getenv("JIRA_SPRINT_FIELD", "customfield_10020")
    auth = (jira_email, raw_token)
    project_jql = f'project = "{project_key}"'

    try:
        with httpx.Client(headers={"Accept": "application/json"}, timeout=30.0) as client:
            issues = _search_all(
                client, base, auth, project_jql,
                ["status", "issuetype", "priority", "duedate", "issuelinks", "labels", "timetracking"],
            )
            open_issues = [issue for issue in issues if not _is_done(issue)]
            sprint_issues = _search_all(
                client, base, auth,
                f"{project_jql} AND sprint in openSprints()",
                ["status", "issuetype", "priority", "duedate", "issuelinks", "labels", "timetracking", "assignee", story_points_field, sprint_field],
            )
    except (httpx.HTTPError, ValueError) as exc:
        raise JiraSignalError(f"Jira Cloud metrics could not be fetched: {exc}") from exc
    finally:
        raw_token = None  # noqa: F841

    total_open = len(open_issues)
    active_sprint_open = [issue for issue in sprint_issues if not _is_done(issue)]
    overdue = sum(1 for issue in active_sprint_open if (due := issue.get("fields", {}).get("duedate")) and date.fromisoformat(due) < date.today())
    def severity(issue: dict) -> str:
        return str((issue.get("fields", {}).get("priority") or {}).get("name", "")).casefold()
    critical = sum(1 for issue in active_sprint_open if issue.get("fields", {}).get("issuetype", {}).get("name", "").casefold() == "bug" and severity(issue) == "critical")
    blocker = sum(1 for issue in active_sprint_open if issue.get("fields", {}).get("issuetype", {}).get("name", "").casefold() == "bug" and severity(issue) == "blocker")
    planned, uses_story_points = _work_units(sprint_issues, story_points_field)
    completed, _ = _work_units(
        [issue for issue in sprint_issues if _is_done(issue)], story_points_field
    )
    actual_completion = completed / planned if planned > 0 else 0.0
    active_statuses = {"in progress", "in review", "review"}
    wip, _ = _work_units([issue for issue in sprint_issues if str((issue.get("fields", {}).get("status") or {}).get("name", "")).casefold() in active_statuses], story_points_field)
    progress_ratio = min(1.0, (completed + wip * 0.5) / planned) if planned else 0.0
    start, end = _sprint_window(sprint_issues, sprint_field)
    now = datetime.now(timezone.utc)
    total_days = max(1.0, (end - start).total_seconds() / 86400) if start and end else 0.0
    elapsed_ratio = min(1.0, max(0.0, (now - start).total_seconds() / 86400 / total_days)) if total_days else 0.0
    deviation = progress_ratio - elapsed_ratio
    # Only a material (>20 percentage point) gap is a velocity penalty.  On
    # normal mid-sprint days, elapsed-time pace is treated as on target.
    velocity = 1.0 if not total_days or deviation >= -0.20 else max(0.0, min(1.0, progress_ratio / max(elapsed_ratio, .01)))
    def is_blocked(issue: dict) -> bool:
        fields = issue.get("fields", {})
        status = str((fields.get("status") or {}).get("name", "")).lower()
        labels = " ".join(str(label).lower() for label in (fields.get("labels") or []))
        return "block" in status or "block" in labels

    def unresolved_dependency(issue: dict) -> bool:
        for link in issue.get("fields", {}).get("issuelinks", []) or []:
            linked = link.get("outwardIssue") or link.get("inwardIssue") or {}
            if linked and not _is_done(linked):
                return True
        return False

    blocked = sum(1 for issue in open_issues if is_blocked(issue))
    dependencies = sum(1 for issue in open_issues if unresolved_dependency(issue))
    original_estimate = sum(float((issue.get("fields", {}).get("timetracking") or {}).get("originalEstimateSeconds") or 0) for issue in sprint_issues)
    remaining_estimate = sum(float((issue.get("fields", {}).get("timetracking") or {}).get("remainingEstimateSeconds") or 0) for issue in sprint_issues)
    assignee_loads: dict[str, int] = {}
    for issue in sprint_issues:
        assignee = (issue.get("fields", {}).get("assignee") or {}).get("accountId") or "unassigned"
        assignee_loads[assignee] = assignee_loads.get(assignee, 0) + (0 if _is_done(issue) else 1)
    average_load = sum(assignee_loads.values()) / len(assignee_loads) if assignee_loads else 0.0
    workload_risk = (max(assignee_loads.values()) / average_load - 1.0) if average_load else 0.0
    return {
        "velocity_percent": round(min(1.0, max(0.0, velocity)), 4),
        "overdue_rate": round(overdue / len(sprint_issues), 4) if sprint_issues else 0.0,
        "bug_ratio": round((critical + blocker) / len(sprint_issues), 4) if sprint_issues else 0.0,
        "active_sprint_issue_count": len(sprint_issues),
        "committed_story_pts": round(planned, 2),
        "completed_story_pts": round(completed + wip * 0.5, 2),
        "actual_completed_story_pts": round(completed, 2),
        "wip_credit_story_pts": round(wip * 0.5, 2),
        "expected_velocity_ratio": round(elapsed_ratio, 4),
        "velocity_deviation": round(deviation, 4),
        "is_sprint_active": bool(sprint_issues),
        "active_sprint_uses_story_points": uses_story_points,
        "open_issue_count": total_open,
        "overdue_issue_count": overdue,
        "open_bug_count": critical + blocker,
        "critical_bug_count": critical,
        "blocker_bug_count": blocker,
        # Current operational evidence used for dashboard risk analysis and
        # persisted beside every prediction. These are never stale score
        # contributions: they are replaced on every Jira sync.
        "sprint_completion": round(velocity, 4),
        "original_estimate_seconds": round(original_estimate, 2),
        "remaining_estimate_seconds": round(remaining_estimate, 2),
        "remaining_estimate_ratio": round(remaining_estimate / original_estimate, 4) if original_estimate else 0.0,
        "blocked_issue_count": blocked,
        "dependency_risk": round(dependencies / total_open, 4) if total_open else 0.0,
        "team_workload_risk": round(max(0.0, workload_risk), 4),
    }


def fetch_jira_capacity_snapshot(jira_url: str, encrypted_jira_token: str, jira_email: str) -> dict:
    """Fetch the live active-sprint work graph needed for capacity analysis.

    This deliberately returns raw Jira *metadata* for current, non-completed
    work only.  It neither writes to PHPS nor substitutes historical sprint
    work for the active delivery picture.
    """
    if not jira_url or not encrypted_jira_token or not jira_email:
        raise JiraSignalError("Jira URL, account email, and API token are required")
    base, project_key = _cloud_base_and_project_key(jira_url)
    token = decrypt_token(encrypted_jira_token)
    story_points_field = os.getenv("JIRA_STORY_POINTS_FIELD", "customfield_10016")
    sprint_field = os.getenv("JIRA_SPRINT_FIELD", "customfield_10020")
    role_field = os.getenv("JIRA_DEVELOPER_ROLE_FIELD")
    fields = [
        "assignee", "reporter", "status", "priority", "issuetype", "parent", "subtasks", "epic", "issuelinks",
        "labels", "components", "comment", "created", "updated", "duedate", "resolutiondate", "timetracking", "sprint", sprint_field,
        story_points_field,
    ]
    if role_field:
        fields.append(role_field)
    project_jql = f'project = "{project_key}"'
    try:
        with httpx.Client(headers={"Accept": "application/json"}, timeout=30.0) as client:
            active = _search_all(
                client, base, (jira_email, token),
                f"{project_jql} AND sprint in openSprints()",
                fields,
            )
            # Current sprint backlog can include unstarted issues assigned to
            # its open sprint.  Jira's openSprints JQL is the authoritative
            # current-sprint membership filter.
            historical = _search_all(
                client, base, (jira_email, token),
                f"{project_jql} AND sprint in closedSprints() AND statusCategory = Done",
                ["status", story_points_field, sprint_field],
            )
            # Keep the project-wide issue inventory and Agile metadata in the
            # snapshot for drill-downs.  Capacity calculations below continue
            # to use *only* ``active`` so backlog/future work cannot distort
            # active sprint health.
            all_issues = _search_all(client, base, (jira_email, token), project_jql, fields)
            boards, sprints = [], []
            try:
                boards_response = client.get(f"{base}/rest/agile/1.0/board", params={"projectKeyOrId": project_key, "maxResults": 50})
                boards_response.raise_for_status()
                boards = boards_response.json().get("values", [])
                for board in boards:
                    response = client.get(f"{base}/rest/agile/1.0/board/{board['id']}/sprint", params={"state": "active,future,closed", "maxResults": 50})
                    response.raise_for_status()
                    sprints.extend(response.json().get("values", []))
            except httpx.HTTPError:
                # Jira projects can expose issues but not Agile board APIs.
                # Keep the valid issue-based capacity snapshot available.
                boards, sprints = [], []
    except httpx.HTTPError as exc:
        raise JiraSignalError(f"Jira capacity data could not be fetched: {exc}") from exc
    finally:
        token = None  # noqa: F841

    def sprint_dates(issue: dict) -> tuple[str | None, str | None, str | None]:
        value = issue.get("fields", {}).get(sprint_field) or issue.get("fields", {}).get("sprint")
        values = value if isinstance(value, list) else [value]
        for sprint in values:
            if isinstance(sprint, dict) and (sprint.get("state", "ACTIVE").upper() == "ACTIVE"):
                return sprint.get("name"), sprint.get("startDate"), sprint.get("endDate")
        return None, None, None

    sprint_name = sprint_start = sprint_end = None
    for issue in active:
        sprint_name, sprint_start, sprint_end = sprint_dates(issue)
        if sprint_name or sprint_start or sprint_end:
            break
    return {
        "issues": active,
        "all_issues": all_issues,
        "historical_completed_issues": historical,
        "boards": boards,
        "sprints": sprints,
        "story_points_field": story_points_field,
        "developer_role_field": role_field,
        "active_sprint": {"name": sprint_name, "start_date": sprint_start, "end_date": sprint_end},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
