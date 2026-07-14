"""Jira Cloud delivery-signal extraction.

The project setting must be a Jira Cloud project URL, for example
``https://example.atlassian.net/jira/software/projects/PHPS``.  Only issues
in that project are queried; a failed Jira request raises an error instead of
quietly publishing placeholder health data.
"""
import logging
import os
import re
from datetime import date
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
    auth = (jira_email, raw_token)
    project_jql = f'project = "{project_key}"'

    try:
        with httpx.Client(headers={"Accept": "application/json"}, timeout=30.0) as client:
            issues = _search_all(
                client, base, auth, project_jql,
                ["status", "issuetype", "duedate"],
            )
            open_issues = [issue for issue in issues if not _is_done(issue)]
            overdue = sum(
                1 for issue in open_issues
                if (due_date := issue.get("fields", {}).get("duedate"))
                and date.fromisoformat(due_date) < date.today()
            )
            bugs = sum(
                1 for issue in open_issues
                if issue.get("fields", {}).get("issuetype", {}).get("name", "").lower() == "bug"
            )

            sprint_issues = _search_all(
                client, base, auth,
                f"{project_jql} AND sprint in openSprints()",
                ["status", story_points_field],
            )
    except (httpx.HTTPError, ValueError) as exc:
        raise JiraSignalError(f"Jira Cloud metrics could not be fetched: {exc}") from exc
    finally:
        raw_token = None  # noqa: F841

    total_open = len(open_issues)
    planned, uses_story_points = _work_units(sprint_issues, story_points_field)
    completed, _ = _work_units(
        [issue for issue in sprint_issues if _is_done(issue)], story_points_field
    )
    # This is the current sprint completion ratio: completed work divided by
    # all work committed to the active sprint.  There is no velocity value to
    # invent when Jira has no active sprint; the pipeline then retains its
    # last measured value (or its neutral initial value).
    velocity = completed / planned if planned > 0 else None
    return {
        "velocity_percent": round(min(1.0, max(0.0, velocity)), 4) if velocity is not None else None,
        "overdue_rate": round(overdue / total_open, 4) if total_open else 0.0,
        "bug_ratio": round(bugs / total_open, 4) if total_open else 0.0,
        "active_sprint_issue_count": len(sprint_issues),
        "active_sprint_uses_story_points": uses_story_points,
        "open_issue_count": total_open,
        "overdue_issue_count": overdue,
        "open_bug_count": bugs,
    }
