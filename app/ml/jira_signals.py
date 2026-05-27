"""
Jira signal extractor — Layer 3b of the PHPS ML pipeline.

Fetches sprint velocity, overdue rate, and bug ratio from a live Jira instance.
All returned values are normalised to [0.0, 1.0].

FR-30 to FR-34: Jira delivery signal extraction.
"""
import logging
from typing import Dict

import httpx

from app.utils.encryption import decrypt_token

logger = logging.getLogger(__name__)


def fetch_jira_signals(
    jira_url: str,
    encrypted_jira_token: str,
    jira_email: str,
) -> Dict[str, float]:
    """
    Fetch three normalised delivery signals from a Jira workspace:

      velocity_percent  — story points completed / planned this sprint  [0, 1]
      overdue_rate      — issues past due date / total open issues       [0, 1]
      bug_ratio         — open bugs / total open issues                  [0, 1]

    Returns safe defaults (0.5, 0.2, 0.1) if Jira is unreachable.

    Decrypts the stored Jira API token before use.
    The decrypted token is never logged or stored.
    """
    # Default values — used when Jira is unavailable
    defaults = {
        "velocity_percent": 0.5,
        "overdue_rate":     0.2,
        "bug_ratio":        0.1,
    }

    if not jira_url or not encrypted_jira_token:
        logger.warning("Jira not configured — using default signal values")
        return defaults

    try:
        raw_token = decrypt_token(encrypted_jira_token)
        base      = jira_url.rstrip("/")
        auth      = (jira_email, raw_token)
        headers   = {"Accept": "application/json"}
        timeout   = 30.0

        # ── Fetch all issues ───────────────────────────────────────────────
        resp = httpx.get(
            f"{base}/rest/api/3/search",
            params={"maxResults": 200, "fields": "status,issuetype,duedate"},
            auth=auth,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        issues = resp.json().get("issues", [])
        total  = len(issues) or 1  # avoid division by zero

        # ── Overdue rate ───────────────────────────────────────────────────
        # An issue is overdue if it has a due date and is not Done
        overdue = sum(
            1
            for i in issues
            if i["fields"].get("duedate")
            and i["fields"]["status"]["name"] != "Done"
        )
        overdue_rate = round(overdue / total, 4)

        # ── Bug ratio ──────────────────────────────────────────────────────
        # Open bugs as a proportion of all open issues
        bugs = sum(
            1
            for i in issues
            if i["fields"]["issuetype"]["name"].lower() == "bug"
            and i["fields"]["status"]["name"] != "Done"
        )
        bug_ratio = round(bugs / total, 4)

        # ── Sprint velocity ────────────────────────────────────────────────
        # Story points completed / story points planned in the current sprint
        velocity = 0.0
        try:
            boards_resp = httpx.get(
                f"{base}/rest/agile/1.0/board",
                params={"type": "scrum"},
                auth=auth,
                headers=headers,
                timeout=timeout,
            )
            boards_resp.raise_for_status()
            boards = boards_resp.json().get("values", [])

            if boards:
                board_id    = boards[0]["id"]
                sprint_resp = httpx.get(
                    f"{base}/rest/agile/1.0/board/{board_id}/sprint",
                    params={"state": "active"},
                    auth=auth,
                    headers=headers,
                    timeout=timeout,
                )
                sprint_resp.raise_for_status()
                sprints = sprint_resp.json().get("values", [])

                if sprints:
                    sprint_id      = sprints[0]["id"]
                    sprint_issues  = httpx.get(
                        f"{base}/rest/agile/1.0/sprint/{sprint_id}/issue",
                        params={"fields": "status,story_points,customfield_10016"},
                        auth=auth,
                        headers=headers,
                        timeout=timeout,
                    ).json().get("issues", [])

                    def _points(issue):
                        """Extract story points from either standard or custom field."""
                        fields = issue["fields"]
                        return (
                            fields.get("story_points")
                            or fields.get("customfield_10016")
                            or 0
                        ) or 0

                    planned   = sum(_points(i) for i in sprint_issues)
                    completed = sum(
                        _points(i)
                        for i in sprint_issues
                        if i["fields"]["status"]["name"] == "Done"
                    )
                    velocity = round(completed / planned, 4) if planned > 0 else 0.0

        except Exception as sprint_err:
            # Sprint velocity is optional — don't fail the whole pipeline
            logger.warning(f"Could not fetch sprint velocity: {sprint_err}")
            velocity = 0.5   # assume average velocity if unavailable

        # Clamp all values to [0, 1]
        return {
            "velocity_percent": min(1.0, max(0.0, velocity)),
            "overdue_rate":     min(1.0, max(0.0, overdue_rate)),
            "bug_ratio":        min(1.0, max(0.0, bug_ratio)),
        }

    except httpx.ConnectError:
        logger.error(f"Cannot connect to Jira at {jira_url}")
        return defaults
    except httpx.TimeoutException:
        logger.error(f"Jira request timed out for {jira_url}")
        return defaults
    except httpx.HTTPStatusError as e:
        logger.error(f"Jira API error {e.response.status_code}: {e.response.text[:200]}")
        return defaults
    except Exception as e:
        logger.error(f"Unexpected error fetching Jira signals: {e}")
        return defaults
    finally:
        # Ensure the decrypted token is not left in any variable
        raw_token = None  # noqa: F841
