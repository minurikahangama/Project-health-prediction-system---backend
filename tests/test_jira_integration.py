import os
import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

# The application database module requires a URL at import time.  Tests only
# use in-memory fakes, so no external database is contacted.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

try:
    from app.api import pipeline
    from app.api import projects
    from app.ml import jira_signals
    from app.services import project_health
    _DEPENDENCY_ERROR = None
except ModuleNotFoundError as exc:
    pipeline = None
    jira_signals = None
    _DEPENDENCY_ERROR = str(exc)


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _JiraClient:
    def __init__(self, issues, sprint_issues):
        self.issues = issues
        self.sprint_issues = sprint_issues
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def post(self, _url, *, json, **_kwargs):
        self.requests.append(("POST", _url, json))
        rows = self.sprint_issues if "openSprints()" in json["jql"] else self.issues
        return _Response({"issues": rows, "total": len(rows)})


class _Query:
    def __init__(self, latest):
        self.latest = latest

    def filter_by(self, **_kwargs):
        return self

    def order_by(self, *_args):
        return self

    def first(self):
        return self.latest


class _Session:
    def __init__(self, latest):
        self.latest = latest
        self.added = []
        self.committed = False

    def query(self, *_args):
        return _Query(self.latest)

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.committed = True


@unittest.skipIf(_DEPENDENCY_ERROR is not None, f"Backend dependencies are unavailable: {_DEPENDENCY_ERROR}")
class JiraIntegrationTests(unittest.TestCase):
    def test_connection_validation_requires_project_access(self):
        with patch.object(projects.httpx, "get", side_effect=[
            _Response({"displayName": "PM", "accountId": "account-1"}),
            _Response({"key": "PHPS", "name": "Health Prediction"}),
        ]) as get:
            result = projects._validate_jira_connection(
                "https://example.atlassian.net/jira/software/projects/PHPS",
                "pm@example.com",
                "token",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["project_key"], "PHPS")
        self.assertEqual(result["project_name"], "Health Prediction")
        self.assertIn("/project/PHPS", get.call_args_list[1].args[0])

    def test_extracts_normalised_delivery_metrics(self):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        issues = [
            {"fields": {"status": {"statusCategory": {"key": "new"}}, "duedate": yesterday, "issuetype": {"name": "Bug"}}},
            {"fields": {"status": {"statusCategory": {"key": "new"}}, "duedate": None, "issuetype": {"name": "Task"}}},
            {"fields": {"status": {"statusCategory": {"key": "done"}}, "duedate": yesterday, "issuetype": {"name": "Bug"}}},
        ]
        sprint_issues = [
            {"fields": {"status": {"statusCategory": {"key": "done"}}, "customfield_10016": 3}},
            {"fields": {"status": {"statusCategory": {"key": "new"}}, "customfield_10016": 2}},
        ]
        client = _JiraClient(issues, sprint_issues)
        with patch.object(jira_signals, "decrypt_token", return_value="token"), patch.object(
            jira_signals.httpx, "Client", return_value=client
        ):
            metrics = jira_signals.fetch_jira_signals(
                "https://example.atlassian.net/jira/software/projects/PHPS", "encrypted", "pm@example.com"
            )

        self.assertEqual(metrics["velocity_percent"], 0.6)
        self.assertEqual(metrics["committed_story_pts"], 5)
        self.assertEqual(metrics["completed_story_pts"], 3)
        self.assertEqual(metrics["remaining_story_pts"], 2)
        # Delivery-risk ratios use active-sprint issues only; the overdue bug
        # in the project-wide fixture is not part of this sprint.
        self.assertEqual(metrics["overdue_rate"], 0.0)
        self.assertEqual(metrics["bug_ratio"], 0.0)
        self.assertTrue(all(url.endswith("/rest/api/3/search/jql") for _, url, _ in client.requests))
        self.assertTrue(all("startAt" not in body for _, _, body in client.requests))

    def test_sync_persists_the_metrics_returned_by_jira(self):
        db = _Session(SimpleNamespace(tone_score=-0.2, urgency_flag=0))
        project = SimpleNamespace(id=12, green_threshold=70.0, red_threshold=40.0)
        metrics = {"velocity_percent": 0.75, "overdue_rate": 0.25, "bug_ratio": 0.125}
        with patch.object(project_health, "compute_health_score", return_value={
            "health_score": 68.0, "rag_status": "AMBER", "divergence_flag": 0,
        }):
            pipeline._run_pipeline_and_save(db, project, anonymised_text=None, jira_signals=metrics)

        stored = db.added[0]
        self.assertTrue(db.committed)
        self.assertEqual(stored.project_id, 12)
        self.assertEqual(stored.velocity_percent, 0.75)
        self.assertEqual(stored.overdue_rate, 0.25)
        self.assertEqual(stored.bug_ratio, 0.125)

    def test_no_active_sprint_reports_null_velocity(self):
        issues = [
            {"fields": {"status": {"statusCategory": {"key": "new"}}, "duedate": None, "issuetype": {"name": "Task"}}},
        ]
        with patch.object(jira_signals, "decrypt_token", return_value="token"), patch.object(
            jira_signals.httpx, "Client", return_value=_JiraClient(issues, [])
        ):
            metrics = jira_signals.fetch_jira_signals(
                "https://example.atlassian.net/jira/software/projects/PHPS", "encrypted", "pm@example.com"
            )

        self.assertIsNone(metrics["velocity_percent"])
        self.assertEqual(metrics["active_sprint_issue_count"], 0)
        self.assertFalse(metrics["active_sprint_uses_story_points"])

    def test_manual_sync_rejects_a_site_url_without_a_project_key(self):
        project = SimpleNamespace(
            jira_url="https://example.atlassian.net",
            encrypted_jira_token="encrypted",
        )
        with patch.object(projects, "_get_project_for_user", return_value=project):
            with self.assertRaises(HTTPException) as raised:
                projects.sync_jira_now(12, db=object(), current_user=object())

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("project key", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
