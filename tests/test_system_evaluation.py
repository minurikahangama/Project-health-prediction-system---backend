import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.services.system_evaluation import SystemEvaluationService


def issue(key, assignee):
    return {"key": key, "fields": {"assignee": {"displayName": assignee}, "customfield_10016": 3,
            "timetracking": {"remainingEstimateSeconds": 3600, "originalEstimateSeconds": 3600},
            "status": {"name": "In Progress", "statusCategory": {"key": "indeterminate"}},
            "priority": {"name": "Medium"}, "issuetype": {"name": "Story"}, "issuelinks": []}}


class SystemEvaluationTests(unittest.TestCase):
    def test_evaluation_uses_available_live_evidence_without_inventing_model_metrics(self):
        now = datetime.now(timezone.utc)
        project = SimpleNamespace(id=1, jira_metrics_snapshot={"committed_story_pts": 3, "actual_completed_story_pts": 1, "remaining_estimated_hours": 1, "overdue_issue_count": 0, "open_bug_count": 0}, jira_capacity_snapshot={"story_points_field": "customfield_10016", "active_sprint": {"name": "Sprint", "start_date": (now - timedelta(days=1)).isoformat(), "end_date": (now + timedelta(days=5)).isoformat()}, "issues": [issue("PHPS-1", "Alex")]})
        score = SimpleNamespace(health_score=100.0, deduction_snapshot={"delivery": {"total": 0}, "communication": {"total": 0}}, shap_explanation={}, recorded_at=now)
        report = SystemEvaluationService.build(project, score, [score], [], [])
        self.assertEqual(report["health_score_validation"]["status"], "Passed")
        self.assertEqual(report["ai_model_evaluation"]["status"], "Unavailable")
        self.assertEqual(report["reliability"]["successful_executions"], 100)
        self.assertIsNotNone(report["capacity_validation"]["capacity_accuracy"])

