"""Regression tests for context-grounded Ask PHPS AI responses."""
import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.services.ai_project_assistant import AIProjectAssistant


def _issue(key, assignee, points, hours, links=None):
    return {"key": key, "fields": {
        "assignee": {"displayName": assignee}, "customfield_10016": points,
        "timetracking": {"originalEstimateSeconds": hours * 3600,
                         "remainingEstimateSeconds": hours * 3600,
                         "timeSpentSeconds": 0},
        "status": {"name": "In Progress", "statusCategory": {"key": "indeterminate"}},
        "priority": {"name": "High"}, "issuetype": {"name": "Story"},
        "issuelinks": links or [],
    }}


class _Query:
    def filter_by(self, **_kwargs):
        return self

    def count(self):
        return 1


class _Database:
    def query(self, _model):
        return _Query()


class AskProjectAssistantTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.project = SimpleNamespace(
            id=1, green_threshold=70.0, red_threshold=40.0,
            deadline=now + timedelta(days=8),
            jira_capacity_snapshot={
                "story_points_field": "customfield_10016",
                "active_sprint": {"name": "Sprint", "start_date": (now - timedelta(days=2)).isoformat(), "end_date": (now + timedelta(days=8)).isoformat()},
                "historical_completed_issues": [],
                "issues": [
                    _issue("PHPS-1", "Alex", 8, 8, [{"type": {"outward": "blocks"}, "outwardIssue": {"key": "PHPS-2"}}]),
                    _issue("PHPS-2", "Emma", 3, 4),
                ],
            },
        )
        self.score = SimpleNamespace(health_score=50.0, rag_status="AMBER", tone_score=-0.2,
                                     urgency_flag=1, velocity_percent=.4, overdue_rate=.3,
                                     bug_ratio=.2, open_issues=2, open_bugs=1)

    def test_answer_contains_current_context_and_question_specific_evidence(self):
        answer = AIProjectAssistant.answer_question("Who is overloaded?", self.project, self.score, _Database())
        self.assertEqual(answer["reasoning"]["focus"], "capacity")
        self.assertIn("Alex", answer["reasoning"]["focus_evidence"]["overloaded"])
        self.assertEqual(answer["evidence"]["dependencies"]["blocked_issues"], ["PHPS-2"])
        self.assertIn("shap_values", answer["reasoning"]["prediction"])

    def test_counterfactual_recomputes_when_current_metrics_change(self):
        before = AIProjectAssistant.answer_question("What if velocity reaches 85%?", self.project, self.score, _Database())
        changed = SimpleNamespace(**{**self.score.__dict__, "velocity_percent": .7})
        after = AIProjectAssistant.answer_question("What if velocity reaches 85%?", self.project, changed, _Database())
        self.assertEqual(before["impact"]["input"]["sprint_velocity"], .85)
        self.assertEqual(after["impact"]["input"]["sprint_velocity"], .85)
        self.assertNotEqual(before["impact"]["health_difference"], after["impact"]["health_difference"])


if __name__ == "__main__":
    unittest.main()
