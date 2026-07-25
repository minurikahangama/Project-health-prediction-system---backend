import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.services.team_capacity import TeamCapacityService


def issue(key, assignee, points, remaining, links=None):
    return {"key": key, "fields": {
        "assignee": {"displayName": assignee}, "customfield_10016": points,
        "timetracking": {"originalEstimateSeconds": remaining, "remainingEstimateSeconds": remaining, "timeSpentSeconds": 0},
        "status": {"name": "In Progress", "statusCategory": {"key": "indeterminate"}},
        "priority": {"name": "High"}, "issuetype": {"name": "Story"}, "issuelinks": links or [],
    }}


class TeamCapacityTests(unittest.TestCase):
    def setUp(self):
        start = datetime.now(timezone.utc) - timedelta(days=2)
        end = start + timedelta(days=10)
        self.snapshot = {
            "story_points_field": "customfield_10016",
            "active_sprint": {"name": "Sprint 9", "start_date": start.isoformat(), "end_date": end.isoformat()},
            "historical_completed_issues": [],
            "issues": [
                issue("PHPS-1", "Alex", 8, 28800, [{"type": {"outward": "blocks"}, "outwardIssue": {"key": "PHPS-2"}}]),
                issue("PHPS-2", "Emma", 3, 14400),
                issue("PHPS-3", "Alex", 5, 7200),
            ],
        }
        self.project = SimpleNamespace(id=1, green_threshold=70.0, red_threshold=40.0, deadline=end)
        self.score = SimpleNamespace(health_score=60.0, rag_status="AMBER", tone_score=0.0, urgency_flag=0, velocity_percent=.5, overdue_rate=.1, bug_ratio=.1, open_issues=3, open_bugs=0)

    def test_capacity_and_dependency_values_are_derived_from_current_issues(self):
        result = TeamCapacityService.analyse(self.snapshot, overload_threshold=90)
        alex = next(row for row in result["team_capacity"] if row["developer"] == "Alex")
        self.assertEqual(alex["current_story_points"], 13)
        self.assertEqual(alex["estimated_remaining_hours"], 10)
        self.assertIn("Alex", result["workload_distribution"]["overloaded_members"])
        dependency = result["dependency_analysis"]
        self.assertEqual(dependency["blocked_issues"], ["PHPS-2"])
        self.assertEqual(dependency["affected_developers"], ["Emma"])
        self.assertEqual(dependency["blocked_story_points"], 3)

    def test_availability_simulation_does_not_mutate_analysis_and_recalculates_health(self):
        analysis = TeamCapacityService.analyse(self.snapshot, overload_threshold=90)
        original = analysis["team_capacity"][0]["estimated_remaining_hours"]
        result = TeamCapacityService.simulate_unavailability(analysis, "Alex", 5, self.project, self.score)
        self.assertEqual(analysis["team_capacity"][0]["estimated_remaining_hours"], original)
        self.assertLess(result["velocity"], self.score.velocity_percent)
        self.assertIn("predicted_health", result)
        self.assertIn("risk_level", result)

    def test_dashboard_and_developer_metrics_are_snapshot_derived(self):
        analysis = TeamCapacityService.analyse(self.snapshot, overload_threshold=90)
        alex = next(row for row in analysis["team_capacity"] if row["developer"] == "Alex")
        self.assertEqual(alex["assigned_issues"], 2)
        self.assertEqual(alex["remaining_story_points"], 13)
        self.assertGreater(alex["available_sprint_hours"], 0)
        dashboard = TeamCapacityService.dashboard(self.snapshot, SimpleNamespace(id=7), self.score)
        self.assertEqual(dashboard["project_id"], 7)
        self.assertEqual(dashboard["burndown"]["remaining_story_points"], 16)
        self.assertEqual(dashboard["dependency_graph"]["blocked_story_points"], 3)
        self.assertIn("delivery_risk", dashboard)

    def test_project_board_members_without_active_sprint_work_remain_visible(self):
        snapshot = {**self.snapshot, "all_issues": [
            *self.snapshot["issues"], issue("PHPS-4", "David", 5, 14400),
        ]}
        result = TeamCapacityService.analyse(snapshot)
        david = next(row for row in result["team_capacity"] if row["developer"] == "David")
        self.assertFalse(david["is_active_in_sprint"])
        self.assertEqual(david["assigned_issues"], 1)
        self.assertEqual(david["assigned_story_points"], 5)

    def test_dashboard_risk_exposes_live_normalised_contributors(self):
        dashboard = TeamCapacityService.dashboard(self.snapshot, self.project, self.score)
        risk = dashboard["delivery_risk"]
        self.assertEqual(len(risk["top_contributors"]), 5)
        self.assertIn("communication_sentiment", risk["components"])
        self.assertIn("blocked_story_points", risk["components"])
        self.assertGreaterEqual(risk["score"], 0)
        self.assertLessEqual(risk["score"], 100)

    def test_no_active_sprint_is_explicit_not_zero_filled(self):
        snapshot = {**self.snapshot, "active_sprint": {"name": None, "start_date": None, "end_date": None}, "issues": []}
        analysis = TeamCapacityService.analyse(snapshot)
        self.assertEqual(analysis["burndown"]["status"], "No active sprint found")
        self.assertIsNone(analysis["burndown"]["remaining_story_points"])
        self.assertTrue(TeamCapacityService.validate_snapshot(snapshot)["valid"])


if __name__ == "__main__":
    unittest.main()
