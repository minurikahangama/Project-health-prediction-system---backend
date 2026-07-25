"""Model-only validation for simulations, recommendations and forecasts."""
import os
import unittest
import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.ml.scorer import compute_health_score
from app.services.ai_insights import AIInsightsService
from app.services.decision_support import DecisionSupportService


class DecisionSupportValidationTests(unittest.TestCase):
    def test_capacity_risk_accepts_no_active_sprint_metrics(self):
        risks = DecisionSupportService._capacity_risks({
            "workload_distribution": {"overload_threshold": 90},
            "team_capacity": [],
            "dependency_analysis": {"blocked_estimated_hours": 0},
            "burndown": {"burndown_deviation": None, "remaining_story_points": None},
            "_total_remaining_seconds": 0,
        })
        self.assertEqual(risks, {"capacity": 0.0, "dependency": 0.0, "burndown": 0.0})

    def setUp(self):
        self.project = SimpleNamespace(
            green_threshold=70.0, red_threshold=40.0, jira_url="https://jira.example.test",
            deadline=datetime.now(timezone.utc) + timedelta(weeks=4),
        )
        initial = compute_health_score(-.1, 0, .6, .2, .25, open_issues=20, open_bugs=5,
                                       email_count=3, transcript_count=1, deadline=self.project.deadline)
        self.score = SimpleNamespace(health_score=initial["health_score"], tone_score=-.1,
            urgency_flag=0, velocity_percent=.6, overdue_rate=.2, bug_ratio=.25,
            open_issues=20, open_bugs=5, recorded_at=datetime.now(timezone.utc),
            rag_status=initial["rag_status"], divergence_flag=0)
        self.values = {"communication_sentiment": -.1, "urgency_flag": 0,
                       "sprint_velocity": .6, "overdue_rate": .2, "bug_ratio": .25}

    def test_each_simulation_calls_the_xgboost_scorer(self):
        with patch("app.services.ai_insights.compute_health_score", wraps=compute_health_score) as predict:
            result = AIInsightsService.simulate(self.project, self.score,
                {**self.values, "sprint_velocity": .8}, 3, 1)
        predict.assert_called_once()
        self.assertEqual(result["prediction_source"], "xgboost")
        self.assertGreater(result["predicted_score"], self.score.health_score)

    def test_recommendation_gain_equals_simulated_score_difference(self):
        decision = DecisionSupportService.build(self.project, self.score, self.values, 3, 1)
        for action in decision["scenarios"]:
            simulated = AIInsightsService.simulate(self.project, self.score,
                {**self.values, action["metric"]: action["target"]}, 3, 1)
            self.assertEqual(action["gain"], round(simulated["predicted_score"] - decision["current_score"], 2))

    def test_planner_targets_change_when_the_simulator_metrics_change(self):
        high_bug = DecisionSupportService.build(self.project, self.score, self.values, 3, 1)
        low_bug_values = {**self.values, "bug_ratio": .05, "sprint_velocity": .8}
        low_bug = DecisionSupportService.build(self.project, self.score, low_bug_values, 3, 1)
        high_bug_action = next(action for action in high_bug["simulation_results"] if action["metric"] == "bug_ratio")
        low_bug_action = next(action for action in low_bug["simulation_results"] if action["metric"] == "bug_ratio")
        self.assertNotEqual(high_bug_action["current"], low_bug_action["current"])
        self.assertNotEqual(high_bug["current_score"], low_bug["current_score"])

    def test_recommendations_are_regenerated_for_changed_project_signals(self):
        baseline = DecisionSupportService.build(self.project, self.score, self.values, 3, 1)
        adjusted = DecisionSupportService.build(
            self.project, self.score, {**self.values, "communication_sentiment": .7}, 3, 1
        )
        self.assertNotEqual(
            [(item["metric"], item["current"], item["gain"]) for item in baseline["recommendations"]],
            [(item["metric"], item["current"], item["gain"]) for item in adjusted["recommendations"]],
        )

    def test_planner_fields_are_calculated_from_simulation_output(self):
        decision = DecisionSupportService.build(self.project, self.score, self.values, 3, 1)
        for action in decision["recommendations"]:
            self.assertEqual(action["title"].split(" to ")[-1], f"{action['target']:.2f}")
            self.assertEqual(action["estimated_duration_days"], max(1, -(-action["estimated_effort"] // 10)))
            self.assertIn("XGBoost simulations", action["why_selected"])

    def test_live_capacity_counterfactuals_are_ranked_with_model_actions(self):
        capacity = {
            "team_capacity": [
                {"developer": "Emma", "estimated_remaining_hours": 10, "workload_percentage": 120},
                {"developer": "David", "estimated_remaining_hours": 2, "workload_percentage": 40},
            ],
            "workload_distribution": {"overload_threshold": 90, "overloaded_members": ["Emma"], "underutilized_members": ["David"]},
            "dependency_analysis": {"blocked_issues": ["PHPS-9"], "blocked_estimated_hours": 4, "affected_developers": ["Emma"]},
            "burndown": {"burndown_deviation": 3, "remaining_story_points": 9},
            "_total_remaining_seconds": 43200,
        }
        decision = DecisionSupportService.build(self.project, self.score, self.values, 3, 1, capacity)
        self.assertLessEqual(len(decision["recommendations"]), 5)
        self.assertTrue(any("developer_capacity" in action["affected_metrics"] for action in decision["recommendations"]))
        self.assertIn("capacity", decision["capacity_risk_scores"])

    def test_planner_has_no_static_action_policy_and_uses_simulations(self):
        source = inspect.getsource(__import__("app.services.decision_support", fromlist=["*"]))
        self.assertNotIn("ACTION_POLICY", source)
        with patch("app.services.ai_insights.compute_health_score", wraps=compute_health_score) as predict:
            decision = DecisionSupportService.build(self.project, self.score, self.values, 3, 1)
        self.assertGreater(predict.call_count, 1)
        self.assertTrue(decision["simulation_results"])
        for action in decision["recommendations"]:
            self.assertEqual(action["prediction_source"], "xgboost")
            self.assertIn("confidence_score", action)
            self.assertIn("estimated_duration_days", action)
        for step in decision["recovery_plan"]:
            self.assertEqual(step["prediction_source"], "xgboost")
            self.assertIn("cumulative_projected_health", step)

    def test_forecast_is_a_progressive_series_of_xgboost_predictions(self):
        decision = DecisionSupportService.build(self.project, self.score, self.values, 3, 1)
        forecast = [row["score"] for row in decision["forecast"]]
        self.assertGreater(forecast[-1], forecast[0])
        self.assertGreater(len(set(forecast)), 1)
        self.assertTrue(all(later >= earlier for earlier, later in zip(forecast, forecast[1:])))


if __name__ == "__main__":
    unittest.main()
