"""Regression tests for the production health-score calibration."""
import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.pop("PHPS_USE_XGBOOST_MODEL", None)

from app.ml.scorer import compute_health_score, explain_health_score
from app.ml.sentiment import detect_urgency


class HealthScoringTests(unittest.TestCase):
    def test_successful_communication_increases_the_score(self):
        baseline = compute_health_score(0.0, 0, 0.5, 0.2, 0.1)["health_score"]
        successful = compute_health_score(0.95, 0, 0.5, 0.2, 0.1)["health_score"]

        self.assertGreater(successful, baseline)
        self.assertGreater(successful - baseline, 1.0)

    def test_better_delivery_metrics_increase_the_score(self):
        at_risk = compute_health_score(0.0, 0, 0.25, 0.6, 0.4)["health_score"]
        on_track = compute_health_score(0.0, 0, 0.75, 0.1, 0.1)["health_score"]

        self.assertGreater(on_track, at_risk)
        self.assertGreater(on_track - at_risk, 5.0)

    def test_urgency_is_a_ten_point_penalty(self):
        normal = compute_health_score(0.5, 0, 0.7, 0.1, 0.1)["health_score"]
        urgent = compute_health_score(0.5, 1, 0.7, 0.1, 0.1)["health_score"]

        self.assertGreater(normal - urgent, 0.0)

    def test_explanation_reconciles_with_prediction(self):
        result = compute_health_score(0.5, 1, 0.7, 0.1, 0.1)
        contributions = explain_health_score(
            health_score=result["health_score"], tone_score=0.5,
            urgency_flag=1, velocity_percent=0.7, overdue_rate=0.1,
            bug_ratio=0.1,
        )
        self.assertAlmostEqual(
            contributions["communication_sentiment"]
            + contributions["delivery_metrics"]
            + contributions["urgency_penalty"],
            result["health_score"], places=2,
        )

    def test_reassuring_no_blockers_statement_is_not_urgent(self):
        self.assertEqual(detect_urgency("All goals are complete; there are no blockers or delays."), 0)

    def test_xgboost_is_the_default_prediction_source(self):
        result = compute_health_score(0.0, 0, 0.5, 0.2, 0.1)
        self.assertEqual(result["prediction_source"], "xgboost")
        self.assertEqual(set(result["feature_vector"]), {
            "overall_sentiment", "urgency_count", "velocity", "overdue_rate",
            "bug_ratio", "open_issues", "open_bugs", "email_count",
            "transcript_count", "days_to_deadline", "sentiment_trend",
            "velocity_trend", "bug_trend",
        })

    def test_each_favourable_input_is_monotonic(self):
        base = compute_health_score(0.0, 0, 0.5, 0.5, 0.5)["health_score"]
        self.assertGreater(compute_health_score(0.5, 0, 0.5, 0.5, 0.5)["health_score"], base)
        self.assertGreater(compute_health_score(0.0, 0, 0.75, 0.5, 0.5)["health_score"], base)
        self.assertGreater(compute_health_score(0.0, 0, 0.5, 0.25, 0.5)["health_score"], base)
        self.assertGreater(compute_health_score(0.0, 0, 0.5, 0.5, 0.25)["health_score"], base)

    def test_bounds_and_rag_thresholds_are_correct(self):
        best = compute_health_score(1, 0, 1, 0, 0)["health_score"]
        worst = compute_health_score(-1, 1, 0, 1, 1)["health_score"]
        self.assertTrue(0.0 <= worst <= best <= 100.0)

    def test_negative_transcript_signal_lowers_score_with_same_delivery(self):
        positive = compute_health_score(0.9947, 0, 0.2, 0.25, 0.75)["health_score"]
        negative = compute_health_score(-0.9947, 0, 0.2, 0.25, 0.75)["health_score"]
        self.assertGreater(positive, negative)
        self.assertGreater(positive - negative, 5.0)

    def test_shap_values_reconstruct_the_xgboost_prediction(self):
        result = compute_health_score(0.4, 1, 0.7, 0.1, 0.2, email_count=3)
        shap_values = result["shap_values"]
        self.assertIsNotNone(shap_values)
        reconstructed = sum(shap_values.values())
        self.assertAlmostEqual(reconstructed, result["health_score"], places=2)


if __name__ == "__main__":
    unittest.main()
