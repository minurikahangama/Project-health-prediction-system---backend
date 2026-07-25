import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.services.health_scoring import communication_deduction, delivery_deductions, health_score, status_for_score


class DeductionEngineTests(unittest.TestCase):
    def test_baseline_without_evidence(self):
        result = health_score(None, [], 70, 40)
        self.assertEqual(result["health_score"], 100)

    def test_velocity_deduction_uses_completed_over_committed(self):
        metrics = {"is_sprint_active": True, "committed_story_pts": 10,
                   "actual_completed_story_pts": 2, "expected_velocity_ratio": .5}
        self.assertEqual(delivery_deductions(metrics)["velocity"], 21)
        metrics["actual_completed_story_pts"] = 5
        self.assertEqual(delivery_deductions(metrics)["velocity"], 0)

    def test_overdue_and_critical_bug_caps(self):
        metrics = {"active_sprint_issue_count": 10, "overdue_issue_count": 1,
                   "critical_bug_count": 0, "blocker_bug_count": 0}
        deductions = delivery_deductions(metrics)
        self.assertEqual(deductions["overdue"], 10)
        metrics.update(overdue_issue_count=5, critical_bug_count=2)
        deductions = delivery_deductions(metrics)
        self.assertEqual(deductions["overdue"], 20)
        self.assertEqual(deductions["bug"], 25)

    def test_communication_recovery_only_offsets_communication(self):
        now = datetime.now(timezone.utc)
        negative = SimpleNamespace(tone_score=-.8, urgency_flag=1, uploaded_at=now)
        positive = SimpleNamespace(tone_score=.8, urgency_flag=0, uploaded_at=now)
        result = communication_deduction([negative, positive], now)
        self.assertAlmostEqual(result["total"], 11.6)  # 18 - 6.4
        self.assertEqual(health_score(None, [negative, positive], 70, 40)["health_score"], 88.4)

    def test_old_communication_decays_and_statuses_are_configurable(self):
        old = SimpleNamespace(tone_score=-1, urgency_flag=0, uploaded_at=datetime.now(timezone.utc) - timedelta(days=20))
        self.assertLess(communication_deduction([old])["total"], 6)
        self.assertEqual(status_for_score(80, 80, 50), "GREEN")
        self.assertEqual(status_for_score(79, 80, 50), "AMBER")
        self.assertEqual(status_for_score(49, 80, 50), "RED")


if __name__ == "__main__":
    unittest.main()
