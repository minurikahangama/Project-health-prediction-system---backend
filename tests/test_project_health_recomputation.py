"""Project-health recomputation must precede explanation and persistence."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.api import pipeline
from app.services import project_health
from app.models.models import HealthScore, ProcessedEmail, TranscriptUpload


class _Query:
    def __init__(self, *, latest=None, rows=()):
        self.latest = latest
        self.rows = list(rows)

    def filter_by(self, **_kwargs):
        return self

    def order_by(self, *_args):
        return self

    def first(self):
        return self.latest

    def all(self):
        return self.rows


class _Session:
    def __init__(self, latest, emails, transcripts):
        self.latest = latest
        self.emails = emails
        self.transcripts = transcripts
        self.added = []
        self.committed = False

    def query(self, model):
        if model is HealthScore:
            return _Query(latest=self.latest)
        if model is ProcessedEmail:
            return _Query(rows=self.emails)
        if model is TranscriptUpload:
            return _Query(rows=self.transcripts)
        raise AssertionError(f"Unexpected query model: {model}")

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.committed = True


class ProjectHealthRecomputationTests(unittest.TestCase):
    @staticmethod
    def _project():
        return SimpleNamespace(id=9, green_threshold=70.0, red_threshold=40.0)

    @staticmethod
    def _latest():
        return SimpleNamespace(
            tone_score=0.0, urgency_flag=0, velocity_percent=0.5,
            overdue_rate=0.2, bug_ratio=0.1,
        )

    def test_aggregates_all_communication_then_saves_a_new_score(self):
        latest = SimpleNamespace(
            tone_score=0.0,
            urgency_flag=0,
            velocity_percent=0.5,
            overdue_rate=0.2,
            bug_ratio=0.1,
        )
        emails = [
            SimpleNamespace(tone_score=0.8, urgency_flag=0),
            SimpleNamespace(tone_score=-0.4, urgency_flag=1),
        ]
        transcripts = [SimpleNamespace(tone_score=-0.2, urgency_flag=0)]
        db = _Session(latest, emails, transcripts)
        project = self._project()

        with patch.object(project_health, "compute_health_score", return_value={
            "health_score": 47.5, "rag_status": "AMBER", "divergence_flag": 1,
        }) as compute:
            result = pipeline._run_pipeline_and_save(
                db, project, anonymised_text=None, jira_signals=None,
                analysis_source="transcript_deleted",
            )

        self.assertTrue(db.committed)
        self.assertEqual(len(db.added), 1)
        stored = db.added[0]
        self.assertAlmostEqual(stored.tone_score, (0.8 - 0.4 - 0.2) / 3)
        self.assertEqual(stored.urgency_flag, 1)
        self.assertEqual(stored.health_score, 47.5)
        contributions = result["contributions"]
        self.assertEqual(
            contributions["communication_sentiment"] + contributions["delivery_metrics"],
            47.5,
        )
        self.assertEqual(
            contributions["email_analysis"] + contributions["transcript_analysis"],
            contributions["communication_sentiment"],
        )
        self.assertEqual(
            contributions["velocity"] + contributions["overdue_rate"] + contributions["bug_ratio"],
            contributions["delivery_metrics"],
        )
        self.assertAlmostEqual(compute.call_args.kwargs["tone_score"], (0.8 - 0.4 - 0.2) / 3)
        self.assertEqual(compute.call_args.kwargs["urgency_flag"], 1)
        self.assertEqual(compute.call_args.kwargs["velocity_percent"], 0.5)

    def test_positive_email_rebuilds_to_a_higher_health_score(self):
        emails = [SimpleNamespace(tone_score=0.1, urgency_flag=0)]
        db = _Session(self._latest(), emails, [])
        first = pipeline._run_pipeline_and_save(db, self._project(), None, None)
        emails.append(SimpleNamespace(tone_score=0.9, urgency_flag=0))
        second = pipeline._run_pipeline_and_save(db, self._project(), None, None)
        self.assertGreater(second["health_score"], first["health_score"])

    def test_negative_transcript_lowers_score_and_deletion_rebuilds_upward(self):
        emails = [SimpleNamespace(tone_score=0.8, urgency_flag=0)]
        transcripts = []
        db = _Session(self._latest(), emails, transcripts)
        before = pipeline._run_pipeline_and_save(db, self._project(), None, None)
        transcripts.append(SimpleNamespace(tone_score=-0.9, urgency_flag=0))
        with_negative = pipeline._run_pipeline_and_save(db, self._project(), None, None)
        transcripts.clear()  # Represents the already-deleted database row.
        after_delete = pipeline._run_pipeline_and_save(db, self._project(), None, None)
        self.assertLess(with_negative["health_score"], before["health_score"])
        self.assertGreater(after_delete["health_score"], with_negative["health_score"])

    def test_urgency_is_rebuilt_from_all_remaining_sources(self):
        transcripts = [SimpleNamespace(tone_score=-0.2, urgency_flag=1)]
        db = _Session(self._latest(), [SimpleNamespace(tone_score=0.5, urgency_flag=0)], transcripts)
        urgent = pipeline._run_pipeline_and_save(db, self._project(), None, None)
        transcripts.clear()
        cleared = pipeline._run_pipeline_and_save(db, self._project(), None, None)
        self.assertEqual(urgent["urgency_flag"], 1)
        self.assertEqual(cleared["urgency_flag"], 0)
        self.assertGreater(cleared["health_score"], urgent["health_score"])


if __name__ == "__main__":
    unittest.main()
