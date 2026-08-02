"""Deletion-specific regression tests for the baseline-first health engine."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.services.feature_processor import prepare_advanced_features
from app.services.transcript_service import TranscriptService
from app.models.models import HealthScore, Project, TranscriptUpload


class _Query:
    def __init__(self, one=None, rows=()): self.one, self.rows = one, list(rows)
    def filter_by(self, **_): return self
    def order_by(self, *_): return self
    def first(self): return self.one
    def all(self): return self.rows


class _Session:
    def __init__(self, transcript, project, latest, remaining):
        self.transcript, self.project, self.latest, self.remaining = transcript, project, latest, remaining
    def query(self, model):
        if model is TranscriptUpload:
            return _Query(one=self.transcript, rows=self.remaining)
        if model is Project: return _Query(one=self.project)
        if model is HealthScore: return _Query(one=self.latest)
        raise AssertionError(model)
    def delete(self, row):
        self.remaining[:] = [item for item in self.remaining if item is not row]
    def flush(self): pass


class HealthEngineDeletionTests(unittest.TestCase):
    def test_transcript_removal_returns_to_pre_transcript_baseline(self):
        baseline = prepare_advanced_features({"tone_score": 0.0, "is_brand_new": True})["ground_truth_score"]
        # Positive communication is a recovery, not a bonus above the 100
        # point baseline.
        raised = prepare_advanced_features({"tone_score": .9, "is_brand_new": True})["ground_truth_score"]
        self.assertEqual(raised, baseline)

        transcript = SimpleNamespace(id=7, project_id=1, storage_filename="x.txt")
        project = SimpleNamespace(id=1)
        session = _Session(transcript, project, SimpleNamespace(health_score=raised), [])
        with patch("app.services.transcript_service.PredictionService") as scorer:
            scorer.return_value.recalculate.return_value = {"health_score": baseline}
            result = TranscriptService(session).delete_transcript("7")
        self.assertEqual(result["health_score"], baseline)
        self.assertEqual(scorer.return_value.recalculate.call_args.kwargs["analysis_source"], "transcript_deleted")

    def test_total_transcript_clear_resets_to_neutral_feature(self):
        processed = prepare_advanced_features({"tone_score": 0.0, "days_since_last_transcript": None})
        self.assertEqual(processed["features"]["norm_decayed_sentiment"], .5)
        self.assertGreater(processed["ground_truth_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
