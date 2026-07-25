"""Transcript evidence lifecycle operations.

Deleting an observation never mutates an old score: it creates a new score
from the remaining evidence immediately, preserving a complete audit trail.
"""
from __future__ import annotations
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from app.models.models import HealthScore, Project, TranscriptUpload
from app.services.project_health import PredictionService

logger = logging.getLogger(__name__)


def invalidate_project_health_cache(project_id: int) -> None:
    """Single cache-invalidation seam; presently responses are uncached."""
    logger.info("Invalidated health response cache project=%s", project_id)


class TranscriptService:
    def __init__(self, db: Session, storage_dir: Path | None = None):
        self.db, self.storage_dir = db, storage_dir

    def delete_transcript(self, transcript_id: str | int, *, project_id: int | None = None) -> dict:
        query = self.db.query(TranscriptUpload).filter_by(id=int(transcript_id))
        if project_id is not None:
            query = query.filter_by(project_id=project_id)
        transcript = query.first()
        if transcript is None:
            raise LookupError("Transcript not found")
        project = self.db.query(Project).filter_by(id=transcript.project_id).first()
        if project is None:
            raise LookupError("Project not found")
        old = self.db.query(HealthScore).filter_by(project_id=project.id).order_by(HealthScore.recorded_at.desc()).first()
        old_score = float(old.health_score) if old is not None else 100.0
        if self.storage_dir is not None:
            path = self.storage_dir / transcript.storage_filename
            if path.is_file():
                path.unlink()
        deleted_id = transcript.id
        self.db.delete(transcript)
        # Flush makes the following collection see only active transcripts;
        # commit remains owned by PredictionService.recalculate.
        if hasattr(self.db, "flush"):
            self.db.flush()
        remaining = self.db.query(TranscriptUpload).filter_by(project_id=project.id).all()
        result = PredictionService(self.db).recalculate(
            project, analysis_source="transcript_deleted",
            neutral_when_no_transcripts=not remaining,
        )
        invalidate_project_health_cache(project.id)
        logger.info("Transcript %s deleted for Project %s. Health Score recalculated from %.2f -> %.2f.", deleted_id, project.id, old_score, result["health_score"])
        return {**result, "remaining_transcript_count": len(remaining), "message": "Transcript removed and health score recalculated"}


def delete_transcript(transcript_id: str, db: Session, *, project_id: int | None = None, storage_dir: Path | None = None) -> dict:
    """Functional entry point required by integrations."""
    return TranscriptService(db, storage_dir).delete_transcript(transcript_id, project_id=project_id)
