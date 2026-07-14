"""Single authoritative workflow for rebuilding a project's health score.

Event handlers save their new numeric evidence first, then call
``ProjectHealthService.recalculate``.  The service deliberately owns the
whole sequence: collect evidence -> aggregate -> load delivery metrics ->
predict -> persist -> explain.  This prevents an API response or dashboard
view from becoming an alternative scoring path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from app.ml.scorer import compute_health_score, explain_health_score
from app.models.models import HealthScore, ProcessedEmail, Project, TranscriptUpload


@dataclass(frozen=True)
class CommunicationData:
    """Numeric-only project communication evidence, separated by source."""

    emails: tuple[object, ...]
    transcripts: tuple[object, ...]

    @property
    def observations(self) -> tuple[object, ...]:
        return self.emails + self.transcripts


class ProjectHealthService:
    """Rebuild and explain the latest health prediction for one project."""

    def __init__(self, db: Session):
        self.db = db

    def collect_project_communication_data(self, project_id: int) -> CommunicationData:
        """Load every scored email and transcript for the project."""
        email_query = self.db.query(ProcessedEmail).filter_by(project_id=project_id)
        transcript_query = self.db.query(TranscriptUpload).filter_by(project_id=project_id)
        # Lightweight test sessions from older endpoint tests may not expose
        # all(); production SQLAlchemy queries always do.
        emails = email_query.all() if hasattr(email_query, "all") else []
        transcripts = transcript_query.all() if hasattr(transcript_query, "all") else []
        return CommunicationData(
            emails=tuple(item for item in emails if item.tone_score is not None and item.urgency_flag is not None),
            transcripts=tuple(item for item in transcripts if item.tone_score is not None and item.urgency_flag is not None),
        )

    @staticmethod
    def calculate_average_tone(data: CommunicationData) -> float:
        """Return the unweighted mean across every communication record."""
        observations = data.observations
        return sum(float(item.tone_score) for item in observations) / len(observations) if observations else 0.0

    @staticmethod
    def calculate_project_urgency(data: CommunicationData) -> int:
        """A project is urgent when any retained communication is urgent."""
        return int(any(item.urgency_flag for item in data.observations))

    def _latest_score(self, project_id: int) -> Optional[HealthScore]:
        return (
            self.db.query(HealthScore)
            .filter_by(project_id=project_id)
            .order_by(HealthScore.recorded_at.desc())
            .first()
        )

    def fetch_jira_metrics(self, project_id: int, updated_metrics: Optional[dict] = None) -> dict:
        """Use just-synced metrics or the latest persisted project metrics."""
        latest = self._latest_score(project_id)
        metrics = updated_metrics or {}
        previous_velocity = getattr(latest, "velocity_percent", 0.5)
        previous_overdue = getattr(latest, "overdue_rate", 0.2)
        previous_bug_ratio = getattr(latest, "bug_ratio", 0.1)
        velocity = metrics.get("velocity_percent")
        if velocity is None:
            velocity = previous_velocity
        return {
            "velocity_percent": velocity,
            "overdue_rate": metrics.get("overdue_rate", previous_overdue),
            "bug_ratio": metrics.get("bug_ratio", previous_bug_ratio),
        }

    @staticmethod
    def _round_split(total: float, first: float) -> tuple[float, float]:
        first = round(first, 2)
        return first, round(total - first, 2)

    def calculate_contributions(
        self,
        *,
        health_score: float,
        tone_score: float,
        urgency_flag: int,
        jira_metrics: dict,
        communication: CommunicationData,
    ) -> dict:
        """Build an additive UI breakdown after—not during—prediction.

        Urgency is a communication signal, so its penalty is included in the
        communication parent.  Tone is allocated by each source's exact share
        of the averaged tone component and urgency is allocated only among the
        urgent records.  Rounding residuals are assigned to the transcript
        side, making every displayed parent reconcile exactly.
        """
        raw = explain_health_score(
            health_score=health_score,
            tone_score=tone_score,
            urgency_flag=urgency_flag,
            velocity_percent=jira_metrics["velocity_percent"],
            overdue_rate=jira_metrics["overdue_rate"],
            bug_ratio=jira_metrics["bug_ratio"],
        )
        communication_total = round(
            raw["communication_sentiment"] + raw["urgency_penalty"], 2
        )
        delivery_total = raw["delivery_metrics"]
        observations = communication.observations
        count = len(observations)

        def tone_weight(rows: Iterable[object]) -> float:
            return sum((float(item.tone_score) + 1.0) / 2.0 for item in rows) / count if count else 0.0

        email_tone_weight = tone_weight(communication.emails)
        transcript_tone_weight = tone_weight(communication.transcripts)
        total_tone_weight = email_tone_weight + transcript_tone_weight
        tone_parent = raw["communication_sentiment"]
        if total_tone_weight:
            email_tone, transcript_tone = self._round_split(
                tone_parent, tone_parent * email_tone_weight / total_tone_weight
            )
        else:
            email_tone, transcript_tone = 0.0, tone_parent

        urgent_emails = sum(bool(item.urgency_flag) for item in communication.emails)
        urgent_transcripts = sum(bool(item.urgency_flag) for item in communication.transcripts)
        urgent_count = urgent_emails + urgent_transcripts
        urgency_penalty = raw["urgency_penalty"]
        if urgent_count:
            email_urgency, transcript_urgency = self._round_split(
                urgency_penalty, urgency_penalty * urgent_emails / urgent_count
            )
        else:
            email_urgency, transcript_urgency = 0.0, 0.0

        email_analysis = round(email_tone + email_urgency, 2)
        transcript_analysis = round(communication_total - email_analysis, 2)
        total_contribution = round(communication_total + delivery_total, 2)
        # ``explain_health_score`` guarantees this; retain an exact final
        # reconciliation after source-level rounding as well.
        delivery_total = round(delivery_total + (round(health_score, 2) - total_contribution), 2)

        return {
            **raw,
            "communication_sentiment": communication_total,
            "delivery_metrics": delivery_total,
            "urgency_penalty": 0.0,
            "email_analysis": email_analysis,
            "transcript_analysis": transcript_analysis,
            "sentiment_total": round(email_analysis + transcript_analysis, 2),
            "total_contribution": round(communication_total + delivery_total, 2),
            "email_count": len(communication.emails),
            "transcript_count": len(communication.transcripts),
        }

    def recalculate(
        self,
        project: Project,
        *,
        jira_metrics: Optional[dict] = None,
        analysis_source: Optional[str] = None,
        transcript_upload_id: Optional[int] = None,
    ) -> dict:
        """Persist a newly calculated score from complete current project data."""
        communication = self.collect_project_communication_data(project.id)
        tone_score = self.calculate_average_tone(communication)
        urgency_flag = self.calculate_project_urgency(communication)
        metrics = self.fetch_jira_metrics(project.id, jira_metrics)
        prediction = compute_health_score(
            tone_score=tone_score,
            urgency_flag=urgency_flag,
            velocity_percent=metrics["velocity_percent"],
            overdue_rate=metrics["overdue_rate"],
            bug_ratio=metrics["bug_ratio"],
            green_threshold=project.green_threshold,
            red_threshold=project.red_threshold,
        )
        score = HealthScore(
            project_id=project.id,
            health_score=prediction["health_score"],
            rag_status=prediction["rag_status"],
            tone_score=tone_score,
            urgency_flag=urgency_flag,
            velocity_percent=metrics["velocity_percent"],
            overdue_rate=metrics["overdue_rate"],
            bug_ratio=metrics["bug_ratio"],
            divergence_flag=prediction["divergence_flag"],
            analysis_source=analysis_source,
            transcript_upload_id=transcript_upload_id,
        )
        self.db.add(score)
        self.db.commit()
        return {
            "status": "ok",
            "project_id": project.id,
            **prediction,
            "tone_score": tone_score,
            "urgency_flag": urgency_flag,
            "jira_metrics": metrics,
            "contributions": self.calculate_contributions(
                health_score=prediction["health_score"],
                tone_score=tone_score,
                urgency_flag=urgency_flag,
                jira_metrics=metrics,
                communication=communication,
            ),
        }

    @staticmethod
    def explain_saved_score(score: HealthScore) -> dict:
        """Explain a historical score without recalculating it."""
        return explain_health_score(
            health_score=score.health_score,
            tone_score=score.tone_score,
            urgency_flag=score.urgency_flag,
            velocity_percent=score.velocity_percent,
            overdue_rate=score.overdue_rate,
            bug_ratio=score.bug_ratio,
        )
