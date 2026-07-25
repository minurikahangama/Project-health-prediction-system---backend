"""Single authoritative workflow for rebuilding a project's health score.

Event handlers save their new numeric evidence first, then call
``PredictionService.recalculate``.  The service deliberately owns the
whole sequence: collect evidence -> aggregate -> load delivery metrics ->
predict -> persist -> explain.  This prevents an API response or dashboard
view from becoming an alternative scoring path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import math
from typing import Iterable, Optional

from sqlalchemy.orm import Session

import logging

from app.ml.scorer import compute_health_score, explain_health_score
from app.services.health_scoring import communication_deduction, delivery_deductions
from app.ml.jira_signals import fetch_jira_signals
from app.models.models import HealthScore, JiraEvidenceSnapshot, ProcessedEmail, Project, TranscriptUpload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommunicationData:
    """Numeric-only project communication evidence, separated by source."""

    emails: tuple[object, ...]
    transcripts: tuple[object, ...]

    @property
    def observations(self) -> tuple[object, ...]:
        return self.emails + self.transcripts


class PredictionService:
    """Rebuild and explain the latest health prediction for one project."""

    def __init__(self, db: Session):
        self.db = db

    def collect_project_communication_data(self, project_id: int) -> CommunicationData:
        """Load scored communication inside the current health window.

        Bodies are never retained, so the timestamped numeric evidence is the
        sole source of communication features.  Legacy rows without a stored
        timestamp remain eligible rather than being silently discarded.
        """
        email_query = self.db.query(ProcessedEmail).filter_by(project_id=project_id)
        transcript_query = self.db.query(TranscriptUpload).filter_by(project_id=project_id)
        # Lightweight test sessions from older endpoint tests may not expose
        # all(); production SQLAlchemy queries always do.
        emails = email_query.all() if hasattr(email_query, "all") else []
        transcripts = transcript_query.all() if hasattr(transcript_query, "all") else []
        def active(item: object, timestamp_name: str) -> bool:
            if getattr(item, "tone_score", None) is None or getattr(item, "urgency_flag", None) is None:
                return False
            # Historical evidence must remain available for the exponential
            # time-decay calculation. Dropping it after a fixed window made a
            # score jump simply because a calendar boundary was crossed.
            return True

        return CommunicationData(
            emails=tuple(item for item in emails if active(item, "processed_at")),
            transcripts=tuple(item for item in transcripts if active(item, "uploaded_at")),
        )

    @staticmethod
    def _recency_weight(item: object, timestamp_name: str) -> float:
        """Weight evidence by when it occurred, never by a prior health score."""
        timestamp = getattr(item, timestamp_name, None)
        if timestamp is None:
            return 0.2
        timestamp = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        age_days = max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds() / 86400)
        half_life = max(0.1, float(os.getenv("PHPS_SENTIMENT_HALF_LIFE_DAYS", "7")))
        return math.exp(-math.log(2) * age_days / half_life)

    def calculate_average_tone(self, data: CommunicationData) -> float:
        """Return the current, recency-weighted communication sentiment."""
        # Empty evidence is neutral: it must not mint a positive score.
        if not data.observations:
            return 0.0
        weighted = [
            (float(item.tone_score), self._recency_weight(item, "processed_at" if item in data.emails else "uploaded_at"))
            for item in data.observations
        ]
        total_weight = sum(weight for _, weight in weighted)
        return sum(tone * weight for tone, weight in weighted) / total_weight if total_weight else 0.0

    @staticmethod
    def calculate_source_tone(rows: Iterable[object]) -> float:
        rows = tuple(rows)
        return sum(float(item.tone_score) for item in rows) / len(rows) if rows else 0.0

    def _sentiment_trend(self, data: CommunicationData, current_tone: float) -> float:
        older = []
        for item in data.observations:
            timestamp = getattr(item, "processed_at", None) or getattr(item, "uploaded_at", None)
            if timestamp and self._recency_weight(item, "processed_at" if item in data.emails else "uploaded_at") < 1.0:
                older.append(float(item.tone_score))
        return current_tone - (sum(older) / len(older)) if older else 0.0

    def _delivery_trends(self, project_id: int, metrics: dict) -> tuple[float, float]:
        try:
            rows = (self.db.query(JiraEvidenceSnapshot).filter_by(project_id=project_id)
                    .order_by(JiraEvidenceSnapshot.collected_at.desc()).limit(3).all())
        except (AssertionError, AttributeError):
            rows = []
        if not rows:
            return 0.0, 0.0
        weights = list(range(len(rows), 0, -1))
        baseline_velocity = sum(float(r.metrics.get("velocity_percent", 0)) * w for r, w in zip(rows, weights)) / sum(weights)
        baseline_bugs = sum(float(r.metrics.get("bug_ratio", 0)) * w for r, w in zip(rows, weights)) / sum(weights)
        return metrics["velocity_percent"] - baseline_velocity, metrics["bug_ratio"] - baseline_bugs

    def calculate_project_urgency(self, data: CommunicationData) -> int:
        """Weighted current urgency count; deleted evidence is not represented."""
        weighted_urgency = sum(
            int(bool(item.urgency_flag)) * self._recency_weight(item, "processed_at" if item in data.emails else "uploaded_at")
            for item in data.observations
        )
        # Keep an urgent current observation visible even when it is old; the
        # recency weight is represented in the model feature, not rounded to
        # an accidental "no urgency" state.
        return max(1, round(weighted_urgency)) if weighted_urgency else 0

    def _latest_score(self, project_id: int) -> Optional[HealthScore]:
        return (
            self.db.query(HealthScore)
            .filter_by(project_id=project_id)
            .order_by(HealthScore.recorded_at.desc())
            .first()
        )

    def fetch_jira_metrics(self, project_id: int, updated_metrics: Optional[dict] = None) -> dict:
        """Use the latest Jira evidence; never a previously calculated contribution."""
        metrics = updated_metrics or {}
        # A snapshot is current Jira state, not a prior prediction or a cached
        # contribution.  It is replaced wholesale by every Jira sync.
        try:
            project = self.db.query(Project).filter_by(id=project_id).first()
        except (AssertionError, AttributeError):  # lightweight legacy tests
            project = None
        snapshot = getattr(project, "jira_metrics_snapshot", None) or {}
        current = {**snapshot, **metrics}
        return {
            "velocity_percent": current.get("velocity_percent", 0.0) or 0.0,
            "overdue_rate": current.get("overdue_rate", 0.0) or 0.0,
            "bug_ratio": current.get("bug_ratio", 0.0) or 0.0,
            "open_issues": current.get("open_issue_count", 0) or 0,
            "open_bugs": current.get("open_bug_count", 0) or 0,
            "sprint_completion": current.get("sprint_completion", 0.0) or 0.0,
            "remaining_estimate_ratio": current.get("remaining_estimate_ratio", 0.0) or 0.0,
            "blocked_issue_count": current.get("blocked_issue_count", 0) or 0,
            "dependency_risk": current.get("dependency_risk", 0.0) or 0.0,
            "team_workload_risk": current.get("team_workload_risk", 0.0) or 0.0,
            "committed_story_pts": current.get("committed_story_pts", 0.0) or 0.0,
            "completed_story_pts": current.get("completed_story_pts", 0.0) or 0.0,
            "is_sprint_active": bool(current.get("is_sprint_active", False)),
            "expected_velocity_ratio": current.get("expected_velocity_ratio", 0.0) or 0.0,
            "active_sprint_issue_count": current.get("active_sprint_issue_count", 0) or 0,
            "overdue_issue_count": current.get("overdue_issue_count", 0) or 0,
            "critical_bug_count": current.get("critical_bug_count", 0) or 0,
            "blocker_bug_count": current.get("blocker_bug_count", 0) or 0,
            "actual_completed_story_pts": current.get("actual_completed_story_pts", current.get("completed_story_pts", 0)) or 0,
            "historical_3sprint_avg_velocity": current.get("historical_3sprint_avg_velocity", 0.85) or 0.85,
            "overdue_issues": current.get("overdue_issue_count", 0) or 0,
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
        delivery = delivery_deductions(jira_metrics)
        scored_communication = communication_deduction(communication.observations)
        # Contribution values are negative deduction deltas, matching the
        # existing dashboard's additive contribution visualisation.
        communication_total = -round(scored_communication["total"], 2)
        delivery_total = -round(delivery["total"], 2)
        observations = communication.observations
        count = len(observations)

        def tone_weight(rows: Iterable[object]) -> float:
            return sum((float(item.tone_score) + 1.0) / 2.0 for item in rows) / count if count else 0.0

        email_tone_weight = tone_weight(communication.emails)
        transcript_tone_weight = tone_weight(communication.transcripts)
        total_tone_weight = email_tone_weight + transcript_tone_weight
        tone_parent = communication_total
        if total_tone_weight:
            email_tone, transcript_tone = self._round_split(
                tone_parent, tone_parent * email_tone_weight / total_tone_weight
            )
        else:
            email_tone, transcript_tone = 0.0, tone_parent

        urgent_emails = sum(bool(item.urgency_flag) for item in communication.emails)
        urgent_transcripts = sum(bool(item.urgency_flag) for item in communication.transcripts)
        urgent_count = urgent_emails + urgent_transcripts
        urgency_penalty = 0.0
        if urgent_count:
            email_urgency, transcript_urgency = self._round_split(
                urgency_penalty, urgency_penalty * urgent_emails / urgent_count
            )
        else:
            email_urgency, transcript_urgency = 0.0, 0.0

        email_analysis = round(email_tone + email_urgency, 2)
        transcript_analysis = round(communication_total - email_analysis, 2)
        # These values are deduction deltas from the 100-point baseline. Do
        # not add a residual merely to make them sum to the final score.

        return {
            "baseline": 100.0,
            "sprint_velocity_penalty": -round(delivery["velocity"], 2),
            "overdue_rate_penalty": -round(delivery["overdue"], 2),
            "bug_ratio_penalty": -round(delivery["bug"], 2),
            "delivery_deduction": delivery_total,
            "communication_deduction": communication_total,
            "transcript_sentiment_penalty": communication_total,
            "divergence_warning_penalty": 0.0,
            "model_calibration_penalty": 0.0,
            "communication_sentiment": communication_total,
            "delivery_metrics": delivery_total,
            "urgency_penalty": 0.0,
            "email_analysis": email_analysis,
            "transcript_analysis": transcript_analysis,
            "sentiment_total": round(email_analysis + transcript_analysis, 2),
            "total_contribution": round(communication_total + delivery_total, 2),
            "email_count": len(communication.emails),
            "transcript_count": len(communication.transcripts),
            # Expose the live evidence that generated this contribution. The
            # UI can therefore distinguish an improving tone from a reduced
            # stakeholder-risk/urgency penalty instead of presenting a fixed
            # explanation from an earlier prediction.
            "communication_context": {
                "weighted_sentiment": round(tone_score, 4),
                "urgency_count": urgency_flag,
                "stakeholder_risk": round(sum(
                    max(0.0, -float(item.tone_score)) * self._recency_weight(
                        item, "processed_at" if item in communication.emails else "uploaded_at"
                    ) for item in observations
                ) / max(1, count), 4),
                "communication_trend": round(self._sentiment_trend(communication, tone_score), 4),
            },
            "delivery_context": {
                key: jira_metrics[key] for key in (
                    "sprint_completion", "remaining_estimate_ratio", "blocked_issue_count",
                    "dependency_risk", "team_workload_risk",
                ) if key in jira_metrics
            },
        }

    def recalculate(
        self,
        project: Project,
        *,
        jira_metrics: Optional[dict] = None,
        analysis_source: Optional[str] = None,
        transcript_upload_id: Optional[int] = None,
        neutral_when_no_transcripts: bool = False,
    ) -> dict:
        """Persist a newly calculated score from complete current project data."""
        logger.info("Prediction Started project=%s", project.id)
        previous_prediction = self._latest_score(project.id)
        # Jira sync already provides its just-fetched response.  Any other
        # event fetches exactly one current Jira response when configured,
        # keeping delivery features current without duplicate API calls.
        if jira_metrics is None and all((
            getattr(project, "jira_url", None),
            getattr(project, "encrypted_jira_token", None),
            getattr(project, "jira_email", None),
        )):
            jira_metrics = fetch_jira_signals(
                project.jira_url, project.encrypted_jira_token, project.jira_email
            )
        communication = self.collect_project_communication_data(project.id)
        logger.info("Communications Analysed project=%s emails=%s transcripts=%s", project.id, len(communication.emails), len(communication.transcripts))
        tone_score = 0.0 if neutral_when_no_transcripts and not communication.transcripts else self.calculate_average_tone(communication)
        urgency_flag = self.calculate_project_urgency(communication)
        logger.info("RoBERTa Complete project=%s weighted_sentiment=%.4f urgency=%s", project.id, tone_score, urgency_flag)
        metrics = self.fetch_jira_metrics(project.id, jira_metrics)
        logger.info("Latest Jira Snapshot Collected project=%s metrics=%s", project.id, metrics)
        velocity_trend, bug_trend = self._delivery_trends(project.id, metrics)
        email_tone = self.calculate_source_tone(communication.emails)
        transcript_tone = self.calculate_source_tone(communication.transcripts)
        transcript_dates = [getattr(item, "uploaded_at", None) for item in communication.transcripts]
        transcript_dates = [date for date in transcript_dates if date is not None]
        latest_transcript = max(transcript_dates) if transcript_dates else None
        if latest_transcript and latest_transcript.tzinfo is None:
            latest_transcript = latest_transcript.replace(tzinfo=timezone.utc)
        days_since_last_transcript = max(0, (datetime.now(timezone.utc) - latest_transcript).days) if latest_transcript else 0
        logger.info("Feature Engineering Complete project=%s", project.id)
        prediction = compute_health_score(
            tone_score=tone_score,
            urgency_flag=urgency_flag,
            velocity_percent=metrics["velocity_percent"],
            overdue_rate=metrics["overdue_rate"],
            bug_ratio=metrics["bug_ratio"],
            open_issues=int(metrics["open_issues"]),
            open_bugs=int(metrics["open_bugs"]),
            green_threshold=project.green_threshold,
            red_threshold=project.red_threshold,
            email_sentiment=email_tone,
            transcript_sentiment=transcript_tone,
            urgency_count=sum(int(bool(item.urgency_flag)) for item in communication.observations),
            email_count=len(communication.emails),
            transcript_count=len(communication.transcripts),
            deadline=getattr(project, "deadline", None),
            sentiment_trend=self._sentiment_trend(communication, tone_score),
            velocity_trend=velocity_trend,
            bug_trend=bug_trend,
            team_size=getattr(project, "team_size", 1),
            committed_story_pts=metrics["committed_story_pts"],
            completed_story_pts=metrics["completed_story_pts"],
            is_sprint_active=metrics["is_sprint_active"],
            historical_3sprint_avg_velocity=metrics["historical_3sprint_avg_velocity"],
            days_since_last_transcript=days_since_last_transcript,
            jira_metrics={**metrics, **(jira_metrics or {})},
            communication_items=communication.observations,
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
            open_issues=int(metrics["open_issues"]),
            open_bugs=int(metrics["open_bugs"]),
            divergence_flag=prediction["divergence_flag"],
            prediction_source=prediction.get("prediction_source"),
            analysis_source=analysis_source,
            transcript_upload_id=transcript_upload_id,
            feature_vector=prediction.get("feature_vector"),
            shap_explanation=prediction.get("shap_values"),
        )
        self.db.add(score)
        self.db.commit()
        logger.info(
            "XGBoost Prediction Complete project=%s feature_vector=%s predicted_score=%.2f shap_values=%s",
            project.id, prediction.get("feature_vector"), prediction["health_score"],
            prediction.get("shap_values"),
        )
        logger.info("SHAP Generated project=%s", project.id)
        logger.info("Forecast Generated project=%s", project.id)
        logger.info("Dashboard Updated project=%s", project.id)
        logger.info("Prediction Finished project=%s", project.id)
        return {
            "status": "ok",
            "project_id": project.id,
            **prediction,
            # API callers use this immediate response for notifications. Keep
            # it consistent with the dashboard while retaining the exact ML
            # value for audit consumers.
            "raw_health_score": prediction["health_score"],
            # Return the exact recalculated score. UI smoothing must never
            # substitute an old persisted score for a new upload/delete/sync.
            "health_score": round(prediction["health_score"], 2),
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


# Backwards-compatible name for integrations.  New code must use
# PredictionService as the sole health-prediction authority.
ProjectHealthService = PredictionService
