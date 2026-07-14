"""
Pipeline router — n8n-facing ingestion endpoints.

These endpoints are called automatically by the n8n workflow — NOT by the frontend.

Endpoints:
  POST /pipeline/ingest-gmail      — called every 4 hours by n8n (FR-16)
  POST /pipeline/ingest-jira       — called every Monday 09:00 by n8n (FR-17)
  POST /pipeline/upload-transcript — called by PM from dashboard (FR-52)

Full 6-layer pipeline per call:
  Layer 2: spaCy PII anonymisation + GDPR deletion log
  Layer 3a: RoBERTa tone score + urgency flag
  Layer 3b: Jira delivery signals (velocity, overdue, bug_ratio)
  Layer 4: XGBoost health score + divergence detection + RAG status
  Layer 5: Save numeric results to health_scores table
"""
import os
import logging
import secrets
import hashlib
import threading
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.utils.database import get_db
from app.utils.time import utcnow
from app.api.auth import get_current_user, require_role
from app.models.models import ProcessedEmail, Project, TranscriptUpload
from app.ml.preprocessor import process_and_delete
from app.ml.sentiment import score_tone, detect_urgency
from app.ml.jira_signals import JiraSignalError, fetch_jira_signals
from app.services.gmail_ingestion import GmailIngestionError, fetch_messages
from app.services.project_health import ProjectHealthService

router = APIRouter()
logger = logging.getLogger(__name__)
TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "transcript_storage"
_email_sync_locks: dict[int, threading.Lock] = {}
_email_sync_locks_guard = threading.Lock()


def _email_sync_lock(project_id: int) -> threading.Lock:
    """Serialize duplicate checks and analysis for one project in this API process."""
    with _email_sync_locks_guard:
        return _email_sync_locks.setdefault(project_id, threading.Lock())


def require_n8n_secret(x_n8n_secret: str = Header(...)):
    """Authenticate n8n without granting it a human user's JWT."""
    configured_secret = os.getenv("N8N_WEBHOOK_SECRET")
    if not configured_secret:
        logger.error("N8N_WEBHOOK_SECRET is not configured")
        raise HTTPException(status_code=503, detail="n8n ingestion is not configured")
    if not secrets.compare_digest(x_n8n_secret, configured_secret):
        raise HTTPException(status_code=401, detail="Invalid n8n webhook secret")


def _project_for_user(project_id: int, user, db: Session) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.org_id != user.org_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if user.role == "pm" and project.created_by_user_id != user.id:
        raise HTTPException(status_code=403, detail="This project is not assigned to you")
    return project


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class GmailMessage(BaseModel):
    id:      str
    body:    str
    subject: str = ""
    sender: str = ""
    received_at: Optional[str] = None


class GmailIngestRequest(BaseModel):
    project_id: int
    messages:   List[GmailMessage]


class JiraIngestRequest(BaseModel):
    project_id: int


def _sync_actor_name(actor=None) -> str:
    """Return a display-safe name for sync audit fields."""
    if actor is None:
        return "System"
    name = getattr(actor, "name", None)
    email = getattr(actor, "email", None)
    role = getattr(actor, "role", None)
    if name and email:
        return f"{name} ({email})"
    if name:
        return name
    if email:
        return email
    if role:
        return str(role).replace("_", " ").title()
    return "System"


def _mark_project_synced(project: Project, source: str, actor=None) -> None:
    synced_at = utcnow()
    synced_by = _sync_actor_name(actor)
    if source == "jira":
        project.last_jira_synced_at = synced_at
        project.last_jira_synced_by = synced_by
    elif source == "email":
        project.last_email_synced_at = synced_at
        project.last_email_synced_by = synced_by


# ── Helper: run full ML pipeline and save result ──────────────────────────────

def _run_pipeline_and_save(
    db:               Session,
    project:          Project,
    anonymised_text:  Optional[str],
    jira_signals:     Optional[dict],
    analysis_source:  Optional[str] = None,
    transcript_upload_id: Optional[int] = None,
) -> dict:
    """Compatibility entry point for the central project-health workflow."""
    if anonymised_text:
        logger.warning(
            "Ignoring transient text in project %s recomputation; source "
            "evidence must be saved before health is rebuilt.",
            project.id,
        )
    result = ProjectHealthService(db).recalculate(
        project,
        jira_metrics=jira_signals,
        analysis_source=analysis_source,
        transcript_upload_id=transcript_upload_id,
    )
    logger.info(
        "Project %s rebuilt: health=%.1f rag=%s divergence=%s",
        project.id,
        result["health_score"],
        result["rag_status"],
        result["divergence_flag"],
    )
    return result

    # Legacy implementation retained below temporarily for patch-context
    # stability; it is unreachable and will be removed in the next patch.
    """
    # ── Layer 3a: Sentiment ────────────────────────────────────────────────
    # Fetch once so an update uses one coherent baseline for delivery signals.
    latest = (
        db.query(HealthScore)
        .filter_by(project_id=project.id)
        .order_by(HealthScore.recorded_at.desc())
        .first()
    )

    # A prediction is always made from the *current evidence*, never by
    # redistributing the previous HealthScore.  Bodies are already deleted;
    # numeric observations are sufficient for a GDPR-safe recalculation.
    email_query = db.query(ProcessedEmail).filter_by(project_id=project.id)
    transcript_query = db.query(TranscriptUpload).filter_by(project_id=project.id)
    # ``all`` is present on real SQLAlchemy queries. The small in-memory
    # sessions used by isolated scorer tests only expose ``first``.
    email_evidence = email_query.all() if hasattr(email_query, "all") else []
    transcript_evidence = transcript_query.all() if hasattr(transcript_query, "all") else []
    evidence = [
        item for item in [*email_evidence, *transcript_evidence]
        if item.tone_score is not None and item.urgency_flag is not None
    ]
    if evidence:
        tone = sum(item.tone_score for item in evidence) / len(evidence)
        urgency = int(any(item.urgency_flag for item in evidence))
    elif anonymised_text:
        # Compatibility for direct pipeline callers that have not yet saved a
        # source observation. Normal ingestion paths save evidence first.
        tone = score_tone(anonymised_text)
        urgency = detect_urgency(anonymised_text)
    else:
        # Legacy projects without numeric evidence retain their last observed
        # sentiment until new evidence is available.
        tone = latest.tone_score if latest else 0.0
        urgency = latest.urgency_flag if latest else 0

    # ── Layer 3b: Jira signals ─────────────────────────────────────────────
    if jira_signals:
        velocity = jira_signals["velocity_percent"]
        overdue  = jira_signals["overdue_rate"]
        bug      = jira_signals["bug_ratio"]
        # No active sprint means Jira has no velocity measurement. Preserve a
        # previous observation instead of converting missing Jira data into a
        # fabricated 0% delivery rate.
        if velocity is None:
            velocity = latest.velocity_percent if latest else 0.5
    else:
        # Use latest cached Jira signals from the database
        velocity = latest.velocity_percent if latest else 0.5
        overdue  = latest.overdue_rate     if latest else 0.2
        bug      = latest.bug_ratio        if latest else 0.1

    # ── Layer 4: XGBoost fusion ────────────────────────────────────────────
    result = compute_health_score(
        tone_score       = tone,
        urgency_flag     = urgency,
        velocity_percent = velocity,
        overdue_rate     = overdue,
        bug_ratio        = bug,
        green_threshold  = project.green_threshold,
        red_threshold    = project.red_threshold,
    )

    # ── Layer 5: Persist — numbers only, never text ────────────────────────
    score_row = HealthScore(
        project_id       = project.id,
        health_score     = result["health_score"],
        rag_status       = result["rag_status"],
        tone_score       = tone,
        urgency_flag     = urgency,
        velocity_percent = velocity,
        overdue_rate     = overdue,
        bug_ratio        = bug,
        divergence_flag  = result["divergence_flag"],
        analysis_source  = analysis_source,
        transcript_upload_id = transcript_upload_id,
    )
    db.add(score_row)
    db.commit()

    logger.info(
        f"Project {project.id} scored: "
        f"health={result['health_score']:.1f} "
        f"rag={result['rag_status']} "
        f"divergence={result['divergence_flag']}"
    )

    return {
        "status":          "ok",
        "project_id":      project.id,
        "health_score":    result["health_score"],
        "rag_status":      result["rag_status"],
        "divergence_flag": result["divergence_flag"],
        "tone_score":      tone,
        "urgency_flag":    urgency,
        "jira_metrics": {
            "velocity_percent": velocity,
            "overdue_rate": overdue,
            "bug_ratio": bug,
        },
        "contributions": explain_health_score(
            health_score=result["health_score"], tone_score=tone,
            urgency_flag=urgency, velocity_percent=velocity,
            overdue_rate=overdue, bug_ratio=bug,
        ),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

    """

@router.get("/projects")
def list_n8n_projects(
    db: Session = Depends(get_db),
    _: None = Depends(require_n8n_secret),
):
    """Return non-sensitive project routing data for n8n workflows."""
    projects = db.query(Project).all()
    return {
        "projects": [
            {
                "project_id": project.id,
                "gmail_filter_email": project.gmail_filter_email,
                "gmail_project_identifier": project.gmail_project_identifier,
                "jira_configured": bool(project.jira_url and project.encrypted_jira_token),
            }
            for project in projects
        ]
    }

@router.post("/ingest-gmail")
def ingest_gmail(
    data: GmailIngestRequest,
    db:   Session = Depends(get_db),
    _: None = Depends(require_n8n_secret),
):
    """
    Called by n8n every 4 hours with Gmail messages for a project.
    Runs the full pipeline: preprocess → sentiment → score → save.
    FR-16: Automated Gmail ingestion.
    """
    project = db.query(Project).filter(Project.id == data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return _ingest_gmail_messages(db, project, data.messages)


def _ingest_gmail_messages(
    db: Session,
    project: Project,
    messages: List[GmailMessage],
    actor=None,
) -> dict:
    # The unique database key prevents duplicate records across processes;
    # this lock also prevents two local retries from analysing the same email
    # while neither has yet reached the persistence step.
    with _email_sync_lock(project.id):
        return _ingest_gmail_messages_locked(db, project, messages, actor)


def _ingest_gmail_messages_locked(
    db: Session,
    project: Project,
    messages: List[GmailMessage],
    actor=None,
) -> dict:
    """Filter, deduplicate, score, and persist messages for one project."""

    if not messages:
        return {
            "status": "ignored",
            "project_id": project.id,
            "message": "No messages to process",
        }

    if not project.gmail_project_identifier:
        raise GmailIngestionError(
            "Set a Gmail project identifier before enabling Gmail ingestion"
        )

    new_messages = []
    seen_message_ids = set()
    duplicate_count = 0

    for message in messages:

        # IMAP and webhook payloads can contain the same message more than
        # once. Deduplicate this batch before running analysis.
        if message.id in seen_message_ids:
            duplicate_count += 1
            logger.info("Duplicate Gmail skipped from current batch: %s", message.id)
            continue
        seen_message_ids.add(message.id)

        existing = (
            db.query(ProcessedEmail)
            .filter_by(project_id=project.id, gmail_message_id=message.id)
            .first()
        )

        if existing:
            duplicate_count += 1
            logger.info(f"Duplicate Gmail skipped: {message.id}")
            continue

        new_messages.append(message)

    if not new_messages:
        return {
            "status": "ignored",
            "project_id": project.id,
            "message": f"All {duplicate_count} emails were already processed.",
        }

    identifier = project.gmail_project_identifier.casefold()

    matching_messages = [
        message
        for message in new_messages
        if identifier in f"{message.subject}\n{message.body}".casefold()
    ]

    if not matching_messages:
        return {
            "status": "ignored",
            "project_id": project.id,
            "message": "No new messages matched this project's Gmail identifier",
        }

    # Persist one numeric observation per message.  Scoring an entire sync as
    # one blob and assigning that result to every email made the aggregation
    # depend on batch size rather than what each communication actually said.
    # Raw text is anonymised and discarded before only its derived values are
    # retained.
    for message in matching_messages:
        anonymised = process_and_delete(
            raw_text=f"Subject: {message.subject}\n{message.body}",
            project_id=project.id,
            event_type="Gmail Ingestion",
            db=db,
        )
        try:
            received_at = parsedate_to_datetime(message.received_at) if message.received_at else None
        except (TypeError, ValueError):
            received_at = None
        db.add(ProcessedEmail(
            project_id=project.id,
            gmail_message_id=message.id,
            subject=message.subject[:998] if message.subject else None,
            sender=message.sender[:255] if message.sender else None,
            received_at=received_at,
            tone_score=score_tone(anonymised),
            urgency_flag=detect_urgency(anonymised),
        ))
    if hasattr(db, "flush"):
        db.flush()
    result = _run_pipeline_and_save(
        db=db, project=project, anonymised_text=None,
        jira_signals=None,
        analysis_source="gmail",
    )
    _mark_project_synced(project, "email", actor)
    db.commit()
    result["processed_messages"] = len(matching_messages)
    result["duplicate_messages"] = duplicate_count
    result["last_email_synced_at"] = project.last_email_synced_at.isoformat()
    result["last_email_synced_by"] = project.last_email_synced_by
    return result


def sync_gmail_project(db: Session, project: Project, actor=None) -> dict:
    """Poll the configured inbox for one project without n8n."""
    if not project.gmail_account_email or not project.encrypted_gmail_token:
        raise GmailIngestionError("Authorize a Gmail account before syncing email")
    if not project.gmail_filter_email or not project.gmail_project_identifier:
        raise GmailIngestionError("Project needs both a Gmail sender address and project identifier")
    rows = fetch_messages(
        sender=project.gmail_filter_email,
        account_email=project.gmail_account_email,
        encrypted_token=project.encrypted_gmail_token,
    )
    return _ingest_gmail_messages(db, project, [GmailMessage(**row) for row in rows], actor=actor)


@router.post("/ingest-jira")
def ingest_jira(
    data: JiraIngestRequest,
    db:   Session = Depends(get_db),
    _: None = Depends(require_n8n_secret),
):
    """
    Called by n8n every Monday at 09:00.
    Fetches fresh Jira signals and runs the health score pipeline.
    FR-17: Automated Jira synchronisation.
    """
    project = db.query(Project).filter(Project.id == data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.jira_url or not project.encrypted_jira_token:
        raise HTTPException(
            status_code=400,
            detail="Jira is not configured for this project",
        )

    return sync_jira_project(db, project)


def sync_jira_project(db: Session, project: Project, actor=None) -> dict:
    """Fetch real Jira delivery signals and save a score for one project."""
    if not project.jira_url or not project.encrypted_jira_token:
        raise JiraSignalError("Jira is not configured for this project")
    # Layer 3b: Fetch fresh Jira signals
    try:
        signals = fetch_jira_signals(
            jira_url=project.jira_url,
            encrypted_jira_token=project.encrypted_jira_token,
            jira_email=project.jira_email or "",
        )
    except JiraSignalError as exc:
        logger.warning("Jira ingestion failed for project %s: %s", project.id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    result = _run_pipeline_and_save(
        db              = db,
        project         = project,
        anonymised_text = None,   # use cached sentiment
        jira_signals    = signals,
        analysis_source = "jira",
    )
    # The manual-sync endpoint returns these source counts so the detailed
    # page can show exactly what Jira records produced the saved metrics.
    result["jira_metrics"].update({
        key: signals[key]
        for key in (
            "active_sprint_issue_count", "active_sprint_uses_story_points",
            "open_issue_count", "overdue_issue_count", "open_bug_count",
        )
    })
    result["jira_metrics"]["velocity_measured"] = signals["velocity_percent"] is not None
    _mark_project_synced(project, "jira", actor)
    db.commit()
    result["last_jira_synced_at"] = project.last_jira_synced_at.isoformat()
    result["last_jira_synced_by"] = project.last_jira_synced_by
    return result


@router.post("/upload-transcript")
async def upload_transcript(
    project_id: int,
    file:       UploadFile = File(...),
    db:         Session = Depends(get_db),
    user = Depends(require_role("pm", "org_admin")),
):
    """
    Called by the PM from the dashboard transcript upload panel.
    Accepts .vtt or .txt files, runs the full pipeline.
    FR-52: Meeting transcript processing.
    """
    # Validate file type
    if not file.filename.lower().endswith((".vtt", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only .vtt and .txt files are accepted",
        )

    project = _project_for_user(project_id, user, db)

    # Read file content
    raw_bytes = await file.read()

    # Validate size (max 10 MB)
    if len(raw_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File is too large. Maximum size is 10MB",
        )

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Decode to text
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")  # fallback encoding

    content_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    # Each upload is an analysis event.  Keep the hash for auditing, but do
    # not deduplicate it: PMs may intentionally upload revised/confirmed
    # meeting notes with unchanged text and expect a fresh score observation.
    extension = Path(file.filename).suffix.lower()
    storage_filename = f"{uuid4().hex}{extension}"
    upload = TranscriptUpload(
        project_id=project.id,
        uploaded_by_user_id=user.id,
        original_filename=Path(file.filename).name,
        storage_filename=storage_filename,
        size_bytes=len(raw_bytes),
        content_sha256=content_sha256,
    )
    db.add(upload)
    db.commit()

    storage_path = TRANSCRIPT_DIR / storage_filename
    try:
        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        storage_path.write_bytes(raw_bytes)
        anonymised = process_and_delete(
            raw_text=raw_text,
            project_id=project_id,
            event_type="Transcript Processing",
            db=db,
        )
        # Store derived evidence before recalculating so this upload affects
        # the new score together with all previously retained evidence.
        upload.tone_score = score_tone(anonymised)
        upload.urgency_flag = detect_urgency(anonymised)
        if hasattr(db, "flush"):
            db.flush()
        result = _run_pipeline_and_save(
            db=db,
            project=project,
            anonymised_text=None,
            jira_signals=None,
            analysis_source="transcript",
            transcript_upload_id=upload.id,
        )
    except Exception:
        # A real processing failure remains retryable; remove the claim and
        # its managed file before propagating the original error.
        db.rollback()
        if storage_path.is_file():
            storage_path.unlink()
        db.delete(upload)
        db.commit()
        raise

    result["message"] = "Transcript processed. Health score updated."
    return result


@router.get("/projects/{project_id}/transcripts")
def list_transcripts(project_id: int, db: Session = Depends(get_db), user=Depends(require_role("pm", "org_admin"))):
    _project_for_user(project_id, user, db)
    rows = db.query(TranscriptUpload).filter_by(project_id=project_id).order_by(TranscriptUpload.uploaded_at.desc()).all()
    return [{"id": row.id, "filename": row.original_filename, "size_bytes": row.size_bytes, "uploaded_at": row.uploaded_at.isoformat()} for row in rows]


@router.delete("/projects/{project_id}/transcripts/{transcript_id}")
def delete_transcript(project_id: int, transcript_id: int, db: Session = Depends(get_db), user=Depends(require_role("pm", "org_admin"))):
    project = _project_for_user(project_id, user, db)
    row = db.query(TranscriptUpload).filter_by(id=transcript_id, project_id=project_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Transcript not found")
    path = TRANSCRIPT_DIR / row.storage_filename
    if path.is_file():
        path.unlink()
    db.delete(row)
    if hasattr(db, "flush"):
        db.flush()
    # Preserve historical predictions and append a new one based on the
    # remaining emails/transcripts and latest Jira signals.
    result = _run_pipeline_and_save(
        db=db, project=project, anonymised_text=None,
        jira_signals=None,
        analysis_source="transcript_deleted",
    )
    result["message"] = "Transcript removed and health score recalculated"
    return result
