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
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.utils.database import get_db
from app.api.auth import get_current_user
from app.models.models import HealthScore, Project
from app.ml.preprocessor import process_and_delete
from app.ml.sentiment import score_tone, detect_urgency
from app.ml.jira_signals import fetch_jira_signals
from app.ml.scorer import compute_health_score

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class GmailMessage(BaseModel):
    body:    str
    subject: str = ""


class GmailIngestRequest(BaseModel):
    project_id: int
    messages:   List[GmailMessage]


class JiraIngestRequest(BaseModel):
    project_id: int


# ── Helper: run full ML pipeline and save result ──────────────────────────────

def _run_pipeline_and_save(
    db:               Session,
    project:          Project,
    anonymised_text:  Optional[str],
    jira_signals:     Optional[dict],
) -> dict:
    """
    Runs Layers 3a, 3b, 4, and 5 of the PHPS pipeline.
    anonymised_text and jira_signals can each be None if not yet available.
    """
    # ── Layer 3a: Sentiment ────────────────────────────────────────────────
    if anonymised_text:
        tone    = score_tone(anonymised_text)
        urgency = detect_urgency(anonymised_text)
        anonymised_text = None   # clear from memory after use
    else:
        # Use latest cached values from the database
        latest = (
            db.query(HealthScore)
            .filter_by(project_id=project.id)
            .order_by(HealthScore.recorded_at.desc())
            .first()
        )
        tone    = latest.tone_score    if latest else 0.0
        urgency = latest.urgency_flag  if latest else 0

    # ── Layer 3b: Jira signals ─────────────────────────────────────────────
    if jira_signals:
        velocity = jira_signals["velocity_percent"]
        overdue  = jira_signals["overdue_rate"]
        bug      = jira_signals["bug_ratio"]
    else:
        # Use latest cached Jira signals from the database
        latest = (
            db.query(HealthScore)
            .filter_by(project_id=project.id)
            .order_by(HealthScore.recorded_at.desc())
            .first()
        )
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
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ingest-gmail")
def ingest_gmail(
    data: GmailIngestRequest,
    db:   Session = Depends(get_db),
):
    """
    Called by n8n every 4 hours with Gmail messages for a project.
    Runs the full pipeline: preprocess → sentiment → score → save.
    FR-16: Automated Gmail ingestion.
    """
    project = db.query(Project).filter(Project.id == data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not data.messages:
        raise HTTPException(status_code=400, detail="No messages to process")

    # Combine all message bodies into one text block
    combined_text = " ".join(
        m.body for m in data.messages if m.body and m.body.strip()
    )
    if not combined_text.strip():
        raise HTTPException(status_code=400, detail="All messages are empty")

    # Layer 2: GDPR — anonymise and delete raw text
    anonymised = process_and_delete(
        raw_text   = combined_text,
        project_id = project.id,
        event_type = "Gmail Ingestion",
        db         = db,
    )

    return _run_pipeline_and_save(
        db              = db,
        project         = project,
        anonymised_text = anonymised,
        jira_signals    = None,   # use cached Jira signals
    )


@router.post("/ingest-jira")
def ingest_jira(
    data: JiraIngestRequest,
    db:   Session = Depends(get_db),
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

    # Layer 3b: Fetch fresh Jira signals
    signals = fetch_jira_signals(
        jira_url             = project.jira_url,
        encrypted_jira_token = project.encrypted_jira_token,
        jira_email           = project.jira_email or "",
    )

    return _run_pipeline_and_save(
        db              = db,
        project         = project,
        anonymised_text = None,   # use cached sentiment
        jira_signals    = signals,
    )


@router.post("/upload-transcript")
async def upload_transcript(
    project_id: int,
    file:       UploadFile = File(...),
    db:         Session = Depends(get_db),
    user = Depends(get_current_user),
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

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.org_id != user.org_id:
        raise HTTPException(status_code=403, detail="Access denied")

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

    # Layer 2: GDPR anonymise and delete raw text
    anonymised = process_and_delete(
        raw_text   = raw_text,
        project_id = project_id,
        event_type = "Transcript Processing",
        db         = db,
    )

    result = _run_pipeline_and_save(
        db              = db,
        project         = project,
        anonymised_text = anonymised,
        jira_signals    = None,
    )
    result["message"] = "Transcript processed. Health score updated."
    return result
