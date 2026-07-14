"""
Projects router — full CRUD for the Project resource.

Endpoints:
  GET    /projects/                       — list projects for the logged-in PM's org
  POST   /projects/                       — create a new project
  GET    /projects/{id}                   — get project details
  PATCH  /projects/{id}                   — update project details / settings
  DELETE /projects/{id}                   — delete a project (with name confirmation)
  GET    /projects/{id}/health-score      — latest score + history for PM Dashboard
  POST   /projects/{id}/test-jira         — test Jira connection
  PATCH  /projects/{id}/pm-note          — update the PM note shown to clients
  POST   /projects/{id}/dismiss-divergence — dismiss divergence banner for session
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, validator
from typing import Optional, List
from datetime import datetime

from app.utils.database import get_db
from app.utils.encryption import encrypt_token, decrypt_token
from app.api.auth import get_current_user, require_role
from app.models.models import Project, HealthScore, User, ProcessedEmail, TranscriptUpload
from app.services.project_health import ProjectHealthService

import httpx

router = APIRouter()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name:             str
    start_date:       datetime
    deadline:         datetime
    team_size:        int
    jira_project_url: Optional[str] = None
    jira_api_token:   Optional[str] = None   # plaintext — encrypted before DB write
    jira_email:       Optional[str] = None
    gmail_account_email: Optional[str] = None
    gmail_app_password: Optional[str] = None
    gmail_filter_email: Optional[str] = None
    gmail_project_identifier: Optional[str] = None
    green_threshold:  float = 70.0
    red_threshold:    float = 40.0
    assigned_pm_id:    Optional[int] = None

    @validator("deadline")
    def deadline_after_start(cls, v, values):
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("Deadline must be after start date")
        return v

    @validator("green_threshold")
    def green_above_red(cls, v, values):
        if "red_threshold" in values and v <= values["red_threshold"]:
            raise ValueError("Green threshold must be above Red threshold")
        return v

    @validator("team_size")
    def team_size_positive(cls, v):
        if v < 1:
            raise ValueError("Team size must be at least 1")
        return v


class ProjectUpdate(BaseModel):
    name:              Optional[str]    = None
    start_date:        Optional[datetime] = None
    deadline:          Optional[datetime] = None
    team_size:         Optional[int]    = None
    green_threshold:   Optional[float]  = None
    red_threshold:     Optional[float]  = None
    gmail_filter_email: Optional[str]   = None
    gmail_project_identifier: Optional[str] = None
    gmail_account_email: Optional[str]  = None
    gmail_app_password: Optional[str]   = None  # encrypted before DB write
    jira_project_url:  Optional[str]    = None
    jira_email:        Optional[str]    = None
    jira_api_token:    Optional[str]    = None  # encrypted before DB write


class PMNoteUpdate(BaseModel):
    pm_note: str

    @validator("pm_note")
    def note_length(cls, v):
        if len(v) > 500:
            raise ValueError("PM note cannot exceed 500 characters")
        return v


class JiraTestRequest(BaseModel):
    jira_project_url: str
    jira_email:     str
    jira_api_token: str


class GmailTestRequest(BaseModel):
    gmail_account_email: str
    gmail_app_password: str


# ── Helper: check project belongs to current user's org ──────────────────────

def _get_project_for_user(project_id: int, user: User, db: Session) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.org_id != user.org_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if user.role == "pm" and project.created_by_user_id != user.id:
        raise HTTPException(status_code=403, detail="This project is not assigned to you")
    return project


def _project_response(project: Project, latest: Optional[HealthScore] = None) -> dict:
    """Serialize the public project shape used by all project endpoints."""
    sync_times = [
        value for value in (project.last_jira_synced_at, project.last_email_synced_at)
        if value is not None
    ]
    last_synced_at = max(sync_times).isoformat() if sync_times else None
    return {
        "id": project.id,
        "name": project.name,
        "start_date": project.start_date.isoformat(),
        "deadline": project.deadline.isoformat(),
        "team_size": project.team_size,
        "jira_project_url": project.jira_url,
        "jira_email": project.jira_email,
        "jira_connected": bool(project.encrypted_jira_token),
        "gmail_connected": bool(project.gmail_account_email and project.encrypted_gmail_token),
        "gmail_account_email": project.gmail_account_email,
        "gmail_filter_email": project.gmail_filter_email,
        "gmail_project_identifier": project.gmail_project_identifier,
        "green_threshold": project.green_threshold,
        "red_threshold": project.red_threshold,
        "pm_note": project.pm_note,
        "assigned_pm_id": project.created_by_user_id,
        "assigned_pm_name": project.pm.name if project.pm else None,
        "health_score": latest.health_score if latest else None,
        "rag_status": latest.rag_status if latest else None,
        "divergence_flag": latest.divergence_flag if latest else 0,
        "velocity_percent": latest.velocity_percent if latest else None,
        "overdue_rate": latest.overdue_rate if latest else None,
        "bug_ratio": latest.bug_ratio if latest else None,
        "recorded_at": latest.recorded_at.isoformat() if latest else None,
        "last_updated_at": last_synced_at or (latest.recorded_at.isoformat() if latest else None),
        "last_synced_at": last_synced_at,
        "last_jira_synced_at": project.last_jira_synced_at.isoformat() if project.last_jira_synced_at else None,
        "last_jira_synced_by": project.last_jira_synced_by,
        "last_email_synced_at": project.last_email_synced_at.isoformat() if project.last_email_synced_at else None,
        "last_email_synced_by": project.last_email_synced_by,
    }


def _create_initial_health_score(project: Project) -> HealthScore:
    """Provide the required unmeasured state until the first data sync."""
    return HealthScore(
        project_id=project.id,
        health_score=0.0,
        rag_status="RED",
        tone_score=0.0,
        urgency_flag=0,
        velocity_percent=0.0,
        overdue_rate=0.0,
        bug_ratio=0.0,
        divergence_flag=0,
    )


def _rag_for_score(score: float, green_threshold: float, red_threshold: float) -> str:
    if score >= green_threshold:
        return "GREEN"
    if score < red_threshold:
        return "RED"
    return "AMBER"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/")
def list_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return all projects in the current user's organisation."""
    query = db.query(Project).filter(Project.org_id == current_user.org_id)
    if current_user.role == "pm":
        query = query.filter(Project.created_by_user_id == current_user.id)
    projects = query.order_by(Project.created_at.desc()).all()

    result = []
    for p in projects:
        # Get latest health score for summary card
        latest = (
            db.query(HealthScore)
            .filter(HealthScore.project_id == p.id)
            .order_by(HealthScore.recorded_at.desc())
            .first()
        )
        result.append(_project_response(p, latest))
    return result


@router.post("/", status_code=201)
def create_project(
    data: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm", "org_admin")),
):
    """Create a project and assign it to a PM in the current organisation."""
    if not (data.jira_project_url and data.jira_email and data.jira_api_token):
        raise HTTPException(
            status_code=400,
            detail="Jira URL, account email, and API token must be validated before creating a project.",
        )
    # Repeat the wizard validation at the write boundary so a client cannot
    # bypass it and save unverified credentials.
    _validate_jira_connection(
        data.jira_project_url, data.jira_email, data.jira_api_token
    )
    if data.gmail_account_email or data.gmail_app_password:
        if not (data.gmail_account_email and data.gmail_app_password):
            raise HTTPException(status_code=400, detail="Provide both the Gmail account email and app password.")
        _validate_gmail_connection(data.gmail_account_email, data.gmail_app_password)
    # Encrypt integration secrets before storage.
    enc_jira = encrypt_token(data.jira_api_token) if data.jira_api_token else None
    enc_gmail = encrypt_token(data.gmail_app_password) if data.gmail_app_password else None

    assigned_pm_id = current_user.id
    if current_user.role == "org_admin":
        if not data.assigned_pm_id:
            raise HTTPException(status_code=400, detail="Select a project manager to assign this project")
        assignee = db.query(User).filter(
            User.id == data.assigned_pm_id,
            User.org_id == current_user.org_id,
            User.role == "pm",
            User.is_active == True,
        ).first()
        if not assignee:
            raise HTTPException(status_code=400, detail="Selected project manager is not active in your organisation")
        assigned_pm_id = assignee.id

    project = Project(
        org_id               = current_user.org_id,
        created_by_user_id   = assigned_pm_id,
        name                 = data.name,
        start_date           = data.start_date,
        deadline             = data.deadline,
        team_size            = data.team_size,
        jira_url             = data.jira_project_url,
        encrypted_jira_token = enc_jira,
        jira_email           = data.jira_email,
        gmail_account_email  = data.gmail_account_email,
        encrypted_gmail_token = enc_gmail,
        gmail_filter_email   = data.gmail_filter_email,
        gmail_project_identifier = data.gmail_project_identifier,
        green_threshold      = data.green_threshold,
        red_threshold        = data.red_threshold,
    )
    db.add(project)
    db.flush()
    # Zero is an explicit "not measured" state, never an estimate.
    db.add(_create_initial_health_score(project))
    db.commit()
    db.refresh(project)

    return _project_response(project)


@router.get("/{project_id}")
def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get full project details (excluding encrypted tokens)."""
    project = _get_project_for_user(project_id, current_user, db)
    latest = (db.query(HealthScore).filter(HealthScore.project_id == project.id)
              .order_by(HealthScore.recorded_at.desc()).first())
    return _project_response(project, latest)


@router.patch("/{project_id}")
def update_project(
    project_id: int,
    data: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm", "org_admin")),
):
    """Update project details or RAG thresholds."""
    project = _get_project_for_user(project_id, current_user, db)

    next_start_date = data.start_date or project.start_date
    next_deadline = data.deadline or project.deadline
    if next_deadline <= next_start_date:
        raise HTTPException(status_code=400, detail="Deadline must be after the start date")
    next_green = data.green_threshold if data.green_threshold is not None else project.green_threshold
    next_red = data.red_threshold if data.red_threshold is not None else project.red_threshold
    if not (0 < next_red < next_green <= 100):
        raise HTTPException(
            status_code=400,
            detail="Red threshold must be greater than 0 and lower than the Green threshold (maximum 100).",
        )

    if data.name is not None:
        project.name = data.name
    if data.start_date is not None:
        project.start_date = data.start_date
    if data.deadline is not None:
        project.deadline = data.deadline
    if data.team_size is not None:
        project.team_size = data.team_size
    if data.green_threshold is not None:
        project.green_threshold = data.green_threshold
    if data.red_threshold is not None:
        project.red_threshold = data.red_threshold

    # Thresholds classify the score, so changing them must immediately update
    # the current and historical RAG statuses—not only future pipeline runs.
    if data.green_threshold is not None or data.red_threshold is not None:
        for score_row in db.query(HealthScore).filter(HealthScore.project_id == project.id):
            score_row.rag_status = _rag_for_score(
                score_row.health_score, next_green, next_red
            )
    if data.gmail_filter_email is not None:
        project.gmail_filter_email = data.gmail_filter_email
    if data.gmail_project_identifier is not None:
        project.gmail_project_identifier = data.gmail_project_identifier.strip() or None
    gmail_settings_changed = data.gmail_account_email is not None or bool(data.gmail_app_password)
    if gmail_settings_changed:
        gmail_account_email = data.gmail_account_email if data.gmail_account_email is not None else project.gmail_account_email
        gmail_app_password = data.gmail_app_password or decrypt_token(project.encrypted_gmail_token or "")
        if not (gmail_account_email and gmail_app_password):
            raise HTTPException(status_code=400, detail="Gmail account email and app password are required to update email authorization.")
        _validate_gmail_connection(gmail_account_email, gmail_app_password)
    if data.gmail_account_email is not None:
        project.gmail_account_email = data.gmail_account_email
    if data.gmail_app_password:
        project.encrypted_gmail_token = encrypt_token(data.gmail_app_password)
    # Jira settings are a single credential set.  Validate the effective
    # values before changing any one of them, including when an existing token
    # is retained while the URL or email changes.
    jira_settings_changed = (
        data.jira_project_url is not None
        or data.jira_email is not None
        or bool(data.jira_api_token)
    )
    if jira_settings_changed:
        jira_url = data.jira_project_url if data.jira_project_url is not None else project.jira_url
        jira_email = data.jira_email if data.jira_email is not None else project.jira_email
        jira_token = data.jira_api_token or decrypt_token(project.encrypted_jira_token or "")
        if not (jira_url and jira_email and jira_token):
            raise HTTPException(
                status_code=400,
                detail="Jira URL, account email, and API token are required to update Jira settings.",
            )
        _validate_jira_connection(jira_url, jira_email, jira_token)

    if data.jira_project_url is not None:
        project.jira_url = data.jira_project_url
    if data.jira_email is not None:
        project.jira_email = data.jira_email
    if data.jira_api_token:
        project.encrypted_jira_token = encrypt_token(data.jira_api_token)

    db.commit()
    db.refresh(project)
    latest = (db.query(HealthScore).filter(HealthScore.project_id == project.id)
              .order_by(HealthScore.recorded_at.desc()).first())
    return _project_response(project, latest)


@router.delete("/{project_id}")
def delete_project(
    project_id: int,
    confirm_name: str,          # must match project name exactly
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm", "org_admin")),
):
    """Delete a project. Requires typing the project name as confirmation."""
    project = _get_project_for_user(project_id, current_user, db)

    if confirm_name != project.name:
        raise HTTPException(
            status_code=400,
            detail="Project name does not match. Deletion cancelled.",
        )

    db.delete(project)
    db.commit()
    return {"message": f"Project '{project.name}' deleted permanently"}


@router.get("/{project_id}/health-score")
def get_health_score(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns the latest health score plus the last 20 historical records
    for rendering all PM Dashboard charts.
    """
    project = _get_project_for_user(project_id, current_user, db)

    latest = (
        db.query(HealthScore)
        .filter(HealthScore.project_id == project_id)
        .order_by(HealthScore.recorded_at.desc())
        .first()
    )
    if not latest:
        # Backfill older projects with the same unmeasured state.
        latest = _create_initial_health_score(project)
        db.add(latest)
        db.commit()
        db.refresh(latest)

    # History for charts — ascending order so charts render left → right
    sync_times = [
        value for value in (project.last_jira_synced_at, project.last_email_synced_at)
        if value is not None
    ]
    last_synced_at = max(sync_times).isoformat() if sync_times else None

    history_rows = (
        db.query(HealthScore)
        .filter(HealthScore.project_id == project_id)
        .order_by(HealthScore.recorded_at.asc())
        .limit(20)
        .all()
    )

    def _contributions(row: HealthScore) -> dict:
        return ProjectHealthService.explain_saved_score(row)

    history = [
        {
            "date":             row.recorded_at.strftime("%b %d"),
            "recorded_at":      row.recorded_at.isoformat(),
            "score":            row.health_score,
            "health_score":     row.health_score,
            "tone":             row.tone_score,
            "urgency_flag":     row.urgency_flag,
            "velocity_percent": row.velocity_percent,
            "overdue_rate":      row.overdue_rate,
            "bug_ratio":         row.bug_ratio,
            "contributions":     _contributions(row),
        }
        for row in history_rows
    ]

    # ── Compute contribution breakdown for the Score Analysis card ──────────
    # Count actual email and transcript records in the database to split
    # the communication_sentiment proportionally by source.
    email_rows = db.query(ProcessedEmail).filter(
        ProcessedEmail.project_id == project_id
    ).all()
    transcript_rows = db.query(TranscriptUpload).filter(
        TranscriptUpload.project_id == project_id
    ).all()
    email_count = len(email_rows)
    transcript_count = len(transcript_rows)

    # ``explain_health_score`` exposes urgency separately for diagnostics.
    # In the dashboard hierarchy it belongs to communication: urgency is
    # derived from email/transcript evidence. This makes the two top-level
    # explanation values reconcile exactly with the newly predicted score.
    raw_contribs = _contributions(latest)
    sentiment_value = round(
        raw_contribs.get("communication_sentiment", 0.0)
        + raw_contribs.get("urgency_penalty", 0.0),
        2,
    )
    # Attribute the latest sentiment result using the actual numeric evidence,
    # rather than merely splitting it by record count. A negative email can
    # therefore reduce the email share, while a positive one raises it.
    def _sentiment_weight(rows) -> float:
        return sum(
            max(0.0, min(1.0, (float(row.tone_score) + 1.0) / 2.0))
            for row in rows
            if row.tone_score is not None
        )

    email_weight = _sentiment_weight(email_rows)
    transcript_weight = _sentiment_weight(transcript_rows)
    total_weight = email_weight + transcript_weight
    if total_weight:
        email_analysis = round(sentiment_value * (email_weight / total_weight), 2)
        transcript_analysis = round(sentiment_value - email_analysis, 2)
    else:
        # Existing projects created before numeric evidence support have no
        # per-source tone values yet. Keep their display stable until fresh
        # email/transcript evidence is ingested.
        total_sources = max(1, email_count + transcript_count)
        email_analysis = round(sentiment_value * (email_count / total_sources), 2)
        transcript_analysis = round(sentiment_value - email_analysis, 2)

    delivery_metrics_value = raw_contribs.get("delivery_metrics", 0.0)
    total_contribution = round(email_analysis + transcript_analysis + delivery_metrics_value, 2)
    # Preserve the individual delivery attribution and diagnostic urgency
    # fields, but return the dashboard's reconciled top-level hierarchy.
    contribs = {
        **raw_contribs,
        "communication_sentiment": round(email_analysis + transcript_analysis, 2),
        "delivery_metrics": delivery_metrics_value,
        # Its effect is already included in communication_sentiment above.
        "urgency_penalty": 0.0,
    }

    # The dashboard uses the same contribution builder as event responses.
    # Explanations are read-only and are generated from the already-persisted
    # prediction; they never participate in score calculation.
    health_service = ProjectHealthService(db)
    dashboard_contributions = health_service.calculate_contributions(
        health_score=latest.health_score,
        tone_score=latest.tone_score,
        urgency_flag=latest.urgency_flag,
        jira_metrics={
            "velocity_percent": latest.velocity_percent,
            "overdue_rate": latest.overdue_rate,
            "bug_ratio": latest.bug_ratio,
        },
        communication=health_service.collect_project_communication_data(project_id),
    )
    contribs = dashboard_contributions
    email_analysis = dashboard_contributions["email_analysis"]
    transcript_analysis = dashboard_contributions["transcript_analysis"]
    delivery_metrics_value = dashboard_contributions["delivery_metrics"]
    total_contribution = dashboard_contributions["total_contribution"]

    return {
        "project_name":    project.name,
        "deadline":        project.deadline.isoformat(),
        # Latest score
        "health_score":    latest.health_score,
        "rag_status":      latest.rag_status,
        "tone_score":      latest.tone_score,
        "urgency_flag":    latest.urgency_flag,
        "velocity_percent": latest.velocity_percent,
        "overdue_rate":    latest.overdue_rate,
        "bug_ratio":       latest.bug_ratio,
        "divergence_flag": latest.divergence_flag,
        "contributions": contribs,
        "recorded_at":     latest.recorded_at.isoformat(),
        "last_updated_at":  last_synced_at or latest.recorded_at.isoformat(),
        "last_synced_at":   last_synced_at,
        "last_jira_synced_at": project.last_jira_synced_at.isoformat() if project.last_jira_synced_at else None,
        "last_jira_synced_by": project.last_jira_synced_by,
        "last_email_synced_at": project.last_email_synced_at.isoformat() if project.last_email_synced_at else None,
        "last_email_synced_by": project.last_email_synced_by,
        # Thresholds (for chart reference lines)
        "green_threshold": project.green_threshold,
        "red_threshold":   project.red_threshold,
        # PM note for share panel
        "pm_note":         project.pm_note,
        # History array for LineChart and AreaChart
        "history":         history,
        # Contribution breakdown for Score Analysis transparency (Requirement 5)
        "email_analysis":         email_analysis,
        "transcript_analysis":    transcript_analysis,
        "sentiment_total":        round(email_analysis + transcript_analysis, 2),
        "delivery_metrics":       delivery_metrics_value,
        "total_contribution":     total_contribution,
        "email_count":            email_count,
        "transcript_count":       transcript_count,
    }


@router.patch("/{project_id}/pm-note")
def update_pm_note(
    project_id: int,
    body: PMNoteUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm", "org_admin")),
):
    """Update the PM note shown on the client dashboard."""
    project = _get_project_for_user(project_id, current_user, db)
    project.pm_note = body.pm_note
    db.commit()
    return {"message": "PM note saved"}


@router.post("/{project_id}/sync-jira")
def sync_jira_now(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm", "org_admin")),
):
    """Fetch Jira signals now and create a fresh health-score record."""
    project = _get_project_for_user(project_id, current_user, db)
    if not project.jira_url or not project.encrypted_jira_token:
        raise HTTPException(status_code=400, detail="Jira is not configured for this project")

    from app.api.pipeline import sync_jira_project
    from app.ml.jira_signals import JiraSignalError, _cloud_base_and_project_key

    try:
        # Older projects may have been saved with only the site URL.  A
        # project key is essential: without it we could not ensure that the
        # figures belong to this project rather than the whole Jira site.
        _cloud_base_and_project_key(project.jira_url)
    except JiraSignalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return sync_jira_project(db, project, actor=current_user)


@router.post("/{project_id}/sync-gmail")
def sync_gmail_now(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm", "org_admin")),
):
    """Poll the monitored Gmail inbox now, for an auditable integration test."""
    project = _get_project_for_user(project_id, current_user, db)
    from app.api.pipeline import sync_gmail_project
    from app.services.gmail_ingestion import GmailIngestionError

    try:
        return sync_gmail_project(db, project, actor=current_user)
    except GmailIngestionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/test-jira")
def test_jira_connection(
    body: JiraTestRequest,
    current_user: User = Depends(get_current_user),
):
    """Test a Jira connection before saving it."""
    return _validate_jira_connection(
        body.jira_project_url, body.jira_email, body.jira_api_token
    )


@router.post("/test-gmail")
def test_gmail_connection(
    body: GmailTestRequest,
    current_user: User = Depends(get_current_user),
):
    """Test Gmail IMAP app-password authorisation without saving it."""
    _validate_gmail_connection(body.gmail_account_email, body.gmail_app_password)
    return {"success": True, "message": "Gmail authorization successful"}


def _validate_gmail_connection(gmail_account_email: str, gmail_app_password: str) -> None:
    from app.services.gmail_ingestion import GmailIngestionError, validate_imap_credentials
    try:
        validate_imap_credentials(gmail_account_email, gmail_app_password)
    except GmailIngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_jira_connection(jira_project_url: str, jira_email: str, jira_api_token: str) -> dict:
    """Validate Jira Cloud account credentials and access to the configured project."""
    try:
        from app.ml.jira_signals import JiraSignalError, _cloud_base_and_project_key
        url, project_key = _cloud_base_and_project_key(jira_project_url)
        resp = httpx.get(
            f"{url}/rest/api/3/myself",
            auth=(jira_email, jira_api_token),
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            # Authentication alone is not enough: the account must be able to
            # browse the exact project encoded in the URL before credentials
            # can be stored and scheduled for syncing.
            project_resp = httpx.get(
                f"{url}/rest/api/3/project/{project_key}",
                auth=(jira_email, jira_api_token),
                headers={"Accept": "application/json"},
                timeout=10,
            )
            if project_resp.status_code == 200:
                data = resp.json()
                project_data = project_resp.json()
                return {
                    "success":      True,
                    "message":      "Jira connection successful",
                    "display_name": data.get("displayName", ""),
                    "account_id":   data.get("accountId", ""),
                    "project_key":  project_data.get("key", project_key),
                    "project_name": project_data.get("name", ""),
                }
            if project_resp.status_code in {403, 404}:
                raise HTTPException(
                    status_code=400,
                    detail="The Jira account cannot access the project in the supplied URL.",
                )
            raise HTTPException(
                status_code=400,
                detail=f"Jira project validation returned status {project_resp.status_code}.",
            )
        if resp.status_code == 401:
            raise HTTPException(
                status_code=400,
                detail="Authentication failed. Check your email and API token.",
            )
        raise HTTPException(
            status_code=400,
            detail=f"Jira returned status {resp.status_code}. Check your URL and permissions.",
        )
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=400,
            detail="Cannot connect to Jira. Check the URL is correct and reachable.",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=400,
            detail="Connection timed out. The Jira server may be slow or unreachable.",
        ) from exc
    except JiraSignalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
