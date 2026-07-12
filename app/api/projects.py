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
from app.models.models import Project, HealthScore, User

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
    return {
        "id": project.id,
        "name": project.name,
        "start_date": project.start_date.isoformat(),
        "deadline": project.deadline.isoformat(),
        "team_size": project.team_size,
        "jira_project_url": project.jira_url,
        "jira_email": project.jira_email,
        "jira_connected": bool(project.encrypted_jira_token),
        "gmail_connected": bool(project.encrypted_gmail_token),
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
    # Encrypt Jira token before storage
    enc_jira = encrypt_token(data.jira_api_token) if data.jira_api_token else None

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
    history_rows = (
        db.query(HealthScore)
        .filter(HealthScore.project_id == project_id)
        .order_by(HealthScore.recorded_at.asc())
        .limit(20)
        .all()
    )

    history = [
        {
            "date":             row.recorded_at.strftime("%b %d"),
            "recorded_at":      row.recorded_at.isoformat(),
            "score":            row.health_score,
            "tone":             row.tone_score,
            "velocity_percent": row.velocity_percent,
        }
        for row in history_rows
    ]

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
        "recorded_at":     latest.recorded_at.isoformat(),
        # Thresholds (for chart reference lines)
        "green_threshold": project.green_threshold,
        "red_threshold":   project.red_threshold,
        # PM note for share panel
        "pm_note":         project.pm_note,
        # History array for LineChart and AreaChart
        "history":         history,
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

    try:
        return sync_jira_project(db, project)
    except JiraSignalError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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
        return sync_gmail_project(db, project)
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


def _validate_jira_connection(jira_project_url: str, jira_email: str, jira_api_token: str) -> dict:
    """Validate Jira credentials without persisting either secret."""
    try:
        from app.ml.jira_signals import JiraSignalError, _cloud_base_and_project_key
        url, _ = _cloud_base_and_project_key(jira_project_url)
        resp = httpx.get(
            f"{url}/rest/api/3/myself",
            auth=(jira_email, jira_api_token),
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success":      True,
                "message":      "Jira connection successful",
                "display_name": data.get("displayName", ""),
                "account_id":   data.get("accountId", ""),
            }
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
