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
    jira_url:         Optional[str] = None
    jira_api_token:   Optional[str] = None   # plaintext — encrypted before DB write
    jira_email:       Optional[str] = None
    gmail_filter_email: Optional[str] = None
    green_threshold:  float = 70.0
    red_threshold:    float = 40.0

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
    deadline:          Optional[datetime] = None
    team_size:         Optional[int]    = None
    green_threshold:   Optional[float]  = None
    red_threshold:     Optional[float]  = None
    gmail_filter_email: Optional[str]   = None


class PMNoteUpdate(BaseModel):
    pm_note: str

    @validator("pm_note")
    def note_length(cls, v):
        if len(v) > 500:
            raise ValueError("PM note cannot exceed 500 characters")
        return v


class JiraTestRequest(BaseModel):
    jira_url:       str
    jira_email:     str
    jira_api_token: str


# ── Helper: check project belongs to current user's org ──────────────────────

def _get_project_for_user(project_id: int, user: User, db: Session) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.org_id != user.org_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return project


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/")
def list_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return all projects in the current user's organisation."""
    projects = (
        db.query(Project)
        .filter(Project.org_id == current_user.org_id)
        .order_by(Project.created_at.desc())
        .all()
    )

    result = []
    for p in projects:
        # Get latest health score for summary card
        latest = (
            db.query(HealthScore)
            .filter(HealthScore.project_id == p.id)
            .order_by(HealthScore.recorded_at.desc())
            .first()
        )
        result.append({
            "id":               p.id,
            "name":             p.name,
            "deadline":         p.deadline.isoformat(),
            "team_size":        p.team_size,
            "green_threshold":  p.green_threshold,
            "red_threshold":    p.red_threshold,
            "health_score":     latest.health_score     if latest else None,
            "rag_status":       latest.rag_status       if latest else None,
            "velocity_percent": latest.velocity_percent if latest else None,
            "overdue_rate":     latest.overdue_rate     if latest else None,
            "bug_ratio":        latest.bug_ratio        if latest else None,
            "divergence_flag":  latest.divergence_flag  if latest else 0,
            "recorded_at":      latest.recorded_at.isoformat() if latest else None,
        })
    return result


@router.post("/", status_code=201)
def create_project(
    data: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm")),
):
    """Create a new project. Only PMs can create projects."""
    # Encrypt Jira token before storage
    enc_jira = encrypt_token(data.jira_api_token) if data.jira_api_token else None

    project = Project(
        org_id               = current_user.org_id,
        created_by_user_id   = current_user.id,
        name                 = data.name,
        start_date           = data.start_date,
        deadline             = data.deadline,
        team_size            = data.team_size,
        jira_url             = data.jira_url,
        encrypted_jira_token = enc_jira,
        jira_email           = data.jira_email,
        gmail_filter_email   = data.gmail_filter_email,
        green_threshold      = data.green_threshold,
        red_threshold        = data.red_threshold,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    return {"id": project.id, "name": project.name, "message": "Project created"}


@router.get("/{project_id}")
def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get full project details (excluding encrypted tokens)."""
    project = _get_project_for_user(project_id, current_user, db)
    return {
        "id":                  project.id,
        "name":                project.name,
        "start_date":          project.start_date.isoformat(),
        "deadline":            project.deadline.isoformat(),
        "team_size":           project.team_size,
        "jira_url":            project.jira_url,
        "jira_email":          project.jira_email,
        "jira_connected":      bool(project.encrypted_jira_token),
        "gmail_connected":     bool(project.encrypted_gmail_token),
        "gmail_filter_email":  project.gmail_filter_email,
        "green_threshold":     project.green_threshold,
        "red_threshold":       project.red_threshold,
        "pm_note":             project.pm_note,
    }


@router.patch("/{project_id}")
def update_project(
    project_id: int,
    data: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm", "org_admin")),
):
    """Update project details or RAG thresholds."""
    project = _get_project_for_user(project_id, current_user, db)

    if data.name is not None:
        project.name = data.name
    if data.deadline is not None:
        project.deadline = data.deadline
    if data.team_size is not None:
        project.team_size = data.team_size
    if data.green_threshold is not None:
        project.green_threshold = data.green_threshold
    if data.red_threshold is not None:
        project.red_threshold = data.red_threshold
    if data.gmail_filter_email is not None:
        project.gmail_filter_email = data.gmail_filter_email

    db.commit()
    return {"message": "Project updated"}


@router.delete("/{project_id}")
def delete_project(
    project_id: int,
    confirm_name: str,          # must match project name exactly
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm")),
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
        raise HTTPException(
            status_code=404,
            detail="No health score data yet. Connect Jira and wait for the first sync.",
        )

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
            "health_score":     row.health_score,
            "tone_score":       row.tone_score,
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
    current_user: User = Depends(require_role("pm")),
):
    """Update the PM note shown on the client dashboard."""
    project = _get_project_for_user(project_id, current_user, db)
    project.pm_note = body.pm_note
    db.commit()
    return {"message": "PM note saved"}


@router.post("/test-jira")
def test_jira_connection(
    body: JiraTestRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Test a Jira connection before saving it.
    Called from New Project Wizard Step 2 before Continue button is enabled.
    """
    try:
        url  = body.jira_url.rstrip("/")
        auth = (body.jira_email, body.jira_api_token)
        resp = httpx.get(
            f"{url}/rest/api/3/myself",
            auth=auth,
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success":      True,
                "display_name": data.get("displayName", ""),
                "account_id":   data.get("accountId", ""),
            }
        elif resp.status_code == 401:
            raise HTTPException(
                status_code=400,
                detail="Authentication failed. Check your email and API token.",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Jira returned status {resp.status_code}. Check your URL.",
            )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=400,
            detail="Cannot connect to Jira. Check the URL is correct and reachable.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=400,
            detail="Connection timed out. The Jira server may be slow or unreachable.",
        )
