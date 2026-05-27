"""
Admin router — organisation management, user creation, audit views.

Endpoints:
  GET    /admin/organisations             — list all orgs (super_admin)
  POST   /admin/organisations             — create org (super_admin)
  GET    /admin/users                     — list users in org (org_admin / super_admin)
  POST   /admin/users                     — create PM or Org Admin account
  PATCH  /admin/users/{id}/deactivate     — deactivate a user
  GET    /admin/projects/active           — list active projects (called by n8n)
  GET    /admin/share-links               — audit all share tokens (super_admin)
  GET    /admin/system-health             — system status overview (super_admin)
  GET    /admin/export-research           — download anonymised research CSV
  GET    /admin/org/projects              — org admin: view all projects in org
  GET    /admin/org/team                  — org admin: view PMs in org
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
from passlib.context import CryptContext
import io
import csv

from app.utils.database import get_db
from app.api.auth import get_current_user, require_role
from app.models.models import (
    Organisation, User, Project, HealthScore, ClientShareToken
)

router   = APIRouter()
pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class OrganisationCreate(BaseModel):
    name:     str
    industry: Optional[str] = None


class UserCreate(BaseModel):
    full_name:          str
    email:              str
    role:               str          # "org_admin" | "pm"
    temporary_password: str
    org_id:             Optional[int] = None   # required when super_admin creates org_admin


# ── Super Admin endpoints ─────────────────────────────────────────────────────

@router.get("/organisations")
def list_organisations(
    db: Session = Depends(get_db),
    _: User = Depends(require_role("super_admin")),
):
    """List all organisations with summary stats."""
    orgs = db.query(Organisation).order_by(Organisation.created_at.desc()).all()
    result = []
    for org in orgs:
        project_count = db.query(func.count(Project.id)).filter(
            Project.org_id == org.id
        ).scalar()

        # Average health score across all org projects
        avg_health = db.query(func.avg(HealthScore.health_score)).join(Project).filter(
            Project.org_id == org.id
        ).scalar()

        result.append({
            "id":             org.id,
            "name":           org.name,
            "industry":       org.industry,
            "total_projects": project_count,
            "avg_health":     round(float(avg_health), 1) if avg_health else None,
            "created_at":     org.created_at.isoformat(),
        })
    return result


@router.post("/organisations", status_code=201)
def create_organisation(
    data: OrganisationCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("super_admin")),
):
    """Create a new organisation."""
    existing = db.query(Organisation).filter(
        Organisation.name == data.name
    ).first()
    if existing:
        raise HTTPException(status_code=400,
                            detail="An organisation with this name already exists")

    org = Organisation(name=data.name, industry=data.industry)
    db.add(org)
    db.commit()
    db.refresh(org)
    return {"id": org.id, "name": org.name, "message": "Organisation created"}


@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("super_admin", "org_admin")),
):
    """
    List users.
    super_admin sees all users.
    org_admin sees only users in their own organisation.
    """
    query = db.query(User)
    if current_user.role == "org_admin":
        query = query.filter(User.org_id == current_user.org_id)
    users = query.order_by(User.created_at.desc()).all()

    return [
        {
            "id":         u.id,
            "name":       u.name,
            "email":      u.email,
            "role":       u.role,
            "org_id":     u.org_id,
            "is_active":  u.is_active,
            "last_login": u.last_login.isoformat() if u.last_login else None,
        }
        for u in users
    ]


@router.post("/users", status_code=201)
def create_user(
    data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("super_admin", "org_admin")),
):
    """
    Create a new user account.
    super_admin can create org_admin accounts for any org.
    org_admin can only create pm accounts within their own org.
    """
    # Role-based restrictions
    if current_user.role == "org_admin":
        if data.role != "pm":
            raise HTTPException(status_code=403,
                                detail="Organisation admins can only create PM accounts")
        target_org_id = current_user.org_id
    else:
        # super_admin
        if data.role not in ("org_admin", "pm"):
            raise HTTPException(status_code=400,
                                detail="Role must be 'org_admin' or 'pm'")
        if data.role == "org_admin" and not data.org_id:
            raise HTTPException(status_code=400,
                                detail="org_id is required when creating an org_admin")
        target_org_id = data.org_id or current_user.org_id

    # Check email uniqueness
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400,
                            detail="A user with this email already exists")

    # Validate password strength
    pwd = data.temporary_password
    if len(pwd) < 8:
        raise HTTPException(status_code=400,
                            detail="Password must be at least 8 characters")

    new_user = User(
        org_id               = target_org_id,
        name                 = data.full_name,
        email                = data.email,
        password_hash        = pwd_ctx.hash(pwd),
        role                 = data.role,
        is_active            = True,
        force_password_change = True,   # new accounts must change password on first login
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "id":      new_user.id,
        "email":   new_user.email,
        "role":    new_user.role,
        "message": f"Account created. Login credentials should be sent to {data.email}.",
    }


@router.patch("/users/{user_id}/deactivate")
def deactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("super_admin", "org_admin")),
):
    """Deactivate a user account. Their projects remain but they cannot log in."""
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    # Org admins can only deactivate users in their own org
    if current_user.role == "org_admin" and target.org_id != current_user.org_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Cannot deactivate yourself
    if target.id == current_user.id:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account")

    target.is_active = False
    db.commit()
    return {"message": f"Account for {target.email} has been deactivated"}


# ── n8n-facing endpoint ───────────────────────────────────────────────────────

@router.get("/projects/active")
def get_active_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Called by n8n every 4 hours to get the list of projects to process.
    Returns id, jira_url, and jira_email for each project.
    """
    projects = (
        db.query(Project)
        .filter(Project.org_id == current_user.org_id)
        .all()
    )
    return [
        {
            "id":         p.id,
            "name":       p.name,
            "jira_url":   p.jira_url,
            "jira_email": p.jira_email,
        }
        for p in projects
    ]


# ── Audit & reporting ─────────────────────────────────────────────────────────

@router.get("/share-links")
def get_share_link_audit(
    status: Optional[str] = None,   # "active" | "expired" | "revoked"
    db: Session = Depends(get_db),
    _: User = Depends(require_role("super_admin")),
):
    """Share link audit log for the Super Admin panel."""
    query = db.query(ClientShareToken)

    if status == "active":
        query = query.filter(ClientShareToken.revoked == False)
    elif status == "revoked":
        query = query.filter(ClientShareToken.revoked == True)

    tokens = query.order_by(ClientShareToken.created_at.desc()).all()
    now    = datetime.utcnow()

    result = []
    for t in tokens:
        if t.revoked:
            computed_status = "revoked"
        elif t.expiry_date and t.expiry_date < now:
            computed_status = "expired"
        else:
            computed_status = "active"

        if status == "expired" and computed_status != "expired":
            continue

        result.append({
            "id":            t.id,
            "project_id":    t.project_id,
            "token_prefix":  t.token_hash[:8] + "...",  # show only first 8 chars
            "status":        computed_status,
            "expiry_date":   t.expiry_date.isoformat() if t.expiry_date else "Never",
            "last_accessed": t.last_accessed.isoformat() if t.last_accessed else None,
            "created_at":    t.created_at.isoformat(),
        })

    return result


@router.get("/system-health")
def get_system_health(
    db: Session = Depends(get_db),
    _: User = Depends(require_role("super_admin")),
):
    """Quick system health overview for the Super Admin System tab."""
    try:
        # Test DB connection
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    total_orgs     = db.query(func.count(Organisation.id)).scalar()
    total_projects = db.query(func.count(Project.id)).scalar()
    total_scores   = db.query(func.count(HealthScore.id)).scalar()

    latest_score = (
        db.query(HealthScore)
        .order_by(HealthScore.recorded_at.desc())
        .first()
    )

    return {
        "database":         db_status,
        "total_orgs":       total_orgs,
        "total_projects":   total_projects,
        "total_scores":     total_scores,
        "last_ingestion":   latest_score.recorded_at.isoformat() if latest_score else None,
        "checked_at":       datetime.utcnow().isoformat(),
    }


@router.get("/export-research")
def export_research_csv(
    db: Session = Depends(get_db),
    _: User = Depends(require_role("super_admin")),
):
    """
    Download anonymised health score data for research analysis.
    Returns a CSV with NO personal data — only org_id, project_id, and ML scores.
    FR-80: Research export endpoint.
    """
    scores = (
        db.query(HealthScore)
        .join(Project, HealthScore.project_id == Project.id)
        .order_by(HealthScore.recorded_at.asc())
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)

    # Headers — NO names, NO emails, NO personal data
    writer.writerow([
        "org_id", "project_id", "health_score", "rag_status",
        "tone_score", "urgency_flag", "velocity_percent",
        "overdue_rate", "bug_ratio", "divergence_flag", "recorded_at",
    ])

    for s in scores:
        project = db.query(Project).filter(Project.id == s.project_id).first()
        writer.writerow([
            project.org_id,
            s.project_id,
            s.health_score,
            s.rag_status,
            s.tone_score,
            s.urgency_flag,
            s.velocity_percent,
            s.overdue_rate,
            s.bug_ratio,
            s.divergence_flag,
            s.recorded_at.isoformat(),
        ])

    output.seek(0)
    filename = f"phps_research_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Org Admin view endpoints ──────────────────────────────────────────────────

@router.get("/org/projects")
def org_admin_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("org_admin")),
):
    """Org Admin view of all projects in their organisation."""
    projects = (
        db.query(Project)
        .filter(Project.org_id == current_user.org_id)
        .order_by(Project.created_at.desc())
        .all()
    )

    result = []
    for p in projects:
        latest = (
            db.query(HealthScore)
            .filter(HealthScore.project_id == p.id)
            .order_by(HealthScore.recorded_at.desc())
            .first()
        )
        pm = db.query(User).filter(User.id == p.created_by_user_id).first()
        result.append({
            "id":           p.id,
            "name":         p.name,
            "pm_name":      pm.name if pm else "Unassigned",
            "start_date":   p.start_date.isoformat(),
            "deadline":     p.deadline.isoformat(),
            "health_score": latest.health_score if latest else None,
            "rag_status":   latest.rag_status   if latest else None,
            "recorded_at":  latest.recorded_at.isoformat() if latest else None,
        })
    return result


@router.get("/org/team")
def org_admin_team(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("org_admin")),
):
    """Org Admin view of all PMs in their organisation."""
    pms = (
        db.query(User)
        .filter(User.org_id == current_user.org_id, User.role == "pm")
        .all()
    )

    result = []
    for pm in pms:
        project_count = db.query(func.count(Project.id)).filter(
            Project.created_by_user_id == pm.id
        ).scalar()
        result.append({
            "id":            pm.id,
            "name":          pm.name,
            "email":         pm.email,
            "is_active":     pm.is_active,
            "project_count": project_count,
            "last_login":    pm.last_login.isoformat() if pm.last_login else None,
        })
    return result
