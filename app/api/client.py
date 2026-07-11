"""
Client router — share link generation, revocation, and the public client dashboard.

Endpoints:
  POST /client/projects/{id}/share-link    — generate a share link (PM only)
  POST /client/projects/{id}/revoke-link   — revoke active link (PM only)
  GET  /client/{token}                     — public client dashboard data

NFR-13: 256-bit cryptographically secure random token
NFR-14: Only SHA-256 hash stored in database — never raw token
NFR-17: Identical HTTP 403 for expired and revoked tokens (no info leakage)
NFR-18: /client path isolated from /dashboard
NFR-16: Client endpoint returns ONLY 6 safe fields — never raw ML signals
"""
import hashlib
import secrets
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.utils.database import get_db
from app.api.auth import get_current_user, require_role
from app.models.models import ClientShareToken, HealthScore, Project, User
from app.utils.time import utcnow

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_token(raw_token: str) -> str:
    """SHA-256 hash a token string. Returns 64-character lowercase hex string."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _get_project_for_pm(project_id: int, user: User, db: Session) -> Project:
    """Fetch a project ensuring it belongs to the user's org."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.org_id != user.org_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return project


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ShareLinkRequest(BaseModel):
    days: int = 30   # 0 = never expires; positive integer = expiry days


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/projects/{project_id}/share-link")
def generate_share_link(
    project_id: int,
    body: ShareLinkRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm")),
):
    """
    Generate a new 256-bit secure share link for the client dashboard.

    FR-53: Generate cryptographically secure token.
    FR-54: Return the share URL once — it is never stored in full.
    FR-59: Revoke all previous active tokens for this project.
    NFR-13: secrets.token_urlsafe(32) generates 256-bit random token.
    NFR-14: Only SHA-256 hash stored.
    """
    _get_project_for_pm(project_id, current_user, db)

    # Generate raw token — returned to PM, never stored
    raw_token = secrets.token_urlsafe(32)   # 32 bytes = 256 bits

    # Compute expiry
    expiry = (
        utcnow() + timedelta(days=body.days)
        if body.days > 0
        else None
    )

    # FR-59: Revoke all existing active tokens for this project
    db.query(ClientShareToken).filter(
        ClientShareToken.project_id == project_id,
        ClientShareToken.revoked == False,
    ).update({"revoked": True})

    # Store ONLY the hash
    new_token = ClientShareToken(
        project_id         = project_id,
        token_hash         = _hash_token(raw_token),
        expiry_date        = expiry,
        created_by_user_id = current_user.id,
        revoked            = False,
    )
    db.add(new_token)
    db.commit()

    frontend_url = __import__("os").getenv("FRONTEND_URL", "http://localhost:5173")

    return {
        "share_url": f"{frontend_url}/client/{raw_token}",
        "expires":   expiry.isoformat() if expiry else "Never",
        "message":   "Share link generated. Send this URL to your client.",
    }


@router.post("/projects/{project_id}/revoke-link")
def revoke_share_link(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm")),
):
    """
    Revoke all active share tokens for a project.
    The client will immediately receive HTTP 403 on next access.
    """
    _get_project_for_pm(project_id, current_user, db)

    updated = db.query(ClientShareToken).filter(
        ClientShareToken.project_id == project_id,
        ClientShareToken.revoked == False,
    ).update({"revoked": True})

    db.commit()

    return {
        "message": f"{updated} token(s) revoked. Client access is now disabled.",
    }


@router.get("/projects/{project_id}/link-status")
def get_link_status(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("pm")),
):
    """Check whether an active share link exists for this project."""
    _get_project_for_pm(project_id, current_user, db)

    active = db.query(ClientShareToken).filter(
        ClientShareToken.project_id == project_id,
        ClientShareToken.revoked == False,
    ).first()

    if not active:
        return {"has_active_link": False}

    # Check expiry
    if active.expiry_date and active.expiry_date < utcnow():
        return {"has_active_link": False}

    return {
        "has_active_link": True,
        "expires":         active.expiry_date.isoformat() if active.expiry_date else "Never",
        "last_accessed":   active.last_accessed.isoformat() if active.last_accessed else None,
        "created_at":      active.created_at.isoformat(),
    }


@router.get("/{token}")
def get_client_view(
    token: str,
    db: Session = Depends(get_db),
):
    """
    PUBLIC endpoint — no authentication required.
    Validates the share token and returns restricted project health data.

    NFR-17: Returns identical HTTP 403 for expired, revoked, and invalid tokens.
    NFR-18: Isolated at /client/ path.
    NFR-16: Returns ONLY 6 fields — no raw ML signals exposed.
    """
    # Hash the incoming token and look it up
    token_hash = _hash_token(token)

    record = db.query(ClientShareToken).filter(
        ClientShareToken.token_hash == token_hash,
        ClientShareToken.revoked    == False,
    ).first()

    # NFR-17: identical 403 for all failure cases — no information leakage
    if not record:
        raise HTTPException(status_code=403, detail="Access denied")

    if record.expiry_date and record.expiry_date < utcnow():
        raise HTTPException(status_code=403, detail="Access denied")

    # Fetch latest health score
    latest = (
        db.query(HealthScore)
        .filter(HealthScore.project_id == record.project_id)
        .order_by(HealthScore.recorded_at.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="No health data available yet")

    # Fetch project for name, note, deadline, and thresholds
    project = db.query(Project).filter(Project.id == record.project_id).first()

    # Fetch history for the simplified trend chart (scores only)
    history_rows = (
        db.query(HealthScore)
        .filter(HealthScore.project_id == record.project_id)
        .order_by(HealthScore.recorded_at.asc())
        .limit(20)
        .all()
    )
    history = [
        {
            "date":  row.recorded_at.strftime("%b"),
            "score": row.health_score,
        }
        for row in history_rows
    ]

    # Update last accessed timestamp for audit log
    record.last_accessed = utcnow()
    db.commit()

    # NFR-16: ONLY these 6 fields — NEVER tone_score, velocity, overdue, bug_ratio, urgency
    return {
        "project_name":    project.name,
        "deadline":        project.deadline.isoformat(),
        "health_score":    round(latest.health_score, 1),
        "rag_status":      latest.rag_status,
        "recorded_at":     latest.recorded_at.isoformat(),
        "pm_note":         project.pm_note,
        "divergence_flag": latest.divergence_flag,
        "green_threshold": project.green_threshold,
        "red_threshold":   project.red_threshold,
        "history":         history,
    }
