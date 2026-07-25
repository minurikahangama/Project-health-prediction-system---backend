"""Versioned Team Capacity & Delivery Intelligence endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.api.auth import get_current_user
from app.models.models import HealthScore, Project
from app.services.capacity_service import CapacityService
from app.services.dependency_service import DependencyService
from app.services.absence_simulator import AbsenceSimulator
from app.services.identity_resolver import IdentityResolver
from app.utils.database import get_db

router = APIRouter()


class AbsenceRequest(BaseModel):
    project_id: int
    developer_id: str
    unavailable_days: float = Field(ge=0)


def _project(project_id: int, user, db: Session):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.org_id != user.org_id or (user.role == "pm" and project.created_by_user_id != user.id):
        raise HTTPException(403, "Access denied")
    if not project.jira_capacity_snapshot:
        raise HTTPException(409, "No Jira capacity snapshot available. Sync Jira before simulating capacity.")
    return project


@router.get("/{project_id}/capacity-intelligence")
def capacity_intelligence(project_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    project = _project(project_id, current_user, db)
    analysis = CapacityService.analyse(project.jira_capacity_snapshot)
    return {"project_id": project.id, "team_capacity": analysis["team_capacity"],
            "dependency_analysis": DependencyService.analyse(analysis),
            "active_sprint": analysis["active_sprint"]}


@router.post("/simulate-absence")
def simulate_absence(body: AbsenceRequest, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    project = _project(body.project_id, current_user, db)
    score = db.query(HealthScore).filter(HealthScore.project_id == project.id).order_by(HealthScore.recorded_at.desc()).first()
    if score is None:
        raise HTTPException(409, "A health score is required before simulating absence.")
    try:
        analysis = CapacityService.analyse(project.jira_capacity_snapshot)
        resolver = IdentityResolver.for_project(db, project.id)
        for member in analysis["team_capacity"]:
            member["developer"] = resolver.resolve(member.get("jira_account_id") or member["developer"])
        for item in analysis["_items"]:
            item["assignee"] = resolver.resolve(item.get("jira_account_id") or item["assignee"])
        return AbsenceSimulator.simulate(analysis, body.developer_id, body.unavailable_days, project, score)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
