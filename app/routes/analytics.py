"""Authenticated API endpoints for calibrated health analytics."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.ml.features import build_feature_dict
from app.ml.models import CalibratedXGBoost, TemporalHealthForecaster
from app.ml.optimizer import optimize_health, recovery_plan
from app.ml.preprocessor import rolling_windows
from app.models.models import (GDPRDeletionLog, HealthScore, JiraEvidenceSnapshot,
                               ProcessedEmail, Project, ProjectTeamMember,
                               TranscriptUpload, User)
from app.services.explainability import ExplainabilityService
from app.services.team_capacity import TeamCapacityService
from app.services.absence_simulator import AbsenceSimulator
from app.services.decision_support import DecisionSupportService
from app.services.ai_project_assistant import AIProjectAssistant
from app.services.identity_resolver import IdentityResolver
from app.utils.database import get_db

router = APIRouter()

def _project(project_id: int, user: User, db: Session) -> Project:
    project = db.query(Project).filter(Project.id == project_id, Project.org_id == user.org_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    return project

def _scores(project_id: int, db: Session) -> list[HealthScore]:
    rows = db.query(HealthScore).filter_by(project_id=project_id).order_by(HealthScore.recorded_at.asc()).all()
    if not rows:
        raise HTTPException(404, "No health observations are available")
    return rows


class CapacitySimulation(BaseModel):
    developer: str
    unavailable_days: float = Field(ge=0)


class CapacityQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def _capacity_state(project_id: int, user: User, db: Session):
    project = _project(project_id, user, db)
    if not project.jira_capacity_snapshot:
        raise HTTPException(409, "No Jira capacity snapshot is available. Sync Jira before opening this dashboard.")
    analysis = TeamCapacityService.analyse(project.jira_capacity_snapshot)
    resolver = IdentityResolver.for_project(db, project.id)
    for member in analysis["team_capacity"]:
        member["developer"] = resolver.resolve(member.get("jira_account_id") or member["developer"])
    for item in analysis["_items"]:
        item["assignee"] = resolver.resolve(item.get("jira_account_id") or item["assignee"])
    dependency = analysis["dependency_analysis"]
    dependency["affected_developers"] = [resolver.resolve(value) for value in dependency["affected_developers"]]
    dependency["blocking_developers"] = [resolver.resolve(value) for value in dependency["blocking_developers"]]
    dependency["identity_map"] = resolver.public_map()
    item_by_key = {item["key"]: item for item in analysis["_items"]}
    dependency["nodes"] = [{"key": key, "summary": item_by_key[key]["summary"],
                            "assignee": item_by_key[key]["assignee"],
                            "waiting_hours": round(item_by_key[key]["remaining_seconds"] / 3600, 2),
                            "is_critical": key in dependency["critical_path"],
                            "downstream": dependency["graph"].get(key, [])}
                           for key in sorted({node for edge in dependency["dependency_chains"] for node in edge}) if key in item_by_key]
    return project, _scores(project_id, db)[-1], analysis


def _capacity_recommendations(project, score, analysis, db: Session) -> dict:
    from app.models.models import ProcessedEmail, TranscriptUpload
    return DecisionSupportService.build(
        project, score,
        email_count=db.query(ProcessedEmail).filter_by(project_id=project.id).count(),
        transcript_count=db.query(TranscriptUpload).filter_by(project_id=project.id).count(),
        capacity_analysis=analysis,
    )

def _features(project: Project, rows: list[HealthScore]) -> list[dict]:
    result, previous = [], None
    for score in rows:
        value = build_feature_dict(overall_sentiment=score.tone_score, urgency_count=score.urgency_flag,
            velocity=score.velocity_percent, overdue_rate=score.overdue_rate, bug_ratio=score.bug_ratio,
            open_issues=score.open_issues, open_bugs=score.open_bugs, email_count=0, transcript_count=0,
            deadline=project.deadline, previous=previous)
        result.append(value); previous = value
    return result

@router.get("/predict")
def predict(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    project, rows = _project(project_id, current_user, db), _scores(project_id, db)
    model = CalibratedXGBoost(); score = model.predict(_features(project, rows)[-1])
    return {"project_id": project.id, "health_score": round(score, 2), "calibrated": model.calibrated,
            "model": "xgboost_regressor", "note": "Prediction is clipped to the 0-100 health scale."}

@router.get("/explain")
def explain(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    project, rows = _project(project_id, current_user, db), _scores(project_id, db)
    return ExplainabilityService.build(project, rows[-1], health_observation_count=len(rows))

@router.get("/forecast")
def forecast(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    project, rows = _project(project_id, current_user, db), _scores(project_id, db)
    features = _features(project, rows); sequence = rolling_windows(features)[-1:]
    result = TemporalHealthForecaster().forecast(sequence, [row.health_score for row in rows])
    return {"project_id": project.id, "horizon_weeks": 3, "trajectory": [{"week": i+1, "health_score": score} for i, score in enumerate(result["scores"])], "model_source": result["model_source"]}

@router.get("/optimize")
def optimize(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    project, rows = _project(project_id, current_user, db), _scores(project_id, db)
    model = CalibratedXGBoost()
    return optimize_health(_features(project, rows)[-1], model.predict)

@router.get("/recovery-plan")
def plan(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    project, rows = _project(project_id, current_user, db), _scores(project_id, db)
    model = CalibratedXGBoost(); optimized = optimize_health(_features(project, rows)[-1], model.predict)
    return {"optimization": optimized, "roadmap": recovery_plan(optimized)}


@router.get("/projects/{project_id}/team-capacity")
def team_capacity_dashboard(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Return the complete dashboard from the newest successfully synced Jira snapshot."""
    project, score, analysis = _capacity_state(project_id, current_user, db)
    from app.models.models import ProcessedEmail, TranscriptUpload
    payload = TeamCapacityService.dashboard(
        project.jira_capacity_snapshot, project, score,
        db.query(ProcessedEmail).filter_by(project_id=project_id).count(),
        db.query(TranscriptUpload).filter_by(project_id=project_id).count(),
    )
    # Publish the identity-resolved analysis rather than the raw Jira account
    # identifiers produced while the snapshot is parsed.
    payload["team_capacity"] = analysis["team_capacity"]
    payload["dependency_graph"] = analysis["dependency_analysis"]
    payload["burndown"] = analysis["burndown"]
    payload["recommendations"] = _capacity_recommendations(project, score, analysis, db)["recommendations"]
    payload["simulation_data"] = None
    payload["identity_map"] = analysis["dependency_analysis"]["identity_map"]
    return payload


@router.post("/projects/{project_id}/team-capacity/simulate")
def simulate_team_capacity(project_id: int, body: CapacitySimulation, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Calculate an absence what-if from the snapshot without modifying Jira or PHPS state."""
    project, score, analysis = _capacity_state(project_id, current_user, db)
    try:
        # Keep legacy chart fields while emitting the same explicit simulator
        # contract used by /api/v1/intelligence/simulate-absence.
        result = TeamCapacityService.simulate_unavailability(analysis, body.developer, body.unavailable_days, project, score)
        rich = AbsenceSimulator.simulate(analysis, body.developer, body.unavailable_days, project, score)
        result.update(rich)
        result["current_health"] = score.health_score
        result["recovery_probability"] = round(max(0, 100 - (result["deadline_delay_days"] or 0) * 10), 2)
        replacement = next((m["developer"] for m in analysis["team_capacity"] if m["developer"] != body.developer and m["capacity_percentage"] < 80), None)
        result["suggested_replacement"] = replacement
        return result
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/projects/{project_id}/team-capacity/developers")
def team_capacity_developers(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _, _, analysis = _capacity_state(project_id, current_user, db)
    return {"developers": analysis["team_capacity"], "workload_distribution": analysis["workload_distribution"]}


@router.get("/projects/{project_id}/team-capacity/burndown")
def team_capacity_burndown(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _, _, analysis = _capacity_state(project_id, current_user, db)
    return analysis["burndown"]


@router.get("/projects/{project_id}/team-capacity/dependencies")
def team_capacity_dependencies(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _, _, analysis = _capacity_state(project_id, current_user, db)
    return analysis["dependency_analysis"]


@router.get("/projects/{project_id}/team-capacity/capacity")
def team_capacity_metrics(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    _, _, analysis = _capacity_state(project_id, current_user, db)
    return analysis["workload_distribution"]


@router.get("/projects/{project_id}/team-capacity/recommendations")
def team_capacity_recommendations(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    project, score, analysis = _capacity_state(project_id, current_user, db)
    planner = _capacity_recommendations(project, score, analysis, db)
    return {"current_health": planner["current_health"], "recommendations": planner["recommendations"], "shap_values": planner["decision_simulator"]["shap_values"]}


@router.post("/projects/{project_id}/team-capacity/ask-ai")
def ask_team_capacity_ai(project_id: int, body: CapacityQuestion, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Build a fresh XGBoost/Jira/communication-grounded answer for one question."""
    project, score, analysis = _capacity_state(project_id, current_user, db)
    try:
        return AIProjectAssistant.answer_question(body.question, project, score, db)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/v1/projects/{project_id}/hard-reset")
@router.post("/projects/{project_id}/team-capacity/reset", include_in_schema=False)
def reset_team_capacity_telemetry(project_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Permanently remove project telemetry and create a clean 100-point baseline."""
    project = _project(project_id, current_user, db)
    deleted = {}
    for model, label in ((ProcessedEmail, "communications"), (TranscriptUpload, "transcripts"),
                         (JiraEvidenceSnapshot, "jira_evidence"), (ProjectTeamMember, "team_members"),
                         (GDPRDeletionLog, "logs"), (HealthScore, "health_scores")):
        deleted[label] = db.query(model).filter_by(project_id=project.id).delete(synchronize_session=False)
    project.jira_capacity_snapshot = None
    project.jira_metrics_snapshot = None
    project.jira_capacity_synced_at = None
    project.last_jira_synced_at = None
    project.last_jira_synced_by = None
    project.last_email_synced_at = None
    project.last_email_synced_by = None
    baseline = HealthScore(project_id=project.id, health_score=100.0, rag_status="GREEN",
                           tone_score=0.0, urgency_flag=0, velocity_percent=1.0,
                           overdue_rate=0.0, bug_ratio=0.0, open_issues=0, open_bugs=0,
                           divergence_flag=0, analysis_source="reset", prediction_source="clean_baseline")
    db.add(baseline)
    db.commit()
    return {"status": "success", "project_id": project.id, "health_score": 100.0,
            "deleted": deleted, "message": "Project telemetry was permanently reset to the clean baseline."}
