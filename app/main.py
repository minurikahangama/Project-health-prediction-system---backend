"""
PHPS FastAPI application entry point.

Registers all routers, middleware, WebSocket endpoint, and startup events.

Run with:
    cd backend
    uvicorn app.main:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""
import asyncio
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.utils.database import engine, SessionLocal
from app.models import models
from app.services.project_health import PredictionService

# Import all routers
from app.api import auth, projects, pipeline, client, admin, profile, intelligence
from app.routes import analytics

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── App instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "PHPS API",
    version     = "2.0.0",
    description = (
        "Project Health Prediction System — "
        "AI-powered software project health monitoring. "
        "BSc Research: Minuri Dahara, BSCS0122011293."
    ),
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)
scheduler_task: asyncio.Task | None = None


# ── Rate limiting ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── CORS ──────────────────────────────────────────────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# ── CORS Configuration (Updated to add port 8080) ─────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL, 
        "http://localhost:3000", 
        "http://localhost:5174",
        "http://localhost:8080",  # Added: Your active React port
        "http://127.0.0.1:5173",  
        "http://127.0.0.1:5174",  
        "http://127.0.0.1:8080",  # Added: Loopback IP variation for port 8080
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "uploads")), name="uploads")

# ── Startup: create all database tables ───────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Create all tables if they do not yet exist."""
    logger.info("Creating database tables if they do not exist...")
    models.Base.metadata.create_all(bind=engine)
    # create_all does not add fields to an existing table; keep local PostgreSQL
    # installations compatible when profile support is introduced.
    from sqlalchemy import text
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS address TEXT"))
        connection.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(50)"))
        connection.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS profile_image_url VARCHAR(500)"))
        connection.execute(text(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS gmail_project_identifier VARCHAR(255)"
        ))
        connection.execute(text(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS gmail_account_email VARCHAR(255)"
        ))
        connection.execute(text(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS last_jira_synced_at TIMESTAMP"
        ))
        connection.execute(text(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS last_jira_synced_by VARCHAR(255)"
        ))
        connection.execute(text(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS last_email_synced_at TIMESTAMP"
        ))
        connection.execute(text(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS last_email_synced_by VARCHAR(255)"
        ))
        connection.execute(text(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS jira_capacity_snapshot JSON"
        ))
        connection.execute(text(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS jira_capacity_synced_at TIMESTAMP"
        ))
        connection.execute(text(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS jira_metrics_snapshot JSON"
        ))
        connection.execute(text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS project_state VARCHAR(32) DEFAULT 'INITIATION'"))
        connection.execute(text(
            "ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS subject VARCHAR(998)"
        ))
        connection.execute(text(
            "ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS sender VARCHAR(255)"
        ))
        connection.execute(text(
            "ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS received_at TIMESTAMP"
        ))
        connection.execute(text(
            "ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS tone_score FLOAT"
        ))
        connection.execute(text(
            "ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS urgency_flag INTEGER"
        ))
        connection.execute(text("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS penalty FLOAT"))
        connection.execute(text("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS recovery FLOAT"))
        connection.execute(text("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS decay_weight FLOAT"))
        connection.execute(text(
            "ALTER TABLE transcript_uploads ADD COLUMN IF NOT EXISTS content_sha256 VARCHAR(64)"
        ))
        connection.execute(text(
            "ALTER TABLE transcript_uploads ADD COLUMN IF NOT EXISTS tone_score FLOAT"
        ))
        connection.execute(text(
            "ALTER TABLE transcript_uploads ADD COLUMN IF NOT EXISTS urgency_flag INTEGER"
        ))
        connection.execute(text("ALTER TABLE transcript_uploads ADD COLUMN IF NOT EXISTS penalty FLOAT"))
        connection.execute(text("ALTER TABLE transcript_uploads ADD COLUMN IF NOT EXISTS recovery FLOAT"))
        connection.execute(text("ALTER TABLE transcript_uploads ADD COLUMN IF NOT EXISTS decay_weight FLOAT"))
        connection.execute(text(
            "ALTER TABLE health_scores ADD COLUMN IF NOT EXISTS analysis_source VARCHAR(20)"
        ))
        connection.execute(text(
            "ALTER TABLE health_scores ADD COLUMN IF NOT EXISTS prediction_source VARCHAR(64)"
        ))
        connection.execute(text(
            "ALTER TABLE health_scores ADD COLUMN IF NOT EXISTS transcript_upload_id INTEGER"
        ))
        connection.execute(text(
            "ALTER TABLE health_scores ADD COLUMN IF NOT EXISTS feature_vector JSON"
        ))
        connection.execute(text(
            "ALTER TABLE health_scores ADD COLUMN IF NOT EXISTS shap_explanation JSON"
        ))
        # Keep existing local/managed PostgreSQL databases compatible with
        # the live dashboard audit fields even before Alembic is run.
        connection.execute(text(
            "ALTER TABLE health_scores ADD COLUMN IF NOT EXISTS deduction_snapshot JSON"
        ))
        connection.execute(text("ALTER TABLE health_scores ADD COLUMN IF NOT EXISTS open_issues INTEGER DEFAULT 0"))
        connection.execute(text("ALTER TABLE health_scores ADD COLUMN IF NOT EXISTS open_bugs INTEGER DEFAULT 0"))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_health_scores_transcript_upload_id "
            "ON health_scores (transcript_upload_id)"
        ))
        # A transcript upload is an analysis event.  Re-uploading the same
        # notes must create a new event and rerun the health calculation, so
        # older installations must not retain the content-deduplication index.
        connection.execute(text(
            "DROP INDEX IF EXISTS uq_project_transcript_content"
        ))
    logger.info("Database ready.")
    global scheduler_task
    from app.services.scheduler import run_scheduler, scheduler_enabled
    if scheduler_enabled():
        scheduler_task = asyncio.create_task(run_scheduler(), name="phps-ingestion-scheduler")
        logger.info("Internal Gmail/Jira scheduler enabled (Asia/Colombo).")
    logger.info(f"API docs available at: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the in-process scheduler cleanly when the API exits."""
    if scheduler_task:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,     prefix="/auth",     tags=["Authentication"])
app.include_router(projects.router, prefix="/projects", tags=["Projects"])
# Versioned dashboard contract. The unversioned router remains for existing
# clients while new integrations can use /api/v1/projects/{id}/health.
app.include_router(projects.router, prefix="/api/v1/projects", tags=["Projects v1"])
app.include_router(intelligence.router, prefix="/api/v1/intelligence", tags=["Capacity Intelligence"])
app.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
app.include_router(client.router,   prefix="/client",   tags=["Client"])
app.include_router(admin.router,    prefix="/admin",    tags=["Admin"])
app.include_router(profile.router,  prefix="/profile",  tags=["Profile"])
app.include_router(analytics.router, prefix="/api", tags=["Analytics"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    return {
        "status":  "PHPS API is running",
        "version": "2.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["System"])
def health_check():
    """Simple liveness probe."""
    return {"status": "healthy"}


# ── WebSocket: real-time PM Dashboard updates (NFR-04: <30s refresh) ─────────
@app.websocket("/ws/{project_id}")
async def websocket_endpoint(ws: WebSocket, project_id: int):
    """
    WebSocket endpoint for real-time PM Dashboard updates.
    Pushes the latest health score every 30 seconds.
    The frontend connects on mount and closes on unmount.
    """
    await ws.accept()
    logger.info(f"WebSocket connected for project {project_id}")

    try:
        while True:
            db = SessionLocal()
            try:
                from app.models.models import HealthScore
                latest = (
                    db.query(HealthScore)
                    .filter(HealthScore.project_id == project_id)
                    .order_by(HealthScore.recorded_at.desc())
                    .first()
                )
                if latest:
                    await ws.send_json({
                        "health_score":    latest.health_score,
                        "rag_status":      latest.rag_status,
                        "tone_score":      latest.tone_score,
                        "urgency_flag":    latest.urgency_flag,
                        "velocity_percent": latest.velocity_percent,
                        "overdue_rate":    latest.overdue_rate,
                        "bug_ratio":       latest.bug_ratio,
                        "divergence_flag": latest.divergence_flag,
                        "recorded_at":     str(latest.recorded_at),
                        "contributions": PredictionService.explain_saved_score(latest),
                    })
            finally:
                db.close()

            await asyncio.sleep(30)   # push update every 30 seconds

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for project {project_id}")
    except Exception as e:
        logger.error(f"WebSocket error for project {project_id}: {e}")
