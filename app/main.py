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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.utils.database import engine, SessionLocal
from app.models import models

# Import all routers
from app.api import auth, projects, pipeline, client, admin

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


# ── Rate limiting ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── CORS ──────────────────────────────────────────────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = [FRONTEND_URL, "http://localhost:3000", "http://localhost:5174"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Startup: create all database tables ───────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Create all tables if they do not yet exist."""
    logger.info("Creating database tables if they do not exist...")
    models.Base.metadata.create_all(bind=engine)
    logger.info("Database ready.")
    logger.info(f"API docs available at: http://localhost:8000/docs")


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,     prefix="/auth",     tags=["Authentication"])
app.include_router(projects.router, prefix="/projects", tags=["Projects"])
app.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
app.include_router(client.router,   prefix="/client",   tags=["Client"])
app.include_router(admin.router,    prefix="/admin",    tags=["Admin"])


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
                    })
            finally:
                db.close()

            await asyncio.sleep(30)   # push update every 30 seconds

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for project {project_id}")
    except Exception as e:
        logger.error(f"WebSocket error for project {project_id}: {e}")
