"""In-process schedules for Gmail polling and Jira health updates."""
from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import os
import time
from zoneinfo import ZoneInfo

from app.models.models import Project
from app.utils.database import SessionLocal

logger = logging.getLogger(__name__)
LOCAL_TIMEZONE = ZoneInfo("Asia/Colombo")


def run_gmail_poll() -> None:
    from app.api.pipeline import sync_gmail_project
    db = SessionLocal()
    try:
        projects = db.query(Project).filter(Project.gmail_filter_email.isnot(None), Project.gmail_project_identifier.isnot(None)).all()
        for project in projects:
            try:
                result = sync_gmail_project(db, project)
                logger.info("Scheduled Gmail sync for project %s: %s", project.id, result["status"])
            except Exception:
                logger.exception("Scheduled Gmail sync failed for project %s", project.id)
    finally:
        db.close()


def run_jira_poll() -> None:
    from app.api.pipeline import sync_jira_project
    db = SessionLocal()
    try:
        projects = db.query(Project).filter(Project.jira_url.isnot(None), Project.encrypted_jira_token.isnot(None), Project.jira_email.isnot(None)).all()
        for project in projects:
            for attempt in range(1, 4):
                try:
                    sync_jira_project(db, project)
                    logger.info("Scheduled Jira sync completed for project %s", project.id)
                    break
                except Exception:
                    db.rollback()
                    if attempt == 3:
                        logger.exception("Scheduled Jira sync failed for project %s after %s attempts", project.id, attempt)
                    else:
                        logger.warning("Scheduled Jira sync failed for project %s (attempt %s/3); retrying", project.id, attempt)
                        time.sleep(attempt * 2)
    finally:
        db.close()


async def run_scheduler() -> None:
    """Run Gmail at 00:00/04:00/... and Jira Monday 09:00 Colombo time."""
    last_gmail_slot: str | None = None
    last_jira_slot: str | None = None
    while True:
        now = datetime.now(LOCAL_TIMEZONE)
        gmail_slot = now.strftime("%Y-%m-%d-%H")
        jira_slot = now.strftime("%Y-%m-%d")
        # Run once in each due hour. This also recovers a service that starts
        # shortly after the exact minute instead of silently missing a cycle.
        if now.hour % 4 == 0 and last_gmail_slot != gmail_slot:
            last_gmail_slot = gmail_slot
            await asyncio.to_thread(run_gmail_poll)
        if now.weekday() == 0 and now.hour == 9 and last_jira_slot != jira_slot:
            last_jira_slot = jira_slot
            await asyncio.to_thread(run_jira_poll)
        await asyncio.sleep(20)


def scheduler_enabled() -> bool:
    return os.getenv("INTERNAL_SCHEDULER_ENABLED", "true").lower() in {"1", "true", "yes"}
