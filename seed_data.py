"""
Database seed script.

Run this ONCE after `alembic upgrade head` to populate initial data.

Usage:
    cd backend
    source venv/Scripts/activate   (Windows)
    python seed_data.py

Creates:
  - 1 Organisation: TechCorp Ltd
  - 1 Super Admin:  admin@phps.com       / Admin@123
  - 1 Org Admin:    orgadmin@techcorp.com / OrgAdmin@123
  - 1 PM:           pm@techcorp.com       / PM@1234567
  - 1 Project:      TechCorp Platform Redesign
  - 6 HealthScore rows (historical data for charts)
"""
import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.utils.database import SessionLocal
from app.models.models import Organisation, User, Project, HealthScore
from passlib.context import CryptContext
from datetime import datetime, timedelta

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def seed():
    db = SessionLocal()
    try:
        print("=" * 60)
        print("PHPS Database Seed")
        print("=" * 60)

        # ── Organisation ──────────────────────────────────────────────
        existing_org = db.query(Organisation).filter_by(name="TechCorp Ltd").first()
        if existing_org:
            print("Organisation already exists — skipping seed.")
            print("Delete the database and re-run to reseed from scratch.")
            return

        org = Organisation(name="TechCorp Ltd", industry="Software")
        db.add(org)
        db.flush()
        print(f"✓ Created organisation: {org.name} (id={org.id})")

        # ── Super Admin ───────────────────────────────────────────────
        super_admin = User(
            org_id               = org.id,
            name                 = "Super Admin",
            email                = "admin@phps.com",
            password_hash        = pwd_ctx.hash("Admin@123"),
            role                 = "super_admin",
            is_active            = True,
            force_password_change = False,
        )
        db.add(super_admin)
        db.flush()
        print(f"✓ Created super_admin: {super_admin.email}")

        # ── Org Admin ─────────────────────────────────────────────────
        org_admin = User(
            org_id               = org.id,
            name                 = "TechCorp Admin",
            email                = "orgadmin@techcorp.com",
            password_hash        = pwd_ctx.hash("OrgAdmin@123"),
            role                 = "org_admin",
            is_active            = True,
            force_password_change = False,
        )
        db.add(org_admin)
        db.flush()
        print(f"✓ Created org_admin:   {org_admin.email}")

        # ── Project Manager ───────────────────────────────────────────
        pm = User(
            org_id               = org.id,
            name                 = "Sarah Johnson",
            email                = "pm@techcorp.com",
            password_hash        = pwd_ctx.hash("PM@1234567"),
            role                 = "pm",
            is_active            = True,
            force_password_change = False,
        )
        db.add(pm)
        db.flush()
        print(f"✓ Created pm:          {pm.email}")

        # ── Project ───────────────────────────────────────────────────
        project = Project(
            org_id             = org.id,
            created_by_user_id = pm.id,
            name               = "TechCorp Platform Redesign",
            start_date         = datetime(2026, 1, 15),
            deadline           = datetime(2026, 8, 31),
            team_size          = 6,
            jira_url           = None,   # fill in after connecting Jira
            green_threshold    = 70.0,
            red_threshold      = 40.0,
            pm_note            = "Great progress this sprint! On track for the August milestone.",
        )
        db.add(project)
        db.flush()
        print(f"✓ Created project:     {project.name} (id={project.id})")

        # ── 6 months of historical health scores (for dashboard charts) ──
        base_date = datetime(2026, 1, 20)
        scores_data = [
            # (health_score, rag,    tone,  urgency, velocity, overdue, bug, divergence)
            (82,  "GREEN", 0.42,  0,  0.88,  0.08,  0.05,  0),
            (78,  "GREEN", 0.28,  0,  0.82,  0.10,  0.07,  0),
            (74,  "GREEN", 0.11,  0,  0.75,  0.13,  0.09,  0),
            (70,  "AMBER",-0.08,  0,  0.72,  0.15,  0.10,  0),
            (69,  "AMBER",-0.19,  1,  0.72,  0.18,  0.12,  1),
            (67,  "AMBER",-0.24,  1,  0.72,  0.18,  0.12,  1),
        ]

        for i, (hs, rag, tone, urg, vel, ovr, bug, div) in enumerate(scores_data):
            score = HealthScore(
                project_id       = project.id,
                health_score     = hs,
                rag_status       = rag,
                tone_score       = tone,
                urgency_flag     = urg,
                velocity_percent = vel,
                overdue_rate     = ovr,
                bug_ratio        = bug,
                divergence_flag  = div,
                recorded_at      = base_date + timedelta(weeks=i * 2),
            )
            db.add(score)

        db.commit()
        print(f"✓ Created 6 historical health scores")

        print()
        print("=" * 60)
        print("Seed complete! Login credentials:")
        print("=" * 60)
        print(f"  Super Admin: admin@phps.com         / Admin@123")
        print(f"  Org Admin:   orgadmin@techcorp.com  / OrgAdmin@123")
        print(f"  PM:          pm@techcorp.com        / PM@1234567")
        print()
        print("API docs: http://localhost:8000/docs")
        print("=" * 60)

    except Exception as e:
        db.rollback()
        print(f"\nError during seed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
