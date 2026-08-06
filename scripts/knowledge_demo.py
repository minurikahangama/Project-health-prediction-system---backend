"""FR-13 offline demo — seeds two projects and runs the agentic assistant.

Run from the backend root:

    # uses a throwaway SQLite file, no external services needed
    set DATABASE_URL=sqlite:///./_kb_demo.db   (PowerShell: $env:DATABASE_URL=...)
    python scripts/knowledge_demo.py

It ingests the bundled seed documents for an "Alpha" and a "Beta" project, then
asks a set of questions that demonstrate: grounded+cited answers, project
isolation, out-of-domain "not available", and authorization denial.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite:///./_kb_demo.db")

from datetime import datetime, timedelta, timezone

from app.utils.database import Base, engine, SessionLocal
import app.models.models as m
import app.knowledge.models  # noqa: F401
from app.knowledge.config import load_config
from app.knowledge.service import KnowledgeAssistant
from app.knowledge.sources.local import LocalFolderSource

SEED = load_config().seed_dir


def main() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    now = datetime.now(timezone.utc)

    org = m.Organisation(name="Demo Org")
    db.add(org); db.commit(); db.refresh(org)
    pm_a = m.User(org_id=org.id, name="PM Alpha", email="a@demo.test", password_hash="x", role="pm")
    pm_b = m.User(org_id=org.id, name="PM Beta", email="b@demo.test", password_hash="x", role="pm")
    db.add_all([pm_a, pm_b]); db.commit(); db.refresh(pm_a); db.refresh(pm_b)

    def project(name, owner):
        p = m.Project(org_id=org.id, created_by_user_id=owner.id, name=name,
                      start_date=now, deadline=now + timedelta(days=30), team_size=4)
        db.add(p); db.commit(); db.refresh(p); return p

    alpha = project("Alpha", pm_a)
    beta = project("Beta", pm_b)

    kb = KnowledgeAssistant(db)
    print("Sync Alpha:", kb.sync_project(pm_a, alpha.id, LocalFolderSource(os.path.join(SEED, "alpha"))).as_dict())
    print("Sync Beta :", kb.sync_project(pm_b, beta.id, LocalFolderSource(os.path.join(SEED, "beta"))).as_dict())

    def ask(user, project_id, q, label):
        r = kb.answer(user, project_id, q)
        print(f"\n[{label}] Q: {q}")
        print(f"  status : {r['status']}")
        print(f"  answer : {r['answer']}")
        if r["citations"]:
            cites = ", ".join(f"{c['doc_type']}:{c['section']}" for c in r["citations"])
            print(f"  cites  : {cites}")
        print("  trace  : " + " -> ".join(t["step"] for t in r["trace"]))

    ask(pm_a, alpha.id, "What are the password requirements?", "grounded")
    ask(pm_a, alpha.id, "What is AlphaVault?", "same-project term")
    ask(pm_b, beta.id, "What is AlphaVault?", "isolation")
    ask(pm_a, alpha.id, "What is the capital of France?", "out-of-domain")
    ask(pm_b, alpha.id, "What are the password requirements?", "authorization")

    db.close()


if __name__ == "__main__":
    main()
