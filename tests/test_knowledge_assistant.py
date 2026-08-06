"""FR-13 — end-to-end tests for the Agentic Document Assistant.

Runs entirely offline: deterministic hashing embeddings + the local extractive
LLM + a portable SQLite database. Covers every FR-13 acceptance criterion.
"""
import os
import tempfile
import unittest

# A file-backed SQLite DB so every session shares the same data (must be set
# before importing the app's database module).
_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = "sqlite:///" + _db_path.replace("\\", "/")

from datetime import datetime, timedelta, timezone

from app.utils.database import Base, engine, SessionLocal
import app.models.models as m
import app.knowledge.models  # noqa: F401 — register FR-13 tables
from app.knowledge.config import load_config
from app.knowledge.service import KnowledgeAssistant
from app.knowledge.models import KbReasoningTrace, KbMessage
from app.knowledge.sources.local import LocalFolderSource

SEED_DIR = load_config().seed_dir


def _seed(name):
    return LocalFolderSource(os.path.join(SEED_DIR, name))


class KnowledgeAssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.db = SessionLocal()
        now = datetime.now(timezone.utc)

        org = m.Organisation(name="Acme")
        cls.db.add(org); cls.db.commit(); cls.db.refresh(org)

        cls.pm_alpha = m.User(org_id=org.id, name="PM Alpha", email="alpha@acme.test",
                              password_hash="x", role="pm")
        cls.pm_beta = m.User(org_id=org.id, name="PM Beta", email="beta@acme.test",
                             password_hash="x", role="pm")
        cls.db.add_all([cls.pm_alpha, cls.pm_beta]); cls.db.commit()
        cls.db.refresh(cls.pm_alpha); cls.db.refresh(cls.pm_beta)

        def _project(name, owner):
            p = m.Project(org_id=org.id, created_by_user_id=owner.id, name=name,
                          start_date=now, deadline=now + timedelta(days=30), team_size=4)
            cls.db.add(p); cls.db.commit(); cls.db.refresh(p)
            return p

        cls.project_alpha = _project("Alpha", cls.pm_alpha)
        cls.project_beta = _project("Beta", cls.pm_beta)

        cls.assistant = KnowledgeAssistant(cls.db)
        cls.assistant.sync_project(cls.pm_alpha, cls.project_alpha.id, source=_seed("alpha"))
        cls.assistant.sync_project(cls.pm_beta, cls.project_beta.id, source=_seed("beta"))

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        engine.dispose()
        try:
            os.remove(_db_path)
        except OSError:
            pass

    # ── Sync ─────────────────────────────────────────────────────────────
    def test_sync_ingested_all_three_doc_types(self):
        stats = self.assistant.sync_project(
            self.pm_alpha, self.project_alpha.id, source=_seed("alpha"))
        # Second sync of unchanged files → nothing re-embedded (incremental).
        self.assertEqual(stats.unchanged_pages, 3)
        self.assertEqual(stats.added_pages, 0)
        self.assertEqual(sorted(stats.doc_types), ["bug", "qa", "requirement"])

    # ── Grounded answer + citation ───────────────────────────────────────
    def test_grounded_answer_with_citation(self):
        r = self.assistant.answer(self.pm_alpha, self.project_alpha.id,
                                  "What are the password requirements?")
        self.assertEqual(r["status"], "answered")
        self.assertGreaterEqual(len(r["citations"]), 1)
        self.assertTrue(any(w in r["answer"].lower()
                            for w in ("eight", "uppercase", "password")))

    # ── Project isolation ────────────────────────────────────────────────
    def test_project_isolation_no_cross_project_answer(self):
        # AlphaVault only exists in the Alpha project; Beta must not answer it.
        r = self.assistant.answer(self.pm_beta, self.project_beta.id,
                                  "What is AlphaVault?")
        self.assertEqual(r["status"], "not_available")
        self.assertEqual(r["citations"], [])

    def test_same_term_answered_in_owning_project(self):
        r = self.assistant.answer(self.pm_alpha, self.project_alpha.id,
                                  "What is AlphaVault?")
        self.assertEqual(r["status"], "answered")
        self.assertGreaterEqual(len(r["citations"]), 1)

    # ── Authorization ────────────────────────────────────────────────────
    def test_unauthorized_project_is_denied(self):
        # PM Beta may not access the Alpha project.
        r = self.assistant.answer(self.pm_beta, self.project_alpha.id,
                                  "What are the password requirements?")
        self.assertEqual(r["status"], "denied")

    # ── Not available ────────────────────────────────────────────────────
    def test_out_of_domain_question_not_available(self):
        r = self.assistant.answer(self.pm_alpha, self.project_alpha.id,
                                  "What is the capital of France?")
        self.assertEqual(r["status"], "not_available")

    # ── Multi-turn memory ────────────────────────────────────────────────
    def test_multi_turn_retains_conversation(self):
        r1 = self.assistant.answer(self.pm_alpha, self.project_alpha.id,
                                   "What are the password requirements?")
        r2 = self.assistant.answer(self.pm_alpha, self.project_alpha.id,
                                   "How long is the reset link valid?",
                                   conversation_id=r1["conversation_id"])
        self.assertEqual(r1["conversation_id"], r2["conversation_id"])
        count = (self.db.query(KbMessage)
                     .filter(KbMessage.conversation_id == r1["conversation_id"]).count())
        self.assertEqual(count, 4)  # 2 user + 2 assistant

    # ── Reasoning trace ──────────────────────────────────────────────────
    def test_reasoning_trace_is_logged(self):
        r = self.assistant.answer(self.pm_alpha, self.project_alpha.id,
                                  "What are the password requirements?")
        steps = [t["step"] for t in r["trace"]]
        for expected in ("authorize", "grade_question", "retrieve", "grade_docs",
                         "generate", "check_grounding", "respond"):
            self.assertIn(expected, steps)
        persisted = (self.db.query(KbReasoningTrace)
                         .filter(KbReasoningTrace.message_id == r["message_id"]).count())
        self.assertGreaterEqual(persisted, len(steps))


if __name__ == "__main__":
    unittest.main()
