"""KnowledgeAssistant — orchestrates authorization, retrieval, the LangGraph
agent, and persistence for FR-13.

This is the single entry point the API layer calls. It builds the providers and
compiled agent once per instance, resolves which projects a user may access,
runs a question through the graph, and records the conversation + reasoning
trace.
"""
from __future__ import annotations

import os
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.models import Project, User
from app.utils.encryption import encrypt_token, decrypt_token
from app.knowledge.config import KnowledgeConfig, load_config, DOC_TYPES
from app.knowledge.models import (
    KbConversation, KbMessage, KbReasoningTrace, KbProjectSource,
)
from app.knowledge.providers.embeddings import build_embedding_provider
from app.knowledge.providers.llm import build_llm_provider
from app.knowledge.vector_index import VectorIndex
from app.knowledge.ingestion import Ingestor, SyncStats
from app.knowledge.agent import AgentRuntime, build_agent
from app.knowledge.sources.base import DocumentSource
from app.knowledge.sources.local import LocalFolderSource
from app.knowledge.sources.confluence import ConfluenceSource


class AuthorizationError(Exception):
    pass


class KnowledgeAssistant:
    def __init__(self, db: Session, config: Optional[KnowledgeConfig] = None):
        self.db = db
        self.config = config or load_config()
        self.embedder = build_embedding_provider(self.config)
        self.llm = build_llm_provider(self.config)
        self.retriever = VectorIndex(db, self.embedder)
        self.ingestor = Ingestor(db, self.embedder, self.config)
        self.runtime = AgentRuntime(self.retriever, self.llm, self.config)
        self.agent = build_agent(self.runtime)

    # ── Authorization ────────────────────────────────────────────────────
    def allowed_project_ids(self, user: User) -> List[int]:
        q = self.db.query(Project.id)
        if user.role == "super_admin":
            pass
        elif user.role == "org_admin":
            q = q.filter(Project.org_id == user.org_id)
        else:  # pm / other: only projects they created, within their org
            q = q.filter(Project.org_id == user.org_id,
                         Project.created_by_user_id == user.id)
        return [row[0] for row in q.all()]

    def list_projects(self, user: User) -> List[dict]:
        ids = self.allowed_project_ids(user)
        if not ids:
            return []
        rows = (self.db.query(Project)
                    .filter(Project.id.in_(ids))
                    .order_by(Project.name).all())
        return [{"id": p.id, "name": p.name} for p in rows]

    # ── Confluence connection ────────────────────────────────────────────
    def _confluence_row(self, project_id: int) -> Optional[KbProjectSource]:
        return (self.db.query(KbProjectSource)
                    .filter(KbProjectSource.project_id == project_id,
                            KbProjectSource.source_type == "confluence")
                    .first())

    def _build_confluence_source(self, row: KbProjectSource) -> ConfluenceSource:
        cfg = row.config or {}
        labels = cfg.get("labels") or {dt: dt for dt in DOC_TYPES}
        return ConfluenceSource(
            base_url=cfg["base_url"], email=cfg["email"],
            api_token=decrypt_token(cfg.get("token_encrypted", "")),
            space_key=row.space_key, doc_type_labels=labels,
        )

    def set_confluence_connection(self, user: User, project_id: int, *,
                                  base_url: str, email: str, api_token: Optional[str],
                                  space_key: str, labels: Optional[dict]) -> None:
        if project_id not in self.allowed_project_ids(user):
            raise AuthorizationError("Not authorized to configure this project.")
        row = self._confluence_row(project_id)
        config = {
            "base_url": base_url.rstrip("/"), "email": email,
            "labels": labels or {dt: dt for dt in DOC_TYPES},
        }
        if api_token:
            config["token_encrypted"] = encrypt_token(api_token)
        elif row:                                   # keep the existing token on edit
            config["token_encrypted"] = (row.config or {}).get("token_encrypted", "")
        else:
            raise AuthorizationError("An API token is required to connect Confluence.")
        if row:
            row.space_key = space_key
            row.config = config
        else:
            self.db.add(KbProjectSource(
                project_id=project_id, source_type="confluence",
                space_key=space_key, doc_type="confluence", config=config,
            ))
        self.db.commit()

    def get_confluence_connection(self, user: User, project_id: int) -> Optional[dict]:
        if project_id not in self.allowed_project_ids(user):
            raise AuthorizationError("Not authorized to view this project.")
        row = self._confluence_row(project_id)
        if not row:
            return None
        cfg = row.config or {}
        return {
            "connected": True, "base_url": cfg.get("base_url"),
            "email": cfg.get("email"), "space_key": row.space_key,
            "labels": cfg.get("labels", {dt: dt for dt in DOC_TYPES}),
        }

    def test_confluence_connection(self, user: User, project_id: int) -> dict:
        if project_id not in self.allowed_project_ids(user):
            raise AuthorizationError("Not authorized to test this project.")
        row = self._confluence_row(project_id)
        if not row:
            return {"success": False, "message": "No Confluence connection configured."}
        ok, message = self._build_confluence_source(row).test()
        return {"success": ok, "message": message}

    # ── Ingestion ────────────────────────────────────────────────────────
    def resolve_source(self, project: Project) -> DocumentSource:
        row = self._confluence_row(project.id)
        if row:
            return self._build_confluence_source(row)
        local = (self.db.query(KbProjectSource)
                     .filter(KbProjectSource.project_id == project.id,
                             KbProjectSource.source_type == "local").first())
        if local and (local.config or {}).get("path"):
            return LocalFolderSource(local.config["path"])
        # Default: a seed folder named after the project id (dev/local testing).
        return LocalFolderSource(os.path.join(self.config.seed_dir, str(project.id)))

    def sync_project(self, user: User, project_id: int,
                     source: Optional[DocumentSource] = None) -> SyncStats:
        if project_id not in self.allowed_project_ids(user):
            raise AuthorizationError("Not authorized to sync this project.")
        project = self.db.query(Project).get(project_id)
        if project is None:
            raise AuthorizationError("Project not found.")
        source = source or self.resolve_source(project)
        return self.ingestor.sync_project(project_id, source)

    # ── Conversation ─────────────────────────────────────────────────────
    def _get_or_create_conversation(self, user: User, project_id: int,
                                    conversation_id: Optional[int]) -> KbConversation:
        if conversation_id is not None:
            conv = self.db.query(KbConversation).get(conversation_id)
            if conv and conv.user_id == user.id and conv.project_id == project_id:
                return conv
        conv = KbConversation(user_id=user.id, project_id=project_id)
        self.db.add(conv)
        self.db.commit()
        self.db.refresh(conv)
        return conv

    # ── History ──────────────────────────────────────────────────────────
    def list_conversations(self, user: User, project_id: int) -> List[dict]:
        if project_id not in self.allowed_project_ids(user):
            raise AuthorizationError("Not authorized to view this project.")
        convs = (self.db.query(KbConversation)
                     .filter(KbConversation.user_id == user.id,
                             KbConversation.project_id == project_id)
                     .order_by(KbConversation.created_at.desc()).all())
        out: List[dict] = []
        for c in convs:
            first = (self.db.query(KbMessage)
                         .filter(KbMessage.conversation_id == c.id,
                                 KbMessage.role == "user")
                         .order_by(KbMessage.created_at).first())
            count = (self.db.query(KbMessage)
                         .filter(KbMessage.conversation_id == c.id).count())
            if count == 0:
                continue   # skip empty threads
            title = (first.content[:60] if first else "New conversation").strip()
            out.append({"id": c.id, "title": title or "New conversation",
                        "created_at": str(c.created_at), "message_count": count})
        return out

    def delete_conversation(self, user: User, conversation_id: int) -> None:
        conv = self.db.query(KbConversation).get(conversation_id)
        if conv is None or conv.user_id != user.id:
            raise AuthorizationError("Conversation not found.")
        # Remove dependent rows explicitly so it works on any database backend.
        (self.db.query(KbReasoningTrace)
             .filter(KbReasoningTrace.conversation_id == conversation_id)
             .delete(synchronize_session=False))
        (self.db.query(KbMessage)
             .filter(KbMessage.conversation_id == conversation_id)
             .delete(synchronize_session=False))
        self.db.delete(conv)
        self.db.commit()

    # ── Ask ──────────────────────────────────────────────────────────────
    def answer(self, user: User, project_id: int, question: str,
               conversation_id: Optional[int] = None,
               doc_types: Optional[List[str]] = None) -> dict:
        allowed = self.allowed_project_ids(user)
        conv = self._get_or_create_conversation(user, project_id, conversation_id)

        self.db.add(KbMessage(conversation_id=conv.id, role="user", content=question))
        self.db.commit()

        state = {
            "question": question,
            "original_question": question,
            "role": user.role,
            "active_project_id": project_id,
            "allowed_project_ids": allowed,
            "doc_types": doc_types,
            "attempts": 0,
            "trace": [],
        }
        result = self.agent.invoke(state)

        assistant_msg = KbMessage(
            conversation_id=conv.id, role="assistant",
            content=result.get("answer", ""),
            citations=result.get("citations", []),
            status=result.get("status", "answered"),
        )
        self.db.add(assistant_msg)
        self.db.commit()
        self.db.refresh(assistant_msg)

        for entry in result.get("trace", []):
            self.db.add(KbReasoningTrace(
                conversation_id=conv.id, message_id=assistant_msg.id,
                step=entry.get("step", ""), detail=entry.get("detail"),
            ))
        self.db.commit()

        return {
            "conversation_id": conv.id,
            "message_id": assistant_msg.id,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "status": result.get("status", "answered"),
            "trace": result.get("trace", []),
        }
