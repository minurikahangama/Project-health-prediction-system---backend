"""SQLAlchemy models for the FR-13 knowledge base.

Every embedded chunk is stamped with ``project_id`` — the isolation key that
guarantees one project never retrieves another project's documents. Embeddings
are stored as JSON (portable across SQLite/PostgreSQL) and ranked with an
in-Python cosine search; production installations can switch ``kb_document_chunks``
to a native ``pgvector`` column for scale (see FR-13_Implementation_Plan.md §4).
"""
from __future__ import annotations

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, JSON, Index,
)
from sqlalchemy.orm import relationship

from app.utils.database import Base
from app.utils.time import utcnow


class KbProjectSource(Base):
    """Maps a project to a Confluence space (or local folder) per document type."""
    __tablename__ = "kb_project_sources"

    id           = Column(Integer, primary_key=True, index=True)
    project_id   = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    source_type  = Column(String(32), nullable=False, default="local")   # local | confluence
    space_key    = Column(String(255), nullable=True)                    # Confluence space / folder name
    doc_type     = Column(String(32), nullable=False)                    # requirement | qa | bug
    config       = Column(JSON, nullable=True)
    created_at   = Column(DateTime, default=utcnow)


class KbDocumentChunk(Base):
    """One embedded passage of a synchronized document."""
    __tablename__ = "kb_document_chunks"
    __table_args__ = (
        Index("ix_kb_chunks_project_doc", "project_id", "doc_type"),
        Index("ix_kb_chunks_project_page", "project_id", "page_id"),
    )

    id           = Column(Integer, primary_key=True, index=True)
    project_id   = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"),
                          nullable=False, index=True)     # ← isolation key
    doc_type     = Column(String(32),  nullable=False)    # requirement | qa | bug
    page_id      = Column(String(255), nullable=False)    # Confluence page id / file name
    page_title   = Column(String(500), nullable=True)
    page_version = Column(String(64),  nullable=False)    # incremental-sync fingerprint
    section      = Column(String(500), nullable=True)     # for citations
    content      = Column(Text,        nullable=False)
    embedding    = Column(JSON,        nullable=False)     # list[float], L2-normalized
    created_at   = Column(DateTime, default=utcnow)


class KbConversation(Base):
    """A multi-turn chat thread, scoped to one user and one project."""
    __tablename__ = "kb_conversations"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    project_id   = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    created_at   = Column(DateTime, default=utcnow)

    messages = relationship("KbMessage", cascade="all, delete-orphan",
                            order_by="KbMessage.created_at")


class KbMessage(Base):
    """A single turn in a conversation."""
    __tablename__ = "kb_messages"

    id              = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("kb_conversations.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    role            = Column(String(16), nullable=False)   # user | assistant
    content         = Column(Text, nullable=False)
    citations       = Column(JSON, nullable=True)          # list[{page_id, section, doc_type}]
    status          = Column(String(24), nullable=True)    # answered | not_available | denied
    created_at      = Column(DateTime, default=utcnow)


class KbReasoningTrace(Base):
    """Auditable record of every agent step (BR: reasoning trace logged)."""
    __tablename__ = "kb_reasoning_trace"

    id              = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("kb_conversations.id", ondelete="CASCADE"),
                             nullable=True, index=True)
    message_id      = Column(Integer, ForeignKey("kb_messages.id", ondelete="CASCADE"),
                             nullable=True, index=True)
    step            = Column(String(48), nullable=False)
    detail          = Column(JSON, nullable=True)
    created_at      = Column(DateTime, default=utcnow)
