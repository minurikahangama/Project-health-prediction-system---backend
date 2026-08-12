"""Project-scoped semantic retrieval over stored document chunks.

Retrieval ALWAYS filters by ``project_id`` (and optional doc types) before
ranking, which is what structurally enforces project isolation — no query can
reach another project's chunks. Ranking is cosine similarity computed in Python
over JSON-stored embeddings; swap in a pgvector ``ORDER BY embedding <=> :q``
query here for large corpora without changing callers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

import numpy as np
from sqlalchemy.orm import Session

from app.knowledge.models import KbDocumentChunk
from app.knowledge.providers.embeddings import EmbeddingProvider

# Generic words that appear in many document names — matching on these alone
# would wrongly "name" every requirement doc, so they don't count as a mention.
_TITLE_STOPWORDS = {
    "requirement", "requirements", "document", "documents", "doc", "docs",
    "spec", "specs", "page", "the", "mvp", "qa", "bug",
}


def _norm(text: Optional[str]) -> str:
    """Lowercase and strip everything but letters/digits (so 'requirement-mvp3'
    and 'requirement mvp3' both collapse to 'requirementmvp3')."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _distinctive_tokens(title: Optional[str]) -> List[str]:
    """Tokens from a doc title that are specific enough to identify it on their
    own — e.g. 'mvp3' from 'requirement-mvp3'. Generic words are dropped."""
    toks = [t for t in re.split(r"[^a-z0-9]+", (title or "").lower()) if t]
    return [t for t in toks
            if t not in _TITLE_STOPWORDS and (any(c.isdigit() for c in t) or len(t) >= 5)]


@dataclass
class Passage:
    score: float
    doc_type: str
    page_id: str
    page_title: Optional[str]
    section: Optional[str]
    content: str

    def as_dict(self) -> dict:
        return asdict(self)


class VectorIndex:
    def __init__(self, db: Session, embedder: EmbeddingProvider):
        self.db = db
        self.embedder = embedder

    def search(
        self,
        project_id: int,
        query: str,
        top_k: int = 6,
        doc_types: Optional[Sequence[str]] = None,
    ) -> List[Passage]:
        q = self.db.query(KbDocumentChunk).filter(
            KbDocumentChunk.project_id == project_id      # ← isolation key
        )
        if doc_types:
            q = q.filter(KbDocumentChunk.doc_type.in_(list(doc_types)))
        rows = q.all()
        if not rows:
            return []

        # ── Title-aware retrieval ────────────────────────────────────────────
        # A document name ("requirement-mvp3") carries no body semantics, so a
        # plain vector search over chunk *content* never surfaces it — the query
        # "summarize requirement-mvp3" scores highest against unrelated docs and
        # the LLM then refuses. When the question explicitly names a document (by
        # its full title or a distinctive token like "mvp3"), restrict ranking to
        # that document's chunks and boost their scores past the relevance gate.
        nq = _norm(query)
        named_ids = set()
        for row in rows:
            full = _norm(row.page_title) or _norm(row.page_id)
            if (full and len(full) >= 4 and full in nq) or any(
                _norm(tok) in nq
                for tok in _distinctive_tokens(row.page_title) + _distinctive_tokens(row.page_id)
            ):
                named_ids.add(row.page_id)
        named = bool(named_ids)
        if named:
            rows = [row for row in rows if row.page_id in named_ids]

        query_vec = np.asarray(self.embedder.embed_one(query), dtype=np.float32)
        qn = np.linalg.norm(query_vec)
        if qn == 0:
            return []

        scored: List[Passage] = []
        for row in rows:
            vec = np.asarray(row.embedding, dtype=np.float32)
            denom = qn * np.linalg.norm(vec)
            score = float(np.dot(query_vec, vec) / denom) if denom > 0 else 0.0
            if named:
                score += 1.0   # explicitly requested → trust it past the relevance gate
            scored.append(Passage(
                score=score, doc_type=row.doc_type, page_id=row.page_id,
                page_title=row.page_title, section=row.section, content=row.content,
            ))
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:top_k]
