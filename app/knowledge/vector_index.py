"""Project-scoped semantic retrieval over stored document chunks.

Retrieval ALWAYS filters by ``project_id`` (and optional doc types) before
ranking, which is what structurally enforces project isolation — no query can
reach another project's chunks. Ranking is cosine similarity computed in Python
over JSON-stored embeddings; swap in a pgvector ``ORDER BY embedding <=> :q``
query here for large corpora without changing callers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

import numpy as np
from sqlalchemy.orm import Session

from app.knowledge.models import KbDocumentChunk
from app.knowledge.providers.embeddings import EmbeddingProvider


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

        query_vec = np.asarray(self.embedder.embed_one(query), dtype=np.float32)
        qn = np.linalg.norm(query_vec)
        if qn == 0:
            return []

        scored: List[Passage] = []
        for row in rows:
            vec = np.asarray(row.embedding, dtype=np.float32)
            denom = qn * np.linalg.norm(vec)
            score = float(np.dot(query_vec, vec) / denom) if denom > 0 else 0.0
            scored.append(Passage(
                score=score, doc_type=row.doc_type, page_id=row.page_id,
                page_title=row.page_title, section=row.section, content=row.content,
            ))
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:top_k]
