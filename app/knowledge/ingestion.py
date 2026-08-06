"""Ingestion pipeline: sync a source's documents into the knowledge base.

Incremental: a page whose version fingerprint is unchanged is skipped; a changed
page is re-chunked and re-embedded; a page that has disappeared from the source
is deleted from the knowledge base (BR: deleted pages removed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from sqlalchemy.orm import Session

from app.knowledge.chunker import chunk_text
from app.knowledge.config import KnowledgeConfig
from app.knowledge.models import KbDocumentChunk
from app.knowledge.providers.embeddings import EmbeddingProvider
from app.knowledge.sources.base import DocumentSource


@dataclass
class SyncStats:
    added_pages: int = 0
    updated_pages: int = 0
    unchanged_pages: int = 0
    deleted_pages: int = 0
    total_chunks: int = 0
    doc_types: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__


class Ingestor:
    def __init__(self, db: Session, embedder: EmbeddingProvider, config: KnowledgeConfig):
        self.db = db
        self.embedder = embedder
        self.config = config

    def _existing_versions(self, project_id: int) -> Dict[str, str]:
        rows = (
            self.db.query(KbDocumentChunk.page_id, KbDocumentChunk.page_version)
            .filter(KbDocumentChunk.project_id == project_id)
            .distinct()
            .all()
        )
        return {page_id: version for page_id, version in rows}

    def _delete_page(self, project_id: int, page_id: str) -> None:
        (self.db.query(KbDocumentChunk)
             .filter(KbDocumentChunk.project_id == project_id,
                     KbDocumentChunk.page_id == page_id)
             .delete(synchronize_session=False))

    def sync_project(self, project_id: int, source: DocumentSource) -> SyncStats:
        stats = SyncStats()
        existing = self._existing_versions(project_id)
        documents = source.fetch()
        seen_pages = set()
        doc_types = set()

        for doc in documents:
            seen_pages.add(doc.page_id)
            doc_types.add(doc.doc_type)
            prior = existing.get(doc.page_id)

            if prior == doc.version:
                stats.unchanged_pages += 1
                continue

            # New or changed → replace all chunks for this page.
            if prior is not None:
                self._delete_page(project_id, doc.page_id)
                stats.updated_pages += 1
            else:
                stats.added_pages += 1

            chunks = chunk_text(doc.body, self.config.chunk_chars, self.config.chunk_overlap)
            if not chunks:
                continue
            embeddings = self.embedder.embed([c.content for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                self.db.add(KbDocumentChunk(
                    project_id=project_id, doc_type=doc.doc_type,
                    page_id=doc.page_id, page_title=doc.title,
                    page_version=doc.version, section=chunk.section,
                    content=chunk.content, embedding=emb,
                ))
                stats.total_chunks += 1

        # Pages removed at the source → delete from knowledge base.
        for page_id in existing:
            if page_id not in seen_pages:
                self._delete_page(project_id, page_id)
                stats.deleted_pages += 1

        self.db.commit()
        stats.doc_types = sorted(doc_types)
        return stats
