"""Local folder document source — used for seeding and offline tests.

Layout:  <root>/<project_key>/<doc_type>.md
e.g.     seed_docs/alpha/requirement.md
         seed_docs/alpha/qa.md
         seed_docs/alpha/bug.md

``doc_type`` is derived from the file stem; the version is a content hash so an
edited file re-syncs and an unchanged file is skipped (incremental sync).
"""
from __future__ import annotations

import hashlib
import os
from typing import List

from app.knowledge.config import DOC_TYPES
from app.knowledge.sources.base import DocumentSource, SourceDocument


class LocalFolderSource(DocumentSource):
    def __init__(self, project_dir: str):
        self.project_dir = project_dir

    def fetch(self) -> List[SourceDocument]:
        docs: List[SourceDocument] = []
        if not os.path.isdir(self.project_dir):
            return docs
        for fname in sorted(os.listdir(self.project_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in (".md", ".txt"):
                continue
            doc_type = stem.lower()
            if doc_type not in DOC_TYPES:
                continue
            path = os.path.join(self.project_dir, fname)
            with open(path, "r", encoding="utf-8") as fh:
                body = fh.read()
            version = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
            title = body.splitlines()[0].lstrip("# ").strip() if body.strip() else fname
            docs.append(SourceDocument(
                page_id=fname, doc_type=doc_type, title=title or fname,
                version=version, body=body,
            ))
        return docs
