"""Common interface for document sources."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SourceDocument:
    page_id: str          # stable id (Confluence page id / file name)
    doc_type: str         # requirement | qa | bug
    title: str
    version: str          # changes whenever content changes (incremental sync)
    body: str             # plain text / markdown


class DocumentSource:
    """A source yields the current set of documents for one project."""

    def fetch(self) -> List[SourceDocument]:
        raise NotImplementedError
