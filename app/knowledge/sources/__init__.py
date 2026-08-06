"""Document sources for ingestion: Confluence (production) and local folder (offline/test)."""
from app.knowledge.sources.base import SourceDocument, DocumentSource
from app.knowledge.sources.local import LocalFolderSource
from app.knowledge.sources.confluence import ConfluenceSource

__all__ = ["SourceDocument", "DocumentSource", "LocalFolderSource", "ConfluenceSource"]
