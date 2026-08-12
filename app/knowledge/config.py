"""Environment-driven settings for the FR-13 knowledge assistant.

All knobs default to the offline/local path so the feature is runnable without
any external service. Set the KB_* variables in .env to switch to Confluence +
a hosted embedding/LLM provider in production.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class KnowledgeConfig:
    # Retrieval
    top_k: int = int(_get("KB_TOP_K", "6"))
    relevance_min: float = float(_get("KB_RELEVANCE_MIN", "0.12"))
    max_retries: int = int(_get("KB_MAX_RETRIES", "1"))

    # Providers
    embedding_provider: str = _get("KB_EMBEDDING_PROVIDER", "hashing")  # hashing | sentence-transformers
    embedding_model: str = _get("KB_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = int(_get("KB_EMBEDDING_DIM", "256"))
    llm_provider: str = _get("KB_LLM_PROVIDER", "local")  # local | gemini | anthropic
    llm_model: str = _get("KB_LLM_MODEL", "")             # empty → per-provider default

    # Ingestion source
    default_source: str = _get("KB_DEFAULT_SOURCE", "local")  # local | confluence
    seed_dir: str = _get(
        "KB_SEED_DIR",
        os.path.join(os.path.dirname(__file__), "seed_docs"),
    )

    # Chunking
    chunk_chars: int = int(_get("KB_CHUNK_CHARS", "700"))
    chunk_overlap: int = int(_get("KB_CHUNK_OVERLAP", "120"))

    # Web search (Tavily) — powers the assistant's optional "Search the web" toggle.
    # Empty key → web search stays disabled and the toggle degrades gracefully.
    tavily_api_key: str = _get("TAVILY_API_KEY", "")
    web_max_results: int = int(_get("KB_WEB_MAX_RESULTS", "5"))
    web_context_top_k: int = int(_get("KB_WEB_CONTEXT_TOP_K", "3"))


def load_config() -> KnowledgeConfig:
    return KnowledgeConfig()


# Canonical document types (BR: doc_type filtering / project isolation).
DOC_TYPES = ("requirement", "qa", "bug")
