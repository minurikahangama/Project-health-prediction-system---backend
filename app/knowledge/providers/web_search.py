"""Tavily web-search provider for the assistant's optional 'Search the web' mode.

Only used when the user explicitly toggles web search on a question (FR-13
extension). The key is read from ``TAVILY_API_KEY``; when it is absent the
provider reports itself disabled so the agent can degrade gracefully instead of
erroring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import List

import httpx

from app.knowledge.config import KnowledgeConfig

logger = logging.getLogger("phps.web_search")

TAVILY_URL = "https://api.tavily.com/search"


@dataclass
class WebResult:
    title: str
    url: str
    content: str

    def as_dict(self) -> dict:
        return asdict(self)


class WebSearchProvider:
    """No-op base — used when web search is not configured."""
    enabled: bool = False

    def search(self, query: str, max_results: int = 5) -> List[WebResult]:
        return []


class TavilySearch(WebSearchProvider):
    def __init__(self, api_key: str, timeout: float = 20.0):
        self._api_key = api_key
        self._timeout = timeout
        self.enabled = bool(api_key)

    def search(self, query: str, max_results: int = 5) -> List[WebResult]:
        if not self.enabled:
            return []
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": False,
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(TAVILY_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Tavily search failed: %s", exc)
            return []

        results: List[WebResult] = []
        for r in data.get("results", []):
            url = (r.get("url") or "").strip()
            if not url:
                continue
            results.append(WebResult(
                title=(r.get("title") or url).strip(),
                url=url,
                content=(r.get("content") or "").strip(),
            ))
        return results


def build_web_search_provider(config: KnowledgeConfig) -> WebSearchProvider:
    if config.tavily_api_key:
        return TavilySearch(config.tavily_api_key)
    return WebSearchProvider()  # disabled no-op
