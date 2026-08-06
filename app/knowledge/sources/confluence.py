"""Confluence Cloud document source (production path).

Fetches pages from a space, filtered to a set of document types by a label or
title convention, and converts Confluence storage HTML to plain text. Uses the
project's ``page.version.number`` as the incremental-sync fingerprint.

Requires network access + an API token, so it is not exercised by offline tests;
the local folder source stands in for those.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import httpx

from app.knowledge.sources.base import DocumentSource, SourceDocument

_TAG = re.compile(r"<[^>]+>")


def _html_to_text(html: str) -> str:
    text = re.sub(r"<(br|/p|/div|/li|/h[1-6])\s*/?>", "\n", html, flags=re.I)
    text = _TAG.sub("", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class ConfluenceSource(DocumentSource):
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        space_key: str,
        doc_type_labels: Dict[str, str],   # doc_type -> confluence label
        timeout: float = 20.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = (email, api_token)
        self.space_key = space_key
        self.doc_type_labels = doc_type_labels
        self.timeout = timeout

    def test(self) -> "tuple[bool, str]":
        """Verify the credentials and space are reachable (used by the UI)."""
        try:
            with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
                resp = client.get(f"{self.base_url}/wiki/rest/api/space/{self.space_key}")
        except httpx.HTTPError as exc:
            return False, f"Could not reach Confluence: {exc}"
        if resp.status_code == 200:
            return True, f"Connected to Confluence space '{self.space_key}'."
        if resp.status_code in (401, 403):
            return False, "Authentication failed. Check the email and API token."
        if resp.status_code == 404:
            return False, f"Space '{self.space_key}' not found. Check the space key."
        return False, f"Unexpected response from Confluence (HTTP {resp.status_code})."

    def _fetch_all_pages(self) -> List[dict]:
        """List every current page in the space directly (not via the search
        index, which lags after publishing). Includes each page's labels."""
        results: List[dict] = []
        start, limit = 0, 100
        with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
            while True:
                resp = client.get(
                    f"{self.base_url}/wiki/rest/api/content",
                    params={
                        "spaceKey": self.space_key, "type": "page", "status": "current",
                        "expand": "body.storage,version,metadata.labels",
                        "limit": limit, "start": start,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                batch = data.get("results", [])
                results.extend(batch)
                if len(batch) < limit:
                    break
                start += len(batch)
        return results

    def _classify(self, page: dict) -> "str | None":
        """Decide a page's document type from its labels first, then its title."""
        label_to_type = {lab.lower(): dt for dt, lab in self.doc_type_labels.items()}
        page_labels = [l.get("name", "").lower()
                       for l in page.get("metadata", {}).get("labels", {}).get("results", [])]
        for lab in page_labels:
            if lab in label_to_type:
                return label_to_type[lab]
        # Fallback: match by page title (e.g. a page literally titled "requirement").
        title = page.get("title", "").strip().lower()
        for dt, lab in self.doc_type_labels.items():
            if title in (dt, lab.lower()):
                return dt
        for dt in self.doc_type_labels:
            if dt in title:
                return dt
        return None

    def fetch(self) -> List[SourceDocument]:
        docs: List[SourceDocument] = []
        for page in self._fetch_all_pages():
            doc_type = self._classify(page)
            if not doc_type:
                continue                       # templates / unrelated pages
            body_html = page.get("body", {}).get("storage", {}).get("value", "")
            docs.append(SourceDocument(
                page_id=str(page["id"]),
                doc_type=doc_type,
                title=page.get("title", ""),
                version=str(page.get("version", {}).get("number", "1")),
                body=_html_to_text(body_html),
            ))
        return docs
