"""Split document text into passages, preserving a section label for citations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    section: str
    content: str


def chunk_text(body: str, chunk_chars: int = 700, overlap: int = 120) -> List[Chunk]:
    """Split on Markdown headings first, then window long sections by characters."""
    lines = body.splitlines()
    sections: List[tuple[str, List[str]]] = []
    current_title = "Introduction"
    current_lines: List[str] = []

    for line in lines:
        m = _HEADING.match(line)
        if m:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = m.group(2).strip() or current_title
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    if not sections:
        sections = [("Introduction", lines)]

    chunks: List[Chunk] = []
    for title, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if not text:
            continue
        if len(text) <= chunk_chars:
            chunks.append(Chunk(section=title, content=text))
            continue
        start = 0
        while start < len(text):
            piece = text[start:start + chunk_chars].strip()
            if piece:
                chunks.append(Chunk(section=title, content=piece))
            start += max(1, chunk_chars - overlap)
    return chunks
