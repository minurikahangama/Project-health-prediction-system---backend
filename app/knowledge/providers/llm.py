"""LLM providers for the agent's grade / rewrite / generate / grounding steps.

Default is a deterministic, extractive ``LocalLLM`` so the agent runs and is
testable with no API key. Set ``KB_LLM_PROVIDER=anthropic`` (and ANTHROPIC_API_KEY)
to use Claude for higher-quality grading and generation in production.

An answer is always assembled strictly from the retrieved passages — the
providers never introduce outside knowledge (BR: no fabrication).
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

from app.knowledge.config import KnowledgeConfig

_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_GREETING = {"hi", "hello", "hey", "thanks", "thank", "ok", "okay", "bye"}
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or",
    "in", "on", "for", "with", "that", "this", "it", "as", "at", "by", "from",
    "what", "which", "who", "how", "why", "when", "does", "do", "can", "we",
    "our", "my", "i", "you", "your", "please", "tell", "me", "about", "explain",
}


def _words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _content_words(text: str) -> List[str]:
    return [w for w in _words(text) if w not in _STOP and len(w) > 1]


class LLMProvider:
    def classify_in_scope(self, question: str) -> bool: ...
    def rewrite_query(self, question: str) -> str: ...
    def generate(self, question: str, passages: List[Dict]) -> Dict: ...
    def check_grounding(self, answer: str, passages: List[Dict]) -> bool: ...


class LocalLLM(LLMProvider):
    """Deterministic extractive assistant — grounded by construction."""

    def classify_in_scope(self, question: str) -> bool:
        toks = _words(question)
        if not toks:
            return False
        if all(t in _GREETING for t in toks):
            return False
        return True

    def rewrite_query(self, question: str) -> str:
        # Reformulate by stripping stop/filler words down to the content terms,
        # which removes noise and sharpens the semantic match on a weak first
        # pass — without inventing new topic words that could match unrelated docs.
        core = _content_words(question)
        if not core:
            return question
        return " ".join(dict.fromkeys(core))

    def generate(self, question: str, passages: List[Dict]) -> Dict:
        q = set(_content_words(question))
        scored = []
        for idx, p in enumerate(passages):
            for sent in _SENT.split(p["content"].strip()):
                sent = sent.strip()
                if not sent:
                    continue
                overlap = len(q & set(_content_words(sent)))
                if overlap > 0:
                    scored.append((overlap, idx, sent))
        scored.sort(key=lambda r: r[0], reverse=True)

        used: List[int] = []
        sentences: List[str] = []
        for _score, idx, sent in scored:
            if sent in sentences:
                continue
            sentences.append(sent)
            if idx not in used:
                used.append(idx)
            if len(sentences) >= 3:
                break

        if not sentences and passages:
            # Nothing overlapped lexically but retrieval still surfaced a passage;
            # fall back to the top passage's first sentence.
            first = _SENT.split(passages[0]["content"].strip())[0].strip()
            sentences = [first]
            used = [0]

        answer = " ".join(sentences)
        return {"answer": answer, "used": used}

    def check_grounding(self, answer: str, passages: List[Dict]) -> bool:
        if not answer.strip() or not passages:
            return False
        # Every content word in the answer must appear in some passage.
        corpus = set()
        for p in passages:
            corpus.update(_content_words(p["content"]))
        answer_words = set(_content_words(answer))
        if not answer_words:
            return False
        unsupported = answer_words - corpus
        return len(unsupported) == 0


class HostedLLM(LLMProvider):
    """Shared prompt logic for API-backed providers (Anthropic, Gemini, ...).

    Subclasses only implement ``_complete``. The cheap question-classification
    step reuses LocalLLM heuristics to save tokens. Generation is instructed to
    answer ONLY from the passages and to emit ``NOT_AVAILABLE`` when they do not
    cover the question (BR: no fabrication).
    """

    def __init__(self):
        self._local = LocalLLM()

    def _complete(self, system: str, prompt: str, max_tokens: int = 700,
                  response_json: bool = False) -> str:
        raise NotImplementedError

    def classify_in_scope(self, question: str) -> bool:
        return self._local.classify_in_scope(question)

    def rewrite_query(self, question: str) -> str:
        out = self._complete(
            "You reformulate a user question into a concise search query. Output only the query.",
            f"Question: {question}\nSearch query:",
            max_tokens=60,
        )
        return out or self._local.rewrite_query(question)

    def generate(self, question: str, passages: List[Dict]) -> Dict:
        context = "\n\n".join(
            f"[{i}] ({p['doc_type']} · {p.get('section') or p['page_id']})\n{p['content']}"
            for i, p in enumerate(passages)
        )
        system = (
            "You are a project documentation assistant. Answer ONLY from the "
            "provided passages. If they do not contain the answer, reply exactly "
            "'NOT_AVAILABLE'. Cite passages inline like [0], [1]."
        )
        answer = self._complete(system, f"Passages:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        if "NOT_AVAILABLE" in answer.upper():
            return {"answer": answer, "used": []}
        used = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)
                       if int(m) < len(passages)})
        if not used:
            used = list(range(min(2, len(passages))))
        return {"answer": answer, "used": used}

    def check_grounding(self, answer: str, passages: List[Dict]) -> bool:
        if "NOT_AVAILABLE" in answer.upper():
            return False
        return bool(answer.strip()) and bool(passages)


class AnthropicLLM(HostedLLM):
    """Claude-backed provider (optional)."""

    def __init__(self, model: str):
        super().__init__()
        import anthropic  # lazy import
        self._client = anthropic.Anthropic()
        self._model = model

    def _complete(self, system: str, prompt: str, max_tokens: int = 700,
                  response_json: bool = False) -> str:
        msg = self._client.messages.create(
            model=self._model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text").strip()


class GeminiLLM(HostedLLM):
    """Google Gemini-backed provider. Uses the current ``google-genai`` SDK and
    reads the API key from ``GEMINI_API_KEY`` (or an explicit argument)."""

    def __init__(self, model: str, api_key: str | None = None):
        super().__init__()
        from google import genai  # lazy import
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env or set KB_LLM_PROVIDER=local."
            )
        self._client = genai.Client(api_key=key)
        self._model = model

    def _complete(self, system: str, prompt: str, max_tokens: int = 700,
                  response_json: bool = False) -> str:
        from google.genai import types  # lazy import
        cfg = dict(system_instruction=system, max_output_tokens=max_tokens, temperature=0.0)
        if response_json:
            # Force strict JSON output so the sprint-plan parser never fails.
            cfg["response_mime_type"] = "application/json"
        resp = self._client.models.generate_content(
            model=self._model, contents=prompt,
            config=types.GenerateContentConfig(**cfg),
        )
        return (resp.text or "").strip()


# Per-provider default model when KB_LLM_MODEL is not set explicitly.
_DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-5", "gemini": "gemini-2.0-flash"}


def build_llm_provider(config: KnowledgeConfig) -> LLMProvider:
    provider = config.llm_provider
    if provider == "anthropic":
        return AnthropicLLM(config.llm_model or _DEFAULT_MODELS["anthropic"])
    if provider == "gemini":
        return GeminiLLM(config.llm_model or _DEFAULT_MODELS["gemini"])
    return LocalLLM()
