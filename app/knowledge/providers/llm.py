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
    def generate_web(self, question: str, web_results: List[Dict],
                     doc_passages: List[Dict]) -> Dict: ...


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

    def generate_web(self, question: str, web_results: List[Dict],
                     doc_passages: List[Dict]) -> Dict:
        # Offline mode: no synthesis — stitch the most relevant web snippets
        # together extractively so the feature still returns something usable.
        if not web_results:
            return {"answer": "", "used_web": [], "used_docs": []}
        q = set(_content_words(question))
        scored = sorted(
            ((len(q & set(_content_words(r.get("content", "")))), i)
             for i, r in enumerate(web_results)),
            reverse=True,
        )
        parts, used_web = [], []
        for _score, idx in scored[:3]:
            snippet = _SENT.split((web_results[idx].get("content") or "").strip())
            text = " ".join(snippet[:2]).strip()
            if text:
                parts.append(text)
                used_web.append(idx)
        return {"answer": " ".join(parts), "used_web": used_web,
                "used_docs": list(range(min(2, len(doc_passages))))}


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
            f"[{i}] (doc: {p.get('page_title') or p['page_id']} · type: {p['doc_type']}"
            f"{' · section: ' + p['section'] if p.get('section') else ''})\n{p['content']}"
            for i, p in enumerate(passages)
        )
        system = (
            "You are a project documentation assistant. The passages below are "
            "excerpts from the project's own documents; each is tagged with its "
            "document name ('doc:'), type, and section. Answer the user's question "
            "using ONLY these passages. When the user asks about, or to summarise or "
            "describe, a named document, treat the passages tagged with that document "
            "name as its content and answer from them — do not refuse just because the "
            "passages don't repeat the document's name. Reply exactly 'NOT_AVAILABLE' "
            "only when the passages genuinely lack the information. Cite passages "
            "inline like [0], [1]."
        )
        answer = self._complete(system, f"Passages:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                                 max_tokens=2048)
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

    def generate_web(self, question: str, web_results: List[Dict],
                     doc_passages: List[Dict]) -> Dict:
        web_ctx = "\n\n".join(
            f"[W{i}] {r.get('title')} ({r.get('url')})\n{r.get('content')}"
            for i, r in enumerate(web_results)
        )
        doc_ctx = "\n\n".join(
            f"[D{i}] (doc: {p.get('page_title') or p['page_id']}"
            f"{' · ' + p['section'] if p.get('section') else ''})\n{p['content']}"
            for i, p in enumerate(doc_passages)
        )
        system = (
            "You are a technical assistant helping a software project team. Use the "
            "WEB RESULTS as the source of how-to / implementation knowledge, and the "
            "PROJECT CONTEXT (excerpts from the team's own requirement/QA/bug docs) to "
            "ground the answer in what THIS project actually needs. Answer the user's "
            "question practically — outline concrete steps, approaches, libraries, and "
            "trade-offs. Tie the guidance back to the project's requirement when the "
            "context is relevant. Cite web sources inline like [W0], [W1] and project "
            "context like [D0]. If the web results do not cover the question, say so briefly."
        )
        prompt = (f"PROJECT CONTEXT:\n{doc_ctx or '(none)'}\n\n"
                  f"WEB RESULTS:\n{web_ctx or '(none)'}\n\n"
                  f"Question: {question}\n\nAnswer:")
        # Implementation answers include code blocks and run long — give them a
        # generous budget so they don't truncate mid-sentence.
        answer = self._complete(system, prompt, max_tokens=8192)
        used_web = sorted({int(m) for m in re.findall(r"\[W(\d+)\]", answer)
                           if int(m) < len(web_results)})
        used_docs = sorted({int(m) for m in re.findall(r"\[D(\d+)\]", answer)
                            if int(m) < len(doc_passages)})
        if not used_web:                      # always attribute the web sources used
            used_web = list(range(min(3, len(web_results))))
        return {"answer": answer, "used_web": used_web, "used_docs": used_docs}


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
        base = dict(system_instruction=system, max_output_tokens=max_tokens, temperature=0.0)
        if response_json:
            base["response_mime_type"] = "application/json"  # force strict JSON

        def _run(cfg: dict) -> str:
            resp = self._client.models.generate_content(
                model=self._model, contents=prompt,
                config=types.GenerateContentConfig(**cfg))
            return (resp.text or "").strip()

        # Disable "thinking" so the full token budget goes to the actual output.
        # Gemini 2.5 models otherwise spend the budget thinking and truncate the
        # answer mid-sentence — this hits chat answers and JSON alike. Grounded,
        # temperature-0 tasks (RAG answers, query rewrite, JSON plans) don't need it.
        try:
            cfg = dict(base)
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            return _run(cfg)
        except Exception:
            pass  # SDK/model may not support thinking_config — fall through
        return _run(base)


# Per-provider default model when KB_LLM_MODEL is not set explicitly.
_DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-5", "gemini": "gemini-2.0-flash"}


def build_llm_provider(config: KnowledgeConfig) -> LLMProvider:
    provider = config.llm_provider
    if provider == "anthropic":
        return AnthropicLLM(config.llm_model or _DEFAULT_MODELS["anthropic"])
    if provider == "gemini":
        return GeminiLLM(config.llm_model or _DEFAULT_MODELS["gemini"])
    return LocalLLM()
