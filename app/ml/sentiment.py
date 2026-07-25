"""RoBERTa sentiment scorer — Layer 3a of the PHPS ML pipeline.

The fine-tuned model returns negative, neutral and positive probabilities.
``score_tone`` exposes them as a single score in [-1, 1] by calculating
``P(positive) - P(negative)``. A small keyword fallback keeps ingestion
available if the model files or ML dependencies are unavailable.
"""
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "inference" / "roberta_phps"
_tokenizer = None
_model = None
_load_attempted = False

SENTIMENT_DICTIONARY = {
    "broken": -0.85, "stuck": -0.70, "blocked": -0.75, "fail": -0.80,
    "crashed": -0.90, "urgent": -0.40, "asap": -0.30, "critical": -0.80,
    "delay": -0.65, "delayed": -0.65, "behind": -0.60, "risk": -0.50,
    "issue": -0.45, "problem": -0.55, "concern": -0.40, "unhappy": -0.70,
    "frustrated": -0.75, "disappointed": -0.70, "overdue": -0.65,
    "escalate": -0.70, "outage": -0.90, "regression": -0.75,
    "success": 0.85, "fixed": 0.80, "working": 0.75, "done": 0.70,
    "great": 0.90, "perfect": 0.95, "complete": 0.60, "resolved": 0.70,
    "progress": 0.65, "improved": 0.70, "improvement": 0.70,
    "ahead": 0.65, "smooth": 0.65, "stable": 0.55, "confident": 0.65,
    "pleased": 0.75, "happy": 0.70, "positive": 0.60, "excellent": 0.90,
    "productive": 0.65, "delivered": 0.70, "achieved": 0.70,
}

URGENCY_KEYWORDS = {
    "urgent", "asap", "critical", "blocked", "immediately", "overdue",
    "escalate", "at risk", "delay", "deadline", "missed", "behind",
    "falling behind", "showstopper", "blocker", "cannot proceed",
    "production down", "outage", "emergency", "p1", "sev1", "sev-1",
    "not working", "broken", "crashed", "failing", "stuck",
}


def _get_model() -> Optional[Tuple[object, object]]:
    """Load the local fine-tuned model once, on its first inference call."""
    global _tokenizer, _model, _load_attempted
    if _load_attempted:
        return (_tokenizer, _model) if _model is not None else None
    _load_attempted = True

    if not MODEL_PATH.is_dir():
        logger.warning("RoBERTa model directory is missing: %s", MODEL_PATH)
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        _model.eval()
        logger.info("Loaded fine-tuned RoBERTa sentiment model from %s", MODEL_PATH)
        return _tokenizer, _model
    except Exception as exc:
        logger.exception("Could not load RoBERTa sentiment model; using keyword fallback: %s", exc)
        _tokenizer = None
        _model = None
        return None


def _keyword_score(text: str) -> float:
    lower = text.lower()
    # Word-boundary matching avoids treating unrelated substrings as evidence
    # (for example, "issue" inside another word) while supporting ordinary
    # project language when the optional RoBERTa package is unavailable.
    scores = [
        weight for word, weight in SENTIMENT_DICTIONARY.items()
        if re.search(rf"\b{re.escape(word)}\b", lower)
    ]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _calibrated_tone(logits) -> float:
    """Temperature-calibrate raw model logits before the [-1, 1] mapping."""
    import torch
    temperature = max(0.05, float(os.getenv("PHPS_SENTIMENT_TEMPERATURE", "1.0")))
    probabilities = torch.softmax(logits / temperature, dim=-1)[0]
    return float(probabilities[2] - probabilities[0])


def score_tone(text: str) -> float:
    """Score communication sentiment with the fine-tuned local RoBERTa model."""
    if not text or not text.strip():
        return 0.0

    loaded = _get_model()
    if loaded is None:
        return _keyword_score(text)

    tokenizer, model = loaded
    try:
        import torch

        # Overlapping windows preserve context at long-transcript boundaries.
        token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        window = min(510, max(1, int(getattr(tokenizer, "model_max_length", 512)) - 2))
        stride = max(1, int(os.getenv("PHPS_SENTIMENT_WINDOW_STRIDE", str(window // 2))))
        chunks = [token_ids[i:i + window] for i in range(0, len(token_ids), stride)] or [[]]
        scores = []
        with torch.no_grad():
            for chunk in chunks:
                # Transformers 5 removed tokenizer preparation helpers. A
                # single RoBERTa sequence is encoded as <s> tokens </s>;
                # construct that one-item batch directly with its configured
                # special-token IDs. Chunks contain at most 510 content
                # tokens, leaving room for both special tokens.
                input_ids = [tokenizer.bos_token_id, *chunk, tokenizer.eos_token_id]
                inputs = {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
                }
                scores.append(_calibrated_tone(model(**inputs).logits))
        return round(sum(scores) / len(scores), 4)
    except Exception as exc:
        logger.exception("RoBERTa inference failed; using keyword fallback: %s", exc)
        return _keyword_score(text)


def detect_urgency(text: str) -> int:
    """Binary urgency flag used alongside model sentiment in score fusion."""
    if not text:
        return 0
    lower = text.lower()
    # A simple substring check made reassuring updates such as "no blockers"
    # and "not at risk" trigger a health penalty.  Remove explicit negated
    # risk statements before checking for genuine escalation language.
    lower = re.sub(
        r"\bno\s+(?:(?:blockers?|blocked|risks?|delays?|issues?|overdue)"
        r"\s*(?:,|and|or)?\s*)+",
        "",
        lower,
    )
    lower = re.sub(
        r"\b(?:no|not|without|zero)\s+(?:active\s+)?"
        r"(?:urgent|urgency|blockers?|blocked|risks?|delays?|issues?|"
        r"overdue|escalations?|outages?|emergencies)\b",
        "",
        lower,
    )
    return int(any(keyword in lower for keyword in URGENCY_KEYWORDS))
