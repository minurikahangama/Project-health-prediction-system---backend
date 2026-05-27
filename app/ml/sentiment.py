"""
RoBERTa sentiment scorer — Layer 3a of the PHPS ML pipeline.

Functions:
  score_tone(text)     → float [-1.0, +1.0]   (FR-26 to FR-27)
  detect_urgency(text) → int   {0, 1}          (FR-28 to FR-29)

Model loading:
  - Looks for fine-tuned model at app/ml/inference/roberta_phps/
  - Falls back to pre-trained cardiffnlp/twitter-roberta-base-sentiment-3
    if the fine-tuned model is not yet available
"""
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

# ── Model path ────────────────────────────────────────────────────────────────

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "inference", "roberta_phps")
FALLBACK   = "cardiffnlp/twitter-roberta-base-sentiment-3"

# Use the fine-tuned model if it exists, otherwise use the pretrained fallback
if os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH):
    _model_name = MODEL_PATH
    logger.info(f"Using fine-tuned RoBERTa model from: {MODEL_PATH}")
else:
    _model_name = FALLBACK
    logger.warning(
        f"Fine-tuned model not found at {MODEL_PATH}. "
        f"Using pretrained fallback: {FALLBACK}. "
        "Fine-tune with: backend/app/ml/training/fine_tune_roberta.py"
    )

# Load the pipeline once at startup — expensive to reload per request
_pipe = None

def _get_pipe():
    """Lazy-load the HuggingFace pipeline on first use."""
    global _pipe
    if _pipe is None:
        try:
            from transformers import pipeline as hf_pipeline
            _pipe = hf_pipeline(
                "text-classification",
                model=_model_name,
                device=-1,       # -1 = CPU, 0 = first GPU if available
                truncation=True,
            )
            logger.info("RoBERTa pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RoBERTa model: {e}")
            _pipe = None
    return _pipe


# ── Urgency keyword list ──────────────────────────────────────────────────────

URGENCY_KEYWORDS = {
    "urgent", "asap", "critical", "blocked", "immediately", "overdue",
    "escalate", "at risk", "delay", "deadline", "missed", "behind",
    "falling behind", "showstopper", "blocker", "cannot proceed",
    "production down", "outage", "emergency", "p1", "sev1", "sev-1",
    "not working", "broken", "crashed", "failing", "stuck",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_words: int = 400) -> List[str]:
    """
    Split long text into chunks of max_words words.
    RoBERTa has a 512 token limit — chunking handles longer documents.
    """
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


def _label_to_score(label: str, confidence: float) -> float:
    """
    Convert HuggingFace label + confidence into a signed tone score.

    Label conventions (varies by model):
      positive / label_2 / LABEL_2 → positive score  (+confidence)
      negative / label_0 / LABEL_0 → negative score  (−confidence)
      neutral  / label_1 / LABEL_1 → near zero        (0.0)
    """
    label_lower = label.lower()
    if "positive" in label_lower or label_lower in ("label_2", "2"):
        return confidence
    elif "negative" in label_lower or label_lower in ("label_0", "0"):
        return -confidence
    else:
        return 0.0


# ── Public functions ──────────────────────────────────────────────────────────

def score_tone(text: str) -> float:
    """
    Run the RoBERTa classifier and return a tone score in [-1.0, +1.0].

    +1.0 = strongly positive communication
     0.0 = neutral / informational
    -1.0 = strongly negative / distressed communication

    For long texts, splits into chunks and averages scores.
    Falls back to 0.0 (neutral) if the model is unavailable.

    FR-26: Tone score computation.
    FR-27: Score range [-1.0, +1.0].
    """
    if not text or not text.strip():
        return 0.0

    pipe = _get_pipe()
    if pipe is None:
        logger.warning("RoBERTa unavailable — returning neutral tone score 0.0")
        return 0.0

    try:
        chunks = _chunk_text(text) if len(text.split()) > 400 else [text]
        scores = []

        for chunk in chunks:
            result     = pipe(chunk, truncation=True, max_length=512)[0]
            tone_value = _label_to_score(result["label"], result["score"])
            scores.append(tone_value)

        avg_score = sum(scores) / len(scores)
        return round(avg_score, 4)

    except Exception as e:
        logger.error(f"Error running RoBERTa inference: {e}")
        return 0.0


def detect_urgency(text: str) -> int:
    """
    Keyword-based urgency detection.

    Returns 1 if any urgency keyword is found in the text, 0 otherwise.
    This is a fast rule-based signal that complements the neural tone score.

    FR-28: Urgency flag detection.
    FR-29: Binary 0/1 output.
    """
    if not text:
        return 0
    text_lower = text.lower()
    return 1 if any(kw in text_lower for kw in URGENCY_KEYWORDS) else 0
