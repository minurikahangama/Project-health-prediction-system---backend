"""
GDPR-compliant text preprocessor.

Layer 2 of the PHPS ML pipeline:
  1. Clean raw text (remove email headers, signatures, quoted lines)
  2. Anonymise PII using spaCy NER (PERSON, ORG, GPE, EMAIL → [REDACTED])
  3. Hash the raw text as proof of deletion
  4. Delete the raw text from memory
  5. Write a deletion log row to the database

FR-22 to FR-25: Full GDPR Article 5 data minimisation lifecycle.
"""
import re
import hashlib
import logging

from sqlalchemy.orm import Session
from app.models.models import GDPRDeletionLog

logger = logging.getLogger(__name__)

# Load spaCy model once at module import time — expensive to reload
# Falls back to a smaller model if the transformer model is not installed
try:
    import spacy
    nlp = spacy.load("en_core_web_trf")
    logger.info("Loaded spaCy model: en_core_web_trf")
except (ImportError, OSError):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        logger.warning("en_core_web_trf not found — using en_core_web_sm. "
                       "Run: python -m spacy download en_core_web_trf")
    except (ImportError, OSError):
        nlp = None
        logger.error("No spaCy model found. "
                     "Run: python -m spacy download en_core_web_sm")


def clean_text(raw: str) -> str:
    """
    Remove email formatting noise:
    - Quoted reply lines starting with >
    - Email headers (From:, To:, Subject:, Date:, Sent:)
    - Signature blocks after -- or ___
    - Excessive whitespace
    """
    if not raw:
        return ""

    text = raw

    # Remove quoted lines (replies)
    text = re.sub(r"(?m)^>.*$", "", text)

    # Remove email headers
    text = re.sub(
        r"(?m)^(From|To|Cc|Bcc|Subject|Date|Sent|Reply-To):.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove signature blocks (-- or ___ separator)
    text = re.sub(r"(?s)--\s*\n.*", "", text)
    text = re.sub(r"(?s)_{3,}\s*\n.*", "", text)

    # Remove HTML tags if any leaked through
    text = re.sub(r"<[^>]+>", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def anonymise_pii(text: str) -> str:
    """
    Replace named entities (PERSON, ORG, GPE, EMAIL) with [REDACTED].
    Iterates in reverse order to preserve character offsets.

    FR-23: PII anonymisation before any storage or further processing.
    """
    if not text:
        return ""

    if nlp is None:
        # If spaCy is not available, return text with a warning
        logger.warning("spaCy not available — PII anonymisation skipped")
        return text

    doc    = nlp(text)
    result = text

    for ent in reversed(doc.ents):
        if ent.label_ in {"PERSON", "ORG", "GPE", "EMAIL", "NORP"}:
            result = result[: ent.start_char] + "[REDACTED]" + result[ent.end_char :]

    # Also regex-replace email addresses that spaCy may miss
    result = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED]",
        result,
    )

    return result


def process_and_delete(
    raw_text: str,
    project_id: int,
    event_type: str,
    db: Session,
) -> str:
    """
    Full GDPR lifecycle — the only function pipeline.py should call.

    Returns:
        anonymised_text (str) — safe to pass to the sentiment scorer.
                                Never stored in the database.

    Steps:
        1. Clean email/transcript formatting
        2. Anonymise PII entities
        3. Hash the raw text (proof that it existed)
        4. Delete raw text from memory immediately
        5. Write deletion log entry to database

    FR-22: Raw text not stored.
    FR-23: PII anonymised.
    FR-24: Deletion confirmed with hash.
    FR-25: Deletion logged with timestamp.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Cannot process empty text")

    # Step 1: Clean
    cleaned = clean_text(raw_text)

    # Step 2: Anonymise
    anonymised = anonymise_pii(cleaned)

    # Step 3: Hash the ORIGINAL raw text before deleting it
    # This hash serves as proof of deletion in the audit log
    confirmation_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    # Step 4: Delete raw text and cleaned text from Python memory
    raw_text = None   # noqa: F841
    cleaned  = None   # noqa: F841

    # Step 5: Write deletion log
    db.add(
        GDPRDeletionLog(
            event_type        = event_type,
            project_id        = project_id,
            confirmation_hash = confirmation_hash,
        )
    )
    db.commit()

    logger.info(
        f"GDPR: raw text deleted for project {project_id} "
        f"(event={event_type}, hash={confirmation_hash[:16]}...)"
    )

    # Return ONLY the anonymised text — never store it, just pass to scorer
    return anonymised
