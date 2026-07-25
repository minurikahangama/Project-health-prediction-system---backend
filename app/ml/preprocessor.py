"""Feature cleaning, calibration and temporal-window assembly for PHPS.

The scaler is fitted only by the training command and saved with the model.
Runtime code only calls ``transform``; this prevents a single project's data
from changing the calibration used by another project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
import hashlib
import logging
from sqlalchemy.orm import Session
from app.models.models import GDPRDeletionLog

from app.ml.features import FEATURE_NAMES

SENTIMENT_FEATURES = {"overall_sentiment", "sentiment_trend"}
SCALER_FEATURES = tuple(name for name in FEATURE_NAMES if name not in SENTIMENT_FEATURES)
DEFAULT_SCALER_PATH = Path(__file__).resolve().parent / "training" / "phps_feature_scaler.joblib"
logger = logging.getLogger(__name__)

# This is intentionally independent from numerical feature calibration.  The
# import remains here to preserve the existing GDPR ingestion contract.
try:
    import spacy
    nlp = spacy.load("en_core_web_trf")
except (ImportError, OSError):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        nlp = None


class FeaturePreprocessor:
    """Clean vectors and min-max calibrate non-sentiment inputs to [0, 1]."""

    def __init__(self, scaler: MinMaxScaler | None = None):
        self.scaler = scaler or MinMaxScaler(clip=True)
        self._fitted = scaler is not None

    @staticmethod
    def clean(features: dict) -> dict[str, float]:
        values = {name: float(features.get(name, 0.0) or 0.0) for name in FEATURE_NAMES}
        for name in ("overall_sentiment", "sentiment_trend"):
            values[name] = float(np.clip(values[name], -1.0, 1.0))
        for name in ("velocity", "overdue_rate", "bug_ratio", "velocity_trend", "bug_trend"):
            values[name] = float(np.clip(values[name], -1.0 if "trend" in name else 0.0, 1.0))
        for name in ("urgency_count", "open_issues", "open_bugs", "email_count", "transcript_count", "days_to_deadline"):
            values[name] = max(0.0, values[name])
        return values

    def fit(self, rows: Iterable[dict]) -> "FeaturePreprocessor":
        cleaned = [self.clean(row) for row in rows]
        if not cleaned:
            raise ValueError("Cannot fit a feature scaler without training rows")
        self.scaler.fit(np.asarray([[row[name] for name in SCALER_FEATURES] for row in cleaned], dtype=float))
        self._fitted = True
        return self

    def transform_one(self, features: dict) -> dict[str, float]:
        row = self.clean(features)
        if not self._fitted:
            # A missing calibration artifact must not silently fit production data.
            return row
        scaled = self.scaler.transform([[row[name] for name in SCALER_FEATURES]])[0]
        row.update({name: float(value) for name, value in zip(SCALER_FEATURES, scaled)})
        return row

    def transform_matrix(self, rows: Iterable[dict]) -> np.ndarray:
        return np.asarray([[self.transform_one(row)[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)

    def save(self, path: Path = DEFAULT_SCALER_PATH) -> None:
        if not self._fitted:
            raise ValueError("Fit the preprocessor before saving it")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)

    @classmethod
    def load(cls, path: Path = DEFAULT_SCALER_PATH) -> "FeaturePreprocessor":
        return cls(joblib.load(path)) if path.is_file() else cls()


def clip_health(value: float) -> float:
    """The only valid health target/prediction range is 0 through 100."""
    return float(np.clip(float(value), 0.0, 100.0))


def rolling_windows(rows: Iterable[dict], window: int = 4) -> np.ndarray:
    """Return chronological fixed-size temporal windows, left-padding early rows."""
    matrix = np.asarray(list(rows), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected rows with {len(FEATURE_NAMES)} features")
    if len(matrix) == 0:
        return np.empty((0, window, len(FEATURE_NAMES)), dtype=np.float32)
    result = []
    for end in range(len(matrix)):
        start = max(0, end - window + 1)
        values = matrix[start:end + 1]
        if len(values) < window:
            values = np.vstack([np.repeat(values[:1], window - len(values), axis=0), values])
        result.append(values)
    return np.asarray(result, dtype=np.float32)


# ── GDPR text preprocessing (legacy public API) ────────────────────────────
def clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r"(?m)^>.*$", "", raw)
    text = re.sub(r"(?m)^(From|To|Cc|Bcc|Subject|Date|Sent|Reply-To):.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?s)--\s*\n.*", "", text)
    text = re.sub(r"(?s)_{3,}\s*\n.*", "", text)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def anonymise_pii(text: str) -> str:
    if not text:
        return ""
    result = text
    if nlp is not None:
        for entity in reversed(nlp(text).ents):
            if entity.label_ in {"PERSON", "ORG", "GPE", "EMAIL", "NORP"}:
                result = result[:entity.start_char] + "[REDACTED]" + result[entity.end_char:]
    return re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED]", result)


def process_and_delete(raw_text: str, project_id: int, event_type: str, db: Session) -> str:
    """Clean and anonymise text, then retain only a deletion-proof hash."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Cannot process empty text")
    anonymised = anonymise_pii(clean_text(raw_text))
    confirmation_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    raw_text = None  # noqa: F841 -- deliberately discard raw content
    db.add(GDPRDeletionLog(event_type=event_type, project_id=project_id, confirmation_hash=confirmation_hash))
    db.commit()
    return anonymised
