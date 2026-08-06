"""Embedding providers.

Default is a deterministic hashing embedding (pure NumPy, no downloads) so the
whole RAG pipeline runs offline and produces stable vectors for tests. Set
``KB_EMBEDDING_PROVIDER=sentence-transformers`` to use a real semantic model in
production.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

from app.knowledge.config import KnowledgeConfig

_TOKEN = re.compile(r"[a-z0-9]+")

# Filler words carry no topical signal. Dropping them before hashing stops
# short questions ("what is the ...") from matching unrelated documents purely
# on shared stopwords — which is what keeps out-of-domain questions unanswerable.
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and", "or",
    "in", "on", "for", "with", "that", "this", "it", "as", "at", "by", "from",
    "what", "which", "who", "how", "why", "when", "does", "do", "can", "we",
    "our", "my", "i", "you", "your", "please", "tell", "me", "about", "explain",
    "shall", "will", "has", "have", "had", "not", "no",
}


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class EmbeddingProvider:
    dim: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class HashingEmbedding(EmbeddingProvider):
    """Bag-of-words feature hashing → L2-normalized vector.

    Token overlap between two texts produces a meaningful cosine similarity,
    which is all the retrieval layer needs, while remaining fully deterministic
    and dependency-light.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t).tolist() for t in texts]


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Real semantic embeddings via sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def build_embedding_provider(config: KnowledgeConfig) -> EmbeddingProvider:
    if config.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedding(config.embedding_model)
    return HashingEmbedding(dim=config.embedding_dim)
