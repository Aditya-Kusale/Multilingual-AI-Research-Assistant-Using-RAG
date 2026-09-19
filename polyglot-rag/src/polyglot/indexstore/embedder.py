"""Multilingual embeddings.

One shared vector space for every language is the whole point: a Hindi query
must be able to retrieve an English passage without a translation hop. That
rules out English-only encoders and translate-then-embed pipelines (which
lose terminology and leave citations pointing at text the user cannot check).

Default: BAAI/bge-m3 (100+ languages, 8192 ctx, strong cross-lingual
alignment). Swap `embedding.model_name` for intfloat/multilingual-e5-large or
a hosted Cohere embed-multilingual-v3 model without touching anything else.

If sentence-transformers is unavailable, we fall back to a deterministic
hashing encoder so the pipeline, the tests and the CI run end to end. The
fallback is NOT semantically useful — it exists to keep the system runnable.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

from ..config import EmbeddingConfig

log = logging.getLogger(__name__)

_FALLBACK_DIM = 384


class Embedder:
    def __init__(self, cfg: EmbeddingConfig | None = None):
        self.cfg = cfg or EmbeddingConfig()
        self._model = None
        self._dim: int | None = None
        self.backend = "uninitialised"

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> "Embedder":
        if self._model is not None or self.backend == "hashing":
            return self
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.cfg.model_name, device=self.cfg.device
            )
            self._dim = int(self._model.get_sentence_embedding_dimension())
            self.backend = "sentence-transformers"
            log.info("Loaded embedder %s (dim=%d)", self.cfg.model_name, self._dim)
        except Exception as exc:
            log.warning(
                "Falling back to hashing embeddings (%s). "
                "Install sentence-transformers for real retrieval quality.", exc
            )
            self.backend = "hashing"
            self._dim = _FALLBACK_DIM
        return self

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.load()
        return int(self._dim)

    # -- encoding ----------------------------------------------------------

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.cfg.passage_prefix)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.cfg.query_prefix)

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        self.load()
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        prepared = [f"{prefix}{t}" if prefix else t for t in texts]

        if self.backend == "sentence-transformers":
            vectors = self._model.encode(
                prepared,
                batch_size=self.cfg.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.cfg.normalize,
                show_progress_bar=len(prepared) > 256,
            )
            return np.asarray(vectors, dtype="float32")

        vectors = np.vstack([_hash_vector(t, self.dim) for t in prepared])
        if self.cfg.normalize:
            vectors = _l2_normalise(vectors)
        return vectors.astype("float32")


def _hash_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic bag-of-character-ngrams projection. Offline fallback only."""
    from ..language import lexical_tokens

    vec = np.zeros(dim, dtype="float32")
    for token in lexical_tokens(text):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0 if h % 2 == 0 else -1.0
    return vec


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
