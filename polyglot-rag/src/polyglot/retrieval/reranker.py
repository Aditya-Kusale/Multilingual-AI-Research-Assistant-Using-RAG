"""Cross-encoder reranking.

This is the single highest-leverage component in a multilingual RAG stack.
A bi-encoder compresses a query and a passage into vectors independently; a
cross-encoder reads both together and can tell that a Hindi question about
"मूल्यांकन" is answered by an English passage on "evaluation protocol" —
a judgement the bi-encoder can only approximate.

Default model: BAAI/bge-reranker-v2-m3 (multilingual, cross-lingual).
Cost: one forward pass per candidate, so we only rerank the top
`reranker.candidates` fused hits (50 by default), not the whole index.
"""

from __future__ import annotations

import logging

from ..config import RerankerConfig
from ..types import ScoredChunk

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, cfg: RerankerConfig | None = None):
        self.cfg = cfg or RerankerConfig()
        self._model = None
        self._available: bool | None = None

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    @property
    def candidates(self) -> int:
        return self.cfg.candidates

    def load(self) -> "CrossEncoderReranker":
        if self._available is not None:
            return self
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.cfg.model_name, device=self.cfg.device)
            self._available = True
            log.info("Loaded reranker %s", self.cfg.model_name)
        except Exception as exc:
            log.warning(
                "Reranker unavailable (%s). Falling back to fusion order — "
                "expect a noticeable precision drop, especially cross-lingually.",
                exc,
            )
            self._available = False
        return self

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        self.load()
        if not self._available:
            return candidates

        pairs = [(query, sc.chunk.text) for sc in candidates]
        scores = self._model.predict(pairs, batch_size=self.cfg.batch_size)

        for sc, score in zip(candidates, scores):
            sc.rerank_score = float(score)

        return sorted(candidates, key=lambda sc: -(sc.rerank_score or 0.0))
