"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

RRF is used rather than a weighted score blend because dense cosine scores
and BM25 scores live on incomparable scales that also shift per language —
any fixed alpha you tune on English will be wrong on Hindi. RRF only uses
ranks, so it is scale-free and needs no per-language calibration.

    rrf_score(d) = Σ_over_retrievers 1 / (k + rank_r(d))
"""

from __future__ import annotations

from ..config import RetrievalConfig
from ..indexstore.bm25_store import BM25Store
from ..indexstore.embedder import Embedder
from ..indexstore.vector_store import VectorStore
from ..types import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]], k: int = 60
) -> list[tuple[Chunk, float, dict[int, int]]]:
    """Fuse ranked lists. Returns (chunk, fused_score, {list_index: rank})."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[int, int]] = {}
    lookup: dict[str, Chunk] = {}

    for list_idx, ranked in enumerate(ranked_lists):
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(cid, {})[list_idx] = rank
            lookup[cid] = chunk

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(lookup[cid], score, ranks[cid]) for cid, score in ordered]


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        cfg: RetrievalConfig | None = None,
        reranker=None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.cfg = cfg or RetrievalConfig()
        self.reranker = reranker

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredChunk]:
        k = k or self.cfg.final_k

        query_vector = self.embedder.encode_queries([query])[0]
        dense_hits = self.vector_store.search(query_vector, self.cfg.dense_k)
        sparse_hits = self.bm25_store.search(query, self.cfg.sparse_k)

        fused = reciprocal_rank_fusion(
            [[c for c, _ in dense_hits], [c for c, _ in sparse_hits]],
            k=self.cfg.rrf_k,
        )

        candidates = [
            ScoredChunk(
                chunk=chunk,
                score=score,
                dense_rank=ranks.get(0),
                sparse_rank=ranks.get(1),
            )
            for chunk, score, ranks in fused
        ]

        if self.reranker is not None and self.reranker.enabled:
            head = candidates[: self.reranker.candidates]
            tail = candidates[len(head):]
            reranked = self.reranker.rerank(query, head)
            if self.cfg.min_rerank_score is not None:
                kept = [
                    sc for sc in reranked
                    if (sc.rerank_score or 0.0) >= self.cfg.min_rerank_score
                ]
                # Never return nothing: an over-strict threshold should degrade
                # to "best available", not to an empty context window.
                reranked = kept or reranked[:1]
            candidates = reranked + tail

        return candidates[:k]
