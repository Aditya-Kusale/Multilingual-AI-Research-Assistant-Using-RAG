"""Sparse lexical retrieval (BM25 Okapi).

Dense retrieval alone reliably misses exact surface forms: author names,
model names, acronyms, dataset identifiers, equation labels. Those are a
large share of real research questions, so BM25 runs alongside the dense
index and the two are fused.

The implementation is self-contained (no rank_bm25 dependency) because the
important part is the tokeniser, not the scorer — see `language.lexical_tokens`,
which emits character bigrams for Chinese/Japanese/Thai instead of returning
one useless token per paragraph.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from pathlib import Path

from ..language import lexical_tokens
from ..types import Chunk


class BM25Store:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.avg_len: float = 0.0

    # -- writing -----------------------------------------------------------

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = lexical_tokens(chunk.text)
            freqs = Counter(tokens)
            self.chunks.append(chunk)
            self.doc_freqs.append(freqs)
            self.doc_lens.append(len(tokens))
            self.df.update(freqs.keys())
        self._recompute()

    def _recompute(self) -> None:
        n = len(self.chunks)
        self.avg_len = (sum(self.doc_lens) / n) if n else 0.0
        # Robertson/Sparck-Jones idf with the standard +1 smoothing so that
        # terms appearing in most documents never go negative.
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }

    # -- reading -----------------------------------------------------------

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_tokens = lexical_tokens(query)
        if not query_tokens:
            return []

        scores: list[float] = [0.0] * len(self.chunks)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (self.doc_lens[i] / (self.avg_len or 1.0))
                )
                scores[i] += idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda pair: -pair[1],
        )[:k]
        return [(self.chunks[i], float(s)) for i, s in ranked]

    def __len__(self) -> int:
        return len(self.chunks)

    # -- persistence -------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "doc_freqs": [dict(c) for c in self.doc_freqs],
            "doc_lens": self.doc_lens,
            "df": dict(self.df),
        }
        with open(directory / "bm25.pkl", "wb") as fh:
            pickle.dump(payload, fh)
        # chunks.jsonl is written by VectorStore; keep a pointer for clarity.
        (directory / "bm25_meta.json").write_text(
            json.dumps({"count": len(self.chunks)}, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path, chunks: list[Chunk]) -> "BM25Store":
        directory = Path(directory)
        with open(directory / "bm25.pkl", "rb") as fh:
            payload = pickle.load(fh)

        store = cls(k1=payload["k1"], b=payload["b"])
        store.chunks = chunks
        store.doc_freqs = [Counter(d) for d in payload["doc_freqs"]]
        store.doc_lens = payload["doc_lens"]
        store.df = Counter(payload["df"])
        store._recompute()
        return store
