"""Dense vector store.

FAISS `IndexFlatIP` over L2-normalised vectors == exact cosine search, which
is the right default up to a few hundred thousand chunks. Past that, swap in
`IndexHNSWFlat` or a managed store; the `search()` contract is unchanged.

A pure-numpy backend is used when FAISS is not installed so nothing in the
pipeline hard-depends on it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..types import Chunk

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._faiss_index = None
        self.backend = "numpy"
        self._try_faiss()

    def _try_faiss(self) -> None:
        try:
            import faiss

            self._faiss_index = faiss.IndexFlatIP(self.dim)
            self.backend = "faiss"
        except Exception as exc:
            log.info("FAISS unavailable (%s); using numpy backend.", exc)

    # -- writing -----------------------------------------------------------

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if len(chunks) == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.chunks.extend(chunks)

        if self.backend == "faiss":
            self._faiss_index.add(vectors)
        else:
            self._matrix = (
                vectors if self._matrix is None
                else np.vstack([self._matrix, vectors])
            )

    # -- reading -----------------------------------------------------------

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")

        if self.backend == "faiss":
            scores, indices = self._faiss_index.search(q, k)
            pairs = zip(indices[0].tolist(), scores[0].tolist())
        else:
            sims = (self._matrix @ q[0]).astype("float32")
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            pairs = ((int(i), float(sims[i])) for i in top)

        return [(self.chunks[i], float(s)) for i, s in pairs if i >= 0]

    def __len__(self) -> int:
        return len(self.chunks)

    # -- persistence -------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        meta = {"dim": self.dim, "backend": self.backend, "count": len(self.chunks)}
        (directory / "vector_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        if self.backend == "faiss":
            import faiss

            faiss.write_index(self._faiss_index, str(directory / "index.faiss"))
        elif self._matrix is not None:
            np.save(directory / "vectors.npy", self._matrix)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        meta = json.loads((directory / "vector_meta.json").read_text(encoding="utf-8"))
        store = cls(dim=int(meta["dim"]))

        with open(directory / "chunks.jsonl", "r", encoding="utf-8") as fh:
            store.chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]

        faiss_path = directory / "index.faiss"
        npy_path = directory / "vectors.npy"

        if store.backend == "faiss" and faiss_path.exists():
            import faiss

            store._faiss_index = faiss.read_index(str(faiss_path))
        elif npy_path.exists():
            store.backend = "numpy"
            store._faiss_index = None
            store._matrix = np.load(npy_path)
        elif faiss_path.exists():
            # Index was built with FAISS but FAISS is missing now.
            raise RuntimeError(
                "Index was written with FAISS but faiss is not installed. "
                "pip install faiss-cpu, or re-run ingest."
            )
        return store
