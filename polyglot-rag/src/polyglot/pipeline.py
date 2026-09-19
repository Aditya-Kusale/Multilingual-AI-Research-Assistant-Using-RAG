"""The assembled system.

    ingest ──> chunk ──> embed ──┬──> FAISS (dense)
                                 └──> BM25  (sparse)

    query  ──> embed ──> dense k=50 ┐
                                    ├──> RRF ──> cross-encoder ──> top-k
               BM25  ──> sparse k=50┘

           ──> prompt (context kept in source language)
           ──> answer in the query's language, with [C#] citations
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Config
from .generation.answerer import Answerer
from .indexstore.bm25_store import BM25Store
from .indexstore.embedder import Embedder
from .indexstore.vector_store import VectorStore
from .ingest.chunker import chunk_documents
from .ingest.loaders import load_path
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import CrossEncoderReranker
from .types import Answer, Chunk, Document, ScoredChunk

log = logging.getLogger(__name__)


class ResearchAssistant:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.embedder = Embedder(self.cfg.embedding)
        self.vector_store: VectorStore | None = None
        self.bm25_store: BM25Store | None = None
        self.reranker = CrossEncoderReranker(self.cfg.reranker)
        self._retriever: HybridRetriever | None = None
        self._answerer: Answerer | None = None

    # -- build -------------------------------------------------------------

    def ingest(self, paths: list[str | Path]) -> dict:
        documents: list[Document] = []
        for path in paths:
            documents.extend(load_path(path))

        if not documents:
            raise ValueError(f"No supported documents found in: {paths}")

        chunks = chunk_documents(documents, self.cfg.chunking)
        log.info("Loaded %d documents -> %d chunks", len(documents), len(chunks))

        self._build_indexes(chunks)
        self.save()

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "languages": _language_histogram(chunks),
            "embedding_backend": self.embedder.backend,
            "vector_backend": self.vector_store.backend if self.vector_store else None,
            "index_dir": str(self.cfg.storage.index_dir),
        }

    def _build_indexes(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.encode_passages([c.text for c in chunks])

        self.vector_store = VectorStore(dim=self.embedder.dim)
        self.vector_store.add(chunks, vectors)

        self.bm25_store = BM25Store()
        self.bm25_store.add(chunks)

        self._retriever = None
        self._answerer = None

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        directory = self.cfg.storage.index_dir
        directory.mkdir(parents=True, exist_ok=True)
        if self.vector_store is None or self.bm25_store is None:
            raise RuntimeError("Nothing to save; run ingest() first.")

        self.vector_store.save(directory)
        self.bm25_store.save(directory)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "embedding_model": self.cfg.embedding.model_name,
                    "embedding_backend": self.embedder.backend,
                    "dim": self.embedder.dim,
                    "chunks": len(self.vector_store),
                    "languages": _language_histogram(self.vector_store.chunks),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, cfg: Config | None = None) -> "ResearchAssistant":
        assistant = cls(cfg)
        directory = assistant.cfg.storage.index_dir
        if not (directory / "chunks.jsonl").exists():
            raise FileNotFoundError(
                f"No index at {directory}. Run: polyglot ingest <path>"
            )

        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            indexed_model = manifest.get("embedding_model")
            if indexed_model and indexed_model != assistant.cfg.embedding.model_name:
                # Querying an index with a different encoder silently returns
                # nonsense, so fail loudly instead.
                raise RuntimeError(
                    f"Index was built with '{indexed_model}' but config says "
                    f"'{assistant.cfg.embedding.model_name}'. Re-ingest or fix config."
                )

        assistant.vector_store = VectorStore.load(directory)
        assistant.bm25_store = BM25Store.load(directory, assistant.vector_store.chunks)
        return assistant

    # -- query -------------------------------------------------------------

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            if self.vector_store is None or self.bm25_store is None:
                raise RuntimeError("Index not built. Run ingest() or load().")
            self._retriever = HybridRetriever(
                embedder=self.embedder,
                vector_store=self.vector_store,
                bm25_store=self.bm25_store,
                cfg=self.cfg.retrieval,
                reranker=self.reranker,
            )
        return self._retriever

    @property
    def answerer(self) -> Answerer:
        if self._answerer is None:
            self._answerer = Answerer(self.cfg.generation)
        return self._answerer

    def retrieve(self, question: str, k: int | None = None) -> list[ScoredChunk]:
        return self.retriever.retrieve(question, k)

    def ask(
        self,
        question: str,
        k: int | None = None,
        answer_language: str | None = None,
    ) -> Answer:
        contexts = self.retrieve(question, k)
        return self.answerer.answer(question, contexts, answer_language)


def _language_histogram(chunks: list[Chunk]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for chunk in chunks:
        histogram[chunk.language] = histogram.get(chunk.language, 0) + 1
    return dict(sorted(histogram.items(), key=lambda kv: -kv[1]))
