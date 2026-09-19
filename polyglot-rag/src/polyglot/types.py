"""Shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Document:
    """A source document before chunking."""

    doc_id: str
    title: str
    text: str
    language: str = "unknown"
    source_path: str = ""
    abstract: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit. `text` is always in the ORIGINAL language."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    language: str
    position: int
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)

    @property
    def breadcrumb(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


@dataclass
class Citation:
    marker: str          # "C1"
    chunk_id: str
    doc_id: str
    title: str
    section: str
    language: str
    snippet: str
    source_path: str = ""


@dataclass
class Answer:
    question: str
    question_language: str
    answer_language: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    contexts: list[ScoredChunk] = field(default_factory=list)
    unsupported: bool = False
