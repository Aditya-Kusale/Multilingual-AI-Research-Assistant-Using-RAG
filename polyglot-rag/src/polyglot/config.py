"""Configuration loading for Polyglot.

Config resolution order (later wins):
  1. defaults in this file
  2. config.yaml
  3. environment variables (POLYGLOT__SECTION__KEY=value)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "POLYGLOT__"


@dataclass
class EmbeddingConfig:
    # Any multilingual bi-encoder. bge-m3 covers 100+ languages in one space.
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 16
    normalize: bool = True
    # e5-family models need these; bge-m3 does not. Leave empty for bge-m3.
    query_prefix: str = ""
    passage_prefix: str = ""


@dataclass
class RerankerConfig:
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    batch_size: int = 16
    # How many fused candidates get sent to the cross-encoder.
    candidates: int = 50


@dataclass
class ChunkingConfig:
    target_tokens: int = 350
    overlap_sentences: int = 1
    min_tokens: int = 40
    # Prepend "Title > Section" to every chunk so isolated chunks stay legible.
    prepend_breadcrumb: bool = True
    # Attach a trimmed abstract to every chunk of a paper.
    attach_abstract: bool = True
    abstract_chars: int = 400


@dataclass
class RetrievalConfig:
    dense_k: int = 50
    sparse_k: int = 50
    rrf_k: int = 60
    final_k: int = 8
    # Drop reranked hits below this score. None disables filtering.
    min_rerank_score: float | None = None


@dataclass
class GenerationConfig:
    provider: str = "anthropic"          # anthropic | openai | echo
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1200
    temperature: float = 0.0
    # "auto" -> answer in the detected language of the question.
    answer_language: str = "auto"


@dataclass
class StorageConfig:
    index_dir: Path = Path("./storage")

    def __post_init__(self) -> None:
        self.index_dir = Path(self.index_dir)


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def load(cls, path: str | Path | None = "config.yaml") -> "Config":
        data: dict[str, Any] = {}
        if path is not None and Path(path).exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        cfg = _build(cls, data)
        _apply_env(cfg)
        return cfg


def _build(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _build(f.type, value)
        elif isinstance(value, dict) and f.name in _NESTED:
            kwargs[f.name] = _build(_NESTED[f.name], value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


_NESTED = {
    "embedding": EmbeddingConfig,
    "reranker": RerankerConfig,
    "chunking": ChunkingConfig,
    "retrieval": RetrievalConfig,
    "generation": GenerationConfig,
    "storage": StorageConfig,
}


def _coerce(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, Path):
        return Path(raw)
    return raw


def _apply_env(cfg: Config) -> None:
    """POLYGLOT__RETRIEVAL__FINAL_K=12 overrides cfg.retrieval.final_k."""
    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = key[len(ENV_PREFIX):].lower().split("__")
        if len(parts) != 2:
            continue
        section, attr = parts
        target = getattr(cfg, section, None)
        if target is None or not hasattr(target, attr):
            continue
        setattr(target, attr, _coerce(getattr(target, attr), raw))
