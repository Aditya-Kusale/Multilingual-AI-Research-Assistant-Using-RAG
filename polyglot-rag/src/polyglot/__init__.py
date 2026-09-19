"""Polyglot — a multilingual AI research assistant built on hybrid RAG."""

from .config import Config
from .pipeline import ResearchAssistant
from .types import Answer, Chunk, Citation, Document, ScoredChunk

__version__ = "1.0.0"
__all__ = [
    "Config",
    "ResearchAssistant",
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "ScoredChunk",
]
