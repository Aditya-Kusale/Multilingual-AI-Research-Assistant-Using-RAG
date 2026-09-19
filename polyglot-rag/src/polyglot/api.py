"""HTTP API.

    uvicorn polyglot.api:app --reload

The index is loaded once at startup and shared across requests; embedding and
reranker models are the expensive part and must not be re-instantiated per
call.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Config
from .pipeline import ResearchAssistant

log = logging.getLogger(__name__)

_state: dict[str, Any] = {"assistant": None, "config": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config.load("config.yaml")
    _state["config"] = cfg
    try:
        _state["assistant"] = ResearchAssistant.load(cfg)
        log.info("Index loaded: %d chunks", len(_state["assistant"].vector_store))
    except FileNotFoundError:
        log.warning("No index found. POST /ingest before querying.")
        _state["assistant"] = None
    yield
    _state.clear()


app = FastAPI(
    title="Polyglot — Multilingual Research Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


def _assistant() -> ResearchAssistant:
    assistant = _state.get("assistant")
    if assistant is None:
        raise HTTPException(status_code=409, detail="No index built. Call POST /ingest.")
    return assistant


# -- schemas ----------------------------------------------------------------


class IngestRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = None
    answer_language: str | None = Field(
        None, description="ISO 639-1 code. Omit to answer in the question's language."
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = 8


# -- routes -----------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    assistant = _state.get("assistant")
    return {
        "status": "ok",
        "index_loaded": assistant is not None,
        "chunks": len(assistant.vector_store) if assistant else 0,
    }


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    cfg = _state["config"]
    assistant = ResearchAssistant(cfg)
    try:
        stats = assistant.ingest(request.paths)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _state["assistant"] = assistant
    return stats


@app.post("/search")
def search(request: SearchRequest) -> dict:
    hits = _assistant().retrieve(request.query, k=request.k)
    return {
        "query": request.query,
        "results": [
            {
                "chunk_id": sc.chunk.chunk_id,
                "document": sc.chunk.title,
                "section": sc.chunk.section,
                "language": sc.chunk.language,
                "fused_score": sc.score,
                "rerank_score": sc.rerank_score,
                "text": sc.chunk.text,
            }
            for sc in hits
        ],
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    answer = _assistant().ask(
        request.question, k=request.k, answer_language=request.answer_language
    )
    return {
        "question": answer.question,
        "question_language": answer.question_language,
        "answer_language": answer.answer_language,
        "answer": answer.text,
        "unsupported": answer.unsupported,
        "citations": [c.__dict__ for c in answer.citations],
    }
