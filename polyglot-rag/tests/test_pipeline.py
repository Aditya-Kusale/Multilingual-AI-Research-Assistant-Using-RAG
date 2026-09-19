"""Tests.

These run with only numpy + PyYAML installed: the embedder and reranker both
degrade to offline fallbacks, so CI needs no model downloads and no API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polyglot.config import ChunkingConfig, Config, GenerationConfig
from polyglot.generation.answerer import Answerer
from polyglot.generation.llm import EchoClient
from polyglot.indexstore.bm25_store import BM25Store
from polyglot.ingest.chunker import chunk_document, split_sections
from polyglot.ingest.loaders import load_path, make_doc_id
from polyglot.language import (
    approx_tokens,
    detect_language,
    dominant_script,
    lexical_tokens,
    split_sentences,
)
from polyglot.pipeline import ResearchAssistant
from polyglot.retrieval.hybrid import reciprocal_rank_fusion
from polyglot.types import Chunk, Document, ScoredChunk

CORPUS = Path(__file__).resolve().parents[1] / "demo" / "corpus"


# -- language utilities -----------------------------------------------------


def test_dominant_script():
    assert dominant_script("hello world") == "Latin"
    assert dominant_script("मूल्यांकन") == "Devanagari"
    assert dominant_script("分词问题") == "Han"
    assert dominant_script("مرحبا") == "Arabic"


def test_cjk_tokens_are_not_one_giant_token():
    """The bug this whole module exists to prevent."""
    tokens = lexical_tokens("中文分词问题")
    assert len(tokens) > 1
    assert "中文" in tokens          # bigram
    assert "中" in tokens            # unigram


def test_latin_tokens_stay_words():
    tokens = lexical_tokens("Reciprocal Rank Fusion, applied.")
    assert tokens == ["reciprocal", "rank", "fusion", "applied"]


def test_cjk_token_estimate_is_not_english_calibrated():
    cjk = "分词问题需要解决" * 10
    latin = "tokenisation matters here " * 10
    # 80 CJK chars must not be counted as ~20 tokens.
    assert approx_tokens(cjk) >= 70
    assert approx_tokens(latin) < len(latin)


def test_sentence_splitting_across_scripts():
    text = "First sentence. दूसरा वाक्य। 第三句。Fourth?"
    parts = split_sentences(text)
    assert len(parts) == 4


def test_language_detection_by_script_fallback():
    assert detect_language("मूल्यांकन के मापदंड") == "hi"
    assert detect_language("分词问题非常重要") == "zh"


# -- chunking ---------------------------------------------------------------


def test_split_sections_finds_headings():
    text = "# Title\n\nIntro text.\n\n## Method\n\nWe did things.\n\n## Results\n\nIt worked."
    sections = split_sections(text)
    titles = [t for t, _ in sections]
    assert "Method" in titles and "Results" in titles


def test_chunks_carry_breadcrumbs():
    doc = Document(
        doc_id="d1",
        title="A Paper",
        text="## Method\n\n" + ("We measured recall carefully. " * 60),
        abstract="An abstract about recall.",
    )
    chunks = chunk_document(doc, ChunkingConfig(target_tokens=80))
    assert len(chunks) > 1
    assert all("[A Paper > Method]" in c.text for c in chunks)
    assert all("[Abstract]" in c.text for c in chunks)


def test_chunk_ids_are_unique_and_ordered():
    doc = Document(doc_id="d1", title="T", text="Sentence one. " * 200)
    chunks = chunk_document(doc, ChunkingConfig(target_tokens=60))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.position for c in chunks] == list(range(len(chunks)))


# -- fusion -----------------------------------------------------------------


def _chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", title="t", section="",
                 text=cid, language="en", position=0)


def test_rrf_rewards_agreement_between_retrievers():
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    sparse = [_chunk("c"), _chunk("b"), _chunk("z")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    order = [c.chunk_id for c, _, _ in fused]
    # "b" is 2nd in both lists and should beat "a" (1st in only one).
    assert order.index("b") < order.index("a")
    assert order[-1] == "z"


def test_rrf_is_scale_free():
    """Score magnitudes must not influence the outcome — only ranks."""
    dense = [_chunk("a"), _chunk("b")]
    sparse = [_chunk("b"), _chunk("a")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    scores = [s for _, s, _ in fused]
    assert pytest.approx(scores[0]) == scores[1]


# -- BM25 -------------------------------------------------------------------


def test_bm25_retrieves_chinese():
    chunks = [
        Chunk("c1", "d1", "zh doc", "", "中文分词是检索的基础问题", "zh", 0),
        Chunk("c2", "d2", "en doc", "", "completely unrelated english text", "en", 1),
    ]
    store = BM25Store()
    store.add(chunks)
    hits = store.search("分词", k=5)
    assert hits and hits[0][0].chunk_id == "c1"


def test_bm25_matches_exact_surface_forms():
    chunks = [
        Chunk("c1", "d1", "a", "", "We evaluate with bge-reranker-v2-m3 here.", "en", 0),
        Chunk("c2", "d2", "b", "", "General discussion of neural retrieval.", "en", 1),
    ]
    store = BM25Store()
    store.add(chunks)
    hits = store.search("bge-reranker-v2-m3", k=5)
    assert hits[0][0].chunk_id == "c1"


# -- loaders ----------------------------------------------------------------


def test_doc_ids_are_path_independent():
    assert make_doc_id(Path("/a/b/paper.md")) == make_doc_id(Path("/x/y/paper.md"))


def test_corpus_loads_with_language_labels():
    docs = load_path(CORPUS)
    assert len(docs) == 4
    languages = {d.language for d in docs}
    assert {"en", "hi", "fr", "zh"} <= languages


# -- citations --------------------------------------------------------------


def test_hallucinated_citation_markers_are_dropped():
    contexts = [ScoredChunk(chunk=_chunk("a"), score=1.0)]
    citations = Answerer._parse_citations("Claim one [C1]. Claim two [C7].", contexts)
    assert [c.marker for c in citations] == ["C1"]


def test_answer_language_follows_question_by_default():
    answerer = Answerer(GenerationConfig(provider="echo"),
                        client=EchoClient(GenerationConfig()))
    contexts = [ScoredChunk(chunk=_chunk("a"), score=1.0)]
    answer = answerer.answer("यह प्रश्न हिंदी में है", contexts)
    assert answer.question_language == "hi"
    assert answer.answer_language == "hi"


def test_empty_context_returns_refusal_not_invention():
    answerer = Answerer(GenerationConfig(provider="echo"),
                        client=EchoClient(GenerationConfig()))
    answer = answerer.answer("anything at all", [])
    assert answer.unsupported is True
    assert answer.citations == []


# -- end to end -------------------------------------------------------------


def test_ingest_search_ask_roundtrip(tmp_path):
    cfg = Config()
    cfg.storage.index_dir = tmp_path / "storage"
    cfg.generation.provider = "echo"
    cfg.reranker.enabled = False          # no model download in CI

    assistant = ResearchAssistant(cfg)
    stats = assistant.ingest([CORPUS])
    assert stats["documents"] == 4
    assert stats["chunks"] > 4

    reloaded = ResearchAssistant.load(cfg)
    assert len(reloaded.vector_store) == stats["chunks"]

    hits = reloaded.retrieve("分词", k=5)
    assert hits, "sparse retrieval should find the Chinese document"
    assert any(h.chunk.language == "zh" for h in hits)

    answer = reloaded.ask("分词问题是什么？", k=3)
    assert answer.text
    assert answer.answer_language == "zh"


def test_index_refuses_mismatched_embedding_model(tmp_path):
    cfg = Config()
    cfg.storage.index_dir = tmp_path / "storage"
    cfg.generation.provider = "echo"
    cfg.reranker.enabled = False

    ResearchAssistant(cfg).ingest([CORPUS])

    cfg.embedding.model_name = "some/other-model"
    with pytest.raises(RuntimeError, match="built with"):
        ResearchAssistant.load(cfg)
