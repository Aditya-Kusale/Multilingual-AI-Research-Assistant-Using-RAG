# Polyglot — Multilingual AI Research Assistant

Ask a research question in Hindi and get a cited answer grounded in English,
French and Chinese papers — written back to you in Hindi.

A complete hybrid-RAG system: structure-aware ingestion, dense + sparse
retrieval fused with RRF, cross-encoder reranking, citation-grounded
generation, and an evaluation harness that reports **per-language** metrics
rather than a single average that hides which languages are broken.

---

## Architecture

```
INGEST
  PDF / MD / TXT / HTML
        │
        ├─ column-aware text extraction, de-hyphenation
        ├─ structure-aware chunking  (sections → sentence packing → overlap)
        ├─ breadcrumb + abstract prepended to every chunk
        │
        ├──────────────► multilingual embeddings ──► FAISS  (dense)
        └──────────────► script-aware tokeniser  ──► BM25   (sparse)

QUERY
  question (any language)
        │
        ├─ dense search    top-50 ┐
        ├─ BM25 search     top-50 ┼─► Reciprocal Rank Fusion
        │                         ┘         │
        │                                   ▼
        │                         cross-encoder rerank (top-50 → top-8)
        ▼
  prompt: passages stay in their ORIGINAL language
          answer language stated separately
        │
        ▼
  answer in the question's language, every claim carrying a [C#] citation
```

---

## Quickstart

```bash
git clone <your-repo> polyglot-rag && cd polyglot-rag
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

cp .env.example .env          # add ANTHROPIC_API_KEY (or set provider: echo)

polyglot ingest ./demo/corpus
polyglot ask "मूल्यांकन के लिए कौन से मापदंड प्रयोग किए गए?"
```

The demo corpus is four short papers — English, Hindi, French and Chinese —
that cite each other's findings, so cross-lingual retrieval is visible
immediately.

Other entry points:

```bash
polyglot search "bge-reranker-v2-m3" --k 5    # retrieval only, with diagnostics
polyglot eval ./demo/eval_set.jsonl --k 1     # per-language metrics
polyglot info                                 # index manifest

uvicorn polyglot.api:app --reload             # HTTP API at :8000/docs
streamlit run app/streamlit_app.py            # web UI
pytest                                        # 20 tests, no downloads needed
```

### Running with nothing installed

`pip install -r requirements-min.txt` (numpy + PyYAML) is enough for the whole
pipeline to run end to end. The embedder falls back to a deterministic hashing
encoder, the reranker becomes a no-op, and generation falls back to extractive
summarisation. Retrieval quality is poor by design — this path exists for CI,
air-gapped demos and tests, and every entry point warns when it is active.

---

## Design decisions, and why

**One shared multilingual vector space (`BAAI/bge-m3`).** A Hindi query has to
retrieve an English passage directly. The alternative — translate everything to
English at ingest — is easier to reason about but loses technical terminology
and leaves citations pointing at text the user cannot verify against the
original.

**Hybrid retrieval, always.** Dense vectors reliably miss exact surface forms:
author names, model names, dataset identifiers. Those make up a large share of
real research questions. `test_bm25_matches_exact_surface_forms` is the
regression test for this — a query for `bge-reranker-v2-m3` is answered by
BM25, not by the encoder.

**RRF rather than a weighted score blend.** Cosine scores and BM25 scores live
on incomparable scales *that also shift per language*. Any alpha tuned on
English will be wrong on Hindi. RRF uses only ranks, so it needs no
per-language calibration.

**Script-aware tokenisation.** BM25 over whitespace tokens returns essentially
nothing for Chinese, Japanese and Thai, because those scripts do not delimit
words with spaces — a whole paragraph becomes one token. Polyglot emits
character unigrams + bigrams for unsegmented scripts instead. This is the
single most common silent failure in multilingual RAG, and it is easy to ship
without noticing because the English half of your eval set still looks fine.

**Token counting is script-aware too.** An English-calibrated `len(text)/4`
under-counts CJK by ~4x, producing chunks several times larger than intended.

**Reranking is where multilingual RAG actually gets fixed.** A bi-encoder
compresses query and passage independently; a cross-encoder reads both together
and can tell that a Hindi question about मूल्यांकन is answered by an English
passage on *evaluation protocol*. Budget for it before tuning chunk size.

**Structure-aware chunking with breadcrumbs.** Every chunk carries
`[Paper Title > Section]` and a trimmed abstract. Without that, a retrieved
chunk reading "we observe a 12% improvement" is retrievable but useless as
evidence.

**Context stays in its source language.** Pre-translating retrieved passages
adds a lossy hop before the model has even seen them. The prompt states the
answer language separately, so the model reads mixed-language context and
writes monolingual output.

**Citations are validated, not trusted.** Markers pointing past the end of the
context list are dropped rather than rendered — see
`test_hallucinated_citation_markers_are_dropped`.

---

## Evaluation: read the per-language table, not the average

Here is the demo eval set run against the deliberately-degraded offline
encoder:

```
Retrieval evaluation @k=1

language        n    recall       mrr      ndcg
-----------------------------------------------
en              7     0.714     0.714     0.714
fr              2     1.000     1.000     1.000
hi              5     0.600     0.600     0.600
zh              2     1.000     1.000     1.000
-----------------------------------------------
OVERALL        16     0.750     0.750     0.750

same-language  n=10   recall=1.000
cross-lingual  n=6    recall=0.333

  ! Cross-lingual recall lags same-language recall by >15 points.
```

An overall 0.750 looks acceptable. It is hiding a system that works perfectly
within a language and fails two times in three across languages — precisely the
failure a single aggregate number conceals. Install
`sentence-transformers` and the cross-lingual row recovers.

Eval sets are JSONL:

```json
{"query_id": "q06", "query": "मूल्यांकन के लिए कौन से मापदंड प्रयोग किए गए?",
 "language": "hi", "gold_doc_ids": ["en_evaluation_protocol-52025e14"],
 "gold_language": "en", "notes": "cross-lingual"}
```

`doc_id`s are derived from the file *name*, not its absolute path, so a
checked-in eval set is portable across machines.

Build 50–100 annotated queries per target language before tuning anything
else. Teams that tune chunk size before measuring per-language recall routinely
optimise the wrong component.

---

## Configuration

Everything lives in `config.yaml`, overridable by environment variable
(`POLYGLOT__RETRIEVAL__FINAL_K=12`).

| Setting | Default | Notes |
|---|---|---|
| `embedding.model_name` | `BAAI/bge-m3` | swap for `intfloat/multilingual-e5-large` — then set `query_prefix: "query: "` and `passage_prefix: "passage: "` |
| `embedding.device` | `cpu` | `cuda` if available; this is the main throughput lever |
| `reranker.enabled` | `true` | the highest-leverage component; disable only to measure its contribution |
| `reranker.candidates` | `50` | one forward pass each — this is your latency budget |
| `chunking.target_tokens` | `350` | tune *after* reranking is working |
| `retrieval.dense_k` / `sparse_k` | `50` | candidates per retriever before fusion |
| `retrieval.final_k` | `8` | passages placed in the prompt |
| `generation.provider` | `anthropic` | `openai` works with any compatible endpoint incl. local vLLM/Ollama |
| `generation.answer_language` | `auto` | `auto` mirrors the question's language |

The index manifest records which encoder built it, and `load()` refuses to open
an index built with a different model — querying a FAISS index with the wrong
encoder returns confident nonsense rather than an error.

---

## Project layout

```
src/polyglot/
  config.py              YAML + env configuration
  types.py               Document, Chunk, ScoredChunk, Citation, Answer
  language.py            script detection, language ID, script-aware tokenising
  ingest/loaders.py      PDF (two-column aware) / MD / TXT / HTML
  ingest/chunker.py      multilingual heading detection, sentence packing
  indexstore/embedder.py multilingual embeddings + offline fallback
  indexstore/vector_store.py  FAISS (numpy fallback) + persistence
  indexstore/bm25_store.py    BM25 with script-aware tokenisation
  retrieval/hybrid.py    reciprocal rank fusion
  retrieval/reranker.py  cross-encoder reranking
  generation/prompts.py  multilingual, citation-enforcing prompts
  generation/llm.py      Anthropic / OpenAI-compatible / offline
  generation/answerer.py answer assembly + citation validation
  evaluation/metrics.py  recall@k, MRR, nDCG, per-language + cross-lingual
  pipeline.py            the assembled system
  cli.py  api.py         CLI and FastAPI
app/streamlit_app.py     web UI
demo/                    4-language corpus + 16-query eval set
tests/                   20 tests, runnable with no model downloads
```

---

## Taking this to production

Roughly in order of payoff:

1. **Replace PDF parsing.** The built-in extractor handles one- and two-column
   layouts, but real papers have tables, figures, footnotes and reference
   lists. Use GROBID or Docling; the `Document` contract does not change.
2. **Move the index off local disk.** `IndexFlatIP` is exact and fine to a few
   hundred thousand chunks. Past that use `IndexHNSWFlat`, Qdrant or pgvector.
   Only `VectorStore.search()` needs to change.
3. **Put the encoder on a GPU and batch it.** Ingest is embarrassingly
   parallel; reranking is the per-query cost.
4. **Add a real segmenter** for Chinese/Japanese/Thai (jieba, fugashi,
   pythainlp) if those are primary languages. Character bigrams recover most
   but not all of the recall.
5. **Expand the eval set continuously.** Add every query the system gets wrong.
   It is the only thing that tells you whether a change helped.
6. **Add groundedness scoring.** `prompts.JUDGE_PROMPT` is ready for an
   LLM-as-judge pass over answers; wire it into CI against a fixed corpus.
7. **Incremental indexing.** `ingest` currently rebuilds. Adding per-document
   upsert means tracking deletions in both stores.

## Known limits

- Without `lingua` installed, Latin-script language ID falls back to a stopword
  profile covering ten languages; anything else is labelled `en`.
- Ingest is full-rebuild only.
- The offline fallback encoder has no semantics — it exists so the system runs
  without downloads, not to retrieve well.
## Key Topics
`multilingual` · `RAG` · `information retrieval` · `cross-lingual search` · `NLP` · `BM25` · `FAISS` · `citation-grounded generation`
## License
Distributed under the [MIT License](LICENSE).
