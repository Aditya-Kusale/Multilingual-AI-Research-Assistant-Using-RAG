"""Evaluation.

The per-language breakdown is the point. An aggregate recall@10 of 0.82 can
hide a language sitting at 0.31 — and that language will look like "the model
is bad at Tamil" when it is actually a tokenisation bug. Always read the
per-language table, not the average.

Metrics
-------
recall@k   share of gold documents retrieved in the top k
mrr@k      1 / rank of the first gold document
ndcg@k     rank-discounted gain, binary relevance
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class EvalExample:
    query_id: str
    query: str
    language: str
    gold_doc_ids: list[str]
    # Optional: for cross-lingual evaluation, the language of the gold docs.
    gold_language: str = ""
    notes: str = ""


@dataclass
class RetrievalResult:
    query_id: str
    language: str
    gold_language: str
    retrieved_doc_ids: list[str]
    gold_doc_ids: list[str]

    @property
    def cross_lingual(self) -> bool:
        return bool(self.gold_language) and self.gold_language != self.language


@dataclass
class MetricSummary:
    n: int = 0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0


def load_dataset(path: str | Path) -> list[EvalExample]:
    """Read a JSONL eval set."""
    examples: list[EvalExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            examples.append(EvalExample(**json.loads(line)))
    return examples


# -- single-query metrics ---------------------------------------------------


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(gold)) / len(set(gold))


def mrr_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in gold_set
    )
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


# -- aggregation ------------------------------------------------------------


def summarise(results: list[RetrievalResult], k: int) -> MetricSummary:
    if not results:
        return MetricSummary()
    n = len(results)
    return MetricSummary(
        n=n,
        recall=sum(recall_at_k(r.retrieved_doc_ids, r.gold_doc_ids, k) for r in results) / n,
        mrr=sum(mrr_at_k(r.retrieved_doc_ids, r.gold_doc_ids, k) for r in results) / n,
        ndcg=sum(ndcg_at_k(r.retrieved_doc_ids, r.gold_doc_ids, k) for r in results) / n,
    )


def evaluate_retrieval(
    assistant, examples: list[EvalExample], k: int = 10
) -> dict[str, Any]:
    """Run every eval query through the retriever and aggregate."""
    results: list[RetrievalResult] = []

    for example in examples:
        hits = assistant.retrieve(example.query, k=k)
        # De-duplicate to document level, preserving rank order.
        seen: list[str] = []
        for sc in hits:
            if sc.chunk.doc_id not in seen:
                seen.append(sc.chunk.doc_id)
        results.append(
            RetrievalResult(
                query_id=example.query_id,
                language=example.language,
                gold_language=example.gold_language,
                retrieved_doc_ids=seen,
                gold_doc_ids=example.gold_doc_ids,
            )
        )

    by_language: dict[str, MetricSummary] = {}
    for language in sorted({r.language for r in results}):
        subset = [r for r in results if r.language == language]
        by_language[language] = summarise(subset, k)

    cross = [r for r in results if r.cross_lingual]
    same = [r for r in results if not r.cross_lingual]

    return {
        "k": k,
        "overall": asdict(summarise(results, k)),
        "by_language": {lang: asdict(m) for lang, m in by_language.items()},
        "cross_lingual": asdict(summarise(cross, k)),
        "same_language": asdict(summarise(same, k)),
        "failures": [
            {"query_id": r.query_id, "language": r.language, "gold": r.gold_doc_ids}
            for r in results
            if recall_at_k(r.retrieved_doc_ids, r.gold_doc_ids, k) == 0.0
        ],
    }


def format_report(report: dict[str, Any]) -> str:
    """Render the eval dict as a readable table."""
    k = report["k"]
    lines = [
        f"Retrieval evaluation @k={k}",
        "",
        f"{'language':<12}{'n':>5}{'recall':>10}{'mrr':>10}{'ndcg':>10}",
        "-" * 47,
    ]
    for language, m in report["by_language"].items():
        lines.append(
            f"{language:<12}{m['n']:>5}{m['recall']:>10.3f}{m['mrr']:>10.3f}{m['ndcg']:>10.3f}"
        )
    overall = report["overall"]
    lines += [
        "-" * 47,
        f"{'OVERALL':<12}{overall['n']:>5}{overall['recall']:>10.3f}"
        f"{overall['mrr']:>10.3f}{overall['ndcg']:>10.3f}",
        "",
    ]
    cross, same = report["cross_lingual"], report["same_language"]
    lines.append(f"same-language  n={same['n']:<4} recall={same['recall']:.3f}")
    lines.append(f"cross-lingual  n={cross['n']:<4} recall={cross['recall']:.3f}")
    if cross["n"] and same["n"] and cross["recall"] < same["recall"] - 0.15:
        lines.append(
            "\n  ! Cross-lingual recall lags same-language recall by >15 points."
            "\n    Usual causes: reranker disabled, or an embedding model with"
            "\n    weak alignment for these language pairs."
        )
    failures = report.get("failures", [])
    if failures:
        lines.append(f"\n{len(failures)} total miss(es): " +
                     ", ".join(f["query_id"] for f in failures[:10]))
    return "\n".join(lines)
