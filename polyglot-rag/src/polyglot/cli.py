"""Command-line interface.

    polyglot ingest ./demo/corpus
    polyglot ask "मूल्यांकन के लिए कौन सा मापदंड प्रयोग हुआ?"
    polyglot search "retrieval augmented generation" --k 5
    polyglot eval ./demo/eval_set.jsonl --k 10
    polyglot info
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config
from .evaluation.metrics import evaluate_retrieval, format_report, load_dataset
from .pipeline import ResearchAssistant


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polyglot",
        description="Multilingual AI research assistant (hybrid RAG).",
    )
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index files or directories.")
    p_ingest.add_argument("paths", nargs="+")

    p_ask = sub.add_parser("ask", help="Ask a question and get a cited answer.")
    p_ask.add_argument("question")
    p_ask.add_argument("--k", type=int, default=None)
    p_ask.add_argument("--lang", default=None,
                       help="Force answer language (ISO 639-1). Default: match the question.")
    p_ask.add_argument("--json", action="store_true")

    p_search = sub.add_parser("search", help="Show retrieval results without generating.")
    p_search.add_argument("query")
    p_search.add_argument("--k", type=int, default=8)

    p_eval = sub.add_parser("eval", help="Run the retrieval evaluation harness.")
    p_eval.add_argument("dataset")
    p_eval.add_argument("--k", type=int, default=10)
    p_eval.add_argument("--json", action="store_true")

    sub.add_parser("info", help="Show index statistics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    cfg = Config.load(args.config)

    if args.command == "ingest":
        assistant = ResearchAssistant(cfg)
        stats = assistant.ingest(args.paths)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        if stats["embedding_backend"] == "hashing":
            print(
                "\n! Built with the offline hashing encoder. Retrieval quality "
                "will be poor.\n  pip install sentence-transformers faiss-cpu, "
                "then re-run ingest.",
                file=sys.stderr,
            )
        return 0

    if args.command == "info":
        assistant = ResearchAssistant.load(cfg)
        manifest = cfg.storage.index_dir / "manifest.json"
        print(manifest.read_text(encoding="utf-8") if manifest.exists()
              else f"{len(assistant.vector_store)} chunks indexed.")
        return 0

    assistant = ResearchAssistant.load(cfg)

    if args.command == "search":
        hits = assistant.retrieve(args.query, k=args.k)
        if not hits:
            print("No results.")
            return 0
        for i, sc in enumerate(hits, start=1):
            chunk = sc.chunk
            rr = f" rerank={sc.rerank_score:.3f}" if sc.rerank_score is not None else ""
            print(f"\n[{i}] {chunk.breadcrumb}  ({chunk.language})")
            print(f"    fused={sc.score:.4f} dense={sc.dense_rank} "
                  f"sparse={sc.sparse_rank}{rr}")
            print(f"    {' '.join(chunk.text.split())[:240]}...")
        return 0

    if args.command == "ask":
        answer = assistant.ask(args.question, k=args.k, answer_language=args.lang)
        if args.json:
            print(json.dumps(
                {
                    "question": answer.question,
                    "question_language": answer.question_language,
                    "answer_language": answer.answer_language,
                    "answer": answer.text,
                    "citations": [c.__dict__ for c in answer.citations],
                },
                indent=2, ensure_ascii=False,
            ))
            return 0

        print(answer.text)
        if answer.citations:
            print("\nSources")
            for c in answer.citations:
                where = f" — {c.section}" if c.section else ""
                print(f"  [{c.marker}] {c.title}{where}  ({c.language})")
                if c.source_path:
                    print(f"        {c.source_path}")
        return 0

    if args.command == "eval":
        examples = load_dataset(args.dataset)
        report = evaluate_retrieval(assistant, examples, k=args.k)
        print(json.dumps(report, indent=2, ensure_ascii=False)
              if args.json else format_report(report))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
