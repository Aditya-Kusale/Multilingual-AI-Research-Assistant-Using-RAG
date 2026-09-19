#!/usr/bin/env bash
# End-to-end smoke test of the whole system.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

echo "== ingest =="
python -m polyglot.cli ingest ./demo/corpus

echo; echo "== retrieval (Chinese query, Chinese source) =="
python -m polyglot.cli search "分词" --k 3

echo; echo "== retrieval (English query, Hindi source) =="
python -m polyglot.cli search "reciprocal rank fusion" --k 3

echo; echo "== answer (Hindi question) =="
python -m polyglot.cli ask "पुनर्क्रमण से कितना सुधार हुआ?" --k 3

echo; echo "== evaluation =="
python -m polyglot.cli eval ./demo/eval_set.jsonl --k 1
