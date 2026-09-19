"""Turn retrieved context into a cited, language-correct answer."""

from __future__ import annotations

import re

from ..config import GenerationConfig
from ..language import detect_language
from ..types import Answer, Citation, ScoredChunk
from .llm import LLMClient, build_client
from .prompts import build_messages

_CITE_RE = re.compile(r"\[C(\d+)\]")

_NO_ANSWER_MARKERS = (
    "do not contain", "does not contain", "no information", "not covered",
    "कोई जानकारी", "नहीं मिली", "没有", "未提及", "aucune information",
    "no se encuentra", "لا تحتوي",
)


class Answerer:
    def __init__(self, cfg: GenerationConfig | None = None, client: LLMClient | None = None):
        self.cfg = cfg or GenerationConfig()
        self.client = client or build_client(self.cfg)

    def answer(
        self,
        question: str,
        contexts: list[ScoredChunk],
        answer_language: str | None = None,
    ) -> Answer:
        question_language = detect_language(question, default="en")

        target = answer_language or self.cfg.answer_language
        if target in (None, "", "auto"):
            target = question_language

        if not contexts:
            return Answer(
                question=question,
                question_language=question_language,
                answer_language=target,
                text=_empty_response(target),
                unsupported=True,
            )

        system, user = build_messages(question, question_language, target, contexts)
        text = self.client.complete(system, user).strip()

        citations = self._parse_citations(text, contexts)
        return Answer(
            question=question,
            question_language=question_language,
            answer_language=target,
            text=text,
            citations=citations,
            contexts=contexts,
            unsupported=self._looks_unsupported(text, citations),
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _parse_citations(text: str, contexts: list[ScoredChunk]) -> list[Citation]:
        """Map [C#] markers back to chunks, dropping hallucinated markers."""
        seen: set[int] = set()
        citations: list[Citation] = []

        for match in _CITE_RE.finditer(text):
            idx = int(match.group(1))
            if idx in seen or not (1 <= idx <= len(contexts)):
                continue
            seen.add(idx)
            chunk = contexts[idx - 1].chunk
            citations.append(
                Citation(
                    marker=f"C{idx}",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    section=chunk.section,
                    language=chunk.language,
                    snippet=" ".join(chunk.text.split())[:300],
                    source_path=chunk.source_path,
                )
            )
        return citations

    @staticmethod
    def _looks_unsupported(text: str, citations: list[Citation]) -> bool:
        if not citations:
            return True
        lowered = text.lower()
        return any(marker.lower() in lowered for marker in _NO_ANSWER_MARKERS)


def strip_invalid_citations(text: str, n_contexts: int) -> str:
    """Remove [C#] markers pointing past the end of the context list."""
    return _CITE_RE.sub(
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_contexts else "",
        text,
    )


_EMPTY_BY_LANG = {
    "en": "I could not find anything in the indexed sources that answers this question.",
    "hi": "अनुक्रमित स्रोतों में इस प्रश्न का उत्तर देने वाली कोई जानकारी नहीं मिली।",
    "fr": "Je n'ai trouvé aucune information dans les sources indexées pour répondre à cette question.",
    "es": "No encontré información en las fuentes indexadas que responda a esta pregunta.",
    "zh": "在已索引的资料中没有找到可以回答该问题的信息。",
    "ar": "لم أجد في المصادر المفهرسة ما يجيب عن هذا السؤال.",
}


def _empty_response(language: str) -> str:
    return _EMPTY_BY_LANG.get(language, _EMPTY_BY_LANG["en"])
