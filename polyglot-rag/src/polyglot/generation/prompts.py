"""Prompts.

Two deliberate decisions encoded here:

1. Retrieved passages stay in their ORIGINAL language inside the prompt.
   Pre-translating context adds a lossy hop before the model has even seen
   it, and it is exactly where technical terminology degrades. Modern
   instruction-tuned models read mixed-language context fine.

2. The answer language is stated explicitly and separately from the context
   language, so an English corpus can be queried in Hindi and answered in
   Hindi without the model drifting into English mid-paragraph.
"""

from __future__ import annotations

from ..types import ScoredChunk

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "ur": "Urdu", "fr": "French",
    "de": "German", "es": "Spanish", "pt": "Portuguese", "it": "Italian",
    "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian", "pl": "Polish",
    "tr": "Turkish", "ar": "Arabic", "fa": "Persian", "he": "Hebrew",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "sw": "Swahili",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


SYSTEM_PROMPT = """You are a research assistant that answers strictly from \
the sources provided to you.

Rules, in priority order:

1. GROUNDING. Every factual claim must be supported by the supplied sources. \
Never use background knowledge to fill a gap. If the sources do not contain \
the answer, say so plainly in the answer language and stop — do not guess, \
and do not pad the answer with generalities.

2. CITATIONS. Attach a marker such as [C1] (or [C1][C3] for several) to every \
claim, placed at the end of the sentence it supports. Use only markers that \
appear in the sources block.

3. ANSWER LANGUAGE. Write the entire answer in {answer_language}, including \
headings and connecting text — even when the sources are in other languages. \
Translate quoted material into {answer_language}, but keep technical terms, \
author names, dataset names and model names in their original form, adding a \
translation in parentheses on first use where it aids comprehension.

4. CONFLICTS. If sources disagree, say so and attribute each position to its \
source rather than silently picking one.

5. BREVITY. Answer the question that was asked. No preamble, no restating the \
question, no summary of what you are about to do."""


CONTEXT_BLOCK = """<sources>
{sources}
</sources>

<question language="{question_language}">
{question}
</question>

Answer in {answer_language}, citing sources with their [C#] markers."""


def format_sources(contexts: list[ScoredChunk]) -> str:
    """Render retrieved chunks with stable [C#] markers and provenance."""
    blocks = []
    for i, sc in enumerate(contexts, start=1):
        chunk = sc.chunk
        header = (
            f'[C{i}] language="{chunk.language}" '
            f'document="{chunk.title}"'
            + (f' section="{chunk.section}"' if chunk.section else "")
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(
    question: str,
    question_language: str,
    answer_language: str,
    contexts: list[ScoredChunk],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt)."""
    answer_lang_name = language_name(answer_language)
    system = SYSTEM_PROMPT.format(answer_language=answer_lang_name)
    user = CONTEXT_BLOCK.format(
        sources=format_sources(contexts) or "(no sources retrieved)",
        question=question,
        question_language=language_name(question_language),
        answer_language=answer_lang_name,
    )
    return system, user


# Used by the evaluation harness for LLM-as-judge groundedness scoring.
JUDGE_PROMPT = """You are grading whether an answer is supported by its sources.

<sources>
{sources}
</sources>

<answer>
{answer}
</answer>

Return ONLY a JSON object, no markdown fences, with keys:
  "supported_claims": integer,
  "unsupported_claims": integer,
  "groundedness": float between 0 and 1,
  "notes": short string

Count a claim as unsupported if it is not stated in or directly entailed by \
the sources, regardless of whether it is true in general."""
