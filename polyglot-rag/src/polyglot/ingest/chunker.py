"""Structure-aware chunking.

Fixed-size chunking cuts sentences in half and strips every chunk of the
context that made it meaningful. We instead:

  1. split the document on detected headings (multilingual patterns),
  2. pack sentences into ~target_tokens chunks within each section,
  3. carry one sentence of overlap between adjacent chunks,
  4. prepend a "Title > Section" breadcrumb and optionally the abstract,
     so a chunk retrieved in isolation still says what paper it came from.

Step 4 matters more than it looks: without it, a chunk reading "we observe a
12% improvement" is retrievable but useless as evidence.
"""

from __future__ import annotations

import re

from ..config import ChunkingConfig
from ..language import approx_tokens, detect_language, split_sentences
from ..types import Chunk, Document

# Numbered headings (1., 2.3, IV.), plus common section names across the
# languages we ship demo data for. Extend per corpus.
_HEADING_PATTERNS = [
    r"^\s*(?:\d{1,2}(?:\.\d{1,2})*\.?)\s+\S.{0,100}$",
    r"^\s*(?:[IVXLC]{1,6}\.)\s+\S.{0,100}$",
    r"^\s*(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"experiments?|results?|discussion|evaluation|conclusions?|references|"
    r"appendix|acknowledg(?:e)?ments?)\s*:?\s*$",
    r"^\s*(résumé|introduction|méthodes?|résultats?|conclusion|travaux connexes)\s*:?\s*$",
    r"^\s*(摘要|引言|相关工作|方法|实验|结果|讨论|结论|参考文献)\s*[:：]?\s*$",
    r"^\s*(सारांश|परिचय|विधि|प्रयोग|परिणाम|निष्कर्ष|संदर्भ)\s*[:：]?\s*$",
    r"^#{1,4}\s+\S.{0,100}$",  # markdown
]

_HEADING_RE = re.compile("|".join(f"(?:{p})" for p in _HEADING_PATTERNS),
                         re.IGNORECASE | re.MULTILINE)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(section_title, section_text), ...]."""
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    buffer: list[str] = []

    for line in lines:
        if _is_heading(line):
            if buffer:
                sections.append((current_title, buffer))
            current_title = line.strip().lstrip("#").strip()
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_title, buffer))

    return [(title, "\n".join(body).strip()) for title, body in sections
            if "\n".join(body).strip()]


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if not _HEADING_RE.match(stripped):
        return False
    # A "heading" ending in a comma or a lowercase run is usually a wrapped
    # sentence that happened to start with a number.
    return not stripped.endswith((",", ";", "،", "、"))


def chunk_document(doc: Document, cfg: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkingConfig()
    sections = split_sections(doc.text) or [("", doc.text)]

    abstract = ""
    if cfg.attach_abstract and doc.abstract:
        abstract = doc.abstract[: cfg.abstract_chars].strip()

    chunks: list[Chunk] = []
    position = 0

    for section_title, section_text in sections:
        for body in _pack_sentences(section_text, cfg):
            if approx_tokens(body) < cfg.min_tokens and chunks:
                # Merge stragglers into the previous chunk rather than
                # indexing a fragment that will never rank well.
                chunks[-1].text += "\n" + body
                continue

            text = _decorate(body, doc, section_title, abstract, cfg)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{position:04d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section_title,
                    text=text,
                    language=detect_language(body[:800], default=doc.language),
                    position=position,
                    source_path=doc.source_path,
                    metadata={"tokens": approx_tokens(text)},
                )
            )
            position += 1

    return chunks


def _pack_sentences(text: str, cfg: ChunkingConfig) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    packs: list[str] = []
    buffer: list[str] = []
    budget = 0

    for sentence in sentences:
        cost = approx_tokens(sentence)
        if buffer and budget + cost > cfg.target_tokens:
            packs.append(" ".join(buffer))
            tail = buffer[-cfg.overlap_sentences:] if cfg.overlap_sentences else []
            buffer = list(tail)
            budget = sum(approx_tokens(s) for s in buffer)
        buffer.append(sentence)
        budget += cost

    if buffer:
        packs.append(" ".join(buffer))
    return packs


def _decorate(body: str, doc: Document, section: str,
              abstract: str, cfg: ChunkingConfig) -> str:
    parts: list[str] = []
    if cfg.prepend_breadcrumb:
        breadcrumb = f"{doc.title} > {section}" if section else doc.title
        parts.append(f"[{breadcrumb}]")
    if abstract and abstract[:60] not in body:
        parts.append(f"[Abstract] {abstract}")
    parts.append(body)
    return "\n".join(parts)


def chunk_documents(docs: list[Document], cfg: ChunkingConfig | None = None) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, cfg))
    return out
