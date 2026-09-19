"""Load source documents into `Document` objects.

PDF handling notes
------------------
Academic PDFs are two-column more often than not. Naive text extraction
interleaves the columns and produces sentences that are gibberish in any
language. We use PyMuPDF's block extraction and sort blocks by column, then
by vertical position, which handles the common single/two-column cases.

For production-grade parsing of real papers (figures, tables, references,
formulas) swap this for GROBID or Docling; the `Document` contract stays the
same.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..language import detect_language
from ..types import Document

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm"}

_ABSTRACT_RE = re.compile(
    r"^\s*(abstract|résumé|resumen|zusammenfassung|摘要|要旨|सारांश|خلاصة|ملخص)\b[:：.\s]*",
    re.IGNORECASE | re.MULTILINE,
)


def make_doc_id(path: Path) -> str:
    """Stable identifier for a source file.

    Deliberately derived from the file NAME rather than the absolute path, so
    the same corpus produces the same ids on every machine. That is what makes
    a checked-in evaluation set with gold `doc_id`s portable.
    """
    digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^\w.-]+", "-", path.stem)[:40].strip("-")
    return f"{slug}-{digest}"


def load_path(path: str | Path) -> list[Document]:
    """Load a single file or every supported file under a directory."""
    path = Path(path)
    if path.is_dir():
        docs: list[Document] = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                docs.extend(load_path(child))
        return docs

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path}")

    # PDF text arrives hard-wrapped at the column width, so line breaks are
    # layout artifacts and must be reflowed. Markdown and plain text use line
    # breaks structurally — reflowing them would swallow every heading into
    # the paragraph below it, and the chunker would see one giant section.
    if suffix == ".pdf":
        text, title = _load_pdf(path)
        reflow = True
    elif suffix in {".html", ".htm"}:
        text, title = _load_html(path)
        reflow = False
    else:
        text, title = _load_text(path)
        reflow = False

    text = _normalise(text, reflow=reflow)
    if not text.strip():
        return []

    return [
        Document(
            doc_id=make_doc_id(path),
            title=title or path.stem,
            text=text,
            language=detect_language(text[:4000]),
            source_path=str(path),
            abstract=extract_abstract(text),
            metadata={"suffix": suffix, "chars": len(text)},
        )
    ]


# --------------------------------------------------------------------------


def _load_text(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = ""
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            title = stripped[:200]
            break
    return text, title


def _load_html(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        title = (soup.title.string or "").strip() if soup.title else ""
        return soup.get_text("\n"), title
    except ImportError:
        stripped = re.sub(r"<[^>]+>", " ", raw)
        return stripped, path.stem


def _load_pdf(path: Path) -> tuple[str, str]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PDF support needs PyMuPDF. Install with: pip install pymupdf"
        ) from exc

    doc = fitz.open(path)
    title = (doc.metadata or {}).get("title") or ""
    pages: list[str] = []

    for page in doc:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)
        text_blocks = [b for b in blocks if len(b) > 4 and isinstance(b[4], str)]
        if not text_blocks:
            continue
        pages.append(_order_blocks(text_blocks, page.rect.width))

    doc.close()
    return "\n\n".join(pages), title.strip()


def _order_blocks(blocks: list, page_width: float) -> str:
    """Sort blocks into reading order, handling the two-column case."""
    midpoint = page_width / 2
    left = [b for b in blocks if (b[0] + b[2]) / 2 < midpoint]
    right = [b for b in blocks if (b[0] + b[2]) / 2 >= midpoint]

    # Heuristic: a genuine two-column layout puts a substantial share of
    # blocks on each side. Otherwise treat the page as a single flow.
    total = len(blocks)
    two_column = total >= 6 and min(len(left), len(right)) / total > 0.25

    if two_column:
        ordered = sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
    else:
        ordered = sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))

    return "\n".join(b[4].strip() for b in ordered if b[4].strip())


# --------------------------------------------------------------------------


def _normalise(text: str, reflow: bool = False) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if reflow:
        # De-hyphenate words broken across line breaks, then join wrapped
        # lines back into paragraphs. Lines that look like headings (markdown
        # hashes, or short numbered lines) are never joined to the next line.
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(
            r"(?<![\n.!?。！？।:])\n(?![\n#]|\s*\d{1,2}[.)]\s)",
            " ",
            text,
        )

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_abstract(text: str, max_chars: int = 1500) -> str:
    """Pull the abstract if the document has one, else the opening paragraph."""
    match = _ABSTRACT_RE.search(text[:8000])
    if match:
        start = match.end()
        chunk = text[start : start + max_chars]
        stop = re.search(r"\n\s*(1[.\s]|I[.\s]|introduction|引言|परिचय)", chunk, re.IGNORECASE)
        if stop:
            chunk = chunk[: stop.start()]
        return chunk.strip()

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 120]
    if not paragraphs:
        return ""
    # Drop a leading markdown/section heading so the abstract reads as prose.
    body = re.sub(r"^\s*#{1,4}[^\n]*\n", "", paragraphs[0])
    return body.strip()[:max_chars]
