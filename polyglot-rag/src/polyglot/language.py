"""Language detection, script detection, and script-aware tokenisation.

These utilities are the difference between a RAG stack that works in English
and one that works in Hindi, Chinese and Arabic too. In particular:

* BM25 over whitespace tokens silently returns nothing for Chinese/Japanese,
  because those scripts do not delimit words with spaces. We emit character
  bigrams for CJK runs instead.
* Token-count heuristics tuned on English badly under-count CJK, producing
  chunks 2-3x larger than intended.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# Script detection
# --------------------------------------------------------------------------

_CJK_RANGES = (
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3400, 0x4DBF),   # CJK Ext A
    (0x4E00, 0x9FFF),   # CJK Unified
    (0xF900, 0xFAFF),   # CJK Compatibility
    (0xAC00, 0xD7AF),   # Hangul syllables
)

_SCRIPT_RANGES = {
    "Latin": ((0x0041, 0x024F),),
    "Cyrillic": ((0x0400, 0x04FF),),
    "Greek": ((0x0370, 0x03FF),),
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F)),
    "Hebrew": ((0x0590, 0x05FF),),
    "Devanagari": ((0x0900, 0x097F),),
    "Bengali": ((0x0980, 0x09FF),),
    "Tamil": ((0x0B80, 0x0BFF),),
    "Telugu": ((0x0C00, 0x0C7F),),
    "Thai": ((0x0E00, 0x0E7F),),
    "Han": _CJK_RANGES,
}


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def dominant_script(text: str) -> str:
    """Return the script covering the most letters in `text`."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for name, ranges in _SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return "Unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


# Scripts with no reliable whitespace word boundary.
_UNSEGMENTED_SCRIPTS = {"Han", "Thai"}


# --------------------------------------------------------------------------
# Language detection
# --------------------------------------------------------------------------

_SCRIPT_FALLBACK = {
    "Devanagari": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Arabic": "ar",
    "Hebrew": "he",
    "Greek": "el",
    "Cyrillic": "ru",
    "Thai": "th",
    "Han": "zh",
    "Latin": "en",
}

_detector = None
_detector_loaded = False


def _get_detector():
    """Lazily load `lingua` if present. It is far better than langdetect on
    short queries, but it is a heavy optional dependency."""
    global _detector, _detector_loaded
    if _detector_loaded:
        return _detector
    _detector_loaded = True
    try:
        from lingua import LanguageDetectorBuilder

        _detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_preloaded_language_models()
            .build()
        )
    except Exception:
        _detector = None
    return _detector


# Script alone cannot separate English from French, Spanish, German and so on.
# When `lingua` is absent we score high-frequency function words instead —
# crude, but it prevents every Latin-script document being labelled "en",
# which would quietly break the per-language evaluation breakdown.
_STOPWORD_PROFILES: dict[str, set[str]] = {
    "en": {"the", "and", "of", "to", "in", "that", "is", "for", "with", "we", "are", "this"},
    "fr": {"le", "la", "les", "des", "une", "est", "nous", "pour", "dans", "sur", "que", "du", "avec"},
    "es": {"el", "la", "los", "las", "una", "es", "para", "con", "por", "que", "del", "este"},
    "pt": {"o", "a", "os", "as", "uma", "para", "com", "que", "não", "dos", "este", "são"},
    "de": {"der", "die", "das", "und", "ist", "wir", "für", "mit", "den", "eine", "auf", "nicht"},
    "it": {"il", "la", "le", "dei", "una", "per", "con", "che", "del", "sono", "questo", "nel"},
    "nl": {"de", "het", "een", "van", "en", "is", "voor", "met", "niet", "dat", "wij", "aan"},
    "id": {"yang", "dan", "untuk", "dengan", "tidak", "ini", "dari", "pada", "adalah", "kami"},
    "vi": {"và", "của", "các", "trong", "được", "một", "không", "này", "cho", "với"},
    "tr": {"ve", "bir", "bu", "için", "ile", "olarak", "daha", "değil", "olan", "gibi"},
}


def _detect_latin(text: str) -> str | None:
    tokens = [t for t in re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)][:400]
    if len(tokens) < 8:
        return None
    counts = {
        lang: sum(1 for t in tokens if t in words)
        for lang, words in _STOPWORD_PROFILES.items()
    }
    best, score = max(counts.items(), key=lambda kv: kv[1])
    # Require a real signal, not one incidental match.
    return best if score >= max(3, len(tokens) // 40) else None


def detect_language(text: str, default: str = "unknown") -> str:
    """Return an ISO 639-1 code.

    Preference order: `lingua` (accurate, optional dependency) -> stopword
    profile for Latin script -> script inference -> `default`.
    """
    stripped = text.strip()
    if len(stripped) < 2:
        return default

    detector = _get_detector()
    if detector is not None:
        try:
            lang = detector.detect_language_of(stripped)
            if lang is not None:
                return lang.iso_code_639_1.name.lower()
        except Exception:
            pass

    script = dominant_script(stripped)
    if script == "Latin":
        guess = _detect_latin(stripped)
        if guess:
            return guess

    return _SCRIPT_FALLBACK.get(script, default)


# --------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------


def approx_tokens(text: str) -> int:
    """Cheap, script-aware token estimate.

    CJK characters are roughly one token each; alphabetic scripts average
    ~4 characters per token. Good enough for chunk sizing without pulling in
    a tokenizer dependency at ingest time.
    """
    cjk = sum(1 for ch in text if is_cjk(ch))
    other = len(text) - cjk
    return cjk + max(1, other // 4) if text else 0


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

# Latin/Devanagari/CJK/Arabic terminators, kept in one pass.
_SENT_END = re.compile(r"(?<=[.!?。！？؟।…])\s*|\n{2,}")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    # CJK text sometimes arrives as one long run with no terminators at all.
    out: list[str] = []
    for p in parts:
        if approx_tokens(p) > 400:
            out.extend(_hard_wrap(p, 300))
        else:
            out.append(p)
    return out


def _hard_wrap(text: str, target_tokens: int) -> list[str]:
    chunks, buf = [], ""
    for ch in text:
        buf += ch
        if approx_tokens(buf) >= target_tokens:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


# --------------------------------------------------------------------------
# Lexical tokenisation for BM25
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def lexical_tokens(text: str, bigrams_for_unsegmented: bool = True) -> list[str]:
    """Tokenise for BM25 across scripts.

    Word-delimited scripts -> normalised lowercase words.
    Unsegmented scripts (Han, Kana, Thai) -> character unigrams + bigrams,
    which is the standard cheap substitute for a real segmenter and recovers
    most of the recall a whitespace tokenizer throws away.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []

    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        if bigrams_for_unsegmented and _needs_char_tokens(word):
            tokens.extend(_char_ngrams(word))
        else:
            tokens.append(word)
    return tokens


def _needs_char_tokens(word: str) -> bool:
    if len(word) < 2:
        return False
    script = dominant_script(word)
    return script in _UNSEGMENTED_SCRIPTS


def _char_ngrams(word: str) -> list[str]:
    chars = list(word)
    grams = list(chars)
    grams += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return grams
