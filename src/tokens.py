"""Model-independent word tokenization shared by all models.

Labels are defined on these tokens, so XLM-R and XGLM are scored on identical targets
whatever their subword vocabularies are.

Rules:
- Letters and combining marks of one script form a word ("didn't", "week-end" stay whole:
  an apostrophe or hyphen between two letters of the same script is kept inside).
- Every Han character is its own token (the mBERT rule), because Chinese, Cantonese and
  Japanese do not mark word boundaries with spaces.
- Hiragana and katakana runs are separate tokens ("しい", "メンバー").
- A run of digits is one token of script "num".
- Punctuation, symbols, emoji and whitespace are not tokens. They are kept in the
  original text and recovered from the character offsets.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

JOINERS = {"'", "’", "-", "‐"}


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # character offset in the sentence
    end: int
    script: str


def char_script(ch: str) -> str | None:
    """Script of a letter or mark, "num" for digits, None for everything else."""
    cat = unicodedata.category(ch)
    if cat == "Nd":
        return "num"
    if not (cat.startswith("L") or cat.startswith("M")):
        return None
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2EBEF:
        return "han"
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    if 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF66 <= cp <= 0xFF9F:
        return "katakana"
    if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
        return "hangul"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF:
        return "arabic"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x0400 <= cp <= 0x052F:
        return "cyrillic"
    if cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF5A:
        return "latin"
    if cat.startswith("M"):
        return "mark"  # combining mark outside the blocks above, attaches to its word
    return "other"


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    start = None
    script = None

    def close(end: int) -> None:
        nonlocal start, script
        if start is not None:
            tokens.append(Token(text[start:end], start, end, script))
        start, script = None, None

    for i, ch in enumerate(text):
        s = char_script(ch)
        if s == "mark" and start is not None:
            continue
        if s == "han":
            close(i)
            tokens.append(Token(ch, i, i + 1, "han"))
        elif s is None:
            nxt = char_script(text[i + 1]) if i + 1 < len(text) else None
            if ch in JOINERS and start is not None and script not in ("num",) and nxt == script:
                continue
            close(i)
        elif s == script:
            continue
        else:
            close(i)
            start, script = i, s
    close(len(text))
    return tokens
