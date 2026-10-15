"""Token-level language identification (LID) for SwitchLingua sentences.

The dataset gives only the two languages of each row, not a language per word, so we
derive one. Every token is tagged with one of the row's two languages:

1. Script rule: if the token's script belongs to only one of the two languages
   (Devanagari in Hindi-English, Latin in Cantonese-English), that decides it.
   This covers 7 of the 15 pairs completely.
2. Frequency rule (both languages written in Latin script): compare the word's Zipf
   frequency in each language with `wordfreq`. A clear winner (gap >= MARGIN) decides it.
   Romanized Hindi: in Hindi-English rows written without any Devanagari, Hindi words are in
   Latin script too. There a corpus-derived word list (src/resources/roman_hindi.tsv, built by
   scripts/build_roman_hindi_lexicon.py) decides, and other words count as English when
   they are common in English (Zipf >= ROMAN_EN_ZIPF).
3. Context rule: tokens left undecided (numbers, names, words common in both languages
   such as "no" or "me" in Spanish-English) take the language of the nearest decided
   token, preferring the left neighbor when the two sides disagree.

Labels may use the whole sentence (including the right context in rule 3). Only the
model's input features must be prefix-only.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from wordfreq import zipf_frequency

from src.tokens import Token

LANG_SCRIPTS = {
    "Arabic": {"arabic"},
    "Cantonese": {"han"},
    "Chinese": {"han"},
    "Hindi": {"devanagari"},
    "Japanese": {"han", "hiragana", "katakana"},
    "Korean": {"hangul"},
    "Russian": {"cyrillic"},
    "English": {"latin"},
    "French": {"latin"},
    "German": {"latin"},
    "Italian": {"latin"},
    "Malay": {"latin"},
    "Philippines": {"latin"},
    "Spanish": {"latin"},
    "Vietnamese": {"latin"},
}

WORDFREQ_CODE = {
    "English": "en", "French": "fr", "German": "de", "Italian": "it",
    "Malay": "ms", "Philippines": "fil", "Spanish": "es", "Vietnamese": "vi",
}

MARGIN = 0.3    # Zipf gap needed for the frequency rule. Kept small because the Malay,
                # Filipino and other web corpora contain many English loanwords, so English
                # words are often only 0.3 to 1.5 Zipf higher in English than in the other list.
MIN_ZIPF = 1.0  # below this in both languages the word is unknown (likely a name)

ROMAN_HINDI_PATH = Path(__file__).resolve().parent / "resources" / "roman_hindi.tsv"
ROMAN_EN_ZIPF = 3.0  # unlisted word in a romanized Hindi row: English if at least this common

# How each label was decided, kept for diagnostics.
SCRIPT, FREQ, CONTEXT, DEFAULT = "script", "freq", "context", "default"


@lru_cache(maxsize=500_000)
def _zipf(word: str, code: str) -> float:
    return zipf_frequency(word, code)


@lru_cache(maxsize=1)
def roman_hindi_lexicon() -> dict[str, str]:
    """word -> "hindi", "english" or "context"."""
    with ROMAN_HINDI_PATH.open(encoding="utf-8", newline="") as f:
        return {r["word"]: r["class"] for r in csv.DictReader(f, delimiter="	")}


def is_romanized_hindi(tokens: list[Token], lang1: str) -> bool:
    """True for Hindi-English text with no Devanagari at all. Pass all tokens of a row."""
    return lang1 == "Hindi" and not any(t.script == "devanagari" for t in tokens)


def _tag_roman_hindi(tok: Token) -> tuple[str | None, str | None]:
    if tok.script != "latin":
        return None, None
    word = tok.text.lower()
    cls = roman_hindi_lexicon().get(word)
    if cls == "hindi":
        return "Hindi", FREQ
    if cls == "english" or (cls is None and _zipf(word, "en") >= ROMAN_EN_ZIPF):
        return "English", FREQ
    return None, None


def tag_token(tok: Token, lang1: str, lang2: str, romanized: bool = False) -> tuple[str | None, str | None]:
    """Language of one token by the script or frequency rule, or (None, None)."""
    if romanized:
        return _tag_roman_hindi(tok)
    in1 = tok.script in LANG_SCRIPTS[lang1]
    in2 = tok.script in LANG_SCRIPTS[lang2]
    if in1 != in2:
        return (lang1 if in1 else lang2), SCRIPT
    if in1 and in2 and tok.script == "latin":
        word = tok.text.lower()
        z1 = _zipf(word, WORDFREQ_CODE[lang1])
        z2 = _zipf(word, WORDFREQ_CODE[lang2])
        if max(z1, z2) >= MIN_ZIPF and abs(z1 - z2) >= MARGIN:
            return (lang1 if z1 > z2 else lang2), FREQ
    return None, None


def tag_sentence(
    tokens: list[Token], lang1: str, lang2: str, romanized: bool = False
) -> tuple[list[str], list[str]]:
    """Language label and decision source for every token in a sentence.

    `romanized` must be decided per row with `is_romanized_hindi`, not per sentence:
    an all-English sentence inside a Devanagari row is ordinary English.
    """
    tagged = [tag_token(t, lang1, lang2, romanized) for t in tokens]
    langs = [lang for lang, _ in tagged]
    sources = [src for _, src in tagged]

    decided = [i for i, lang in enumerate(langs) if lang is not None]
    if not decided:
        return [lang1] * len(tokens), [DEFAULT] * len(tokens)

    out_langs, out_sources = list(langs), list(sources)
    for i, lang in enumerate(langs):
        if lang is not None:
            continue
        left = next((langs[j] for j in range(i - 1, -1, -1) if langs[j] is not None), None)
        right = next((langs[j] for j in range(i + 1, len(langs)) if langs[j] is not None), None)
        out_langs[i] = left if left is not None else right
        out_sources[i] = CONTEXT
    return out_langs, out_sources
