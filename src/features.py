"""Prefix-only features for the baseline models.

The stored language tags (`langs`) are labels: rule 3 of src/lid.py may fill an undecided
token from its RIGHT neighbor, which uses the future. Features must not, so they are built
from "causal tags": each token's own script/frequency decision (lid_source is script or
freq), with undecided tokens filled from the nearest decided token on the LEFT within the
prefix (or, before any decided token, from the first decided token so far; failing that
the row's first language).

Features at token t (all computed from tokens 0..t and the text typed up to word t+1):
- strata (language pair, known in advance), one-hot
- current language is the second language of the pair
- length of the current same-language run, bucketed 1, 2, 3, 4-6, 7+
- number of switches so far, bucketed 0, 1, 2, 3+
- position t, bucketed 0, 1, 2, 3-5, 6-10, 11+
- punctuation typed after word t: comma-like, sentence-final, other
- script of word t
"""

from __future__ import annotations

import numpy as np

from src.labels import IGNORE

DECIDED = {"script", "freq"}
COMMA = set(",，、;；:：")
FINAL = set(".!?。！？…")


def causal_tags(langs: list[str], sources: list[str], lang1: str) -> list[list[str]]:
    """causal_tags(...)[t][j] = tag of token j as seen with only tokens 0..t available."""
    raw = [lang if src in DECIDED else None for lang, src in zip(langs, sources)]
    out = []
    for t in range(len(raw)):
        prefix = raw[: t + 1]
        first = next((x for x in prefix if x is not None), lang1)
        seen, last = [], None
        for x in prefix:
            last = x if x is not None else last
            seen.append(last if last is not None else first)
        out.append(seen)
    return out


def bucket(x: int, edges: list[int]) -> int:
    """Index of the first edge >= x (edges are inclusive upper bounds), else len(edges)."""
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)


def punct_after(text: str, end: int, next_start: int) -> tuple[int, int, int]:
    gap = set(text[end:next_start]) - {" ", "\t", "\n"}
    return int(bool(gap & COMMA)), int(bool(gap & FINAL)), int(bool(gap - COMMA - FINAL))


SCRIPTS = ["latin", "han", "hiragana", "katakana", "hangul", "arabic", "devanagari", "cyrillic", "num", "other"]


def sentence_features(text, starts, ends, tokens_scripts, langs, sources, y_sw, lang1, lang2):
    """Rows of raw (unencoded) features for every target position of one sentence."""
    tags = causal_tags(langs, sources, lang1)
    rows = []
    for t in range(len(langs) - 1):
        if y_sw[t] == IGNORE:
            continue
        seen = tags[t]
        cur = seen[t]
        run = 1
        while run <= t and seen[t - run] == cur:
            run += 1
        switches = sum(seen[j] != seen[j - 1] for j in range(1, t + 1))
        comma, final, other = punct_after(text, ends[t], starts[t + 1])
        rows.append((int(cur == lang2), bucket(run, [1, 2, 3, 6]), bucket(switches, [0, 1, 2]),
                     bucket(t, [0, 1, 2, 5, 10]), comma, final, other,
                     SCRIPTS.index(tokens_scripts[t]) if tokens_scripts[t] in SCRIPTS else SCRIPTS.index("other")))
    return rows


FEATURE_NAMES = ["cur_is_lang2", "run_bucket", "switches_bucket", "pos_bucket", "punct_comma", "punct_final",
                 "punct_other", "script"]
CATEGORICAL = ["run_bucket", "switches_bucket", "pos_bucket", "script"]


def as_array(rows: list[tuple]) -> np.ndarray:
    return np.asarray(rows, dtype=np.int16).reshape(-1, len(FEATURE_NAMES))
