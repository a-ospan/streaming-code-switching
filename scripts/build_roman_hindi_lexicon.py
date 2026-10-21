"""Build the romanized Hindi word list used by src/lid.py.

56% of Hindi-English rows are written entirely in Latin script ("mujhe bahut pasand aayi"),
so the script rule cannot separate Hindi from English there. wordfreq has no romanized Hindi
list, so we derive one from the corpus itself, using the Hindi-English rows split in two:

- reference rows contain Devanagari. Their Latin-script words are English insertions,
  so they give the frequency of English words in this kind of text.
- romanized rows contain no Devanagari. Their Latin-script words are Hindi or English.

For each word, score = log10(relative frequency in romanized rows / relative frequency in
reference rows), with add-one smoothing. English words occur in both and score near
log10(share of English in romanized rows), about -0.4. Romanized Hindi words ("hai",
"mujhe", "main") almost never occur in the reference rows and score far higher.
Comparing with English inserted in the same corpus, rather than with general English,
avoids flagging words this corpus simply overuses ("honestly", "especially").

The reference rows are small (about 81k tokens), so mid-frequency Hindi words cannot reach
HINDI_SCORE. Words scoring between ENGLISH_SCORE and HINDI_SCORE are settled with general
English frequency: frequent in English -> English ("news", "hospital"), otherwise Hindi
("tarah", "zabardast"). Short words (3 letters or fewer) that are also common English
words are real homographs ("sun", "ban", "le") and are left to the context rule.

Output: src/resources/roman_hindi.tsv (word, counts, score, English Zipf, class), with class
"hindi", "english" or "context". The file is small and committed, so labels are
reproducible and the list can be reviewed by hand.

Run: python -m scripts.build_roman_hindi_lexicon
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from wordfreq import zipf_frequency

from src.data import load_raw, parse_generation_result
from src.tokens import tokenize

OUT = Path(__file__).resolve().parents[1] / "src" / "resources" / "roman_hindi.tsv"

MIN_COUNT = 5      # words seen fewer times in romanized rows are not listed
HINDI_SCORE = 1.0  # at least 10x more frequent in romanized rows -> Hindi
ENGLISH_SCORE = 0.3  # at or below this -> English (not written to the file)
ENGLISH_ZIPF = 3.5   # in the middle zone, this English frequency or more -> English
SHORT = 3            # in the middle zone, short common English words -> context


def main() -> None:
    df = load_raw()
    df = df[df.first_language == "Hindi"]
    roman: Counter[str] = Counter()
    reference: Counter[str] = Counter()
    n_roman_rows = 0
    for raw in df.data_generation_result:
        toks = [t for sent in parse_generation_result(raw) for t in tokenize(sent)]
        latin = [t.text.lower() for t in toks if t.script == "latin"]
        if any(t.script == "devanagari" for t in toks):
            reference.update(latin)
        else:
            roman.update(latin)
            n_roman_rows += 1

    n_roman, n_ref = sum(roman.values()), sum(reference.values())
    rows = []
    for word, c in roman.items():
        if c < MIN_COUNT:
            continue
        score = math.log10((c + 1) / n_roman) - math.log10((reference[word] + 1) / n_ref)
        z_en = zipf_frequency(word, "en")
        if score >= HINDI_SCORE:
            cls = "hindi"
        elif score <= ENGLISH_SCORE:
            continue
        elif z_en < ENGLISH_ZIPF:
            cls = "hindi"
        else:
            cls = "context" if len(word) <= SHORT else "english"
        rows.append((word, c, reference[word], score, z_en, cls))

    rows.sort(key=lambda r: -r[1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        f.write("word\tcount_roman\tcount_reference\tscore\tzipf_en\tclass\n")
        for word, c, r, score, z_en, cls in rows:
            f.write(f"{word}\t{c}\t{r}\t{score:.2f}\t{z_en:.2f}\t{cls}\n")
    print(f"romanized rows: {n_roman_rows}/{len(df)}, Latin tokens: {n_roman} romanized, {n_ref} reference")
    print(f"written to {OUT}: {dict(Counter(r[5] for r in rows))}")


if __name__ == "__main__":
    main()
