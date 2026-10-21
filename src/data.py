"""Load SwitchLingua_text and flatten it into one row per sentence.

The `data_generation_result` column is a stringified Python object written by an
LLM generator. Most rows are a list of sentence strings, but some rows (mostly
Cantonese multi-turn) hold dicts such as {"topic": ..., "instances": [...]} or
{"speaker": ..., "message": ...}, nested lists, or a bare string. `extract_sentences`
handles all of these and keeps only the text fields.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

import pandas as pd

DATASET_ID = "Shelton1013/SwitchLingua_text"
# Pinned dataset version. The splits in results/splits identify rows by position, so every
# machine must load the same version in the same order. Change only together with new splits.
DATASET_REVISION = "6e048d27e3b2cb3a7a926288d4915745a0ab2801"

# Dict keys that hold metadata, not utterance text.
METADATA_KEYS = {"topic", "speaker", "role", "name", "id", "turn"}

# Sentences longer than this (whitespace words) are generator artifacts, e.g. a whole
# conversation pasted into one string. They are flagged, not silently dropped.
MAX_WORDS = 120

META_COLUMNS = [
    "topic", "tense", "perspective", "cs_ratio", "gender", "age", "education_level",
    "first_language", "second_language", "conversation_type", "cs_function", "cs_type", "score",
]


def pair_name(first_language: str, second_language: str) -> str:
    return f"{first_language}-{second_language}"


def load_raw_dataset(columns: list[str] | None = None):
    """The train split as a memory-mapped Hugging Face Dataset, with a `row_id` column.

    Loads from the local cache (no network, no token). Use this with `iter_raw_batches`
    when memory is tight; `load_raw` materializes everything in pandas.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)
    ds = ds.select_columns(columns or META_COLUMNS + ["data_generation_result"])
    return ds.add_column("row_id", list(range(len(ds))))


def iter_raw_batches(ds, batch_size: int):
    """Yield pandas DataFrames of `batch_size` rows from a dataset made by `load_raw_dataset`."""
    for start in range(0, len(ds), batch_size):
        yield ds.select(range(start, min(start + batch_size, len(ds)))).to_pandas()


def load_raw() -> pd.DataFrame:
    """Load the whole train split into pandas."""
    df = load_raw_dataset().to_pandas()
    return df[["row_id"] + [c for c in df.columns if c != "row_id"]]


def extract_sentences(obj) -> list[str]:
    """Recursively collect sentence strings from a parsed generation result."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [s for item in obj for s in extract_sentences(item)]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() if k not in METADATA_KEYS for s in extract_sentences(v)]
    return []


def parse_generation_result(raw: str) -> list[str]:
    """Parse one `data_generation_result` value into a list of sentences.

    Values that are not valid Python literals are plain sentences (91 Cantonese rows).
    """
    try:
        obj = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        obj = raw
    return [s.strip() for s in extract_sentences(obj)]


@dataclass
class FlattenReport:
    rows_in: int
    sentences_out: int
    dropped_empty: pd.Series
    dropped_too_long: pd.Series
    duplicate_sentences: pd.Series

    def __str__(self) -> str:
        lines = [f"rows in: {self.rows_in}, sentences out: {self.sentences_out}"]
        for name in ("dropped_empty", "dropped_too_long", "duplicate_sentences"):
            counts = getattr(self, name)
            lines.append(f"{name}: {int(counts.sum())} " + (str(counts.to_dict()) if counts.sum() else ""))
        return "\n".join(lines)


def flatten(df: pd.DataFrame) -> tuple[pd.DataFrame, FlattenReport]:
    """One row per sentence, with the source row's metadata and a `pair` column.

    Empty and over-long sentences are removed and counted per pair in the report.
    Duplicate sentences are kept but counted, so the split step can keep them on one side.
    """
    meta = [c for c in df.columns if c != "data_generation_result"]
    records = []
    for row in df.itertuples(index=False):
        r = row._asdict()
        for i, sent in enumerate(parse_generation_result(r["data_generation_result"])):
            records.append({**{c: r[c] for c in meta}, "sent_idx": i, "text": sent})
    out = pd.DataFrame.from_records(records)
    out.insert(1, "pair", [pair_name(a, b) for a, b in zip(out.first_language, out.second_language)])

    n_words = out.text.str.split().str.len()
    empty = n_words == 0
    too_long = n_words > MAX_WORDS
    report = FlattenReport(
        rows_in=len(df),
        sentences_out=int((~empty & ~too_long).sum()),
        dropped_empty=out[empty].pair.value_counts(),
        dropped_too_long=out[too_long].pair.value_counts(),
        duplicate_sentences=out[~empty & ~too_long & out.text.duplicated()].pair.value_counts(),
    )
    out = out[~empty & ~too_long].reset_index(drop=True)
    return out, report
