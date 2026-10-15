"""Build the processed dataset and the fixed splits.

Steps: load the cached dataset, flatten to sentences, tokenize, tag each token's language,
build y_sw / y_dur, and assign each source row to train/val/test.

Memory: the full dataset does not fit comfortably in RAM on an 8 GB laptop, so this works
in two passes over the memory-mapped dataset:
  1. light pass: sentence texts only, to group duplicates and fix the splits;
  2. heavy pass: tokenize, tag and label BATCH_ROWS rows at a time, writing each batch
     to its own parquet file.

Outputs:
  data/processed/sentences/part-*.parquet  one row per sentence with tokens, tags, labels (gitignored)
  results/splits/row_splits.csv.gz         row_id -> split, committed so the team uses the same splits
  results/data_summary.csv                 per stratum and split: sizes, switch rate, duration bins

Existing splits are reused, never regenerated, so they stay fixed once created.
Use --force to rebuild the processed parquet files after a change in tokenization or tagging.

Run: python -m scripts.prepare_data [--seed 13] [--limit-rows N] [--force]
Read: pd.read_parquet("data/processed/sentences")
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import flatten, iter_raw_batches, load_raw_dataset
from src.labels import IGNORE, make_labels
from src.lid import is_romanized_hindi, tag_sentence
from src.splits import SPLIT_FRACTIONS, assign_splits
from src.tokens import char_script, tokenize

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "sentences"
SPLITS = ROOT / "results" / "splits" / "row_splits.csv.gz"
SUMMARY = ROOT / "results" / "data_summary.csv"
BATCH_ROWS = 20_000


def strata_name(pair: str, romanized: bool) -> str:
    if pair == "Hindi-English":
        return "Hindi-English (romanized)" if romanized else "Hindi-English (Devanagari)"
    return pair


def romanized_rows(sentences: pd.DataFrame) -> set[int]:
    """Hindi-English rows with no Devanagari letter in any sentence (same rule as is_romanized_hindi)."""
    hindi = sentences[sentences.first_language == "Hindi"]
    has_deva = hindi.groupby("row_id").text.agg(
        lambda texts: any(char_script(c) == "devanagari" for t in texts for c in t))
    return set(has_deva.index[~has_deva])


def make_splits(ds, seed: int) -> pd.DataFrame:
    """Light pass: sentence texts only, then row-level splits stratified by pair and script."""
    parts = []
    for batch in iter_raw_batches(ds.select_columns(["row_id", "first_language", "second_language",
                                                      "data_generation_result"]), BATCH_ROWS):
        sents, _ = flatten(batch)
        roman = romanized_rows(sents)
        sents["strata"] = [strata_name(p, r in roman) for p, r in zip(sents.pair, sents.row_id)]
        parts.append(sents[["row_id", "strata", "text"]])
    sentences = pd.concat(parts, ignore_index=True)
    return assign_splits(sentences, "strata", seed=seed)[["row_id", "strata", "group", "split"]]


def process(sentences: pd.DataFrame) -> pd.DataFrame:
    """Add token, tag and label columns. Romanized Hindi is decided per source row."""
    tokens = [tokenize(t) for t in sentences.text]
    row_tokens: dict[int, list] = {}
    for row_id, toks in zip(sentences.row_id, tokens):
        row_tokens.setdefault(row_id, []).extend(toks)
    roman = {r for r, toks in row_tokens.items() if is_romanized_hindi(toks, "Hindi")}

    cols = {k: [] for k in ("tokens", "starts", "ends", "langs", "lid_source", "y_sw", "y_dur", "romanized")}
    for toks, row_id, l1, l2 in zip(tokens, sentences.row_id, sentences.first_language, sentences.second_language):
        romanized = l1 == "Hindi" and row_id in roman
        langs, sources = tag_sentence(toks, l1, l2, romanized)
        y_sw, y_dur = make_labels(langs)
        cols["tokens"].append([t.text for t in toks])
        cols["starts"].append([t.start for t in toks])
        cols["ends"].append([t.end for t in toks])
        cols["langs"].append(langs)
        cols["lid_source"].append(sources)
        cols["y_sw"].append(y_sw)
        cols["y_dur"].append(y_dur)
        cols["romanized"].append(romanized)
    out = sentences.assign(**cols)
    out["n_tokens"] = out.tokens.str.len()
    out["strata"] = [strata_name(p, r) for p, r in zip(out.pair, out.romanized)]
    return out


def batch_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per stratum and split: sentence/token counts, target positions, switches, duration bins."""
    rows = []
    for (strata, split), g in df.groupby(["strata", "split"]):
        sw = np.concatenate([np.asarray(x) for x in g.y_sw])
        dur = np.concatenate([np.asarray(x) for x in g.y_dur])
        rows.append(dict(strata=strata, split=split, sentences=len(g), tokens=int(g.n_tokens.sum()),
                         positions=int((sw != IGNORE).sum()), switches=int((sw == 1).sum()),
                         dur_1_2=int((dur == 0).sum()), dur_3_6=int((dur == 1).sum()),
                         dur_7plus=int((dur == 2).sum())))
    return pd.DataFrame(rows)


def summarize(counts: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    summ = counts.groupby(["strata", "split"]).sum().reset_index()
    rows = splits.groupby(["strata", "split"]).size().rename("rows").reset_index()
    summ = rows.merge(summ, on=["strata", "split"])
    summ["switch_rate"] = summ.switches / summ.positions
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit-rows", type=int, default=None, help="process only the first N rows (smoke test)")
    ap.add_argument("--force", action="store_true", help="rebuild the processed parquet files if they exist")
    args = ap.parse_args()

    t0 = time.time()
    ds = load_raw_dataset()

    if args.limit_rows:
        batch = ds.select(range(args.limit_rows)).to_pandas()
        sents, report = flatten(batch)
        print(report)
        df = process(sents).assign(split="all")
        print(f"processed {len(df)} sentences in {time.time() - t0:.0f}s")
        counts = batch_counts(df)
        counts["switch_rate"] = counts.switches / counts.positions
        print(counts.to_string(index=False))
        return

    if PROCESSED.exists() and not args.force:
        raise SystemExit(f"{PROCESSED} exists. Use --force to rebuild it (splits are kept).")

    if SPLITS.exists():
        splits = pd.read_csv(SPLITS)
        print(f"reusing existing splits from {SPLITS}")
    else:
        splits = make_splits(ds, args.seed)
        SPLITS.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(SPLITS, index=False)
        (SPLITS.parent / "split_config.json").write_text(
            json.dumps({"seed": args.seed, "fractions": SPLIT_FRACTIONS, "unit": "source row",
                        "strata": "language pair, Hindi-English split by script",
                        "duplicates": "rows sharing a sentence text stay in one split"}, indent=2))
        print(f"created splits -> {SPLITS} ({time.time() - t0:.0f}s)")
    split_of = splits.set_index("row_id").split

    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)  # only reached with --force; these files are derived and rebuildable
    PROCESSED.mkdir(parents=True)

    counts, reports = [], []
    for i, batch in enumerate(iter_raw_batches(ds, BATCH_ROWS)):
        sents, report = flatten(batch)
        reports.append(report)
        df = process(sents)
        df["split"] = df.row_id.map(split_of)
        assert df.split.notna().all(), "some rows have no split"
        assert (df.strata == df.row_id.map(splits.set_index("row_id").strata)).all(), "strata mismatch"
        df.to_parquet(PROCESSED / f"part-{i:05d}.parquet", index=False)
        counts.append(batch_counts(df))
        print(f"batch {i}: {len(batch)} rows, {len(df)} sentences ({time.time() - t0:.0f}s)", flush=True)

    dropped = {k: int(sum(getattr(r, k).sum() for r in reports)) for k in ("dropped_empty", "dropped_too_long")}
    print(f"dropped sentences: {dropped}")
    summ = summarize(pd.concat(counts), splits)
    summ.to_csv(SUMMARY, index=False)
    print(summ.to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
