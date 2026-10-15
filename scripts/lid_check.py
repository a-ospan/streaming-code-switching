"""Hand validation of the token-level language tags.

make:  sample sentences per pair and write one CSV per pair with one token per line and
       the predicted language. Annotators fill the `gold` column only where the prediction
       is wrong (an empty `gold` means the prediction is correct).
score: read the filled CSVs and report token accuracy and switch-point F1 per pair.

Usage:
    python -m scripts.lid_check make  [--per-pair 30] [--seed 13]
    python -m scripts.lid_check score
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from src.data import load_raw, pair_name, parse_generation_result
from src.labels import make_labels
from src.lid import is_romanized_hindi, tag_sentence
from src.tokens import tokenize

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "lid_check"


def sample_sentences(df: pd.DataFrame, per_pair: int, seed: int) -> list[dict]:
    """Random sentences per pair. Hindi gets half romanized, half Devanagari rows."""
    rng = random.Random(seed)
    df = df.assign(pair=[pair_name(a, b) for a, b in zip(df.first_language, df.second_language)])
    picked = []
    for pair, group in df.groupby("pair"):
        rows = []
        for r in group.sample(frac=1, random_state=seed).itertuples():
            sents = parse_generation_result(r.data_generation_result)
            row_tokens = [t for s in sents for t in tokenize(s)]
            romanized = is_romanized_hindi(row_tokens, r.first_language)
            idx = rng.randrange(len(sents))
            rows.append(dict(pair=pair, row_id=r.row_id, sent_idx=idx, text=sents[idx], romanized=romanized,
                             lang1=r.first_language, lang2=r.second_language))
            if len(rows) >= 4 * per_pair:
                break
        if pair == "Hindi-English":
            half = per_pair // 2
            rows = [x for x in rows if x["romanized"]][:half] + [x for x in rows if not x["romanized"]][:per_pair - half]
        picked.extend(rows[:per_pair])
    return picked


def make(per_pair: int, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in sample_sentences(load_raw(), per_pair, seed):
        toks = tokenize(s["text"])
        langs, sources = tag_sentence(toks, s["lang1"], s["lang2"], s["romanized"])
        for i, (tok, lang, src) in enumerate(zip(toks, langs, sources)):
            lines.append(dict(pair=s["pair"], row_id=s["row_id"], sent_idx=s["sent_idx"], romanized=s["romanized"],
                              token_idx=i, token=tok.text, predicted=lang, source=src, gold="",
                              sentence=s["text"] if i == 0 else ""))
    out = pd.DataFrame(lines)
    for pair, part in out.groupby("pair"):
        path = OUT_DIR / f"{pair}.csv"
        if path.exists():
            print(f"skip {path.name}: already exists (may hold annotations)")
            continue
        part.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows Arabic, CJK, etc.
    print(f"{out.groupby(['pair', 'row_id', 'sent_idx']).ngroups} sentences, {len(out)} tokens -> {OUT_DIR}")


def score() -> None:
    results = []
    for path in sorted(OUT_DIR.glob("*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        df["gold"] = df.gold.where(df.gold.str.strip() != "", df.predicted)
        tp = fp = fn = 0
        for _, sent in df.groupby(["row_id", "sent_idx"], sort=False):
            pred_sw, _ = make_labels(sent.predicted.tolist())
            gold_sw, _ = make_labels(sent.gold.tolist())
            for p, g in zip(pred_sw, gold_sw):
                tp += p == 1 and g == 1
                fp += p == 1 and g == 0
                fn += p == 0 and g == 1
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        results.append(dict(pair=path.stem, tokens=len(df), corrected=int((df.gold != df.predicted).sum()),
                            token_acc=(df.gold == df.predicted).mean(), switch_precision=prec, switch_recall=rec,
                            switch_f1=2 * prec * rec / (prec + rec) if prec + rec else float("nan")))
    res = pd.DataFrame(results)
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT_DIR / "summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["make", "score"])
    ap.add_argument("--per-pair", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    make(args.per_pair, args.seed) if args.command == "make" else score()


if __name__ == "__main__":
    main()
