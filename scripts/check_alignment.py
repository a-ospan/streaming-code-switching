"""Measure how well each model's tokenizer fits the word-level targets, per pair.

For a sample of sentences per stratum, report:
- valid_share: target positions usable for prediction (see src/streaming.py);
- straddle / retokenized / truncated: positions lost and why;
- switch_lost: share of true switch points among the lost positions' targets;
- unk_rate: share of subwords that are the unknown token (vocabulary gaps);
- subwords_per_word: tokenizer fertility.

Run: python -m scripts.check_alignment [--per-strata 300] [--seed 13]
Output: results/alignment/<model>.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

from src.labels import IGNORE  # noqa: E402
from src.streaming import encode_sentence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "sentences"
OUT_DIR = ROOT / "results" / "alignment"
MODELS = ["FacebookAI/xlm-roberta-base", "facebook/xglm-564M"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-strata", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    keys = pd.read_parquet(PROCESSED, columns=["row_id", "sent_idx", "strata"],
                           filters=[("split", "==", "train"), ("n_tokens", ">", 1)])
    picked = keys.groupby("strata").sample(args.per_strata, random_state=args.seed)
    sample = pd.read_parquet(PROCESSED, columns=["row_id", "sent_idx", "strata", "text", "starts", "ends", "y_sw",
                                                 "y_dur", "n_tokens"],
                             filters=[("row_id", "in", sorted(set(picked.row_id)))])
    sample = sample.merge(picked[["row_id", "sent_idx"]], on=["row_id", "sent_idx"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        rows = []
        for r in sample.itertuples():
            y_sw, y_dur = list(r.y_sw), list(r.y_dur)
            e = encode_sentence(r.text, list(r.starts), list(r.ends), y_sw, y_dur, tok)
            targets = [t for t in range(len(y_sw)) if y_sw[t] != IGNORE]
            lost = [t for t in targets if e.pred_index[t] < 0]
            special = set(tok.all_special_ids) - {tok.unk_token_id}
            content = [i for i in e.input_ids if i not in special]
            rows.append(dict(strata=r.strata, targets=len(targets), lost=len(lost),
                             switch_targets=sum(y_sw[t] == 1 for t in targets),
                             switch_lost=sum(y_sw[t] == 1 for t in lost),
                             subwords=len(content), words=r.n_tokens,
                             unk=sum(i == tok.unk_token_id for i in content), **e.invalid))
        res = pd.DataFrame(rows).groupby("strata").sum(numeric_only=True)
        out = pd.DataFrame({
            "valid_share": 1 - res.lost / res.targets,
            "straddle": res.straddle / res.targets,
            "retokenized": res.retokenized / res.targets,
            "truncated": res.truncated / res.targets,
            "switch_lost": res.switch_lost / res.switch_targets,
            "unk_rate": res.unk / res.subwords,
            "subwords_per_word": res.subwords / res.words,
        })
        path = OUT_DIR / f"{name.split('/')[-1]}.csv"
        out.round(4).to_csv(path)
        print(f"== {name} -> {path}")
        print(out.round(3).to_string())


if __name__ == "__main__":
    main()
