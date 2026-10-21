"""PyTorch dataset, batching and pair-balanced sampling for training.

Language pairs differ a lot in size (Cantonese-English is 64% of sentences), so training
draws sentences with temperature sampling: pair p is drawn with probability proportional
to n_p ** alpha. alpha = 1 keeps the natural imbalance, alpha = 0 samples all pairs equally,
and alpha = 0.3 is the compromise used for XLM-R and mT5. Sentences within a pair are
drawn uniformly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.labels import IGNORE
from src.streaming import encode_sentence


def pair_sampling_probs(pairs: pd.Series, alpha: float) -> pd.Series:
    """Probability of drawing each pair under temperature sampling."""
    sizes = pairs.value_counts().astype(float)
    weights = sizes**alpha
    return weights / weights.sum()


def temperature_sampler(pairs: pd.Series, alpha: float, num_samples: int, seed: int) -> WeightedRandomSampler:
    """Sampler over sentence indices: each sentence gets P(pair) / n_pair."""
    probs = pair_sampling_probs(pairs, alpha)
    sizes = pairs.value_counts()
    weights = pairs.map(probs / sizes).to_numpy(dtype=np.float64, copy=True)
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples, replacement=True,
                                 generator=gen)


class SwitchDataset(Dataset):
    """Encodes sentences lazily with a model's tokenizer. One item = one sentence."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128, check_prefix: bool = True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.check_prefix = check_prefix

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        e = encode_sentence(r.text, list(r.starts), list(r.ends), list(r.y_sw), list(r.y_dur),
                            self.tokenizer, self.max_length, self.check_prefix)
        return {"input_ids": e.input_ids, "sw_labels": e.sw_labels, "dur_labels": e.dur_labels,
                "pair": r.pair, "strata": r.strata, "index": i}


def make_collate(pad_id: int):
    """Right-pad a batch. Causal models never attend to later (padded) positions."""

    def collate(items: list[dict]) -> dict:
        n = max(len(x["input_ids"]) for x in items)
        ids = np.full((len(items), n), pad_id, dtype=np.int64)
        mask = np.zeros((len(items), n), dtype=np.int64)
        sw = np.full((len(items), n), IGNORE, dtype=np.int64)
        dur = np.full((len(items), n), IGNORE, dtype=np.int64)
        for b, x in enumerate(items):
            m = len(x["input_ids"])
            ids[b, :m], mask[b, :m] = x["input_ids"], 1
            sw[b, :m], dur[b, :m] = x["sw_labels"], x["dur_labels"]
        return {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask),
                "sw_labels": torch.from_numpy(sw), "dur_labels": torch.from_numpy(dur),
                "pair": [x["pair"] for x in items], "strata": [x["strata"] for x in items],
                "index": torch.tensor([x["index"] for x in items])}

    return collate
