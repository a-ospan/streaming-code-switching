import pandas as pd
import pytest

from src.dataset import make_collate, pair_sampling_probs, temperature_sampler
from src.labels import IGNORE


def pairs_series():
    return pd.Series(["Cantonese-English"] * 6000 + ["Spanish-English"] * 250 + ["Hindi-English"] * 250)


def test_alpha_one_keeps_natural_shares():
    p = pair_sampling_probs(pairs_series(), alpha=1.0)
    assert p["Cantonese-English"] == pytest.approx(6000 / 6500)


def test_alpha_zero_is_uniform():
    p = pair_sampling_probs(pairs_series(), alpha=0.0)
    assert p.tolist() == pytest.approx([1 / 3] * 3)


def test_alpha_point_three_matches_formula():
    p = pair_sampling_probs(pairs_series(), alpha=0.3)
    ratio = p["Cantonese-English"] / p["Spanish-English"]
    assert ratio == pytest.approx((6000 / 250) ** 0.3)  # about 2.6, not 24


def test_sampler_draws_pairs_at_target_rates():
    s = pairs_series()
    idx = list(temperature_sampler(s, alpha=0.0, num_samples=30000, seed=0))
    shares = s.iloc[idx].value_counts(normalize=True)
    assert shares.between(0.31, 0.36).all()


def test_collate_pads_with_ignore():
    collate = make_collate(pad_id=1)
    batch = collate([
        {"input_ids": [0, 5, 6, 2], "sw_labels": [X := IGNORE, 1, 0, X], "dur_labels": [X, 0, X, X],
         "pair": "a", "strata": "a", "index": 0},
        {"input_ids": [0, 7, 2], "sw_labels": [X, 0, X], "dur_labels": [X, X, X], "pair": "b", "strata": "b",
         "index": 1},
    ])
    assert batch["input_ids"].tolist() == [[0, 5, 6, 2], [0, 7, 2, 1]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert batch["sw_labels"][1].tolist() == [X, 0, X, X]
