"""Align word-level targets to a model's subwords for prefix-only (streaming) prediction.

Each sentence is encoded once with the model's own tokenizer. With a causal attention mask,
the hidden state at subword i depends only on subwords 0..i, so one forward pass yields a
prediction for every prefix. This is equivalent to feeding growing windows [x1..xt]
one at a time, and much faster.

The prediction for word t is read at the last subword that lies entirely inside the prefix,
meaning it ends at or before the start of word t+1. Punctuation typed right after word t
("sé,") is part of the prefix; whitespace before word t+1 is not, because the tokenizer
attaches it to the next word.

A position is only valid if the subwords up to that point are exactly what tokenizing the
prefix alone would produce. Two cases break this and are marked IGNORE (counted, not hidden):
- straddle: one subword covers the end of word t and the start of word t+1 (possible in
  text without spaces, such as Chinese characters or "Threads发文");
- retokenized: no subword crosses the boundary, but the tokenizer split the prefix
  differently because it saw the following characters.
Either case would let information from t+1 leak into the prediction at t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.labels import IGNORE


@dataclass
class Encoded:
    input_ids: list[int]
    sw_labels: list[int]   # per subword: y_sw at prediction points, IGNORE elsewhere
    dur_labels: list[int]  # per subword: y_dur at prediction points, IGNORE elsewhere
    pred_index: list[int]  # per word: subword index of its prediction, -1 if invalid
    invalid: dict[str, int] = field(default_factory=dict)


def prefix_ends(starts: list[int], ends: list[int], text: str) -> list[int]:
    """Character end of each word's prefix: the start of the next word (end of text for the last)."""
    return [starts[t + 1] if t + 1 < len(starts) else len(text) for t in range(len(starts))]


def encode_sentence(
    text: str,
    starts: list[int],
    ends: list[int],
    y_sw: list[int],
    y_dur: list[int],
    tokenizer,
    max_length: int = 128,
    check_prefix: bool = True,
) -> Encoded:
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    # Special tokens (<s>, </s>, padding) carry no text. The unknown token does: it stands for
    # characters missing from the vocabulary (e.g. Cantonese 冇 in XGLM), so it counts as content.
    special = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
    n_lead = 0
    while n_lead < len(ids) and ids[n_lead] in special:
        n_lead += 1
    content = [i for i in range(len(ids)) if ids[i] not in special]

    sw = [IGNORE] * len(ids)
    dur = [IGNORE] * len(ids)
    pred = [-1] * len(starts)
    invalid = {"straddle": 0, "retokenized": 0, "truncated": 0}
    bounds = prefix_ends(starts, ends, text)

    for t in range(len(starts) - 1):  # the last word has no target
        if y_sw[t] == IGNORE:
            continue
        inside = [i for i in content if offsets[i][1] <= bounds[t]]
        k = inside[-1] if inside else None
        if k is None or offsets[k][1] < ends[t]:
            # No subword ends at word t: either it was truncated away or a subword straddles the boundary.
            truncated = not any(offsets[i][0] < bounds[t] < offsets[i][1] for i in content)
            invalid["truncated" if truncated else "straddle"] += 1
            continue
        if check_prefix:
            prefix_ids = tokenizer(text[: bounds[t]].rstrip(), add_special_tokens=False)["input_ids"]
            if prefix_ids != ids[n_lead : k + 1]:
                invalid["retokenized"] += 1
                continue
        pred[t] = k
        sw[k] = y_sw[t]
        dur[k] = y_dur[t]
    return Encoded(ids, sw, dur, pred, invalid)
