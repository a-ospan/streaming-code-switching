"""Predictive targets built from per-token language tags.

At token t (0-based), using the true tags of the whole sentence:
- y_sw[t]  = 1 if lang[t+1] != lang[t], else 0.
- y_dur[t] = length bin of the segment that starts at t+1, only where y_sw[t] = 1.

The last token has no t+1, so both targets are IGNORE there. y_dur is IGNORE wherever
y_sw is not 1, which matches CrossEntropyLoss(ignore_index=-1) and the rule that
duration is only evaluated at true switch points.

A segment that runs to the end of the sentence is counted as is (it may be cut short
by the sentence boundary). This is a known right-censoring effect to mention in the report.
"""

from __future__ import annotations

IGNORE = -1

# (upper bound inclusive, class id): 1-2 tokens -> 0, 3-6 -> 1, 7+ -> 2
DURATION_BINS = ((2, 0), (6, 1))
DURATION_NAMES = ("1-2", "3-6", "7+")


def duration_class(length: int) -> int:
    for upper, cls in DURATION_BINS:
        if length <= upper:
            return cls
    return len(DURATION_BINS)


def segment_length_from(langs: list[str], start: int) -> int:
    """Number of consecutive tokens with the same language starting at `start`."""
    end = start
    while end < len(langs) and langs[end] == langs[start]:
        end += 1
    return end - start


def make_labels(langs: list[str]) -> tuple[list[int], list[int]]:
    """Return (y_sw, y_dur) for one sentence, one value per token."""
    n = len(langs)
    y_sw = [IGNORE] * n
    y_dur = [IGNORE] * n
    for t in range(n - 1):
        switch = langs[t + 1] != langs[t]
        y_sw[t] = int(switch)
        if switch:
            y_dur[t] = duration_class(segment_length_from(langs, t + 1))
    return y_sw, y_dur
