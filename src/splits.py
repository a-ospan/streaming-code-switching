"""Fixed train/val/test splits at the source-row (conversation) level.

- The unit is the source row, so sentences of one conversation never sit in two splits.
- Rows that share any identical sentence text are grouped together (union-find), so a
  duplicated sentence cannot appear in both train and test.
- Stratified by `strata` (the language pair, with Hindi-English split by script), so each
  stratum gets the same train/val/test proportions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def duplicate_groups(sentences: pd.DataFrame) -> pd.Series:
    """Map row_id -> group id, where rows sharing any sentence text share a group."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row_ids in sentences.groupby("text").row_id.unique():
        first = find(int(row_ids[0]))
        for other in row_ids[1:]:
            root = find(int(other))
            if root != first:
                parent[root] = first
    rows = sentences.row_id.unique()
    return pd.Series([find(int(r)) for r in rows], index=rows, name="group")


def assign_splits(sentences: pd.DataFrame, strata_col: str, seed: int) -> pd.DataFrame:
    """Return one line per row_id with its stratum, duplicate group and split."""
    groups = duplicate_groups(sentences)
    rows = sentences.drop_duplicates("row_id").set_index("row_id")[[strata_col]]
    rows["group"] = groups
    rng = np.random.default_rng(seed)
    out = []
    for stratum, part in rows.groupby(strata_col, sort=True):
        # Group sizes in rows; shuffle groups, then fill train, val, test by cumulative share.
        sizes = part.groupby("group").size()
        order = sizes.index.to_numpy()[rng.permutation(len(sizes))]
        cum = sizes.loc[order].cumsum().to_numpy() / sizes.sum()
        bounds = np.cumsum(list(SPLIT_FRACTIONS.values()))
        names = np.array(list(SPLIT_FRACTIONS))
        split_of_group = dict(zip(order, names[np.searchsorted(bounds, cum - 1e-12, side="left")]))
        out.append(part.assign(split=part.group.map(split_of_group)))
    return pd.concat(out).reset_index().sort_values("row_id").reset_index(drop=True)
