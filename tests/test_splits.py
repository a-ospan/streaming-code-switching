import pandas as pd

from src.splits import assign_splits, duplicate_groups


def make_sentences(n_rows_per_pair=200):
    lines = []
    for pair in ["A-English", "B-English"]:
        for r in range(n_rows_per_pair):
            row_id = len(lines) // 2
            lines += [dict(row_id=row_id, pair=pair, text=f"{pair} row {row_id} s{i}") for i in range(2)]
    return pd.DataFrame(lines)


def test_duplicate_text_links_rows_transitively():
    df = pd.DataFrame(dict(row_id=[0, 1, 1, 2, 3], text=["x", "x", "y", "y", "z"]))
    g = duplicate_groups(df)
    assert g[0] == g[1] == g[2]
    assert g[3] != g[0]


def test_splits_are_stratified_and_deterministic():
    df = make_sentences()
    a = assign_splits(df, "pair", seed=13)
    b = assign_splits(df, "pair", seed=13)
    assert a.equals(b)
    shares = a.groupby("pair").split.value_counts(normalize=True).unstack()
    assert (shares["train"].between(0.78, 0.82)).all()
    assert (shares["test"].between(0.08, 0.12)).all()


def test_duplicates_never_cross_splits():
    df = make_sentences()
    dup = df.iloc[[0]].assign(row_id=399, text=df.text.iloc[0])  # row 399 repeats row 0's sentence
    df = pd.concat([df[df.row_id != 399], dup, df[df.row_id == 399].iloc[1:]])
    s = assign_splits(df, "pair", seed=13).set_index("row_id").split
    assert s[0] == s[399]
