"""Hand-checked label examples. Tags come from reading each sentence; labels were worked out by hand."""

from src.labels import IGNORE as X, duration_class, make_labels


def test_duration_bins():
    assert [duration_class(n) for n in range(1, 10)] == [0, 0, 1, 1, 1, 1, 2, 2, 2]


def test_spanish_english_single_switch():
    # Yo no sé , but I think so  ->  switch after "sé" into a 4-token English segment
    langs = ["es", "es", "es", "en", "en", "en", "en"]
    y_sw, y_dur = make_labels(langs)
    assert y_sw == [0, 0, 1, 0, 0, 0, X]
    assert y_dur == [X, X, 1, X, X, X, X]


def test_arabic_tag_switch_at_end():
    # الموضوع معقد شوي، you know?  ->  2-token tag at the end
    y_sw, y_dur = make_labels(["ar", "ar", "ar", "en", "en"])
    assert y_sw == [0, 0, 1, 0, X]
    assert y_dur == [X, X, 0, X, X]


def test_cantonese_insertion_switches_out_and_back():
    # 呢 個 IG update 好 正  ->  into English (2 tokens), back to Cantonese (2 tokens)
    y_sw, y_dur = make_labels(["yue", "yue", "en", "en", "yue", "yue"])
    assert y_sw == [0, 1, 0, 1, 0, X]
    assert y_dur == [X, 0, X, 0, X, X]


def test_long_clausal_switch():
    # Spanish word followed by an 8-token English clause
    y_sw, y_dur = make_labels(["es"] + ["en"] * 8)
    assert y_sw[0] == 1 and y_dur[0] == 2
    assert y_sw[-1] == X


def test_single_token_sentence_has_no_targets():
    assert make_labels(["en"]) == ([X], [X])


def test_last_token_never_labeled():
    y_sw, y_dur = make_labels(["de", "fr"])
    assert y_sw == [1, X] and y_dur == [0, X]
