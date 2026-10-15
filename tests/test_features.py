from src.features import causal_tags, sentence_features
from src.labels import make_labels
from src.lid import tag_sentence
from src.tokens import tokenize


def feats(text, l1, l2):
    toks = tokenize(text)
    langs, sources = tag_sentence(toks, l1, l2)
    y_sw, _ = make_labels(langs)
    return toks, sentence_features(text, [t.start for t in toks], [t.end for t in toks], [t.script for t in toks],
                                   langs, sources, y_sw, l1, l2)


def test_causal_tags_never_fill_from_the_right():
    # token 1 ("no") is undecided; the label fills it from the left, a prefix ending at it must too
    langs = ["Spanish", "Spanish", "English"]
    sources = ["freq", "context", "freq"]
    assert causal_tags(langs, sources, "Spanish")[1] == ["Spanish", "Spanish"]
    # a leading undecided token takes the first decided language seen so far, never a later one
    tags = causal_tags(["English", "English"], ["context", "freq"], "Spanish")
    assert tags[0] == ["Spanish"]                 # nothing decided yet -> the row's first language
    assert tags[1] == ["English", "English"]


def test_features_do_not_depend_on_the_future():
    text = "Yo no sé, but I think so"
    toks, base = feats(text, "Spanish", "English")
    for t in range(len(toks) - 1):
        altered = text[: toks[t + 1].start] + "completamente distinto aquí ahora"
        _, alt = feats(altered, "Spanish", "English")
        assert alt[t] == base[t], f"feature at word {t} changed when only later words changed"


def test_hand_checked_values():
    _, f = feats("Yo no sé, but I think so", "Spanish", "English")
    # at "sé,": Spanish, run of 3, no switches yet, position 2, comma typed, Latin script
    assert f[2] == (0, 2, 0, 2, 1, 0, 0, 0)
    # at "but": English, run 1, one switch so far, position 3 (bucket 3-5)
    assert f[3] == (1, 0, 1, 3, 0, 0, 0, 0)
