"""Alignment and leakage tests with the real tokenizers (skipped if they are not cached)."""

import os

import pytest

from src.labels import IGNORE, make_labels
from src.streaming import encode_sentence
from src.tokens import tokenize

os.environ.setdefault("HF_HUB_OFFLINE", "1")
transformers = pytest.importorskip("transformers")

MODELS = ["FacebookAI/xlm-roberta-base", "facebook/xglm-564M"]


@pytest.fixture(scope="module", params=MODELS)
def tok(request):
    try:
        return transformers.AutoTokenizer.from_pretrained(request.param)
    except OSError:
        pytest.skip(f"{request.param} tokenizer not cached")


def encode(text, langs, tok, **kw):
    toks = tokenize(text)
    assert len(toks) == len(langs)
    y_sw, y_dur = make_labels(langs)
    return toks, encode_sentence(text, [t.start for t in toks], [t.end for t in toks], y_sw, y_dur, tok, **kw)


def test_labels_land_on_last_subword_of_each_prefix(tok):
    text = "Yo no sé, but I think so"
    toks, e = encode(text, ["es", "es", "es", "en", "en", "en", "en"], tok)
    assert e.pred_index[-1] == -1                       # last word has no target
    assert all(k >= 0 for k in e.pred_index[:-1])
    assert e.sw_labels[e.pred_index[2]] == 1            # switch after "sé,"
    assert e.dur_labels[e.pred_index[2]] == 1           # 4-token English segment -> class 1
    assert tok.convert_ids_to_tokens(e.input_ids[e.pred_index[2]]) == ","  # comma after "sé" is in the prefix
    assert sum(lab != IGNORE for lab in e.sw_labels) == len(toks) - 1


@pytest.mark.parametrize("text,langs", [
    ("Yo no sé, but I think so", ["es", "es", "es", "en", "en", "en", "en"]),
    ("呢個IG update好正", ["yue", "yue", "en", "en", "yue", "yue"]),
    ("भारत के institutes को improvement चाहिए", ["hi", "hi", "en", "hi", "en", "hi"]),
])
def test_no_lookahead_in_inputs(tok, text, langs):
    """Changing everything after word t must not change the inputs up to its prediction point."""
    toks, e = encode(text, langs, tok)
    for t, k in enumerate(e.pred_index):
        if k < 0:
            continue
        altered = text[: toks[t + 1].start] + "zzz qqq 完全不同"
        alt_toks, alt = encode(altered, ["x"] * len(tokenize(altered)), tok)
        assert alt.input_ids[: k + 1] == e.input_ids[: k + 1], f"leak at word {t} ({toks[t].text})"


def test_truncation_is_counted(tok):
    text = " ".join(["hola", "world"] * 100)
    _, e = encode(text, ["es", "en"] * 100, tok, max_length=32, check_prefix=False)
    assert e.invalid["truncated"] > 0
    assert len(e.input_ids) == 32


def test_unknown_characters_count_as_content(tok):
    """A vocabulary gap must not be mistaken for a special token (XGLM lacks Cantonese 冇)."""
    text = "你有冇聽過 news"
    toks, e = encode(text, ["yue"] * 5 + ["en"], tok)
    assert e.invalid["truncated"] == 0
    assert e.pred_index[4] >= 0 and e.sw_labels[e.pred_index[4]] == 1  # switch before "news" is kept
