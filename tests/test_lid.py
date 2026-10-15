"""Hand-checked LID examples. Expected labels were assigned by reading each sentence."""

from src.lid import is_romanized_hindi, tag_sentence
from src.tokens import tokenize


def words(text):
    return [t.text for t in tokenize(text)]


def tags(text, l1, l2):
    toks = tokenize(text)
    langs, _ = tag_sentence(toks, l1, l2)
    short = {l1: l1[:2].upper(), l2: l2[:2].upper()}
    return [f"{t.text}/{short[lang]}" for t, lang in zip(toks, langs)]


# ---- tokenization ----

def test_tokenize_latin_keeps_apostrophes_and_hyphens():
    assert words("It's week-end, right?") == ["It's", "week-end", "right"]


def test_tokenize_han_one_char_per_token():
    assert words("聽日係個big day, 我哋") == ["聽", "日", "係", "個", "big", "day", "我", "哋"]


def test_tokenize_japanese_kana_runs_and_han():
    assert words("新しいメンバー") == ["新", "しい", "メンバー"]


def test_tokenize_devanagari_keeps_vowel_signs():
    assert words("भारत के institutes") == ["भारत", "के", "institutes"]


def test_tokenize_offsets_and_numbers():
    text = "Hola 2024!"
    toks = tokenize(text)
    assert [(t.text, t.start, t.end, t.script) for t in toks] == [("Hola", 0, 4, "latin"), ("2024", 5, 9, "num")]


# ---- language tags ----

def test_script_pair_arabic_tag_switch():
    assert tags("الموضوع معقد شوي، you know?", "Arabic", "English") == [
        "الموضوع/AR", "معقد/AR", "شوي/AR", "you/EN", "know/EN"]


def test_script_pair_cantonese_inserted_word():
    assert tags("呢個IG update好正", "Cantonese", "English") == [
        "呢/CA", "個/CA", "IG/EN", "update/EN", "好/CA", "正/CA"]


def test_freq_pair_spanish_with_ambiguous_word():
    # "no" is frequent in both languages; context keeps it Spanish.
    assert tags("Yo no sé, but I think so", "Spanish", "English") == [
        "Yo/SP", "no/SP", "sé/SP", "but/EN", "I/EN", "think/EN", "so/EN"]


def test_freq_pair_malay():
    assert tags("Saya rasa public should use fragrance macam tu", "Malay", "English") == [
        "Saya/MA", "rasa/MA", "public/EN", "should/EN", "use/EN", "fragrance/EN", "macam/MA", "tu/MA"]


def test_number_takes_left_context():
    assert tags("Das kostet 50 euros, mais c'est cher", "German", "French") == [
        "Das/GE", "kostet/GE", "50/GE", "euros/FR", "mais/FR", "c'est/FR", "cher/FR"]


def test_romanized_hindi_row():
    text = "Logon ko acting bahut pasand aayi, the buzz was amazing"
    toks = tokenize(text)
    assert is_romanized_hindi(toks, "Hindi")
    langs, _ = tag_sentence(toks, "Hindi", "English", romanized=True)
    assert [f"{t.text}/{l[:2]}" for t, l in zip(toks, langs)] == [
        "Logon/Hi", "ko/Hi", "acting/En", "bahut/Hi", "pasand/Hi", "aayi/Hi",
        "the/En", "buzz/En", "was/En", "amazing/En"]


def test_devanagari_row_keeps_script_rule():
    # "main" is romanized Hindi for "I", but in a Devanagari row Latin words are English.
    toks = tokenize("यह main reason है")
    assert not is_romanized_hindi(toks, "Hindi")
    assert tags("यह main reason है", "Hindi", "English") == ["यह/HI", "main/EN", "reason/EN", "है/HI"]
