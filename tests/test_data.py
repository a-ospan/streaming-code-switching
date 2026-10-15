import pandas as pd

from src.data import flatten, parse_generation_result


def test_plain_list():
    raw = "['أنا تابعت الخبر، honestly الموضوع معقد.', 'you know?']"
    assert parse_generation_result(raw) == ["أنا تابعت الخبر، honestly الموضوع معقد.", "you know?"]


def test_unparseable_string_is_one_sentence():
    raw = "聽日係個big day, 我哋要準備好迎接新挑戰！"
    assert parse_generation_result(raw) == [raw]


def test_dict_with_instances_skips_topic():
    raw = "{'topic': 'business', 'instances': ['最近好忙。 Can you believe it?', '係呀。']}"
    assert parse_generation_result(raw) == ["最近好忙。 Can you believe it?", "係呀。"]


def test_speaker_message_dicts_and_nested_lists():
    raw = "[{'speaker': 'A', 'message': 'Hola, how are you?'}, ['Bien, thanks.', 'Y tú?']]"
    assert parse_generation_result(raw) == ["Hola, how are you?", "Bien, thanks.", "Y tú?"]


def test_flatten_drops_empty_and_long_and_reports_by_pair():
    df = pd.DataFrame({
        "row_id": [0, 1],
        "first_language": ["Spanish", "Malay"],
        "second_language": ["English", "English"],
        "data_generation_result": ["['Hola, right?', '  ']", "['" + "kata " * 200 + "', 'Saya okay, you know?']"],
    })
    out, report = flatten(df)
    assert out.text.tolist() == ["Hola, right?", "Saya okay, you know?"]
    assert out.pair.tolist() == ["Spanish-English", "Malay-English"]
    assert out.sent_idx.tolist() == [0, 1]
    assert report.dropped_empty.to_dict() == {"Spanish-English": 1}
    assert report.dropped_too_long.to_dict() == {"Malay-English": 1}
