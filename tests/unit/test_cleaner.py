from bidmate_rag.preprocessing.cleaner import clean_text


def test_clean_text_removes_known_noise_and_preserves_table_spacing() -> None:
    raw = """Warning: Cannot access the `require` function
Warning: Cannot polyfill `DOMMatrix`
문장  하나<br>둘
| 중복 | 중복 | 중복 |
| keep  two | cell |
\ue000깨짐\u00a0문자



끝"""

    cleaned = clean_text(raw)

    assert "Warning:" not in cleaned
    assert "<br>" not in cleaned
    assert "문장 하나\n둘" in cleaned
    assert "| 중복 |" in cleaned
    assert "| keep  two | cell |" in cleaned
    assert "\ue000" not in cleaned
    assert "\u00a0" not in cleaned
    assert "\n\n\n" not in cleaned
