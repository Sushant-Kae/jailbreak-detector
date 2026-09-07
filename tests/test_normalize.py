from src.normalize import normalize_text


def test_lowercases_text():
    assert normalize_text("Hello WORLD") == "hello world"


def test_collapses_whitespace():
    assert normalize_text("hello    world\n\tfoo") == "hello world foo"


def test_strips_leading_trailing_whitespace():
    assert normalize_text("   hello   ") == "hello"


def test_handles_none():
    assert normalize_text(None) == ""


def test_handles_empty_string():
    assert normalize_text("") == ""


def test_handles_whitespace_only():
    assert normalize_text("   \n\t  ") == ""


def test_handles_non_string_input():
    assert normalize_text(123) == "123"
