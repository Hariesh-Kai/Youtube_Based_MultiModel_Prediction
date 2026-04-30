from backend.suggestor import fallback_suggestion


def test_fallback_suggestion_contains_context():
    text = fallback_suggestion("Adult", "Male", "neutrality", "office", ["English"])
    assert "office" in text
    assert "English" in text
