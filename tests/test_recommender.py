from backend.recommender import build_recommendation


def test_build_recommendation_returns_six_songs():
    result = build_recommendation("Adult", "Female", "happiness", "library", ["English"])
    assert result.recommended_genre == "pop"
    assert len(result.songs) == 6
