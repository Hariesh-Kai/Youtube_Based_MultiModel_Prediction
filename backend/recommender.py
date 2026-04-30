from backend.schemas import RecommendationResponse, Song

EMOTION_GENRES = {
    "angry": ["rock", "metal", "energetic"],
    "disgust": ["calm", "ambient", "relaxing"],
    "fear": ["uplifting", "motivational", "soft"],
    "happiness": ["pop", "dance", "party"],
    "neutrality": ["acoustic", "lofi", "chill"],
    "sadness": ["melancholic", "slow", "soft"],
    "surprise": ["upbeat", "electronic", "exciting"],
}


def build_recommendation(age: str, gender: str, emotion: str, environment: str, languages: list[str]) -> RecommendationResponse:
    genres = EMOTION_GENRES.get(emotion, ["trending"])
    recommended_genre = genres[0]
    primary_language = languages[0] if languages else "English"
    songs = [
        Song(
            title=f"{primary_language.title()} {recommended_genre.title()} Track {i}",
            reason=f"Matched for {age} {gender} in {environment}",
        )
        for i in range(1, 7)
    ]
    return RecommendationResponse(
        age=age,
        gender=gender,
        emotion=emotion,
        environment=environment,
        recommended_genre=recommended_genre,
        songs=songs,
    )
