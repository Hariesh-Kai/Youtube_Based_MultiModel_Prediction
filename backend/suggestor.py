import json
import os
import urllib.request

from backend.recommender import EMOTION_GENRES


def fallback_suggestion(age: str, gender: str, emotion: str, environment: str, languages: list[str]) -> str:
    primary_lang = languages[0] if languages else "English"
    genres = EMOTION_GENRES.get(emotion, ["trending"])[:2]
    return (
        f"Try a {primary_lang} playlist mixing {genres[0]} and {genres[-1]} for a {age.lower()} "
        f"{gender.lower()} in a {environment.replace('_', ' ')} setting. Start with 3 upbeat tracks, "
        "then transition into calmer songs for balance."
    )


def openai_suggestion(age: str, gender: str, emotion: str, environment: str, languages: list[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    prompt = (
        "You are a music recommendation assistant. Create one concise recommendation paragraph "
        f"based on profile: age={age}, gender={gender}, emotion={emotion}, "
        f"environment={environment}, languages={languages}."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Be concise and practical."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()
