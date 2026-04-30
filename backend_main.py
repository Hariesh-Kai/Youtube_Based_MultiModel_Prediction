from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.recommender import build_recommendation
from backend.schemas import RecommendationResponse, SuggestionRequest, SuggestionResponse
from backend.suggestor import fallback_suggestion, openai_suggestion

app = FastAPI(title="Emotion Music API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/suggest", response_model=SuggestionResponse)
def suggest(body: SuggestionRequest):
    try:
        text = openai_suggestion(body.age, body.gender, body.emotion, body.environment, body.languages)
        return SuggestionResponse(suggestion=text, provider="openai")
    except Exception:
        text = fallback_suggestion(body.age, body.gender, body.emotion, body.environment, body.languages)
        return SuggestionResponse(suggestion=text, provider="fallback")


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    image: UploadFile = File(...),
    age: str = Form(...),
    gender: str = Form(...),
    emotion: str = Form(...),
    environment: str = Form(...),
    languages: str = Form(...),
):
    _ = await image.read()
    language_list = [lang.strip() for lang in languages.split(",") if lang.strip()]
    return build_recommendation(age, gender, emotion, environment, language_list)
