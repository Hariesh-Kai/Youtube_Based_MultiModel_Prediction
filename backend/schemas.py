from typing import List
from pydantic import BaseModel


class Song(BaseModel):
    title: str
    reason: str


class RecommendationResponse(BaseModel):
    age: str
    gender: str
    emotion: str
    environment: str
    recommended_genre: str
    songs: List[Song]


class SuggestionRequest(BaseModel):
    age: str
    gender: str
    emotion: str
    environment: str
    languages: List[str]


class SuggestionResponse(BaseModel):
    suggestion: str
    provider: str
