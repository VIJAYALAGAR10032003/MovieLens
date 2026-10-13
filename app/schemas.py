"""Pydantic request / response schemas."""
from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="MovieLens user ID")
    top_n:   int = Field(10, ge=1, le=50)
    model:   str = Field("hybrid", description="baseline|svd|content|hybrid")


class RecommendItem(BaseModel):
    movieId: int
    title:   Optional[str] = None
    genres:  Optional[str] = None
    predicted_rating: float


class RecommendResponse(BaseModel):
    user_id:  int
    model:    str
    count:    int
    items:    List[RecommendItem]


class PredictRequest(BaseModel):
    user_id:  int = Field(..., gt=0)
    movie_id: int = Field(..., gt=0)
    model:    str = Field("hybrid")


class PredictResponse(BaseModel):
    user_id:          int
    movie_id:         int
    model:            str
    predicted_rating: float
    title:            Optional[str] = None


class SimilarResponse(BaseModel):
    movie_id: int
    count:    int
    items:    List[RecommendItem]


class MetricsResponse(BaseModel):
    metrics: List[dict]


class HealthResponse(BaseModel):
    status:       str
    models_loaded: dict
    n_users:      int
    n_items:      int