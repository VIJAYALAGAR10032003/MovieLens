"""
FastAPI backend for MovieLens recommender.

Run:
    uvicorn app.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import OUT_DIR, TOP_K
from app.recommender import recommender
from app.schemas import (
    RecommendRequest, RecommendResponse,
    PredictRequest, PredictResponse,
    SimilarResponse, MetricsResponse, HealthResponse,
)

import pandas as pd


app = FastAPI(
    title="MovieLens 25M Recommender API",
    description="Hybrid CF + Content-Based recommendation service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Load models once when API starts."""
    try:
        recommender.load()
        print("[API] Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"[API] WARNING: {e}")
        print("[API] Run `python train.py` before starting the API.")


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------
@app.get("/", tags=["health"])
def root():
    return {"service": "MovieLens Recommender", "status": "ok",
            "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse(
        status="ready" if recommender.loaded else "not_loaded",
        models_loaded={
            "baseline":  recommender.loaded,
            "svd":       recommender.svd_type if recommender.loaded else None,
            "content":   recommender.loaded,
            "hybrid":    recommender.loaded,
        },
        n_users=recommender.n_users if recommender.loaded else 0,
        n_items=recommender.n_items if recommender.loaded else 0,
    )


# ---------------------------------------------------------
# Recommend Top-N
# ---------------------------------------------------------
@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(req: RecommendRequest):
    if not recommender.loaded:
        raise HTTPException(503, "Models not loaded. Run `python train.py` first.")
    items = recommender.recommend(req.user_id, req.top_n, req.model)
    return RecommendResponse(
        user_id=req.user_id,
        model=req.model,
        count=len(items),
        items=items,
    )


# ---------------------------------------------------------
# Predict single rating
# ---------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(req: PredictRequest):
    if not recommender.loaded:
        raise HTTPException(503, "Models not loaded.")
    rating = recommender.predict(req.user_id, req.movie_id, req.model)
    title_row = recommender.movies[recommender.movies["movieId"] == req.movie_id]
    title = title_row["title"].iloc[0] if len(title_row) else None
    return PredictResponse(
        user_id=req.user_id,
        movie_id=req.movie_id,
        model=req.model,
        predicted_rating=round(float(rating), 4),
        title=title,
    )


# ---------------------------------------------------------
# Similar movies
# ---------------------------------------------------------
@app.get("/similar/{movie_id}", response_model=SimilarResponse,
         tags=["similar"])
def similar(movie_id: int, top_n: int = 10):
    if not recommender.loaded:
        raise HTTPException(503, "Models not loaded.")
    items = recommender.similar_movies(movie_id, top_n)
    return SimilarResponse(movie_id=movie_id, count=len(items), items=items)


# ---------------------------------------------------------
# Metrics (from training)
# ---------------------------------------------------------
@app.get("/metrics", response_model=MetricsResponse, tags=["metrics"])
def metrics():
    path = os.path.join(OUT_DIR, "metrics.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "metrics.csv not found. Run `python train.py`.")
    df = pd.read_csv(path)
    return MetricsResponse(metrics=df.to_dict("records"))


# ---------------------------------------------------------
# Serve output charts (optional)
# ---------------------------------------------------------
@app.get("/charts/{filename}", tags=["charts"])
def chart(filename: str):
    path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Chart not found.")
    return FileResponse(path)