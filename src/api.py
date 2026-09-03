"""FastAPI inference service.

Endpoints:
  GET  /health   liveness + whether the model is loaded (for load balancers)
  GET  /metrics  serving stats + frozen training metrics (for monitoring)
  POST /score    score one transaction
  POST /score/batch  score many

Run locally:  uvicorn src.api:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .config import settings
from . import db
from .schema import (HealthResponse, MetricsResponse, ScoreResponse, Transaction)

STATE: dict = {"artifact": None}


def _load_model() -> None:
    try:
        STATE["artifact"] = joblib.load(settings.model_path)
    except FileNotFoundError:
        STATE["artifact"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    _load_model()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


def _require_model() -> dict:
    art = STATE["artifact"]
    if art is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python -m src.train`.")
    return art


def _score_frame(art: dict, rows: list[dict]) -> list[ScoreResponse]:
    cols = art["feature_columns"]
    threshold = art["threshold"]
    version = art["trained_at"]
    X = pd.DataFrame(rows)[cols]
    probs = art["pipeline"].predict_proba(X.values)[:, 1]
    out = []
    for row, p in zip(rows, probs):
        is_fraud = bool(p >= threshold)
        if settings.log_predictions:
            db.log_prediction(row, float(p), is_fraud, threshold, version)
        out.append(ScoreResponse(
            fraud_probability=round(float(p), 6),
            is_fraud=is_fraud,
            threshold=threshold,
            model_version=version,
        ))
    return out


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    art = STATE["artifact"]
    return HealthResponse(
        status="ok",
        model_loaded=art is not None,
        model_version=art["trained_at"] if art else None,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    art = _require_model()
    s = db.summary()
    return MetricsResponse(**s, model_metrics=art["metrics"])


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    art = _require_model()
    return _score_frame(art, [txn.model_dump()])[0]


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(txns: list[Transaction]) -> list[ScoreResponse]:
    art = _require_model()
    if not txns:
        raise HTTPException(status_code=400, detail="Empty batch.")
    return _score_frame(art, [t.model_dump() for t in txns])
