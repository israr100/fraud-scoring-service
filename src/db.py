"""Prediction logging via SQLAlchemy.

Every scored request is written to a `predictions` table. This is what makes it
a *system* rather than a demo: the logged scores are the raw material for
monitoring, drift detection, and offline evaluation once labels arrive.

Runs on SQLite locally and Postgres in the cloud with only FRAUD_DATABASE_URL
changing.
"""
from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import (Boolean, DateTime, Float, Integer, String, Text,
                        create_engine, func, select)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .config import settings

_engine = create_engine(settings.database_url, future=True)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    fraud_probability: Mapped[float] = mapped_column(Float)
    is_fraud: Mapped[bool] = mapped_column(Boolean, index=True)
    threshold: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64))
    features: Mapped[str] = mapped_column(Text)  # JSON snapshot for audit/drift


def init_db() -> None:
    Base.metadata.create_all(_engine)


def log_prediction(features: dict, prob: float, is_fraud: bool, threshold: float,
                   model_version: str) -> None:
    with Session(_engine) as s:
        s.add(Prediction(
            fraud_probability=prob,
            is_fraud=is_fraud,
            threshold=threshold,
            model_version=model_version,
            features=json.dumps(features),
        ))
        s.commit()


def summary() -> dict:
    with Session(_engine) as s:
        total = s.scalar(select(func.count(Prediction.id))) or 0
        flagged = s.scalar(
            select(func.count(Prediction.id)).where(Prediction.is_fraud.is_(True))) or 0
    return {
        "predictions_served": int(total),
        "flagged_fraud": int(flagged),
        "flag_rate": round(flagged / total, 4) if total else 0.0,
    }
