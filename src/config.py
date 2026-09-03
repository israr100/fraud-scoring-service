"""Central configuration, read from environment variables.

Twelve-factor style: every deployable knob is an env var with a sane default,
so the same image runs locally (SQLite) and on AWS (Postgres) with no code change.
"""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_", env_file=".env", extra="ignore")

    # Model artifact
    model_path: str = str(MODELS_DIR / "model.joblib")

    # Database: SQLite locally, Postgres in the cloud.
    # e.g. FRAUD_DATABASE_URL=postgresql+psycopg://user:pass@host:5432/fraud
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'predictions.db'}"

    # Decision threshold. Tuned on validation data for cost-weighted F-beta,
    # NOT the naive 0.5 — see train.py.
    decision_threshold: float = 0.5

    # Service metadata
    app_name: str = "fraud-scoring-service"
    log_predictions: bool = True


settings = Settings()
