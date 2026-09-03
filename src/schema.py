"""Request/response schemas — the API contract, validated by pydantic."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, create_model

from .data import FEATURE_COLUMNS

# Build the request model programmatically so the API contract always stays in
# lock-step with FEATURE_COLUMNS. `extra="forbid"` rejects unknown fields so a
# malformed client fails loudly at the edge instead of silently scoring garbage.
#
# Transaction: one transaction's features (V1..V28 PCA components + Amount).
Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid"),
    **{name: (float, Field(..., description=f"Feature {name}")) for name in FEATURE_COLUMNS},
)


class ScoreResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class MetricsResponse(BaseModel):
    predictions_served: int
    flagged_fraud: int
    flag_rate: float
    model_metrics: dict
