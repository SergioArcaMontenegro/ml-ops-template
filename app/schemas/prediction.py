# app/schemas/prediction.py
"""Request/response schemas for tabular inference endpoint."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ModelBackendName(str, Enum):
    """Model backends supported by the service."""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ONNX = "onnx"
    HEURISTIC = "heuristic"


# Contractual feature order: must match the exact column order
# used during offline training.
FEATURE_ORDER: tuple[str, ...] = (
    "customer_tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "contract_type_code",
    "has_dependents",
    "avg_session_duration_min",
)


class TabularFeatures(BaseModel):
    """Input feature vector for the tabular model.

    All fields are required with strict typing: implicit type coercion
    (e.g., numeric string into float) is strictly rejected.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    customer_tenure_months: Annotated[int, Field(ge=0, le=600)]
    monthly_charges: Annotated[float, Field(ge=0.0, le=100_000.0)]
    total_charges: Annotated[float, Field(ge=0.0)]
    num_support_tickets: Annotated[int, Field(ge=0, le=1000)]
    contract_type_code: Annotated[int, Field(ge=0, le=2)]
    has_dependents: bool
    avg_session_duration_min: Annotated[float, Field(ge=0.0, le=1440.0)]

    @model_validator(mode="after")
    def validate_charges_consistency(self) -> TabularFeatures:
        """Business invariant: total_charges cannot be less than monthly_charges
        if tenure is >= 1 month.
        """
        if self.customer_tenure_months >= 1 and self.total_charges < self.monthly_charges:
            raise ValueError(
                "total_charges cannot be less than monthly_charges "
                "when customer_tenure_months >= 1"
            )
        return self

    def to_ordered_tuple(self) -> tuple[float, ...]:
        """Returns values in the exact order expected by the model."""
        raw = self.model_dump()
        return tuple(float(raw[name]) for name in FEATURE_ORDER)


class PredictionRequest(BaseModel):
    """Payload accepted by POST /v1/predict."""

    model_config = ConfigDict(strict=True, extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    features: TabularFeatures
    backend_override: ModelBackendName | None = Field(
        default=None,
        description="Allows overriding backend for A/B testing or benchmarking.",
    )

    @field_validator("request_id", mode="before")
    @classmethod
    def reject_nil_uuid(cls, value: object) -> object:
        if isinstance(value, str):
            if value == "00000000-0000-0000-0000-000000000000":
                raise ValueError("request_id cannot be the nil UUID")
            try:
                return UUID(value)
            except Exception as exc:
                raise ValueError(f"Invalid UUID format: {value}") from exc
        return value


class PredictionResponse(BaseModel):
    """Response returned by the inference endpoint."""

    model_config = ConfigDict(strict=True)

    request_id: UUID
    churn_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    predicted_label: bool
    model_version: str
    backend: ModelBackendName
    inference_latency_ms: Annotated[float, Field(ge=0.0)]
    is_fallback: bool = False
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class BatchPredictionRequest(BaseModel):
    """Payload for batch predictions."""

    model_config = ConfigDict(strict=True, extra="forbid")

    requests: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Response returned for batch predictions."""

    model_config = ConfigDict(strict=True)

    predictions: list[PredictionResponse]
    total_batch_latency_ms: Annotated[float, Field(ge=0.0)]


class ErrorDetail(BaseModel):
    """Uniform error structure for all API error responses."""

    model_config = ConfigDict(strict=True)

    request_id: UUID
    error_code: str
    message: str
    fields: list[dict[str, str]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
