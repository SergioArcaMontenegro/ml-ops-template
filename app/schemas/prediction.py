# app/schemas/prediction.py
"""Esquemas de request/response para el endpoint de inferencia tabular."""

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
    """Backends de modelo soportados por el servicio."""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ONNX = "onnx"
    HEURISTIC = "heuristic"


# El orden de esta lista es contractual: debe coincidir exactamente
# con el orden de columnas usado durante el entrenamiento offline.
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
    """Vector de features de entrada para el modelo tabular.

    Todos los campos son obligatorios y de tipo estricto: no se acepta
    coerción implícita (p. ej. un string numérico para un campo float).
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
        """Invariante de negocio: total_charges no puede ser menor que
        un único mes de monthly_charges si la antigüedad es >= 1 mes.
        """
        if self.customer_tenure_months >= 1 and self.total_charges < self.monthly_charges:
            raise ValueError(
                "total_charges no puede ser menor que monthly_charges "
                "cuando customer_tenure_months >= 1"
            )
        return self

    def to_ordered_tuple(self) -> tuple[float, ...]:
        """Devuelve los valores en el orden exacto esperado por el modelo."""
        raw = self.model_dump()
        return tuple(float(raw[name]) for name in FEATURE_ORDER)


class PredictionRequest(BaseModel):
    """Payload aceptado por POST /v1/predict."""

    model_config = ConfigDict(strict=True, extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    features: TabularFeatures
    backend_override: ModelBackendName | None = Field(
        default=None,
        description="Permite forzar un backend específico en A/B testing o benchmarking.",
    )

    @field_validator("request_id", mode="before")
    @classmethod
    def reject_nil_uuid(cls, value: object) -> object:
        if isinstance(value, str):
            if value == "00000000-0000-0000-0000-000000000000":
                raise ValueError("request_id no puede ser el UUID nulo")
            try:
                return UUID(value)
            except Exception as exc:
                raise ValueError(f"Formato de UUID inválido: {value}") from exc
        return value


class PredictionResponse(BaseModel):
    """Respuesta del endpoint de inferencia."""

    model_config = ConfigDict(strict=True)

    request_id: UUID
    churn_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    predicted_label: bool
    model_version: str
    backend: ModelBackendName
    inference_latency_ms: Annotated[float, Field(ge=0.0)]
    is_fallback: bool = False
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchPredictionRequest(BaseModel):
    """Payload para inferencia en batch."""

    model_config = ConfigDict(strict=True, extra="forbid")

    requests: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Respuesta para inferencia en batch."""

    model_config = ConfigDict(strict=True)

    predictions: list[PredictionResponse]
    total_batch_latency_ms: Annotated[float, Field(ge=0.0)]


class ErrorDetail(BaseModel):
    """Estructura homogénea para todas las respuestas de error de la API."""

    model_config = ConfigDict(strict=True)

    request_id: UUID
    error_code: str
    message: str
    fields: list[dict[str, str]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
