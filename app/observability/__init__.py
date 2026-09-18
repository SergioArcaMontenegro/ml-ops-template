"""Módulo de observabilidad, métricas Prometheus y detección de drift."""

from app.observability.metrics import (
    CIRCUIT_STATE_GAUGE,
    FALLBACK_PREDICTIONS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    MODEL_INFERENCE_SECONDS,
    PREDICTION_PROBABILITY_HISTOGRAM,
    record_feature_distribution,
)

__all__ = [
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION_SECONDS",
    "MODEL_INFERENCE_SECONDS",
    "PREDICTION_PROBABILITY_HISTOGRAM",
    "CIRCUIT_STATE_GAUGE",
    "FALLBACK_PREDICTIONS_TOTAL",
    "record_feature_distribution",
]
