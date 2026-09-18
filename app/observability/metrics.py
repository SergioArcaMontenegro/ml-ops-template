# app/observability/metrics.py
"""Definición canónica de métricas Prometheus para el servicio de inferencia."""

from __future__ import annotations

from typing import Any
from prometheus_client import Counter, Gauge, Histogram

# Métricas HTTP (Capítulo 4)
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total de peticiones HTTP recibidas",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Latencia total de las peticiones HTTP en segundos",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# Métricas de Inferencia Pura (Capítulo 4)
MODEL_INFERENCE_SECONDS = Histogram(
    "model_inference_duration_seconds",
    "Latencia de inferencia pura del modelo (excluyendo serialización/deserialización)",
    ["backend"],
    buckets=(0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1),
)

PREDICTION_PROBABILITY_HISTOGRAM = Histogram(
    "prediction_probability",
    "Distribución de las probabilidades calculadas por el modelo",
    ["backend"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Histograma para distribución de variables críticas (Capítulo 4)
FEATURE_TENURE_HISTOGRAM = Histogram(
    "feature_customer_tenure_months",
    "Distribución del feature customer_tenure_months",
    buckets=(0, 6, 12, 24, 36, 48, 60, 120),
)

FEATURE_CHARGES_HISTOGRAM = Histogram(
    "feature_monthly_charges",
    "Distribución del feature monthly_charges",
    buckets=(0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0),
)

# Métricas de Resiliencia y Fallback (Capítulo 6)
CIRCUIT_STATE_GAUGE = Gauge(
    "circuit_breaker_state",
    "Estado del circuit breaker: 0=CLOSED, 1=HALF_OPEN, 2=OPEN",
)

FALLBACK_PREDICTIONS_TOTAL = Counter(
    "fallback_predictions_total",
    "Total de predicciones servidas por el backend heurístico de fallback",
    ["reason"],
)

# Métricas de Data y Concept Drift (Capítulo 5)
DATASET_DRIFT_DETECTED_GAUGE = Gauge(
    "dataset_drift_detected",
    "Indica si el job de Evidently ha detectado data drift significativo (1=Sí, 0=No)",
)

DATA_DRIFT_SHARE_GAUGE = Gauge(
    "data_drift_share",
    "Fracción de features que presentan drift estadístico respecto a la referencia",
)

CONCEPT_DRIFT_SCORE_GAUGE = Gauge(
    "concept_drift_score",
    "Deterioro en métrica de evaluación (ROC AUC) respecto al baseline histórico",
)


def record_feature_distribution(features: dict[str, Any]) -> None:
    """Registra valores en los histogramas de features con cardinalidad acotada."""
    if "customer_tenure_months" in features:
        FEATURE_TENURE_HISTOGRAM.observe(float(features["customer_tenure_months"]))
    if "monthly_charges" in features:
        FEATURE_CHARGES_HISTOGRAM.observe(float(features["monthly_charges"]))
