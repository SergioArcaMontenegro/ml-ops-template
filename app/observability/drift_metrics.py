# app/observability/drift_metrics.py
"""Exportador de métricas de data drift y concept drift a Prometheus y Pushgateway."""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from app.observability.metrics import (
    CONCEPT_DRIFT_SCORE_GAUGE,
    DATA_DRIFT_SHARE_GAUGE,
    DATASET_DRIFT_DETECTED_GAUGE,
)

logger = logging.getLogger("mlops.drift")


def update_drift_gauges(
    dataset_drift: bool, drift_share: float, concept_drift: float = 0.0
) -> None:
    """Actualiza los gauges locales de Prometheus en memoria."""
    DATASET_DRIFT_DETECTED_GAUGE.set(1.0 if dataset_drift else 0.0)
    DATA_DRIFT_SHARE_GAUGE.set(float(drift_share))
    CONCEPT_DRIFT_SCORE_GAUGE.set(float(concept_drift))


def push_drift_metrics_to_gateway(
    gateway_url: str,
    job_name: str,
    dataset_drift: bool,
    drift_share: float,
    concept_drift: float = 0.0,
) -> None:
    """Empuja métricas calculadas en jobs batch externos hacia Prometheus Pushgateway."""
    registry = CollectorRegistry()

    g_detected = Gauge(
        "dataset_drift_detected", "Drift detectado a nivel dataset", registry=registry
    )
    g_share = Gauge("data_drift_share", "Proporción de features con drift", registry=registry)
    g_concept = Gauge(
        "concept_drift_score", "Pérdida de AUC respecto a baseline", registry=registry
    )

    g_detected.set(1.0 if dataset_drift else 0.0)
    g_share.set(float(drift_share))
    g_concept.set(float(concept_drift))

    try:
        push_to_gateway(gateway_url, job=job_name, registry=registry)
        logger.info("Métricas de drift enviadas con éxito a Pushgateway (%s)", gateway_url)
    except Exception as exc:
        logger.warning("No se pudo enviar métricas a Pushgateway (%s): %s", gateway_url, exc)
