# app/services/inference.py
"""Servicio desacoplado de inferencia con soporte multi-backend,
circuit breaker y degradación controlada a fallback heurístico.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter

from app.observability.metrics import (
    CIRCUIT_STATE_GAUGE,
    FALLBACK_PREDICTIONS_TOTAL,
    MODEL_INFERENCE_SECONDS,
    PREDICTION_PROBABILITY_HISTOGRAM,
    record_feature_distribution,
)
from app.schemas.prediction import (
    FEATURE_ORDER,
    ModelBackendName,
    PredictionRequest,
    PredictionResponse,
)
from app.services.backends import ModelBackend
from app.services.circuit_breaker import CircuitBreaker, CircuitState
from app.services.heuristic_backend import HeuristicBackend

logger = logging.getLogger("mlops.inference")


@dataclass(slots=True)
class InferenceService:
    backends: dict[ModelBackendName, ModelBackend]
    default_backend: ModelBackendName
    decision_threshold: float = 0.5
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    fallback_backend: ModelBackend = field(default_factory=HeuristicBackend)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        raw_backend = request.backend_override or self.default_backend
        backend_name = raw_backend if isinstance(raw_backend, ModelBackendName) else ModelBackendName(str(raw_backend))
        backend_label = backend_name.value
        
        feature_dict = request.features.model_dump()
        record_feature_distribution(feature_dict)
        feature_vector = request.features.to_ordered_tuple()

        # Actualizar métrica gauge del circuit breaker (0=closed, 1=half_open, 2=open)
        state = self.circuit_breaker.state
        CIRCUIT_STATE_GAUGE.set(0 if state == CircuitState.CLOSED else (1 if state == CircuitState.HALF_OPEN else 2))

        is_fallback = False
        active_backend = self.backends.get(backend_name)

        # Si el backend solicitado no existe o el circuit breaker no permite llamadas al primario:
        if not active_backend or not self.circuit_breaker.allow_request_to_primary():
            is_fallback = True
            active_backend = self.fallback_backend
            backend_label = "heuristic"
            FALLBACK_PREDICTIONS_TOTAL.labels(reason="circuit_open_or_missing_backend").inc()

        t0 = perf_counter()
        try:
            probability = active_backend.predict_proba(feature_vector)
            if not is_fallback:
                self.circuit_breaker.record_success()
        except Exception as exc:
            logger.error(
                "Fallo en backend primario %s: %s. Ejecutando fallback heurístico.",
                backend_label,
                exc,
                extra={"request_id": str(request.request_id)},
            )
            self.circuit_breaker.record_failure()
            is_fallback = True
            active_backend = self.fallback_backend
            backend_label = "heuristic"
            FALLBACK_PREDICTIONS_TOTAL.labels(reason="primary_backend_exception").inc()
            probability = self.fallback_backend.predict_proba(feature_vector)

        latency_ms = (perf_counter() - t0) * 1000.0

        # Registrar métricas Prometheus
        MODEL_INFERENCE_SECONDS.labels(backend=backend_label).observe(latency_ms / 1000.0)
        PREDICTION_PROBABILITY_HISTOGRAM.labels(backend=backend_label).observe(probability)

        predicted_label = probability >= self.decision_threshold

        return PredictionResponse(
            request_id=request.request_id,
            churn_probability=round(probability, 4),
            predicted_label=predicted_label,
            model_version=active_backend.version,
            backend=ModelBackendName(backend_label),
            inference_latency_ms=round(latency_ms, 3),
            is_fallback=is_fallback,
        )
