# app/observability/drift_gate.py
"""Consulta periódica de las métricas de drift para forzar el fallback
automático cuando el concept drift supera un umbral crítico de negocio.
"""

from __future__ import annotations

import asyncio
import logging

from app.observability.metrics import CONCEPT_DRIFT_SCORE_GAUGE
from app.services.circuit_breaker import CircuitBreaker

logger = logging.getLogger("mlops.drift_gate")

_CHECK_INTERVAL_SECONDS = 30


async def periodic_drift_gate(
    circuit_breaker: CircuitBreaker, critical_threshold: float = 0.08
) -> None:
    """Fuerza la apertura manual del circuito si el último concept drift
    reportado supera el umbral crítico, garantizando degradación controlada.
    """
    while True:
        await asyncio.sleep(_CHECK_INTERVAL_SECONDS)

        try:
            # Leer el valor del Gauge localmente
            current_value = CONCEPT_DRIFT_SCORE_GAUGE._value.get()  # noqa: SLF001
            if current_value is not None and current_value > critical_threshold:
                logger.critical(
                    "Concept drift crítico detectado (%.4f > %.4f). Forzando apertura del circuit breaker.",
                    current_value,
                    critical_threshold,
                )
                circuit_breaker.force_open()
        except Exception as exc:
            logger.debug("Error en chequeo de drift gate: %s", exc)
