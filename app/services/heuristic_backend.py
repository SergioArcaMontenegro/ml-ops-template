# app/services/heuristic_backend.py
"""Backend de fallback basado en una regla de negocio simple, sin ML.

Se activa cuando el circuit breaker abre el circuito hacia el backend
primario o cuando el drift supera el umbral crítico (sección 6.6).
No debe depender de ningún artefacto externo ni librería con estado
propio: su única responsabilidad es no fallar nunca.
"""

from __future__ import annotations

import math

from app.schemas.prediction import FEATURE_ORDER

# Pesos fijos, calibrados manualmente por el equipo de negocio a partir
# de reglas históricas conocidas ANTES de que existiera el modelo de ML.
# Se versionan en código, no en un artefacto de modelo, precisamente
# para que este backend no comparta ningún punto de fallo con el
# pipeline de entrenamiento/despliegue del modelo principal.
_HEURISTIC_WEIGHTS: dict[str, float] = {
    "customer_tenure_months": -0.01,
    "monthly_charges": 0.004,
    "total_charges": -0.00002,
    "num_support_tickets": 0.08,
    "contract_type_code": -0.15,
    "has_dependents": -0.05,
    "avg_session_duration_min": -0.001,
}
_HEURISTIC_BIAS = 0.35


def _sigmoid(x: float) -> float:
    # Implementación manual con math.exp de la librería estándar (cero dependencias externas)
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


class HeuristicBackend:
    """Backend de reglas fijas, usado como fallback de última instancia."""

    version = "heuristic-fallback-v1"

    def predict_proba(self, features: tuple[float, ...]) -> float:
        linear_combination = _HEURISTIC_BIAS
        for feature_name, value in zip(FEATURE_ORDER, features, strict=True):
            linear_combination += _HEURISTIC_WEIGHTS[feature_name] * value

        probability = _sigmoid(linear_combination)
        return max(0.0, min(1.0, probability))
