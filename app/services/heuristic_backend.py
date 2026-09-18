# app/services/heuristic_backend.py
"""Fallback backend based on simple business heuristics, without ML.

Activated when the circuit breaker trips open or when concept drift
exceeds critical thresholds. Has zero external dependencies: its sole
responsibility is to never fail.
"""

from __future__ import annotations

import math

from app.schemas.prediction import FEATURE_ORDER

# Static heuristic weights calibrated by business domain knowledge
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
    # Pure Python implementation with math.exp from standard library (zero external dependencies)
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


class HeuristicBackend:
    """Fixed-rule fallback backend used as a last line of defense."""

    version = "heuristic-fallback-v1"

    def predict_proba(self, features: tuple[float, ...]) -> float:
        linear_combination = _HEURISTIC_BIAS
        for feature_name, value in zip(FEATURE_ORDER, features, strict=True):
            linear_combination += _HEURISTIC_WEIGHTS[feature_name] * value

        probability = _sigmoid(linear_combination)
        return max(0.0, min(1.0, probability))
