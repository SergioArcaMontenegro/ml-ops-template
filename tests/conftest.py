"""Configuración y fixtures comunes para pytest."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.prediction import ModelBackendName, TabularFeatures
from app.services.circuit_breaker import CircuitBreaker
from app.services.heuristic_backend import HeuristicBackend
from app.services.inference import InferenceService


@pytest.fixture
def valid_features_dict() -> dict[str, Any]:
    return {
        "customer_tenure_months": 12,
        "monthly_charges": 70.0,
        "total_charges": 840.0,
        "num_support_tickets": 2,
        "contract_type_code": 1,
        "has_dependents": True,
        "avg_session_duration_min": 35.5,
    }


@pytest.fixture
def valid_tabular_features(valid_features_dict: dict[str, Any]) -> TabularFeatures:
    return TabularFeatures(**valid_features_dict)


@pytest.fixture
def heuristic_service() -> InferenceService:
    heuristic = HeuristicBackend()
    return InferenceService(
        backends={},
        default_backend=ModelBackendName.HEURISTIC,
        decision_threshold=0.5,
        circuit_breaker=CircuitBreaker(),
        fallback_backend=heuristic,
    )


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client
