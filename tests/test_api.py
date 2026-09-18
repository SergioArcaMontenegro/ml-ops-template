"""Tests de integración para la API HTTP de FastAPI."""

from uuid import uuid4
from fastapi.testclient import TestClient


def test_healthz_endpoint(client: TestClient) -> None:
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}


def test_ready_endpoint(client: TestClient) -> None:
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


def test_metrics_endpoint(client: TestClient) -> None:
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "http_requests_total" in res.text


def test_predict_endpoint_success(client: TestClient, valid_features_dict: dict) -> None:
    payload = {
        "request_id": str(uuid4()),
        "features": valid_features_dict,
    }
    res = client.post("/v1/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "churn_probability" in data
    assert "predicted_label" in data
    assert "is_fallback" in data
    assert res.headers.get("x-request-id") is not None


def test_predict_endpoint_validation_error(client: TestClient, valid_features_dict: dict) -> None:
    bad_payload = {
        "request_id": str(uuid4()),
        "features": {
            **valid_features_dict,
            "customer_tenure_months": "doce",  # Tipo inválido
        },
    }
    res = client.post("/v1/predict", json=bad_payload)
    assert res.status_code == 422
    data = res.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert len(data["fields"]) > 0
