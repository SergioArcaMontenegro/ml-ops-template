"""Tests unitarios para la capa de servicio de inferencia desacoplada."""

from uuid import uuid4

from app.schemas.prediction import ModelBackendName, PredictionRequest, TabularFeatures
from app.services.heuristic_backend import HeuristicBackend
from app.services.inference import InferenceService


def test_heuristic_backend_execution(valid_tabular_features: TabularFeatures) -> None:
    backend = HeuristicBackend()
    vector = valid_tabular_features.to_ordered_tuple()
    prob = backend.predict_proba(vector)
    assert 0.0 <= prob <= 1.0
    assert backend.version == "heuristic-fallback-v1"


def test_inference_service_predict(
    heuristic_service: InferenceService, valid_tabular_features: TabularFeatures
) -> None:
    req = PredictionRequest(request_id=uuid4(), features=valid_tabular_features)
    resp = heuristic_service.predict(req)

    assert resp.request_id == req.request_id
    assert 0.0 <= resp.churn_probability <= 1.0
    assert isinstance(resp.predicted_label, bool)
    assert resp.backend == ModelBackendName.HEURISTIC
    assert resp.inference_latency_ms >= 0.0
    assert resp.is_fallback is True
