# app/main.py
"""FastAPI production inference microservice entrypoint."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import BackgroundTasks, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import settings
from app.exceptions.handlers import (
    InferenceServiceError,
    inference_service_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.middleware.logging import StructuredLoggingMiddleware, configure_json_logging
from app.middleware.metrics import MetricsMiddleware
from app.observability.drift_gate import periodic_drift_gate
from app.observability.feature_logger import record_features
from app.observability.streaming_drift import (
    initialize_reference,
    periodic_streaming_drift_check,
    record_to_window,
)
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelBackendName,
    PredictionRequest,
    PredictionResponse,
)
from app.services.backends import ModelBackend, XGBoostBackend
from app.services.circuit_breaker import CircuitBreaker
from app.services.heuristic_backend import HeuristicBackend
from app.services.inference import InferenceService
from app.services.onnx_backend import ONNXBackend

configure_json_logging()
logger = logging.getLogger("mlops.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manages service lifecycle: initializes backends, circuit breaker, and drift tasks."""
    logger.info("Initializing MLOps inference service...")

    # 1. Instantiate available backends
    heuristic = HeuristicBackend()
    backends: dict[ModelBackendName, ModelBackend] = {
        ModelBackendName.HEURISTIC: heuristic,
    }

    # Attempt to load ONNX backend
    if settings.model_path.exists():
        try:
            backends[ModelBackendName.ONNX] = ONNXBackend(
                model_path=settings.model_path,
                version=settings.model_version,
            )
            logger.info("ONNX backend initialized from %s", settings.model_path)
        except Exception as exc:
            logger.warning("Failed to load ONNX backend: %s", exc)

    # Attempt to load XGBoost backend
    if settings.xgboost_model_path.exists():
        try:
            backends[ModelBackendName.XGBOOST] = XGBoostBackend(
                model_path=settings.xgboost_model_path,
                version=settings.model_version,
            )
            logger.info("XGBoost backend initialized from %s", settings.xgboost_model_path)
        except Exception as exc:
            logger.warning("Failed to load XGBoost backend: %s", exc)

    # Default backend
    default_backend_name = ModelBackendName.HEURISTIC
    if settings.model_backend in [b.value for b in ModelBackendName]:
        requested_name = ModelBackendName(settings.model_backend)
        if requested_name in backends:
            default_backend_name = requested_name

    circuit_breaker = CircuitBreaker(
        failure_threshold=settings.circuit_breaker_failure_threshold,
        recovery_timeout_seconds=settings.circuit_breaker_recovery_timeout_sec,
    )

    inference_service = InferenceService(
        backends=backends,
        default_backend=default_backend_name,
        decision_threshold=settings.decision_threshold,
        circuit_breaker=circuit_breaker,
        fallback_backend=heuristic,
    )
    app.state.inference_service = inference_service

    # 2. Initialize drift reference dataset
    initialize_reference(settings.reference_data_path)

    # 3. Launch background monitoring tasks
    drift_check_task = asyncio.create_task(periodic_streaming_drift_check())
    drift_gate_task = asyncio.create_task(
        periodic_drift_gate(circuit_breaker, settings.drift_critical_threshold)
    )

    yield

    logger.info("Stopping background tasks...")
    drift_check_task.cancel()
    drift_gate_task.cancel()
    await asyncio.gather(drift_check_task, drift_gate_task, return_exceptions=True)
    logger.info("Service shutdown cleanly.")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ASGI middleware stack
app.add_middleware(MetricsMiddleware)
app.add_middleware(StructuredLoggingMiddleware)

# Exception handlers
app.add_exception_handler(RequestValidationError, cast(Any, validation_exception_handler))
app.add_exception_handler(InferenceServiceError, cast(Any, inference_service_exception_handler))
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.post("/v1/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(
    request: Request,
    payload: PredictionRequest,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    """Primary tabular inference endpoint."""
    request.state.request_id = payload.request_id
    service: InferenceService = request.app.state.inference_service
    response = service.predict(payload)

    # Asynchronous background logging for data drift & audit
    background_tasks.add_task(
        record_features,
        payload.features,
        response.churn_probability,
    )
    record_to_window(payload.features)

    # Shadow Traffic support (Chapter 9)
    if settings.shadow_mode or request.headers.get(settings.shadow_header_name) == "true":
        logger.info(
            "Shadow mode request processed",
            extra={"shadow": True, "request_id": str(payload.request_id)},
        )

    return response


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    payload: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
) -> BatchPredictionResponse:
    """Batch prediction endpoint."""
    service: InferenceService = request.app.state.inference_service
    t0 = time.perf_counter()
    responses: list[PredictionResponse] = []

    for req in payload.requests:
        res = service.predict(req)
        responses.append(res)
        background_tasks.add_task(record_features, req.features, res.churn_probability)
        record_to_window(req.features)

    total_batch_latency_ms = (time.perf_counter() - t0) * 1000.0
    return BatchPredictionResponse(
        predictions=responses,
        total_batch_latency_ms=round(total_batch_latency_ms, 3),
    )


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz() -> dict[str, str]:
    """Basic health check endpoint for Docker/Kubernetes liveness."""
    return {"status": "healthy"}


@app.get("/ready", status_code=status.HTTP_200_OK)
async def ready(request: Request) -> JSONResponse:
    """Readiness probe checking inference service availability."""
    service: InferenceService | None = getattr(request.app.state, "inference_service", None)
    if not service:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "service_not_initialized"},
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})


@app.get("/metrics")
async def metrics() -> Response:
    """Exposes metrics in standard Prometheus format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
