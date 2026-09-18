# app/exceptions/handlers.py
"""Manejadores de excepción custom para respuestas de error homogéneas."""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.prediction import ErrorDetail


class InferenceServiceError(Exception):
    """Excepción base para errores de la capa de servicio de inferencia."""


def _get_request_id(request: Request) -> UUID:
    raw = getattr(request.state, "request_id", None)
    if isinstance(raw, UUID):
        return raw
    if isinstance(raw, str):
        try:
            return UUID(raw)
        except Exception:
            pass
    return uuid4()


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Traduce errores de validación de Pydantic a un contrato de error estable."""
    request_id = _get_request_id(request)

    fields = [
        {
            "location": ".".join(str(part) for part in error["loc"]),
            "issue": str(error["msg"]),
        }
        for error in exc.errors()
    ]

    detail = ErrorDetail(
        request_id=request_id,
        error_code="VALIDATION_ERROR",
        message="El payload no cumple el esquema esperado.",
        fields=fields,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(detail),
    )


async def inference_service_exception_handler(
    request: Request, exc: InferenceServiceError
) -> JSONResponse:
    """Traduce fallos de la capa de dominio a un 503 Service Unavailable."""
    request_id = _get_request_id(request)

    detail = ErrorDetail(
        request_id=request_id,
        error_code="INFERENCE_UNAVAILABLE",
        message=str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=jsonable_encoder(detail),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Red de seguridad final: nunca debe filtrarse un traceback al cliente."""
    request_id = _get_request_id(request)

    detail = ErrorDetail(
        request_id=request_id,
        error_code="INTERNAL_ERROR",
        message="Error interno no controlado. Ha sido registrado para su análisis.",
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(detail),
    )
