# app/exceptions/handlers.py
"""Custom exception handlers for uniform API error responses."""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.prediction import ErrorDetail


class InferenceServiceError(Exception):
    """Base exception for domain inference layer failures."""


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
    """Translates Pydantic validation errors into a stable error contract."""
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
        message="Request payload does not match expected schema.",
        fields=fields,
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(detail),
    )


async def inference_service_exception_handler(
    request: Request, exc: InferenceServiceError
) -> JSONResponse:
    """Translates domain layer failures into HTTP 503 Service Unavailable."""
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
    """Final safety net: unhandled exceptions never leak tracebacks to clients."""
    request_id = _get_request_id(request)

    detail = ErrorDetail(
        request_id=request_id,
        error_code="INTERNAL_ERROR",
        message="Internal server error. The incident has been logged.",
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(detail),
    )
