# app/middleware/logging.py
"""Middleware ASGI de logging estructurado en JSON con request tracing."""

from __future__ import annotations

import json
import logging
import time
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("inference.access")


class JSONLogFormatter(logging.Formatter):
    """Formatea cada registro de log como una línea JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configura el logger raíz de acceso para emitir JSON por stdout."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False


class StructuredLoggingMiddleware:
    """Middleware ASGI que mide latencia total y emite un log estructurado."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        start_time = time.perf_counter()
        status_code_holder: dict[str, int] = {"value": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_code_holder["value"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "request_completed",
                extra={
                    "extra_fields": {
                        "request_id": request_id,
                        "path": scope.get("path"),
                        "method": scope.get("method"),
                        "status_code": status_code_holder["value"],
                        "latency_ms": round(elapsed_ms, 3),
                    }
                },
            )
