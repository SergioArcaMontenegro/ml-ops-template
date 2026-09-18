# app/middleware/metrics.py
"""Middleware ASGI que instrumenta cada petición HTTP en Prometheus."""

from __future__ import annotations

import time
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability.metrics import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL


class MetricsMiddleware:
    """Registra latencia y conteo de peticiones HTTP con labels de baja cardinalidad."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # No registrar métricas para el propio endpoint /metrics para no sesgar
        path = scope.get("path", "")
        if path == "/metrics":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()
        status_holder: dict[str, int] = {"value": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["value"] = message["status"]
            await send(message)

        route_path = scope.get("route").path if scope.get("route") else path or "unknown"

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            method = scope.get("method", "UNKNOWN")

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method, path=route_path
            ).observe(elapsed_seconds)

            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                path=route_path,
                status_code=str(status_holder["value"]),
            ).inc()
