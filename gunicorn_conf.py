# gunicorn_conf.py
"""Configuración de Gunicorn para servir la aplicación FastAPI en producción."""

from __future__ import annotations

import multiprocessing
import os

bind = os.environ.get("BIND", "0.0.0.0:8000")
worker_class = "uvicorn.workers.UvicornWorker"

# Fórmula estándar (2*CPU + 1) acotada por un máximo operativo para no
# saturar la memoria cuando el modelo cargado en cada worker es pesado.
_cpu_count = multiprocessing.cpu_count()
workers = min(int(os.environ.get("WEB_CONCURRENCY", _cpu_count * 2 + 1)), 8)

# Recicla workers periódicamente para mitigar fugas de memoria de
# librerías nativas (numpy/xgboost) que no siempre liberan memoria C.
max_requests = 10_000
max_requests_jitter = 500

timeout = 30
graceful_timeout = 30
keepalive = 5

accesslog = None  # El logging de acceso lo gestiona StructuredLoggingMiddleware
errorlog = "-"
loglevel = "info"

preload_app = False  # False: cada worker carga el modelo de forma independiente
                      # en su propio lifespan, evitando compartir estado de C
                      # extensions (xgboost) entre procesos via fork().
