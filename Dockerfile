# syntax=docker/dockerfile:1.7

########################################
# Stage 1: builder
########################################
FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.4.29 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build

COPY pyproject.toml requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system=false --target=/opt/venv/lib/python3.11/site-packages -r requirements.txt || \
    pip install --no-cache-dir --prefix=/opt/venv -r requirements.txt

COPY app/ ./app/
COPY gunicorn_conf.py ./

########################################
# Stage 2: runtime
########################################
FROM python:3.11-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --no-create-home --shell /usr/sbin/nologin appuser

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app:/opt/venv/lib/python3.11/site-packages" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEB_CONCURRENCY=4

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appgroup /build/app ./app
COPY --from=builder --chown=appuser:appgroup /build/gunicorn_conf.py ./

# Directorio para modelos locales
RUN mkdir -p /app/models && chown -R appuser:appgroup /app/models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else sys.exit(1)"]

ENTRYPOINT ["gunicorn", "app.main:app", "--config", "gunicorn_conf.py"]
