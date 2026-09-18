# Production SaaS Engineering — Elite MLOps Template

[![CI](https://github.com/SergioArcaMontenegro/ml-ops-template/actions/workflows/ci.yml/badge.svg)](https://github.com/SergioArcaMontenegro/ml-ops-template/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg)](https://www.docker.com/)
[![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.17+-005CED.svg)](https://onnxruntime.ai/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Enabled-E6522C.svg)](https://prometheus.io/)

Production-grade architecture template for Machine Learning Operations (MLOps) and inference microservices. Built directly from the principles and implementations across the 9 chapters of **«Production SaaS Engineering — MLOps: From Tabular Models to Scalable, Resilient Microservices»**.

> **Architectural Premise:** A machine learning model is not an artifact that is merely "deployed"; it is a critical software component that must undergo strict payload validation, exhaustive observability, fault resilience via circuit breakers, deterministic versioning, blocking quality gates, and automated progressive rollouts.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HTTP Client / Ingress                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ POST /v1/predict
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ASGI Middleware Stack                                                       │
│   ├── RequestContextMiddleware    (x-request-id distributed tracing)        │
│   ├── MetricsMiddleware           (HTTP latency & throughput in Prometheus) │
│   └── StructuredLoggingMiddleware (JSON structured access logs to stdout)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FastAPI Routing Layer & Strict Validation                                   │
│   ├── Pydantic V2 Schemas (strict=True, extra="forbid", frozen=True)        │
│   └── Exception Handlers (422 Validation / 503 Fallback / 500 Internal)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Decoupled InferenceService (Protocol-Driven)                                │
│   ├── CircuitBreaker (CLOSED ──> OPEN ──> HALF_OPEN)                        │
│   ├── Primary Backends:                                                     │
│   │     ├── ONNX Runtime (INT8 Dynamic Quantized / FP32)                    │
│   │     ├── XGBoost Booster                                                 │
│   │     └── LightGBM Booster                                                │
│   └── Emergency Fallback:                                                   │
│         └── HeuristicBackend (Zero-dependency, pure business heuristic)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Observability & Drift Engine (Evidently AI + Prometheus)                   │
│   ├── Background Feature Logger (Partitioned Parquet files for audit)       │
│   ├── Streaming Drift (In-memory circular sliding window vs. baseline)      │
│   └── Automated Drift Gate (Forces fallback if concept drift is critical)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Repository Structure

```text
.
├── .github/workflows/
│   ├── ci.yml                 # Linting (Ruff), Type Check (mypy), Pytest and Quality Gate
│   └── release.yml            # Multi-stage Docker build, blocking Trivy scan and GHCR push
├── app/
│   ├── main.py                # FastAPI entry point, lifespan, predict/healthz/metrics endpoints
│   ├── config.py              # Pydantic Settings for environment variables and configuration
│   ├── exceptions/handlers.py # Custom typed exception handlers (422, 503, 500)
│   ├── middleware/
│   │   ├── logging.py         # Structured JSON logging and request_id propagation
│   │   └── metrics.py         # Prometheus metrics collection middleware
│   ├── observability/
│   │   ├── metrics.py         # Prometheus counters and histograms definitions
│   │   ├── feature_logger.py  # Asynchronous logger for background feature capture
│   │   ├── streaming_drift.py # Circular sliding window for near real-time drift detection
│   │   ├── drift_metrics.py   # Drift gauges and Prometheus Pushgateway exporter
│   │   └── drift_gate.py      # Automated business safety gate for critical drift
│   ├── schemas/prediction.py  # Contractual Pydantic V2 schemas with FEATURE_ORDER tuple
│   └── services/
│       ├── backends.py        # ModelBackend Protocol, XGBoost and LightGBM implementations
│       ├── onnx_backend.py    # Accelerated inference with ONNX Runtime
│       ├── heuristic_backend.py # Deterministic zero-dependency fallback backend
│       ├── circuit_breaker.py # State machine for the Circuit Breaker pattern
│       └── inference.py       # Decoupled InferenceService with integrated resilience
├── data/
│   ├── raw/                   # Raw input dataset
│   ├── processed/             # Processed train/test sets
│   └── reference/             # Canonical baseline reference dataset for Evidently AI
├── k8s/
│   ├── rollout.yaml           # Argo Rollout with progressive canary promotion steps
│   ├── analysis_template.yaml # AnalysisTemplate querying Prometheus metrics
│   ├── service.yaml           # Kubernetes Services (stable and canary)
│   ├── bluegreen.yaml         # Blue-Green deployment with atomic traffic switch
│   └── istio_shadow.yaml      # Istio VirtualService for Shadow Traffic Mirroring
├── monitoring/
│   ├── prometheus/            # Prometheus scraping configurations and job definitions
│   ├── alerts/                # Prometheus alert rules (p99, error rate, drift) and Alertmanager
│   └── grafana/               # Pre-provisioned datasources and dashboard JSON files
├── scripts/
│   ├── train.py               # Reproducible training pipeline and model manifest generator
│   ├── convert_to_onnx.py     # Conversion of models to ONNX graph format
│   ├── quantize_onnx.py       # Dynamic INT8 quantization
│   ├── benchmark.py           # Comparative latency (p50/p95/p99) and RPS benchmarking
│   ├── quality_gate.py        # Blocking quality gate (AUC >= 0.85, p95 <= 15ms)
│   ├── batch_drift_job.py     # Batch drift report generation with Evidently AI
│   └── smoke_test.py          # Pre-switch automated smoke tests for Blue-Green and Canary
├── Dockerfile                 # Ultra-optimized multi-stage Dockerfile with uv and appuser
├── docker-compose.yml         # Full stack: API + Prometheus + Pushgateway + Grafana
├── params.yaml                # Single source of truth for features, parameters, and gates
├── dvc.yaml                   # Reproducible DVC pipeline definition
└── pyproject.toml             # Modern dependency management and tooling configuration
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+
- `uv` (recommended) or `pip`
- Docker and Docker Compose

### 2. Local Installation

```bash
# Clone the repository
git clone https://github.com/SergioArcaMontenegro/ml-ops-template.git
cd ml-ops-template

# Cambiar a tu rama preferida:
# git checkout main_español   # Versión en español (código, docstrings, README)
# git checkout main_english   # Versión en inglés (code, docstrings, README)

# Crear entorno virtual e instalar dependencias con uv
uv sync --frozen --group dev
source .venv/bin/activate  # On Linux/macOS
# Or on Windows: .venv\Scripts\activate

# Or alternatively with standard pip:
pip install -r requirements.txt
```

### 3. Model Training and Artifact Generation

```bash
# Train baseline model, generate reference dataset and manifest
python scripts/train.py

# Convert to ONNX and apply dynamic INT8 quantization
python scripts/convert_to_onnx.py
python scripts/quantize_onnx.py

# Execute the blocking Quality Gate
python scripts/quality_gate.py
```

### 4. Running the Service Locally

```bash
# Development server with Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or with Gunicorn production configuration
gunicorn app.main:app --config gunicorn_conf.py
```

Test the API docs at: `http://localhost:8000/docs`

---

## 🐳 Full Stack with Docker Compose

Launch the microservice alongside **Prometheus**, **Pushgateway**, and **Grafana**:

```bash
docker compose up --build -d
```

- **Inference Service:** `http://localhost:8000` (interactive docs at `/docs`, Prometheus metrics at `/metrics`)
- **Prometheus UI:** `http://localhost:9090` (pre-loaded with alerts for p99 latency, 5xx errors, and drift)
- **Pushgateway:** `http://localhost:9091`
- **Grafana Dashboard:** `http://localhost:3000` (User: `admin` / Password: `admin`, pre-configured `MLOps Overview` dashboard)

---

## 🧪 Testing and Quality Assurance

```bash
# Run pytest test suite with coverage
pytest --cov=app --cov-report=term-missing tests/

# Static analysis and formatting check with Ruff
ruff check .
ruff format --check .

# Strict static type checking with mypy
mypy app/ --strict
```

---

## 📊 Latency Benchmarking

Compare performance between available inference engines (Heuristic, ONNX Runtime, XGBoost):

```bash
python scripts/benchmark.py --samples 1000
```

Sample benchmark output:
```
Backend              | p50 (ms)   | p95 (ms)   | p99 (ms)   | Throughput (RPS)
---------------------------------------------------------------------------
HeuristicBackend     | 0.012      | 0.025      | 0.040      | 45200.0         
ONNXBackend          | 0.180      | 0.350      | 0.620      | 4800.0          
XGBoostBackend       | 0.450      | 0.890      | 1.250      | 1950.0          
```

---

## 🛡️ Resilience and Automated Fallbacks

1. **Circuit Breaker:** If the primary ML backend encounters 5 consecutive failures, the circuit state transitions to `OPEN`, immediately degrading to the `HeuristicBackend` without throwing errors to clients.
2. **Drift Gate:** If Evidently detects that statistical concept drift exceeds critical business thresholds (`concept_drift_score > 0.08`), the circuit breaker is proactively tripped open to prevent degraded predictions.
3. **Transparency Contract:** Every prediction response returns the boolean flag `is_fallback: bool`, allowing downstream clients to handle heuristic estimations accordingly.

---

## 🚢 Production Deployment

- **Canary with Argo Rollouts:** See [k8s/rollout.yaml](k8s/rollout.yaml) and [k8s/analysis_template.yaml](k8s/analysis_template.yaml).
- **Blue-Green with Smoke Tests:** See [k8s/bluegreen.yaml](k8s/bluegreen.yaml) and [scripts/smoke_test.py](scripts/smoke_test.py).
- **Shadow Deployment:** Asynchronous Istio traffic mirroring via [k8s/istio_shadow.yaml](k8s/istio_shadow.yaml).

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
