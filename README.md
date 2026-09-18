# Ingeniería de SaaS en Producción — Plantilla de MLOps de Élite

[![CI](https://github.com/SergioArcaMontenegro/ml-ops-template/actions/workflows/ci.yml/badge.svg)](https://github.com/SergioArcaMontenegro/ml-ops-template/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg)](https://www.docker.com/)
[![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.17+-005CED.svg)](https://onnxruntime.ai/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Enabled-E6522C.svg)](https://prometheus.io/)

Plantilla de arquitectura de nivel de producción para microservicios de inferencia y operaciones de Machine Learning (MLOps). Diseñada a partir de los 9 capítulos del libro **«Ingeniería de SaaS en Producción — MLOps: De Modelos Tabulares a Microservicios Escalables y Resilientes»**.

> **Premisa de arquitectura:** Un modelo de machine learning no es un artefacto que se "despliega", es un componente de software crítico que debe someterse a validación estricta de payloads, observabilidad exhaustiva, resiliencia ante fallos con circuit breakers, versionado determinista, gates de calidad bloqueantes y despliegue progresivo automatizado.

---

## 🏛️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Cliente HTTP / Ingress                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ POST /v1/predict
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ASGI Middleware Stack                                                       │
│   ├── RequestContextMiddleware    (x-request-id tracing)                    │
│   ├── MetricsMiddleware           (Latencia HTTP & Throughput Prometheus)   │
│   └── StructuredLoggingMiddleware (Logs en JSON estructurado por stdout)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Capa FastAPI Routing & Validación Estricta                                  │
│   ├── Pydantic V2 Schemas (strict=True, extra="forbid", frozen=True)        │
│   └── Exception Handlers (422 Validación / 503 Fallback / 500 Interno)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ InferenceService (Desacoplado vía Protocol)                                 │
│   ├── CircuitBreaker (CLOSED ──> OPEN ──> HALF_OPEN)                        │
│   ├── Backends Primarios:                                                   │
│   │     ├── ONNX Runtime (INT8 Cuantizado / FP32)                           │
│   │     ├── XGBoost Booster                                                 │
│   │     └── LightGBM Booster                                                │
│   └── Fallback de Emergencia:                                               │
│         └── HeuristicBackend (Reglas de negocio puras, sin deps frágiles)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Observabilidad & Drift Engine (Evidently AI + Prometheus)                   │
│   ├── Background Feature Logger (Parquet particionado para auditoría)       │
│   ├── Streaming Drift (Ventana deslizante circular contra referencia)       │
│   └── Drift Gate Automático (Dispara apertura de fallback si hay degradación)│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Estructura del Repositorio

```text
.
├── .github/workflows/
│   ├── ci.yml                 # Linting (Ruff), Type Check (mypy), Pytest y Quality Gate
│   └── release.yml            # Build multi-stage, escaneo Trivy bloqueante y push a GHCR
├── app/
│   ├── main.py                # Entrada FastAPI, lifespan, endpoints predict/healthz/metrics
│   ├── config.py              # Pydantic Settings para configuración y variables de entorno
│   ├── exceptions/handlers.py # Manejadores custom tipados (422, 503, 500)
│   ├── middleware/
│   │   ├── logging.py         # Logging estructurado JSON y request_id
│   │   └── metrics.py         # Recolección de métricas Prometheus
│   ├── observability/
│   │   ├── metrics.py         # Definición de contadores e histogramas Prometheus
│   │   ├── feature_logger.py  # Logger asíncrono para captura de features
│   │   ├── streaming_drift.py # Ventana deslizante circular para drift en tiempo real
│   │   ├── drift_metrics.py   # Actualización de gauges y Pushgateway
│   │   └── drift_gate.py      # Guardia automática de fallback ante drift crítico
│   ├── schemas/prediction.py  # Esquemas Pydantic V2 contractuales con orden FEATURE_ORDER
│   └── services/
│       ├── backends.py        # Protocol ModelBackend, XGBoost y LightGBM
│       ├── onnx_backend.py    # Inferencia acelerada con ONNX Runtime
│       ├── heuristic_backend.py # Fallback determinista sin dependencias externas
│       ├── circuit_breaker.py # Máquina de estados del Circuit Breaker
│       └── inference.py       # InferenceService desacoplado con observabilidad
├── data/
│   ├── raw/                   # Datos brutos de entrada
│   ├── processed/             # Conjuntos procesados de train/test
│   └── reference/             # Dataset de referencia canónico para Evidently AI
├── k8s/
│   ├── rollout.yaml           # Argo Rollout con estrategia Canary progresiva
│   ├── analysis_template.yaml # AnalysisTemplate con métricas Prometheus
│   ├── service.yaml           # Servicios Kubernetes (stable y canary)
│   ├── bluegreen.yaml         # Despliegue Blue-Green con switch atómico
│   └── istio_shadow.yaml      # VirtualService de Istio para Shadow Deployment
├── monitoring/
│   ├── prometheus/            # Configuración de scrape y jobs de Prometheus
│   ├── alerts/                # Reglas de alerta (p99, error rate, drift) y Alertmanager
│   └── grafana/               # Dashboards y datasources listos para provisioning
├── scripts/
│   ├── train.py               # Pipeline de entrenamiento reproducible y manifiesto
│   ├── convert_to_onnx.py     # Conversión de modelos a grafo ONNX
│   ├── quantize_onnx.py       # Cuantización dinámica INT8
│   ├── benchmark.py           # Benchmark comparativo de latencia p50/p95/p99 y RPS
│   ├── quality_gate.py        # Quality gate bloqueante (AUC >= 0.85, p95 <= 15ms)
│   ├── batch_drift_job.py     # Análisis batch de drift con Evidently AI
│   └── smoke_test.py          # Smoke tests pre-switch para Blue-Green y Canary
├── Dockerfile                 # Multi-stage ultra-optimizado con uv y usuario appuser
├── docker-compose.yml         # Stack completo: API + Prometheus + Pushgateway + Grafana
├── params.yaml                # Fuente única de verdad de features, params y umbrales
├── dvc.yaml                   # Pipeline reproducible de DVC
└── pyproject.toml             # Gestión moderna de dependencias y herramientas
```

---

## 🚀 Inicio Rápido

### 1. Requisitos Previos
- Python 3.11+
- `uv` (recomendado) o `pip`
- Docker y Docker Compose

### 2. Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/SergioArcaMontenegro/ml-ops-template.git
cd ml-ops-template

# Crear entorno virtual e instalar dependencias con uv
uv sync --frozen --group dev
source .venv/bin/activate  # En Linux/macOS
# O en Windows: .venv\Scripts\activate

# O alternativamente con pip clásico:
pip install -r requirements.txt
```

### 3. Entrenar el Modelo y Generar Artefactos

```bash
# Entrenar modelo baseline, generar dataset de referencia y manifiesto
python scripts/train.py

# Convertir a ONNX y aplicar cuantización dinámica INT8
python scripts/convert_to_onnx.py
python scripts/quantize_onnx.py

# Ejecutar el Quality Gate bloqueante
python scripts/quality_gate.py
```

### 4. Ejecutar el Servicio en Local

```bash
# Servidor de desarrollo con Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# O con configuración de producción de Gunicorn
gunicorn app.main:app --config gunicorn_conf.py
```

Prueba la API en: `http://localhost:8000/docs`

---

## 🐳 Stack Completo con Docker Compose

Levanta el microservicio junto con **Prometheus**, **Pushgateway** y **Grafana** preconfigurados:

```bash
docker compose up --build -d
```

- **Servicio de Inferencia:** `http://localhost:8000` (docs en `/docs`, métricas en `/metrics`)
- **Prometheus:** `http://localhost:9090` (con alertas de latencia p99, tasa de error 5xx y drift)
- **Pushgateway:** `http://localhost:9091`
- **Grafana:** `http://localhost:3000` (Usuario: `admin` / Contraseña: `admin`, dashboard `MLOps Overview` precargado)

---

## 🧪 Pruebas y Calidad

```bash
# Ejecutar suite de pruebas con pytest y cobertura
pytest --cov=app --cov-report=term-missing tests/

# Análisis estático y formateo con Ruff
ruff check .
ruff format --check .

# Verificación de tipos estricta con mypy
mypy app/ --strict
```

---

## 📊 Benchmarking de Latencia

Compara el rendimiento de los backends instalados (Heuristic, ONNX Runtime, XGBoost):

```bash
python scripts/benchmark.py --samples 1000
```

Ejemplo de salida esperada:
```
Backend              | p50 (ms)   | p95 (ms)   | p99 (ms)   | Throughput (RPS)
---------------------------------------------------------------------------
HeuristicBackend     | 0.012      | 0.025      | 0.040      | 45200.0         
ONNXBackend          | 0.180      | 0.350      | 0.620      | 4800.0          
XGBoostBackend       | 0.450      | 0.890      | 1.250      | 1950.0          
```

---

## 🛡️ Resiliencia y Fallbacks Automáticos

1. **Circuit Breaker:** Si el backend primario acumula 5 fallos consecutivos, el circuito pasa a estado `OPEN`, degradando de forma transparente al `HeuristicBackend`.
2. **Drift Gate:** Si Evidently detecta que el drift supera el umbral crítico de negocio (`concept_drift_score > 0.08`), el circuito se abre preventivamente para evitar inferencias degradadas.
3. **Contrato de Transparencia:** Toda predicción incluye el flag `is_fallback: bool` para que los consumidores sepan si el resultado proviene del modelo de ML o del fallback determinista.

---

## 🚢 Despliegue en Kubernetes

- **Canary con Argo Rollouts:** Ver [k8s/rollout.yaml](k8s/rollout.yaml) y [k8s/analysis_template.yaml](k8s/analysis_template.yaml).
- **Blue-Green con Smoke Tests:** Ver [k8s/bluegreen.yaml](k8s/bluegreen.yaml) y script [scripts/smoke_test.py](scripts/smoke_test.py).
- **Shadow Deployment:** Duplicación asíncrona de tráfico en Istio con [k8s/istio_shadow.yaml](k8s/istio_shadow.yaml).

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
