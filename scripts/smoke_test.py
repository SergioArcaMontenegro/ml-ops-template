#!/usr/bin/env python3
# scripts/smoke_test.py
"""Smoke test automatizado pre-switch para despliegues Blue-Green y Canary (Capítulo 9).

Verifica /healthz, latencia de inferencia y validación estricta de esquemas
antes de conmutar el tráfico de producción.
"""

from __future__ import annotations

import argparse
import sys
import time
from uuid import uuid4

import httpx


def run_smoke_test(base_url: str = "http://localhost:8000") -> bool:
    print(f"-> Iniciando Smoke Test contra {base_url}...")

    with httpx.Client(base_url=base_url, timeout=5.0) as client:
        # 1. Healthcheck
        try:
            r_health = client.get("/healthz")
            if r_health.status_code != 200:
                print(f"[FAIL] /healthz devolvió {r_health.status_code}")
                return False
            print("[PASS] /healthz responde 200 OK")
        except Exception as exc:
            print(f"[FAIL] Error conectando a /healthz: {exc}")
            return False

        # 2. Inferencia válida
        payload = {
            "request_id": str(uuid4()),
            "features": {
                "customer_tenure_months": 24,
                "monthly_charges": 65.5,
                "total_charges": 1572.0,
                "num_support_tickets": 1,
                "contract_type_code": 1,
                "has_dependents": False,
                "avg_session_duration_min": 45.0,
            },
        }

        t0 = time.perf_counter()
        try:
            r_pred = client.post("/v1/predict", json=payload)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            if r_pred.status_code != 200:
                print(f"[FAIL] /v1/predict devolvió {r_pred.status_code}: {r_pred.text}")
                return False
            data = r_pred.json()
            if "churn_probability" not in data or "predicted_label" not in data:
                print(f"[FAIL] Respuesta malformada: {data}")
                return False
            print(
                f"[PASS] /v1/predict responde 200 OK en {lat_ms:.2f}ms (Prob: {data['churn_probability']})"
            )
        except Exception as exc:
            print(f"[FAIL] Error en petición de inferencia: {exc}")
            return False

        # 3. Validación de rechazo (campo prohibido o tipo inválido) -> debe devolver 422
        bad_payload = {
            "request_id": str(uuid4()),
            "features": {
                "customer_tenure_months": "tres_meses",  # String inválido
                "monthly_charges": 50.0,
                "total_charges": 150.0,
                "num_support_tickets": 0,
                "contract_type_code": 0,
                "has_dependents": True,
                "avg_session_duration_min": 10.0,
            },
        }
        try:
            r_bad = client.post("/v1/predict", json=bad_payload)
            if r_bad.status_code != 422:
                print(
                    f"[FAIL] Validación estricta falló: se esperaba 422 y se obtuvo {r_bad.status_code}"
                )
                return False
            print("[PASS] Validación estricta Pydantic V2 rechaza tipos erróneos con 422")
        except Exception as exc:
            print(f"[FAIL] Error en test de validación: {exc}")
            return False

    print("\n[SUCCESS] Todos los smoke tests han pasado con éxito. Entorno apto para conmutación.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test de servicio de inferencia")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    ok = run_smoke_test(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
