#!/usr/bin/env python3
# scripts/benchmark.py
"""Compara latencia (p50/p95/p99) y throughput entre backends de inferencia (Capítulo 3)."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Garantizar acceso al paquete 'app' independientemente del directorio de ejecución
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.schemas.prediction import FEATURE_ORDER
from app.services.backends import ModelBackend, XGBoostBackend
from app.services.heuristic_backend import HeuristicBackend
from app.services.onnx_backend import ONNXBackend


@dataclass(slots=True)
class BenchmarkResult:
    backend_name: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    rps: float


def run_backend_benchmark(
    backend: ModelBackend,
    backend_name: str,
    feature_matrix: list[tuple[float, ...]],
    warmup: int = 50,
) -> BenchmarkResult:
    # Calentamiento
    for row in feature_matrix[:warmup]:
        backend.predict_proba(row)

    latencies_ms: list[float] = []
    t_start = time.perf_counter()

    for row in feature_matrix:
        start = time.perf_counter()
        backend.predict_proba(row)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    total_time = time.perf_counter() - t_start
    rps = len(feature_matrix) / total_time if total_time > 0 else 0.0

    return BenchmarkResult(
        backend_name=backend_name,
        p50_ms=round(float(np.percentile(latencies_ms, 50)), 3),
        p95_ms=round(float(np.percentile(latencies_ms, 95)), 3),
        p99_ms=round(float(np.percentile(latencies_ms, 99)), 3),
        mean_ms=round(float(np.mean(latencies_ms)), 3),
        rps=round(rps, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de latencia y throughput")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    # Cargar o generar datos de test
    test_path = Path("data/processed/test.parquet")
    if test_path.exists():
        df = pd.read_parquet(test_path)[list(FEATURE_ORDER)]
        features = [tuple(r) for r in df.values]
    else:
        # Vector sintético
        rng = np.random.default_rng(42)
        features = [
            tuple(rng.uniform(1.0, 50.0, size=len(FEATURE_ORDER))) for _ in range(args.samples)
        ]

    backends: list[tuple[str, ModelBackend]] = [
        ("HeuristicBackend", HeuristicBackend()),
    ]

    onnx_path = Path("models/model.onnx")
    if onnx_path.exists():
        backends.append(("ONNXBackend", ONNXBackend(onnx_path, version="v1.0.0")))

    xgb_path = Path("models/xgboost_model.json")
    if xgb_path.exists():
        backends.append(("XGBoostBackend", XGBoostBackend(xgb_path, version="v1.0.0")))

    print(
        f"\n{'Backend':<20} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10} | {'Throughput (RPS)':<16}"
    )
    print("-" * 75)
    for name, b in backends:
        res = run_backend_benchmark(b, name, features)
        print(
            f"{res.backend_name:<20} | {res.p50_ms:<10} | {res.p95_ms:<10} | {res.p99_ms:<10} | {res.rps:<16}"
        )
    print("-" * 75 + "\n")


if __name__ == "__main__":
    main()
