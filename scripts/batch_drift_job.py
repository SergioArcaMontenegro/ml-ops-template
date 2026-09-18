#!/usr/bin/env python3
# scripts/batch_drift_job.py
"""Job periódico de análisis de Data Drift con Evidently AI (Capítulo 5).

Compara el lote de producción actual contra el dataset de referencia,
genera informe HTML y empuja métricas hacia Prometheus Pushgateway.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Garantizar acceso al paquete 'app' independientemente del directorio de ejecución
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from app.observability.drift_metrics import push_drift_metrics_to_gateway
from app.schemas.prediction import FEATURE_ORDER


def run_batch_drift(
    reference_path: Path = Path("data/reference/reference_dataset.parquet"),
    current_path: Path = Path("data/processed/test.parquet"),
    report_output: Path = Path("reports/drift_report.html"),
    gateway_url: str = "http://localhost:9091",
) -> None:
    if not reference_path.exists():
        print(f"[ERROR] No existe dataset de referencia en {reference_path}")
        return
    if not current_path.exists():
        print(f"[ERROR] No existe dataset actual en {current_path}")
        return

    features = list(FEATURE_ORDER)
    ref_df = pd.read_parquet(reference_path)[features]
    curr_df = pd.read_parquet(current_path)[features]

    print(
        f"Ejecutando DataDriftPreset sobre {len(curr_df)} registros contra referencia ({len(ref_df)})..."
    )
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_output))
    print(f"-> Reporte HTML guardado en {report_output}")

    res = report.as_dict()
    metric_res = res["metrics"][0]["result"]
    dataset_drift = bool(metric_res["dataset_drift"])
    drift_share = float(metric_res["drift_share"])

    print("========================================")
    print(f"Dataset Drift Detectado: {dataset_drift}")
    print(f"Proporción de Features con Drift: {drift_share:.2%}")
    print("========================================")

    push_drift_metrics_to_gateway(
        gateway_url=gateway_url,
        job_name="batch_drift_analysis",
        dataset_drift=dataset_drift,
        drift_share=drift_share,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis batch de drift con Evidently")
    parser.add_argument(
        "--ref", type=Path, default=Path("data/reference/reference_dataset.parquet")
    )
    parser.add_argument("--current", type=Path, default=Path("data/processed/test.parquet"))
    parser.add_argument("--output", type=Path, default=Path("reports/drift_report.html"))
    parser.add_argument("--gateway", type=str, default="http://localhost:9091")
    args = parser.parse_args()

    run_batch_drift(args.ref, args.current, args.output, args.gateway)


if __name__ == "__main__":
    main()
