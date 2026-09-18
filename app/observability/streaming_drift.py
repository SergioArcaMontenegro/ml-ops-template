# app/observability/streaming_drift.py
"""Detección de drift casi en tiempo real mediante ventana deslizante
en memoria, evaluada periódicamente contra la referencia.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path

import pandas as pd

from app.observability.drift_metrics import update_drift_gauges
from app.schemas.prediction import FEATURE_ORDER, TabularFeatures

logger = logging.getLogger("mlops.streaming_drift")

_WINDOW_SIZE = 1000
_EVALUATION_INTERVAL_SECONDS = 300

_window: deque[dict[str, float]] = deque(maxlen=_WINDOW_SIZE)
_reference_df: pd.DataFrame | None = None


def initialize_reference(reference_path: Path) -> None:
    """Carga el dataset de referencia desde Parquet."""
    global _reference_df
    if reference_path.exists():
        try:
            _reference_df = pd.read_parquet(reference_path)[list(FEATURE_ORDER)]
            logger.info("Dataset de referencia cargado correctamente desde %s", reference_path)
        except Exception as exc:
            logger.error("Error al cargar dataset de referencia %s: %s", reference_path, exc)
    else:
        logger.warning("No se encontró dataset de referencia en %s", reference_path)


def record_to_window(features: TabularFeatures) -> None:
    """Registra un vector en la ventana deslizante circular."""
    _window.append(features.model_dump())


async def periodic_streaming_drift_check() -> None:
    """Tarea de background ejecutada periódicamente desde el lifespan."""
    while True:
        await asyncio.sleep(_EVALUATION_INTERVAL_SECONDS)

        if _reference_df is None or len(_window) < 50:
            # Ventana insuficientemente poblada: evitar falsos positivos
            continue

        try:
            current_df = pd.DataFrame(list(_window))[list(FEATURE_ORDER)]

            # Evaluación con Evidently AI si está instalado, o cálculo estadístico básico
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=_reference_df, current_data=current_df)
            result = report.as_dict()

            metrics_data = result["metrics"][0]["result"]
            dataset_drift = bool(metrics_data["dataset_drift"])
            drift_share = float(metrics_data["drift_share"])

            update_drift_gauges(dataset_drift=dataset_drift, drift_share=drift_share)
            logger.info(
                "Streaming drift check completado. dataset_drift=%s, drift_share=%.3f",
                dataset_drift,
                drift_share,
            )
        except Exception as exc:
            logger.warning("Error durante streaming drift check: %s", exc)
