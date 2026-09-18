# app/observability/feature_logger.py
"""Logger asíncrono de features de entrada, dedicado a análisis de drift.

Escribe a un sink append-only (ficheros Parquet particionados) de forma
no bloqueante respecto al hot path de inferencia.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.schemas.prediction import FEATURE_ORDER, TabularFeatures

_BUFFER: list[dict[str, Any]] = []
_BUFFER_LOCK = asyncio.Lock()
_FLUSH_THRESHOLD = 500


async def record_features(features: TabularFeatures, predicted_probability: float) -> None:
    """Añade un registro al buffer en memoria; descarga a disco por lotes."""
    row = features.model_dump()
    row["predicted_probability"] = predicted_probability
    row["logged_at"] = datetime.now(timezone.utc).isoformat()

    should_flush = False
    batch: list[dict[str, Any]] = []

    async with _BUFFER_LOCK:
        _BUFFER.append(row)
        if len(_BUFFER) >= _FLUSH_THRESHOLD:
            should_flush = True
            batch = _BUFFER.copy()
            _BUFFER.clear()

    if should_flush:
        await asyncio.to_thread(_flush_to_disk, batch)


def _flush_to_disk(batch: list[dict[str, Any]], target_dir: Path | None = None) -> None:
    output_dir = target_dir or Path("data/features_logged")
    output_dir.mkdir(parents=True, exist_ok=True)

    partition_hour = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
    output_path = output_dir / f"features_{partition_hour}.parquet"

    columns = list(FEATURE_ORDER) + ["predicted_probability", "logged_at"]
    df = pd.DataFrame(batch, columns=columns)

    if output_path.exists():
        try:
            existing = pd.read_parquet(output_path)
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass

    df.to_parquet(output_path, index=False)
