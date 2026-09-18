# app/services/backends.py
"""Backends de modelo intercambiables tras un protocolo común."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelBackend(Protocol):
    """Contrato mínimo que debe cumplir cualquier backend de modelo."""

    version: str

    def predict_proba(self, features: tuple[float, ...]) -> float:
        """Devuelve la probabilidad de la clase positiva para un único vector."""
        ...


class XGBoostBackend:
    """Backend basado en xgboost.Booster cargado desde un fichero binario o JSON."""

    def __init__(self, model_path: Path, version: str) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo XGBoost en {model_path}")
        import xgboost as xgb

        self._booster = xgb.Booster()
        self._booster.load_model(str(model_path))
        self.version = version

    def predict_proba(self, features: tuple[float, ...]) -> float:
        import xgboost as xgb

        from app.schemas.prediction import FEATURE_ORDER

        matrix = xgb.DMatrix(
            np.asarray([features], dtype=np.float32), feature_names=list(FEATURE_ORDER)
        )
        prediction = self._booster.predict(matrix)
        return float(prediction[0])


class LightGBMBackend:
    """Backend basado en lightgbm.Booster."""

    def __init__(self, model_path: Path, version: str) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo LightGBM en {model_path}")
        import lightgbm as lgb

        self._booster = lgb.Booster(model_file=str(model_path))
        self.version = version

    def predict_proba(self, features: tuple[float, ...]) -> float:
        matrix = np.asarray([features], dtype=np.float32)
        prediction = self._booster.predict(matrix)
        return float(prediction[0])
