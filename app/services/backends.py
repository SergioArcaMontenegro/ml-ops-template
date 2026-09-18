# app/services/backends.py
"""Interchangeable model backends conforming to a shared protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelBackend(Protocol):
    """Minimal contract required for any model backend implementation."""

    version: str

    def predict_proba(self, features: tuple[float, ...]) -> float:
        """Returns positive class probability for a single feature vector."""
        ...


class XGBoostBackend:
    """Backend powered by an xgboost.Booster loaded from disk."""

    def __init__(self, model_path: Path, version: str) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model file not found at {model_path}")
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
    """Backend powered by a lightgbm.Booster."""

    def __init__(self, model_path: Path, version: str) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"LightGBM model file not found at {model_path}")
        import lightgbm as lgb

        self._booster = lgb.Booster(model_file=str(model_path))
        self.version = version

    def predict_proba(self, features: tuple[float, ...]) -> float:
        matrix = np.asarray([features], dtype=np.float32)
        prediction = self._booster.predict(matrix)
        return float(prediction[0])
