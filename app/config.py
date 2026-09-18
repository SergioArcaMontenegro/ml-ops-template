"""Centralized application configuration using Pydantic Settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "MLOps Production Serving"
    app_env: str = "production"
    log_level: str = "INFO"

    # Backends and model paths
    model_backend: str = "heuristic"  # xgboost, onnx, lightgbm, heuristic
    model_path: Path = Path("models/model.onnx")
    xgboost_model_path: Path = Path("models/xgboost_model.json")
    lightgbm_model_path: Path = Path("models/lightgbm_model.txt")
    model_version: str = "v1.0.0"

    # Resilience and Fallbacks
    decision_threshold: float = 0.5
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_sec: float = 30.0

    # Drift and Observability
    reference_data_path: Path = Path("data/reference/reference_dataset.parquet")
    drift_critical_threshold: float = 0.05
    pushgateway_url: str = "http://pushgateway:9091"

    # Shadow Mode (Chapter 9)
    shadow_mode: bool = False
    shadow_header_name: str = "x-shadow-traffic"


settings = Settings()
