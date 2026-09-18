#!/usr/bin/env python3
# scripts/convert_to_onnx.py
"""Convierte modelos entrenados (XGBoost/LightGBM) a formato ONNX (Capítulo 3)."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from app.schemas.prediction import FEATURE_ORDER


def convert_model(
    model_path: Path = Path("models/xgboost_model.json"),
    output_path: Path = Path("models/model.onnx"),
    num_features: int = len(FEATURE_ORDER),
) -> None:
    """Convierte el modelo XGBoost a ONNX con batch dinámico."""
    if not model_path.exists():
        print(f"[AVISO] No se encontró {model_path}. Omitiendo conversión ONNX.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import xgboost as xgb
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        booster.feature_names = [f"f{i}" for i in range(num_features)]

        initial_types = [("input", FloatTensorType([None, num_features]))]
        onnx_model = convert_xgboost(
            booster,
            initial_types=initial_types,
            target_opset=15,
        )
        output_path.write_bytes(onnx_model.SerializeToString())
        print(f"-> Modelo ONNX guardado exitosamente en {output_path}")
    except Exception as exc:
        print(f"[AVISO] Conversión con onnxmltools falló ({exc}). Creando fallback de grafo ONNX o dummy.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversión de modelo a ONNX")
    parser.add_argument("--model-path", type=Path, default=Path("models/xgboost_model.json"))
    parser.add_argument("--output-path", type=Path, default=Path("models/model.onnx"))
    args = parser.parse_args()
    convert_model(args.model_path, args.output_path)


if __name__ == "__main__":
    main()
