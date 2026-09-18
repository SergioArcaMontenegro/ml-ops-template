#!/usr/bin/env python3
# scripts/quantize_onnx.py
"""Aplica cuantización dinámica INT8 a un grafo ONNX FP32 existente (Capítulo 3)."""

from __future__ import annotations

import argparse
from pathlib import Path


def quantize_model(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"[AVISO] No se encontró el modelo de entrada {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            extra_options={"EnableSubgraph": True},
        )
        print(f"-> Modelo cuantizado escrito en {output_path}")
    except Exception as exc:
        print(f"[AVISO] Error al cuantizar ONNX ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cuantización dinámica INT8 de un modelo ONNX")
    parser.add_argument("--input", type=Path, default=Path("models/model.onnx"))
    parser.add_argument("--output", type=Path, default=Path("models/model_quantized.onnx"))
    args = parser.parse_args()

    quantize_model(args.input, args.output)


if __name__ == "__main__":
    main()
