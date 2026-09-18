# app/services/onnx_backend.py
"""Backend de inferencia sobre ONNX Runtime, compatible con ModelBackend."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ONNXBackend:
    """Backend que ejecuta un grafo ONNX (FP32 o INT8) vía ONNX Runtime.

    Cumple el protocolo `ModelBackend` definido en app.services.backends,
    por lo que es intercambiable con XGBoostBackend/LightGBMBackend sin
    tocar InferenceService.
    """

    def __init__(
        self,
        model_path: Path,
        version: str,
        intra_op_num_threads: int = 1,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo ONNX en {model_path}")

        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_op_num_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self.version = version

    def predict_proba(self, features: tuple[float, ...]) -> float:
        input_array = np.asarray([features], dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: input_array})

        # La salida de un clasificador ONNX puede ser un array de probabilidades
        # o una lista de diccionarios por clase.
        if len(outputs) > 1:
            probabilities_output = outputs[1]
            if isinstance(probabilities_output, list) and isinstance(probabilities_output[0], dict):
                # Formato mapa: [{0: 0.1, 1: 0.9}]
                return float(probabilities_output[0].get(1, 0.5))
            elif hasattr(probabilities_output, "shape") and len(probabilities_output.shape) == 2:
                # Formato matriz: [[0.1, 0.9]]
                return float(probabilities_output[0][1])

        # Fallback a outputs[0]
        first_out = outputs[0]
        if hasattr(first_out, "ndim") and first_out.ndim == 2 and first_out.shape[1] > 1:
            return float(first_out[0][1])
        return float(first_out[0])
