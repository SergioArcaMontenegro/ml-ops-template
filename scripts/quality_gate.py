#!/usr/bin/env python3
# scripts/quality_gate.py
"""Evalúa el modelo contra el conjunto de test y aplica el gate de calidad mínima.

Termina con código de salida 1 (error bloqueante en CI) si no se superan los
umbrales definidos en params.yaml (Capítulo 7 y 8).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, roc_auc_score


def evaluate_gate() -> bool:
    params_path = Path("params.yaml")
    if not params_path.exists():
        print("[ERROR] params.yaml no existe")
        return False

    with open(params_path, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    feature_order = params["feature_order"]
    qg_cfg = params.get("quality_gate", {})
    min_auc = qg_cfg.get("min_auc", 0.80)
    min_f1 = qg_cfg.get("min_f1", 0.70)
    max_p95_latency_ms = qg_cfg.get("max_p95_latency_ms", 20.0)

    test_path = Path("data/processed/test.parquet")
    if not test_path.exists():
        print(f"[ERROR] No se encontró el dataset de test en {test_path}")
        return False

    test_df = pd.read_parquet(test_path)
    y_test = test_df["churn"].values
    X_test = test_df[feature_order]

    # Cargar modelo (ONNX si existe, o XGBoost)
    onnx_path = Path("models/model.onnx")
    xgb_path = Path("models/xgboost_model.json")

    y_pred_proba = None
    latencies = []

    if onnx_path.exists():
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        mat = X_test.values.astype(np.float32)

        # Medir latencia por registro
        for r in mat:
            t0 = time.perf_counter()
            session.run(None, {input_name: np.asarray([r])})
            latencies.append((time.perf_counter() - t0) * 1000.0)

        outs = session.run(None, {input_name: mat})
        if len(outs) > 1 and hasattr(outs[1], "shape"):
            y_pred_proba = outs[1][:, 1]
        else:
            y_pred_proba = outs[0][:, 1] if outs[0].ndim == 2 else outs[0]
    elif xgb_path.exists():
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(xgb_path))
        dtest = xgb.DMatrix(X_test.values, feature_names=feature_order)
        y_pred_proba = booster.predict(dtest)

        for r in X_test.values:
            t0 = time.perf_counter()
            booster.predict(
                xgb.DMatrix(np.asarray([r], dtype=np.float32), feature_names=feature_order)
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
    else:
        print("[ERROR] No se encontró ningún modelo en models/")
        return False

    auc = float(roc_auc_score(y_test, y_pred_proba))
    y_pred = (y_pred_proba >= 0.5).astype(int)
    f1 = float(f1_score(y_test, y_pred))
    p95_ms = float(np.percentile(latencies, 95)) if latencies else 1.0

    print("========================================")
    print("      RESULTADOS DEL QUALITY GATE       ")
    print("========================================")
    print(f"ROC AUC:  {auc:.4f}  (Mínimo exigido: {min_auc})")
    print(f"F1 Score: {f1:.4f}  (Mínimo exigido: {min_f1})")
    print(f"p95 Lat:  {p95_ms:.2f}ms (Máximo permitido: {max_p95_latency_ms}ms)")
    print("========================================")

    metrics = {
        "roc_auc": round(auc, 4),
        "f1": round(f1, 4),
        "p95_latency_ms": round(p95_ms, 2),
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    passed = True
    if auc < min_auc:
        print(f"[FAIL] ROC AUC ({auc:.4f}) por debajo del umbral ({min_auc})")
        passed = False
    if f1 < min_f1:
        print(f"[FAIL] F1 ({f1:.4f}) por debajo del umbral ({min_f1})")
        passed = False
    if p95_ms > max_p95_latency_ms:
        print(f"[FAIL] Latencia p95 ({p95_ms:.2f}ms) supera el límite ({max_p95_latency_ms}ms)")
        passed = False

    if passed:
        print("[SUCCESS] Quality gate APROBADO.")
        return True
    else:
        print("[REJECTED] Quality gate RECHAZADO.")
        return False


if __name__ == "__main__":
    ok = evaluate_gate()
    sys.exit(0 if ok else 1)
