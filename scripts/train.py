#!/usr/bin/env python3
# scripts/train.py
"""Pipeline de entrenamiento reproducible para el modelo de churn (Capítulo 7).

Genera datos sintéticos o lee data/raw/, entrena un XGBoostClassifier,
registra artefactos, dataset de referencia y manifiesto de modelo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb

from app.schemas.prediction import FEATURE_ORDER


def generate_synthetic_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Genera datos de churn sintéticos respetando los esquemas del contrato."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 72, size=n_samples)
    monthly = np.round(rng.uniform(20.0, 120.0, size=n_samples), 2)
    # total_charges >= monthly_charges si tenure >= 1
    total = np.round(monthly * np.maximum(1, tenure) + rng.uniform(0, 50, size=n_samples), 2)
    tickets = rng.poisson(lam=1.5, size=n_samples)
    contract = rng.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    dependents = rng.choice([True, False], size=n_samples, p=[0.3, 0.7])
    avg_session = np.round(rng.uniform(5.0, 120.0, size=n_samples), 1)

    # Probabilidad de churn sintética con señal nítida
    logit = (
        0.5
        - 0.08 * tenure
        + 0.04 * monthly
        + 0.6 * tickets
        - 1.5 * contract
        - 0.5 * dependents.astype(int)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (prob >= 0.5).astype(int)

    df = pd.DataFrame({
        "customer_tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "num_support_tickets": tickets,
        "contract_type_code": contract,
        "has_dependents": dependents,
        "avg_session_duration_min": avg_session,
        "churn": churn,
    })
    return df


def train_model(params_path: Path = Path("params.yaml")) -> None:
    """Ejecuta el pipeline de preparación y entrenamiento."""
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    data_cfg = params.get("data", {})
    model_cfg = params.get("model", {})
    feature_cols = list(FEATURE_ORDER)

    # Rutas
    raw_path = Path(data_cfg.get("raw_path", "data/raw/customer_churn.csv"))
    train_path = Path(data_cfg.get("train_path", "data/processed/train.parquet"))
    test_path = Path(data_cfg.get("test_path", "data/processed/test.parquet"))
    ref_path = Path(data_cfg.get("reference_path", "data/reference/reference_dataset.parquet"))
    
    train_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_data(n_samples=3000, seed=data_cfg.get("random_state", 42))
        df.to_csv(raw_path, index=False)
        print(f"-> Datos sintéticos creados en {raw_path}")
    else:
        df = pd.read_csv(raw_path)

    # Guardar dataset de referencia para evidently (sin columna churn)
    ref_df = df[feature_cols].copy()
    ref_df.to_parquet(ref_path, index=False)
    print(f"-> Dataset de referencia guardado en {ref_path}")

    # Train / Test split
    X = df[feature_cols]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg.get("test_size", 0.2), random_state=data_cfg.get("random_state", 42)
    )

    X_train.to_parquet(train_path, index=False)
    test_df = X_test.copy()
    test_df["churn"] = y_test
    test_df.to_parquet(test_path, index=False)
    print(f"-> Train ({len(X_train)}) y Test ({len(X_test)}) guardados en data/processed/")

    # Entrenar XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=model_cfg.get("n_estimators", 150),
        max_depth=model_cfg.get("max_depth", 4),
        learning_rate=model_cfg.get("learning_rate", 0.05),
        subsample=model_cfg.get("subsample", 0.8),
        colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
        random_state=model_cfg.get("random_state", 42),
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)

    # Evaluación
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, y_pred_proba))
    f1 = float(f1_score(y_test, y_pred))

    print(f"-> Métricas en test: ROC AUC = {auc:.4f}, F1 = {f1:.4f}")

    # Guardar modelo
    model_file = Path("models/xgboost_model.json")
    clf.save_model(str(model_file))
    print(f"-> Modelo guardado en {model_file}")

    # Manifiesto de registro de modelo
    with open(model_file, "rb") as mf:
        model_hash = hashlib.sha256(mf.read()).hexdigest()

    manifest = {
        "model_version": "v1.0.0",
        "backend": "xgboost",
        "sha256": model_hash,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "roc_auc": round(auc, 4),
            "f1": round(f1, 4),
        },
        "feature_order": feature_cols,
    }
    with open("models/model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("-> Manifiesto de modelo guardado en models/model_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    train_model()
