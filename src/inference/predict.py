"""Inferencia: cargar el artefacto entrenado y predecir sobre datos nuevos.

Este modulo no entrena nada. Carga el pipeline serializado y lo usa.
"""

import logging

import joblib
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_model(path=None):
    """Carga el pipeline serializado."""
    path = path or config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            "No existe el modelo en %s. Correr primero "
            "python -m src.training.train" % path
        )
    pipeline = joblib.load(path)
    logger.info("Modelo cargado desde %s", path)
    return pipeline


def get_risk_level(probabilidad):
    """Traduce la probabilidad a LOW / MEDIUM / HIGH."""
    if probabilidad < config.RISK_LOW_MAX:
        return "LOW"
    if probabilidad < config.RISK_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def predict(pipeline, datos):
    """Predice sobre uno o varios clientes.

    datos puede ser un dict (un cliente) o un DataFrame (varios).
    """
    if isinstance(datos, dict):
        datos = pd.DataFrame([datos])

    if config.ID_COL in datos.columns:
        datos = datos.drop(columns=[config.ID_COL])

    probabilidades = pipeline.predict_proba(datos)[:, 1]

    salidas = []
    for proba in probabilidades:
        salidas.append({
            "prediction": "CHURN" if proba >= config.DECISION_THRESHOLD
                          else "NO_CHURN",
            "probability": round(float(proba), 4),
            "risk_level": get_risk_level(proba),
        })

    return salidas
