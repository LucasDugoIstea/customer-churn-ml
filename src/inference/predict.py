"""Inferencia: cargar el artefacto y predecir sobre datos nuevos.

Este modulo no entrena nada. Carga el pipeline ya serializado y lo usa. Es la
pieza que despues va a consumir la API de FastAPI en la entrega 2.
"""

import logging

import joblib
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Se cachea el modelo para no leer el .joblib del disco en cada prediccion
_modelo = None


def load_model(path=None):
    """Carga el pipeline serializado.

    La primera vez lo lee del disco; despues devuelve el que ya esta en
    memoria.
    """
    global _modelo
    path = path or config.MODEL_PATH

    if _modelo is None:
        if not path.exists():
            raise FileNotFoundError(
                "No existe el modelo en %s. Corre primero "
                "'python -m src.training.train'" % path
            )
        _modelo = joblib.load(path)
        logger.info("Modelo cargado desde %s", path)

    return _modelo


def get_risk_level(probabilidad):
    """Traduce la probabilidad a LOW / MEDIUM / HIGH.

    Los cortes salen de config.py. La consigna pide que la salida derivada
    tenga umbrales definidos y justificados por el equipo.
    """
    if probabilidad < config.RISK_LOW_MAX:
        return "LOW"
    if probabilidad < config.RISK_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def predict(datos, threshold=None):
    """Predice sobre uno o varios clientes.

    datos puede ser un dict (un cliente) o un DataFrame (varios).
    Devuelve una lista de dicts con el formato que pide la consigna.
    """
    threshold = threshold if threshold is not None else config.DECISION_THRESHOLD
    modelo = load_model()

    if isinstance(datos, dict):
        datos = pd.DataFrame([datos])

    # Si viene el customerID se guarda aparte, porque no es un predictor
    ids = None
    if config.ID_COL in datos.columns:
        ids = datos[config.ID_COL].tolist()
        datos = datos.drop(columns=[config.ID_COL])

    probabilidades = modelo.predict_proba(datos)[:, 1]

    salidas = []
    for i, proba in enumerate(probabilidades):
        salida = {
            "prediction": "CHURN" if proba >= threshold else "NO_CHURN",
            "probability": round(float(proba), 4),
            "risk_level": get_risk_level(proba),
        }
        if ids:
            salida["customer_id"] = ids[i]
        salidas.append(salida)

    return salidas


def predict_batch(csv_path):
    """Predice sobre un CSV completo. Sirve para probar con scoring_batch."""
    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return predict(df)
