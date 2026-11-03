"""Inferencia: usar el pipeline entrenado sobre clientes nuevos.

Este modulo no entrena nada ni sabe de HTTP. Es la logica que usa la API y que
se testea sola: de la probabilidad a la prediccion y al nivel de riesgo.
"""

import pandas as pd

from src import config
from src.evaluation.metrics import positive_proba


def get_risk_level(probabilidad):
    """Traduce la probabilidad a LOW / MEDIUM / HIGH.

    LOW termina en el umbral de decision, asi un cliente marcado CHURN nunca
    sale con riesgo LOW. Los cortes estan en params.yaml.
    """
    if probabilidad < config.RISK_LOW_MAX:
        return "LOW"
    if probabilidad < config.RISK_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def predict(pipeline, datos):
    """Predice sobre uno o varios clientes.

    datos puede ser un dict (un cliente) o un DataFrame (varios). Devuelve una
    lista con un diccionario por cliente, en tipos de Python (no de numpy),
    para que se puedan serializar a JSON.
    """
    if isinstance(datos, dict):
        datos = pd.DataFrame([datos])

    # El pipeline espera exactamente las columnas con las que se entreno, en
    # ese orden. Si falta alguna, falla aca con un KeyError claro.
    datos = datos[config.FEATURES]

    salidas = []
    for proba in positive_proba(pipeline, datos):
        proba = float(proba)
        salidas.append({
            "prediction": "CHURN" if proba >= config.DECISION_THRESHOLD else "NO_CHURN",
            "probability": round(proba, 4),
            "risk_level": get_risk_level(proba),
        })
    return salidas
