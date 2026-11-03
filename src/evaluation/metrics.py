"""Calculo de metricas.

La consigna dice que accuracy no se acepta como unica metrica, asi que se
calculan tambien precision, recall, f1, roc_auc y la matriz de confusion.
"""

import json
import logging

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(pipeline, X_test, y_test, threshold=0.5):
    """Evalua el pipeline sobre el test y devuelve un diccionario de metricas.

    La clase predicha sale de comparar la probabilidad con el umbral de
    decision (el mismo que usa la API), no del predict() de sklearn, que
    siempre corta en 0.5.

    roc_auc se calcula sobre la probabilidad y no sobre la clase predicha,
    porque mide que tan bien ordena el modelo, independientemente del umbral.
    """
    y_proba = positive_proba(pipeline, X_test)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metricas = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    return metricas


def positive_proba(pipeline, X):
    """Probabilidad de la clase positiva (churn = 1).

    No se asume que es la columna 1: se busca en classes_, como pide la slide
    de serving.
    """
    columna = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(X)[:, columna]


def save_metrics(metricas, path):
    """Guarda las metricas en JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    logger.info("Metricas guardadas en %s", path)
