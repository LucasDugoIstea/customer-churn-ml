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


def evaluate_model(pipeline, X_test, y_test):
    """Evalua el pipeline sobre el test y devuelve un diccionario de metricas.

    roc_auc se calcula sobre la probabilidad y no sobre la clase predicha,
    porque mide que tan bien ordena el modelo, independientemente del umbral.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

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


def save_metrics(metricas, path):
    """Guarda las metricas en JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    logger.info("Metricas guardadas en %s", path)
