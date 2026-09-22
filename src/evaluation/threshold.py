"""Analisis del umbral de decision.

El modelo devuelve una probabilidad. Convertirla en CHURN / NO_CHURN necesita
un corte, y 0.50 es solo el corte por defecto, no el correcto.

La consigna pide discutir que impacto tiene un falso negativo (decir que un
cliente se queda cuando en realidad se iba). Este script muestra como cambian
las metricas al mover el umbral, para poder tomar esa decision con datos y no
a ojo.

Se ejecuta con:

    python -m src.evaluation.threshold
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.load import load_data, split_data
from src.inference.predict import load_model


def tabla_por_umbral(pipeline, X_test, y_test, umbrales=None):
    """Devuelve las metricas para cada umbral candidato."""
    if umbrales is None:
        umbrales = np.arange(0.20, 0.75, 0.05)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    filas = []
    for u in umbrales:
        y_pred = (y_proba >= u).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        filas.append({
            "umbral": round(float(u), 2),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
            "falsos_negativos": int(fn),
            "falsos_positivos": int(fp),
        })

    return filas


def main():
    df = load_data()
    _, X_test, _, y_test = split_data(df)
    pipeline = load_model()

    filas = tabla_por_umbral(pipeline, X_test, y_test)

    print("\nImpacto del umbral de decision")
    print("=" * 72)
    print("%-8s %-10s %-9s %-7s %-14s %s" % (
        "umbral", "precision", "recall", "f1", "falsos neg.", "falsos pos."))
    print("-" * 72)
    for f in filas:
        print("%-8.2f %-10.3f %-9.3f %-7.3f %-14d %d" % (
            f["umbral"], f["precision"], f["recall"], f["f1"],
            f["falsos_negativos"], f["falsos_positivos"]))
    print("=" * 72)
    print("\nBajar el umbral detecta mas clientes en riesgo (menos falsos")
    print("negativos) pero molesta a mas clientes que no se iban a ir")
    print("(mas falsos positivos). Donde cortar depende de cuanto cuesta")
    print("cada error en el negocio.\n")


if __name__ == "__main__":
    main()
