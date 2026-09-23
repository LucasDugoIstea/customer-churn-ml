"""Entrenamiento y comparacion de modelos.

Se ejecuta desde consola con:

    python -m src.training.train
"""

import logging

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.data.load import load_data, split_data
from src.evaluation.metrics import evaluate_model, save_metrics
from src.features.build import build_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_models():
    """Los tres modelos a comparar: baseline, lineal y arbol."""
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        ),
        "forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=config.RANDOM_STATE,
        ),
    }


def train_model(model, X_train, y_train):
    """Entrena un pipeline completo y lo devuelve."""
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline, path=None):
    """Serializa el pipeline entero, no solo el estimador."""
    path = path or config.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Model saved: %s", path)


def main():
    logger.info("Training started")

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    resultados = {}
    pipelines = {}

    for nombre, model in get_models().items():
        pipeline = train_model(model, X_train, y_train)
        metricas = evaluate_model(pipeline, X_test, y_test)

        resultados[nombre] = metricas
        pipelines[nombre] = pipeline
        logger.info("%s: roc_auc=%.4f recall=%.4f precision=%.4f f1=%.4f",
                    nombre, metricas["roc_auc"], metricas["recall"],
                    metricas["precision"], metricas["f1"])

    # El candidato se elige por ROC-AUC porque mide que tan bien ordena el
    # modelo, sin depender del umbral que se use despues.
    mejor = max(resultados, key=lambda n: resultados[n]["roc_auc"])
    logger.info("Modelo candidato: %s (roc_auc = %.4f)",
                mejor, resultados[mejor]["roc_auc"])

    save_model(pipelines[mejor])

    reporte = {"modelo_candidato": mejor, "resultados": resultados}
    save_metrics(reporte, config.METRICS_PATH)

    logger.info("Training finished")


if __name__ == "__main__":
    main()
