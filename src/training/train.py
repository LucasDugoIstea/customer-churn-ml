"""Entrenamiento y comparacion de modelos.

Se ejecuta desde consola con:

    python -m src.training.train

No depende de ningun notebook. Entrena los tres modelos que pide la consigna
(baseline, lineal y arbol), los compara, y serializa el mejor como artefacto.
"""

import argparse
import logging

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.data.load import load_data, split_data
from src.evaluation.metrics import evaluate_model, print_report, save_metrics
from src.features.build import build_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_models():
    """Devuelve los modelos a comparar.

    El baseline es obligatorio: sin un piso contra el cual comparar, no se
    puede decir si un modelo es bueno. DummyClassifier con la clase
    mayoritaria siempre predice 'no se va', que en este dataset ya acierta
    bastante justamente porque hay desbalance.
    """
    return {
        "baseline": DummyClassifier(
            strategy="most_frequent",
        ),
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


def main(metrica_seleccion="roc_auc"):
    """Flujo completo de entrenamiento."""
    logger.info("Training started")

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    resultados = {}
    pipelines = {}

    for nombre, model in get_models().items():
        logger.info("Entrenando: %s", nombre)
        pipeline = train_model(model, X_train, y_train)
        metricas = evaluate_model(pipeline, X_test, y_test)

        resultados[nombre] = metricas
        pipelines[nombre] = pipeline
        print_report(nombre, metricas)

    # El candidato se elige por la metrica que se pase por parametro.
    # Por defecto roc_auc, pero ojo: la decision final tiene que estar
    # justificada segun el negocio, no solo por el numero mas alto.
    mejor = max(resultados, key=lambda n: resultados[n][metrica_seleccion])
    logger.info("Modelo candidato: %s (%s = %.4f)",
                mejor, metrica_seleccion, resultados[mejor][metrica_seleccion])

    save_model(pipelines[mejor])

    reporte = {
        "modelo_candidato": mejor,
        "metrica_seleccion": metrica_seleccion,
        "resultados": resultados,
    }
    save_metrics(reporte, config.METRICS_PATH)

    logger.info("Training finished")
    return reporte


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena los modelos de churn")
    parser.add_argument(
        "--metrica",
        default="roc_auc",
        choices=["roc_auc", "f1", "recall", "precision", "accuracy"],
        help="Metrica con la que se elige el modelo candidato",
    )
    args = parser.parse_args()
    main(metrica_seleccion=args.metrica)
