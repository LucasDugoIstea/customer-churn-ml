"""Entrenamiento, comparacion y registro de modelos.

Se ejecuta desde consola con:

    python -m src.training.train

Cada modelo de params.yaml es un run de MLflow con sus parametros, metricas,
artefactos y el pipeline completo (preprocesamiento + estimador). Al final el
mejor queda registrado en el Model Registry con el alias candidate. Pasarlo a
production es un paso aparte: scripts/promote_model.py.
"""

import logging

import mlflow
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.data.load import load_data, split_data
from src.evaluation.metrics import evaluate_model, positive_proba, save_metrics
from src.features.build import build_pipeline
from src.training.registry import dataset_version, git_commit, register_candidate, setup_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# De nombre de params.yaml a clase de scikit-learn
ALGORITHMS = {
    "dummy": DummyClassifier,
    "logistic": LogisticRegression,
    "forest": RandomForestClassifier,
}

# Version del codigo del pipeline. Se sube si cambia el preprocesamiento.
PIPELINE_VERSION = "2"

# MLflow 3 guarda los modelos de sklearn con skops, que a diferencia de pickle
# no ejecuta codigo arbitrario al cargar. Solo deja pasar tipos conocidos; el
# ColumnTransformer y los arboles del random forest usan estos, que hay que
# declarar a mano.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.compose._column_transformer._RemainderColsList",
    "sklearn.tree._tree.Tree",
]


def build_model(model_type, params):
    """Instancia el estimador; el random_state sale de params.yaml."""
    cls = ALGORITHMS[model_type]
    if model_type != "dummy":
        params = dict(params, random_state=config.RANDOM_STATE)
    return cls(**params)


def train_model(model, X_train, y_train):
    """Entrena un pipeline completo y lo devuelve."""
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)
    return pipeline


def run_experiment(spec, X_train, X_test, y_train, y_test):
    """Un run de MLflow: entrena, evalua y registra todo lo que explica el resultado."""
    with mlflow.start_run(run_name=spec["run_name"]) as run:
        model = build_model(spec["type"], spec["params"])
        pipeline = train_model(model, X_train, y_train)
        metricas = evaluate_model(pipeline, X_test, y_test, config.DECISION_THRESHOLD)

        mlflow.set_tags({
            "git_commit": git_commit(),
            "dataset_version": dataset_version(),
            "pipeline_version": PIPELINE_VERSION,
            "model_type": spec["type"],
        })
        mlflow.log_params({
            "model_type": spec["type"],
            **spec["params"],
            "random_state": config.RANDOM_STATE,
            "test_size": config.TEST_SIZE,
            "threshold": config.DECISION_THRESHOLD,
        })
        mlflow.log_metrics({
            k: v for k, v in metricas.items() if k != "confusion_matrix"
        })
        mlflow.log_dict(metricas["confusion_matrix"], "confusion_matrix.json")
        mlflow.log_dict({
            "schema_version": config.SCHEMA_VERSION,
            "numeric": config.NUMERIC_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
        }, "feature_schema.json")

        # Se registra el pipeline entero, con la firma de entrada/salida y un
        # ejemplo valido: es el contrato del modelo. Las numericas se declaran
        # float porque un entero no puede representar un faltante.
        ejemplo = X_train.head(3).astype({c: "float64" for c in config.NUMERIC_FEATURES})
        firma = infer_signature(ejemplo, positive_proba(pipeline, ejemplo))
        mlflow.sklearn.log_model(
            pipeline, name="model", signature=firma, input_example=ejemplo,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        logger.info("%-15s roc_auc=%.4f recall=%.4f precision=%.4f f1=%.4f",
                    spec["run_name"], metricas["roc_auc"], metricas["recall"],
                    metricas["precision"], metricas["f1"])
        return run.info.run_id, metricas


def main():
    logger.info("Training started")
    setup_mlflow()

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    resultados = {}
    run_ids = {}
    for spec in config.PARAMS["models"]:
        run_id, metricas = run_experiment(spec, X_train, X_test, y_train, y_test)
        resultados[spec["run_name"]] = metricas
        run_ids[spec["run_name"]] = run_id

    # El baseline no compite: es el piso contra el que se compara.
    metrica = config.PARAMS["seleccion"]["metrica"]
    competidores = [n for n in resultados if not n.startswith("baseline")]
    mejor = max(competidores, key=lambda n: resultados[n][metrica])
    logger.info("Modelo candidato: %s (%s = %.4f)", mejor, metrica, resultados[mejor][metrica])

    version = register_candidate(run_ids[mejor], resultados[mejor])

    reporte = {
        "modelo_candidato": mejor,
        "run_id": run_ids[mejor],
        "registered_model": config.REGISTERED_MODEL,
        "version": version,
        "threshold": config.DECISION_THRESHOLD,
        "resultados": resultados,
    }
    save_metrics(reporte, config.METRICS_PATH)

    logger.info("Training finished")


if __name__ == "__main__":
    main()
