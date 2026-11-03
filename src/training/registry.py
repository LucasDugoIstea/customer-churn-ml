"""Conexion con MLflow y gobierno del modelo en el Model Registry.

Registrar la version que sale de un run, ponerle la metadata minima y un
alias. La version es fija y el alias se puede mover: la API carga el alias, asi
que cambiar de version (o volver atras) no obliga a tocar la API.
"""

import logging
import subprocess

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from src import config

logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name=None, artifacts_dir=None):
    """Apunta MLflow al servidor que corresponda y elige el experimento.

    En local (sin MLFLOW_TRACKING_URI) los artefactos van a mlartifacts/ en la
    raiz del proyecto, asi no dependen de desde que carpeta se corra. Los tests
    pasan otra carpeta para no tocar la del proyecto.
    """
    experiment_name = experiment_name or config.EXPERIMENT_NAME
    artifacts_dir = artifacts_dir or config.LOCAL_ARTIFACTS_DIR
    uri = config.tracking_uri()
    mlflow.set_tracking_uri(uri)

    if uri.startswith("sqlite:") and mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifacts_dir.as_uri())

    mlflow.set_experiment(experiment_name)
    logger.info("MLflow: %s | experimento: %s", uri, experiment_name)


def git_commit():
    """SHA del commit actual, para saber que codigo genero el modelo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.BASE_DIR,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        # Las salidas del propio pipeline (las reescribe dvc repro) no cuentan
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".",
             ":!reports/metrics.json", ":!dvc.lock"],
            cwd=config.BASE_DIR, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    # Si hay cambios sin commitear, el SHA solo no alcanza para reconstruir
    return sha + ("-dirty" if dirty else "")


def dataset_version():
    """md5 del dataset segun DVC: identifica exactamente que datos se usaron."""
    try:
        with open(config.RAW_DATA_DVC_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f)["outs"][0]["md5"]
    except (OSError, KeyError, IndexError, TypeError):
        return "unknown"


def register_candidate(run_id, metricas, model_name=None):
    """Registra el modelo del run como version nueva y le pone alias candidate.

    Devuelve el numero de version que asigno MLflow.
    """
    model_name = model_name or config.REGISTERED_MODEL
    client = MlflowClient()

    result = mlflow.register_model("runs:/%s/model" % run_id, model_name)
    version = result.version

    metadata = {
        "owner": "Lucas Dugo",
        "dataset_version": dataset_version(),
        "git_commit": git_commit(),
        "primary_metric": "roc_auc=%.4f" % metricas["roc_auc"],
        "recall_churn": "%.4f" % metricas["recall"],
        "threshold": str(config.DECISION_THRESHOLD),
        "positive_class": config.POSITIVE_CLASS,
        "schema_version": config.SCHEMA_VERSION,
    }
    for key, value in metadata.items():
        client.set_model_version_tag(model_name, version, key, value)

    client.set_registered_model_alias(model_name, "candidate", version)
    logger.info("Registrado %s version %s con alias candidate", model_name, version)
    return version


def promote(alias, version, decision_note, model_name=None):
    """Mueve un alias (por ejemplo production) a una version.

    La promocion es una decision explicita, no automatica: por eso es un paso
    aparte del entrenamiento y deja anotado por que se hizo.
    """
    model_name = model_name or config.REGISTERED_MODEL
    client = MlflowClient()

    if not str(version).isdigit():
        # version puede ser otro alias, por ejemplo "candidate"
        version = client.get_model_version_by_alias(model_name, version).version

    client.set_registered_model_alias(model_name, alias, version)
    client.set_model_version_tag(model_name, version, "decision_note", decision_note)
    logger.info("Alias %s -> %s version %s", alias, model_name, version)
    return version
