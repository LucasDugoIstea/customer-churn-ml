"""Carga del modelo desde el Model Registry de MLflow.

La API no carga un archivo de models/: carga el alias del registry, por
defecto models:/CustomerChurnModel@production. Si production pasa de la
version 1 a la 2, el codigo de la API no cambia.
"""

import logging
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from src import config

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """El pipeline y los datos para saber que version esta respondiendo."""

    pipeline: object
    uri: str
    name: str
    version: str


def resolve_version(model_uri):
    """Traduce models:/nombre@alias o models:/nombre/N al numero de version."""
    referencia = model_uri.removeprefix("models:/")
    if "@" in referencia:
        name, alias = referencia.split("@", 1)
        version = MlflowClient().get_model_version_by_alias(name, alias).version
        return name, str(version)
    if "/" in referencia:
        name, version = referencia.split("/", 1)
        return name, version
    return referencia, "unknown"


def load_model(model_uri=None):
    """Carga el pipeline completo (preprocesamiento + estimador).

    Se usa mlflow.sklearn y no mlflow.pyfunc porque la API necesita
    predict_proba, y el modelo pyfunc solo expone predict.

    Si el modelo no existe o no carga, falla aca y no en el primer request.
    """
    model_uri = model_uri or config.model_uri()
    mlflow.set_tracking_uri(config.tracking_uri())

    pipeline = mlflow.sklearn.load_model(model_uri)
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("El modelo %s no tiene predict_proba" % model_uri)

    name, version = resolve_version(model_uri)
    logger.info("Modelo cargado: %s (version %s)", model_uri, version)
    return LoadedModel(pipeline=pipeline, uri=model_uri, name=name, version=version)
