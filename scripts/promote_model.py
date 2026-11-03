"""Promueve un modelo del registry: mueve el alias production.

    python -m scripts.promote_model

Que version y por que se lee de params.yaml (seccion promocion), asi la
decision queda versionada en Git junto con el codigo. Para hacer rollback se
pone en version el numero de la version anterior y se vuelve a correr.

Es un paso aparte del entrenamiento a proposito: pasar a produccion tiene que
ser una decision explicita, no algo automatico.
"""

import logging

from src import config
from src.training.registry import promote, setup_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    setup_mlflow()
    p = config.PARAMS["promocion"]
    version = promote(p["alias"], p["version"], p["decision_note"])
    logger.info("Listo: la API va a cargar %s@%s (version %s)",
                config.REGISTERED_MODEL, p["alias"], version)


if __name__ == "__main__":
    main()
