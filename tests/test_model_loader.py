"""Carga del modelo desde el Model Registry.

Parece simple pero es critico: ruta equivocada, alias inexistente, version de
libreria incompatible. Se prueba contra un registry temporal con un modelo
entrenado y promovido con el mismo codigo del proyecto.
"""

import pytest

from src.inference.model_loader import load_model, resolve_version


def test_model_loads(model, registered_model):
    assert hasattr(model.pipeline, "predict_proba")
    assert model.version == registered_model["version"]
    assert model.uri.endswith("@production")


def test_resolve_version_por_alias(registered_model):
    _, version = resolve_version(registered_model["uri"])
    assert version == registered_model["version"]


def test_resolve_version_por_numero():
    assert resolve_version("models:/CustomerChurnModel/3") == ("CustomerChurnModel", "3")


def test_alias_inexistente_falla_al_cargar(registered_model):
    """Si el alias no existe, tiene que fallar al cargar, no mas adelante."""
    with pytest.raises(Exception):
        load_model("models:/CustomerChurnModel@no-existe")
