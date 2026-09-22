"""Tests del pipeline de datos y entrenamiento.

Son pocos y basicos a proposito: verifican que las piezas que el resto del
proyecto da por sentadas efectivamente se cumplan. La bateria completa que
pide la consigna (contratos de la API, health check, entradas invalidas) va en
la entrega 2, cuando exista FastAPI.
"""

import pandas as pd
import pytest

from src import config
from src.data.load import load_data, split_data, validate_columns
from src.features.build import build_pipeline
from sklearn.dummy import DummyClassifier


def test_load_data_trae_filas():
    df = load_data()
    assert len(df) > 0
    assert config.TARGET in df.columns


def test_totalcharges_queda_numerica():
    """TotalCharges viene como texto con blancos; tiene que salir numerica."""
    df = load_data()
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])


def test_validate_columns_falla_si_falta_una():
    df = pd.DataFrame({"tenure": [1, 2]})
    with pytest.raises(ValueError):
        validate_columns(df)


def test_split_saca_el_id():
    """customerID no puede quedar como predictor."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    assert config.ID_COL not in X_train.columns
    assert config.TARGET not in X_train.columns


def test_split_es_reproducible():
    """Con la misma semilla, la particion tiene que dar igual."""
    df = load_data()
    X1, _, _, _ = split_data(df)
    X2, _, _, _ = split_data(df)
    assert list(X1.index) == list(X2.index)


def test_split_respeta_la_proporcion():
    """stratify tiene que mantener el balance de clases entre train y test."""
    df = load_data()
    _, _, y_train, y_test = split_data(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_pipeline_entrena_y_predice():
    """El pipeline completo tiene que poder entrenar y devolver predicciones."""
    df = load_data()
    X_train, X_test, y_train, _ = split_data(df)

    pipeline = build_pipeline(DummyClassifier(strategy="most_frequent"))
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)


def test_pipeline_aguanta_categoria_desconocida():
    """Una categoria que no estaba en el entrenamiento no debe romper nada.

    Es el caso que cubre handle_unknown='ignore' en el OneHotEncoder.
    """
    df = load_data()
    X_train, X_test, y_train, _ = split_data(df)

    pipeline = build_pipeline(DummyClassifier(strategy="most_frequent"))
    pipeline.fit(X_train, y_train)

    raro = X_test.head(1).copy()
    raro["Contract"] = "Contrato que no existe"

    preds = pipeline.predict(raro)
    assert len(preds) == 1
