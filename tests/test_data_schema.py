"""Contrato minimo de datos: columnas y tipos.

Si el dataset cambia o una columna llega con otro nombre, tiene que fallar
aca, al principio, y con un mensaje claro.
"""

import pandas as pd
import pytest

from src import config
from src.data.load import load_data, validate_columns


def test_required_columns_exist(raw_df):
    assert set(config.REQUIRED_COLUMNS).issubset(raw_df.columns)


def test_target_exists_in_training(raw_df):
    assert config.TARGET in raw_df.columns
    assert set(raw_df[config.TARGET].unique()) <= {"Yes", "No"}


def test_totalcharges_queda_numerica(raw_df):
    """TotalCharges viene como texto con blancos; tiene que salir numerica,
    con los blancos como NaN para que los impute el pipeline."""
    assert pd.api.types.is_numeric_dtype(raw_df["TotalCharges"])
    assert raw_df["TotalCharges"].isna().any()


def test_validate_columns_falla_si_falta_una():
    df = pd.DataFrame({"tenure": [1, 2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(df)


@pytest.mark.skipif(not config.RAW_DATA_PATH.exists(), reason="sin dvc pull no esta el dataset real")
def test_dataset_real_cumple_el_contrato():
    df = load_data()
    validate_columns(df)
    assert len(df) > 0
