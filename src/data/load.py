"""Carga y particion del dataset.

Responsabilidad unica: leer archivos, validar que vengan las columnas que
esperamos y dividir en train/test. Nada de entrenar ni de transformar.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)


def load_data(path=None):
    """Lee el CSV historico y devuelve un DataFrame.

    Se hace una unica limpieza de tipo aca: TotalCharges viene como texto
    porque tiene celdas en blanco. Con errors='coerce' esos blancos quedan
    como NaN y despues los imputa el pipeline, que es donde corresponde.
    """
    path = path or config.RAW_DATA_PATH
    df = pd.read_csv(path)
    logger.info("Rows loaded: %d", len(df))

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    faltantes = df["TotalCharges"].isna().sum()
    if faltantes:
        logger.info("TotalCharges con faltantes: %d", faltantes)

    return df


def validate_columns(df, required_columns=None):
    """Falla temprano si el dataset no trae las columnas del contrato."""
    required_columns = required_columns or config.REQUIRED_COLUMNS
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: %s" % sorted(missing))


def split_data(df):
    """Separa X e y y hace la particion train/test.

    El customerID se saca porque es un identificador, no un predictor: el
    README del dataset lo pide explicitamente.

    Se usa stratify porque las clases estan desbalanceadas (hay bastante mas
    No que Yes) y sin eso el test podria quedar con una proporcion distinta a
    la del train.
    """
    validate_columns(df)

    X = df.drop(columns=[config.TARGET, config.ID_COL])
    y = (df[config.TARGET] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    logger.info("Train: %d filas | Test: %d filas", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
