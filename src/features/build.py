"""Construccion del preprocesamiento y del pipeline.

Todo el preprocesamiento vive adentro de un Pipeline de scikit-learn. La razon
es la de la clase 3: si el preprocesamiento queda afuera, entrenamiento e
inferencia pueden terminar aplicando transformaciones distintas, y eso es como
predecir con otro modelo.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor():
    """Arma el ColumnTransformer.

    Las numericas y las categoricas no se procesan igual, por eso cada grupo
    tiene su propio pipeline y despues se unen.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown='ignore' para que una categoria que no estaba en el
        # entrenamiento no tire abajo el servicio en produccion
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, config.NUMERIC_FEATURES),
        ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
    ])

    return preprocessor


def build_pipeline(model):
    """Devuelve el pipeline completo: preprocesamiento + estimador.

    Se recibe el modelo por parametro para poder reusar la misma funcion con
    el baseline, la regresion logistica y el random forest.
    """
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
