"""Particion y pipeline de preprocesamiento.

El modelo casi nunca recibe las columnas crudas: si la cadena de imputacion,
encoding y escalado se rompe, el modelo deja de funcionar aunque el archivo
exista y este bien guardado.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.data.load import split_data
from src.features.build import build_pipeline


def test_split_saca_el_id_y_el_target(raw_df):
    """customerID no puede quedar como predictor."""
    X_train, X_test, _, _ = split_data(raw_df)
    assert config.ID_COL not in X_train.columns
    assert config.TARGET not in X_train.columns


def test_split_es_reproducible(raw_df):
    """Con la misma semilla, la particion tiene que dar igual."""
    X1, _, _, _ = split_data(raw_df)
    X2, _, _, _ = split_data(raw_df)
    assert list(X1.index) == list(X2.index)


def test_split_respeta_la_proporcion(raw_df):
    """stratify tiene que mantener el balance de clases entre train y test."""
    _, _, y_train, y_test = split_data(raw_df)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_pipeline_tiene_los_pasos_esperados():
    pipeline = build_pipeline(LogisticRegression())
    assert list(pipeline.named_steps) == ["preprocess", "model"]


def test_preprocessor_handles_missing_values(raw_df):
    """Ajusta y predice con faltantes en TotalCharges; salen tantas filas como entran."""
    X_train, _, y_train, _ = split_data(raw_df)
    assert X_train["TotalCharges"].isna().any()

    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(X_train, y_train)
    assert len(pipeline.predict(X_train.head(5))) == 5


def test_pipeline_aguanta_categoria_desconocida(raw_df):
    """Una categoria que no estaba en el entrenamiento no rompe la inferencia
    (handle_unknown='ignore' en el OneHotEncoder). La API igual la rechaza
    antes con Pydantic; esta es la segunda red."""
    X_train, X_test, y_train, _ = split_data(raw_df)
    pipeline = build_pipeline(DummyClassifier(strategy="most_frequent"))
    pipeline.fit(X_train, y_train)

    raro = X_test.head(1).copy()
    raro["Contract"] = "Contrato que no existe"
    assert len(pipeline.predict(raro)) == 1
